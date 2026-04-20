"""InSceneDti: a kwneuro.Dti backed by a vtkMRMLDiffusionTensorVolumeNode.

kwneuro (via dipy) stores diffusion tensors in the "lower triangular"
form — 6 components per voxel ordered `(Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)` —
because that's what `dipy.reconst.dti.lower_triangular` returns.

Slicer's DTI node stores the full symmetric 3x3 tensor (9 row-major
components) via VTK's dedicated `PointData.Tensors` attribute.
`slicer.util.arrayFromVolume` on a DTI node reads that and reshapes to
`(kji..., 3, 3)` (see `Base/Python/slicer/util.py:1806-1807`).

`InSceneDti` is a kwneuro.Dti subclass. The inherited `.volume` field is
an `InMemoryVolumeResource` holding the 6-LT representation, built at
construction from the scene node's 9-component tensor data. That volume
is therefore a snapshot: if the underlying scene node's tensor data
changes afterwards, the `.volume` on this `InSceneDti` does not update.
Rewrap the node to refresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

import slicer

from kwneuro.dti import Dti
from kwneuro.resource import InMemoryVolumeResource

from kwneuro_slicer_bridge.conversions import (
    affine_to_ijk_to_ras_matrix,
    ijk_to_ras_matrix_to_affine,
)


# Index table mapping the dipy 6-element lower-triangular ordering
# (xx, xy, yy, xz, yz, zz) into a full symmetric 3x3.
_LT_TO_FULL_INDEX = np.array(
    [
        [0, 1, 3],  # Dxx Dxy Dxz
        [1, 2, 4],  # Dyx=Dxy Dyy Dyz
        [3, 4, 5],  # Dzx=Dxz Dzy=Dyz Dzz
    ],
    dtype=np.int64,
)


def _lower_triangular_to_full(lt: NDArray[np.floating]) -> NDArray[np.floating]:
    """Expand a (..., 6) dipy-lower-triangular array to (..., 3, 3)."""
    if lt.shape[-1] != 6:
        msg = f"last axis must be length 6, got {lt.shape[-1]}"
        raise ValueError(msg)
    return lt[..., _LT_TO_FULL_INDEX]


def _full_to_lower_triangular(full: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compress a (..., 3, 3) array to dipy-lower-triangular (..., 6)."""
    if full.shape[-2:] != (3, 3):
        msg = f"last two axes must be (3, 3), got {full.shape[-2:]}"
        raise ValueError(msg)
    return np.stack(
        [
            full[..., 0, 0],  # Dxx
            full[..., 0, 1],  # Dxy
            full[..., 1, 1],  # Dyy
            full[..., 0, 2],  # Dxz
            full[..., 1, 2],  # Dyz
            full[..., 2, 2],  # Dzz
        ],
        axis=-1,
    )


def _numpy_tensor_array_to_vtk_image(full: NDArray[np.floating]) -> Any:
    """Convert an `(nx, ny, nz, 3, 3)` tensor array to a vtkImageData
    whose PointData.Tensors holds 9 components per voxel interleaved.
    """
    import vtk
    from vtk.util import numpy_support

    if full.ndim != 5 or full.shape[-2:] != (3, 3):
        msg = f"Expected (nx, ny, nz, 3, 3); got {full.shape}"
        raise ValueError(msg)
    nx, ny, nz = full.shape[:3]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)

    full_9 = full.reshape(nx, ny, nz, 9)
    transposed = np.asfortranarray(full_9.transpose(3, 0, 1, 2))  # (9, nx, ny, nz)
    flat = transposed.flatten(order="F")
    vtk_array = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetNumberOfComponents(9)
    image_data.GetPointData().SetTensors(vtk_array)
    return image_data


def _read_full_tensor_from_node(node: Any) -> NDArray[np.floating]:
    """Read a DTI node's tensors as an `(nx, ny, nz, 3, 3)` array."""
    from vtk.util import numpy_support

    image_data = node.GetImageData()
    tensors = image_data.GetPointData().GetTensors()
    if tensors is None:
        msg = (
            f"DTI node {node.GetID()!r} has no PointData.Tensors attached; "
            f"wrap a node created via InSceneDti.from_dti."
        )
        raise ValueError(msg)
    nx, ny, nz = image_data.GetDimensions()
    raw = numpy_support.vtk_to_numpy(tensors)  # (N, 9) C-order
    stage1 = raw.reshape(nz, ny, nx, 9)
    stage2 = stage1.reshape(nz, ny, nx, 3, 3)
    return np.ascontiguousarray(stage2.transpose(2, 1, 0, 3, 4))


def _read_affine_from_node(node: Any) -> NDArray[np.floating]:
    import vtk

    ijk_to_ras = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(ijk_to_ras)
    return ijk_to_ras_matrix_to_affine(ijk_to_ras)


@dataclass(init=False)
class InSceneDti(Dti):
    """A kwneuro.Dti whose tensor data lives in a vtkMRMLDiffusionTensorVolumeNode.

    Is-a `Dti`. The inherited `.volume` is an `InMemoryVolumeResource`
    snapshot of the node's tensor data (converted from 9-component full
    3x3 form to 6-element lower-triangular form at construction).
    """

    node_id: str
    _node: Any = field(default=None, repr=False, compare=False)

    def __init__(self, node_id: str, _node: Any = None) -> None:
        self._node = _node if _node is not None else slicer.mrmlScene.GetNodeByID(node_id)
        if self._node is None:
            msg = f"MRML node {node_id!r} not found in the scene"
            raise ValueError(msg)
        self.node_id = node_id
        full = _read_full_tensor_from_node(self._node)
        lt = _full_to_lower_triangular(full).astype(np.float32, copy=False)
        volume = InMemoryVolumeResource(
            array=lt,
            affine=_read_affine_from_node(self._node),
            metadata={
                "slicer_node_id": self.node_id,
                "slicer_node_name": self._node.GetName(),
            },
        )
        super().__init__(volume=volume)

    # --- Accessors for the scene-side extras ---

    def get_node(self) -> Any:
        return self._node

    def get_tensor_array(self) -> NDArray[np.floating]:
        """Return a fresh read of the full `(nx, ny, nz, 3, 3)` tensor array.

        Unlike `.volume` (which is a snapshot from construction), this
        reads current node state each call.
        """
        return _read_full_tensor_from_node(self.get_node())

    # --- Conversions ---

    def to_in_memory(self) -> Dti:
        """Return a plain `kwneuro.Dti` detached from the scene.

        The `.volume` on this InSceneDti is already an `InMemoryVolumeResource`
        snapshot, so this mostly rewraps into a plain `Dti` without the
        scene-side `node_id` field.
        """
        return Dti(volume=self.volume)

    # --- Factories ---

    @staticmethod
    def from_dti(dti: Dti, name: str = "kwneuro_dti", show: bool = False) -> InSceneDti:
        """Push a kwneuro.Dti into the scene as a vtkMRMLDiffusionTensorVolumeNode."""
        lt = dti.volume.get_array()
        if lt.ndim != 4 or lt.shape[-1] != 6:
            msg = f"Dti volume must be 4D with last axis length 6, got {lt.shape}"
            raise ValueError(msg)
        full = _lower_triangular_to_full(lt).astype(np.float32, copy=False)
        image_data = _numpy_tensor_array_to_vtk_image(full)

        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDiffusionTensorVolumeNode", name)
        node.SetIJKToRASMatrix(affine_to_ijk_to_ras_matrix(dti.volume.get_affine()))
        node.SetAndObserveImageData(image_data)

        if show:
            slicer.util.setSliceViewerLayers(background=node)

        return InSceneDti(node_id=node.GetID(), _node=node)

    @staticmethod
    def from_node(node: Any) -> InSceneDti:
        node_id = node.GetID()
        if node_id is None:
            msg = "Node must be added to the scene before wrapping"
            raise ValueError(msg)
        return InSceneDti(node_id=node_id, _node=node)
