"""InSceneDwi: a kwneuro.Dwi backed by a vtkMRMLDiffusionWeightedVolumeNode.

`slicer.util.loadVolume` silently drops the 4th dimension when loading a
4D NIfTI because it defaults to `vtkMRMLScalarVolumeNode`. This class
targets `vtkMRMLDiffusionWeightedVolumeNode` instead, which preserves
the gradient dimension and exposes bval / bvec / measurement frame as
first-class node attributes.

Inheriting from `kwneuro.dwi.Dwi` means any pipeline function that
takes a `Dwi` accepts an `InSceneDwi` directly — no conversion step.
The volume is a live `InSceneVolumeResource` view of the node; bval
and bvec are snapshots taken at construction (copied from the node's
gradient/b-value attributes into in-memory resources).

The measurement frame defaults to identity. kwneuro's `Dwi` does not
model a gradient frame explicitly (dipy convention: gradients are
already in the scan frame), so identity round-trips correctly.
DICOM-origin users who care about glyph-space orientation can pass
an explicit `measurement_frame=` to `from_dwi`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import slicer

from kwneuro.dwi import Dwi
from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
from kwneuro.resource import InMemoryBvalResource, InMemoryBvecResource

from kwneuro_slicer_bridge.conversions import (
    affine_to_ijk_to_ras_matrix,
    numpy_to_vtk_image,
)
from kwneuro_slicer_bridge.volume import InSceneVolumeResource


def _read_bval_from_node(node: Any) -> NDArray[np.floating]:
    from vtk.util import numpy_support
    vtk_bvalues = node.GetBValues()
    if vtk_bvalues is None:
        msg = (
            f"DWI node {node.GetID()!r} has no b-values attached; "
            f"wrap a node created via SetBValues / InSceneDwi.from_dwi."
        )
        raise ValueError(msg)
    return np.asarray(numpy_support.vtk_to_numpy(vtk_bvalues), dtype=np.float64).copy()


def _read_bvec_from_node(node: Any) -> NDArray[np.floating]:
    from vtk.util import numpy_support
    vtk_gradients = node.GetDiffusionGradients()
    if vtk_gradients is None:
        msg = (
            f"DWI node {node.GetID()!r} has no diffusion gradients attached; "
            f"wrap a node created via SetDiffusionGradients / InSceneDwi.from_dwi."
        )
        raise ValueError(msg)
    arr = numpy_support.vtk_to_numpy(vtk_gradients)
    return np.asarray(arr, dtype=np.float64).copy()


@dataclass(init=False)
class InSceneDwi(Dwi):
    """A kwneuro.Dwi whose data lives in a vtkMRMLDiffusionWeightedVolumeNode.

    Is-a `Dwi`, so it drops directly into any pipeline function that takes
    a `Dwi`. The inherited `volume` field is a live `InSceneVolumeResource`
    view; `bval` and `bvec` are snapshots taken from the node at
    construction.

    Parameters
    ----------
    node_id : str
        MRML ID of an existing vtkMRMLDiffusionWeightedVolumeNode in the
        scene.
    """

    node_id: str
    """MRML ID of the backing vtkMRMLDiffusionWeightedVolumeNode."""

    _node: Any = field(default=None, repr=False, compare=False)

    def __init__(self, node_id: str, _node: Any = None) -> None:
        self._node = _node if _node is not None else slicer.mrmlScene.GetNodeByID(node_id)
        if self._node is None:
            msg = f"MRML node {node_id!r} not found in the scene"
            raise ValueError(msg)
        self.node_id = node_id
        super().__init__(
            volume=InSceneVolumeResource(node_id=node_id, _node=self._node),
            bval=InMemoryBvalResource(array=_read_bval_from_node(self._node)),
            bvec=InMemoryBvecResource(array=_read_bvec_from_node(self._node)),
        )

    # --- Accessors for the scene-side extras ---

    def get_node(self) -> Any:
        return self._node

    # --- Conversions ---

    def to_in_memory(self) -> Dwi:
        """Return a plain `kwneuro.Dwi` fully detached from the scene.

        Copies the volume data into an `InMemoryVolumeResource`; bval
        and bvec are already in-memory (they were snapshotted at
        construction) so those are shared.
        """
        return Dwi(
            volume=self.volume.to_in_memory(),  # type: ignore[attr-defined]
            bval=self.bval,
            bvec=self.bvec,
        )

    # --- Factories ---

    @staticmethod
    def from_dwi(
        dwi: Dwi,
        name: str = "kwneuro_dwi",
        show: bool = False,
        measurement_frame: NDArray[np.floating] | None = None,
    ) -> InSceneDwi:
        """Push a kwneuro.Dwi into the scene as a vtkMRMLDiffusionWeightedVolumeNode.

        :param measurement_frame: Optional 3x3 matrix describing how the
            gradient frame relates to RAS. Defaults to identity.
        """
        import vtk
        from vtk.util import numpy_support

        loaded = dwi.load() if not dwi.volume.is_loaded else dwi
        volume_array = loaded.volume.get_array()
        if volume_array.ndim != 4:
            msg = f"Dwi volume must be 4D, got shape {volume_array.shape}"
            raise ValueError(msg)

        node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLDiffusionWeightedVolumeNode", name,
        )
        node.SetIJKToRASMatrix(affine_to_ijk_to_ras_matrix(loaded.volume.get_affine()))
        node.SetAndObserveImageData(numpy_to_vtk_image(volume_array))

        # Go through the abstract BvalResource/BvecResource interface
        # (`.get()`) rather than `.array` so the types line up with
        # `Dwi.bval: BvalResource` / `Dwi.bvec: BvecResource`. `.array`
        # only lives on the InMemory concrete subclasses.
        bval_array = np.asarray(loaded.bval.get(), dtype=np.float64)
        bvec_array = np.asarray(loaded.bvec.get(), dtype=np.float64)
        n_gradients = bval_array.shape[0]
        node.SetNumberOfGradients(n_gradients)

        vtk_gradients = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(bvec_array), deep=True, array_type=vtk.VTK_DOUBLE,
        )
        node.SetDiffusionGradients(vtk_gradients)

        vtk_bvalues = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(bval_array), deep=True, array_type=vtk.VTK_DOUBLE,
        )
        node.SetBValues(vtk_bvalues)

        frame = np.eye(3) if measurement_frame is None else np.asarray(
            measurement_frame, dtype=np.float64,
        )
        if frame.shape != (3, 3):
            msg = f"measurement_frame must be 3x3, got shape {frame.shape}"
            raise ValueError(msg)
        mat = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                mat.SetElement(i, j, frame[i, j])
        node.SetMeasurementFrameMatrix(mat)

        if show:
            slicer.util.setSliceViewerLayers(background=node)

        return InSceneDwi(node_id=node.GetID(), _node=node)

    @staticmethod
    def from_nifti_path(
        volume_path: Path,
        bval_path: Path,
        bvec_path: Path,
        name: str = "kwneuro_dwi",
        show: bool = False,
    ) -> InSceneDwi:
        """Load a DWI from NIfTI + FSL bval/bvec into the scene.

        Bypasses `slicer.util.loadVolume`, which would silently drop the
        4th dimension; routes via kwneuro's on-disk resources.
        """
        dwi = Dwi(
            NiftiVolumeResource(Path(volume_path)),
            FslBvalResource(Path(bval_path)),
            FslBvecResource(Path(bvec_path)),
        ).load()
        return InSceneDwi.from_dwi(dwi, name=name, show=show)

    @staticmethod
    def from_node(node: Any) -> InSceneDwi:
        """Wrap an existing vtkMRMLDiffusionWeightedVolumeNode."""
        node_id = node.GetID()
        if node_id is None:
            msg = "Node must be added to the scene before wrapping"
            raise ValueError(msg)
        return InSceneDwi(node_id=node_id, _node=node)
