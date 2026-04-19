"""Phase 0 SlicerVolumeResource — adapted from the starter gist at
https://gist.github.com/ebrahimebrahim/57d4f7f2999b29138a9ec4146febb7f3.

Shared between experiments so we only keep one copy. This file is
deliberately kept close to the gist; it is NOT the final Phase 1 module —
Phase 1 will promote (and iterate on) this design as part of the real
`KWNeuro` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

import slicer  # Must be imported inside Slicer Python.

from kwneuro.resource import InMemoryVolumeResource, VolumeResource


def _numpy_to_vtk_image(array: NDArray[np.number]) -> Any:
    import vtk
    from vtk.util import numpy_support

    array = np.asfortranarray(array)
    image_data = vtk.vtkImageData()
    if array.ndim not in (3, 4):
        raise ValueError(f"Array must be 3D or 4D, got {array.ndim}D")
    image_data.SetDimensions(array.shape[0], array.shape[1], array.shape[2])
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    if array.ndim == 4:
        flat_array = array.reshape(-1, order="F")
        n_components = array.shape[3]
    else:
        flat_array = array.flatten(order="F")
        n_components = 1
    vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetNumberOfComponents(n_components)
    image_data.GetPointData().SetScalars(vtk_array)
    return image_data


def _vtk_image_to_numpy(image_data: Any) -> NDArray[np.number]:
    from vtk.util import numpy_support
    dims = image_data.GetDimensions()
    scalars = image_data.GetPointData().GetScalars()
    n_components = scalars.GetNumberOfComponents()
    array = numpy_support.vtk_to_numpy(scalars)
    if n_components > 1:
        array = array.reshape(dims[0], dims[1], dims[2], n_components, order="F")
    else:
        array = array.reshape(dims[0], dims[1], dims[2], order="F")
    return array


def _affine_to_ijk_to_ras_matrix(affine: NDArray[np.floating]) -> Any:
    import vtk
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, affine[i, j])
    return matrix


def _ijk_to_ras_matrix_to_affine(matrix: Any) -> NDArray[np.floating]:
    affine = np.eye(4)
    for i in range(4):
        for j in range(4):
            affine[i, j] = matrix.GetElement(i, j)
    return affine


@dataclass
class SlicerVolumeResource(VolumeResource):
    is_loaded: ClassVar[bool] = True
    node_id: str
    _node: Any = field(default=None, repr=False, compare=False)

    def load(self) -> "SlicerVolumeResource":
        return self

    def _get_node(self) -> Any:
        if self._node is None:
            self._node = slicer.mrmlScene.GetNodeByID(self.node_id)
            if self._node is None:
                raise ValueError(f"MRML node {self.node_id!r} not in scene")
        return self._node

    def get_array(self) -> NDArray[np.number]:
        node = self._get_node()
        return _vtk_image_to_numpy(node.GetImageData())

    def get_affine(self) -> NDArray[np.floating]:
        import vtk
        node = self._get_node()
        ijk_to_ras = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijk_to_ras)
        return _ijk_to_ras_matrix_to_affine(ijk_to_ras)

    def get_metadata(self) -> dict[str, Any]:
        # Phase 0 finding: kwneuro.util.update_volume_metadata writes metadata
        # keys straight into a nib.Nifti1Header, which rejects non-NIfTI field
        # names. The gist returned Slicer-specific keys here, which broke the
        # denoise pipeline. Returning an empty dict sidesteps the issue; Phase
        # 1 should decide whether to extract NIfTI-compatible metadata from
        # the Slicer storage node or change kwneuro to filter keys.
        return {}

    @staticmethod
    def from_resource(vol: VolumeResource, name: str = "kwneuro_volume", show: bool = False) -> "SlicerVolumeResource":
        if not vol.is_loaded:
            vol = vol.load()
        array = vol.get_array()
        affine = vol.get_affine()
        cls_name = "vtkMRMLVectorVolumeNode" if array.ndim == 4 else "vtkMRMLScalarVolumeNode"
        node = slicer.mrmlScene.AddNewNodeByClass(cls_name, name)
        node.SetIJKToRASMatrix(_affine_to_ijk_to_ras_matrix(affine))
        node.SetAndObserveImageData(_numpy_to_vtk_image(array))
        if show:
            slicer.util.setSliceViewerLayers(background=node)
        return SlicerVolumeResource(node_id=node.GetID(), _node=node)

    @staticmethod
    def from_node(node: Any) -> "SlicerVolumeResource":
        node_id = node.GetID()
        if node_id is None:
            raise ValueError("Node must be added to scene before wrapping")
        return SlicerVolumeResource(node_id=node_id, _node=node)


__all__ = ["SlicerVolumeResource", "InMemoryVolumeResource"]
