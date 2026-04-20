"""InSceneVolumeResource: a kwneuro VolumeResource backed by a Slicer MRML volume node."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

import slicer

from kwneuro.resource import InMemoryVolumeResource, VolumeResource

from kwneuro_slicer_bridge.conversions import (
    affine_to_ijk_to_ras_matrix,
    ijk_to_ras_matrix_to_affine,
    numpy_to_vtk_image,
    vtk_image_to_numpy,
)


@dataclass
class InSceneVolumeResource(VolumeResource):
    """A kwneuro VolumeResource whose data lives in a Slicer MRML volume node.

    Sits alongside kwneuro's `InMemoryVolumeResource` and `NiftiVolumeResource`
    as a third storage location (scene vs memory vs disk). 3D arrays are
    backed by `vtkMRMLScalarVolumeNode`; 4D arrays by `vtkMRMLVectorVolumeNode`.
    DWI-shaped data should prefer `InSceneDwi`, which uses Slicer's native
    `vtkMRMLDiffusionWeightedVolumeNode` and carries bval/bvec as node
    attributes.

    `is_loaded` is True because the data is already in memory (VTK
    structures). `load()` is a no-op; converting back to a pure-Python
    kwneuro resource is done via `to_in_memory()`.

    Cache correctness
    -----------------
    Scene-backed resources are not fingerprint-stable. The `_node` field
    (a live VTK object) fails kwneuro's fingerprinter and the whole
    resource is silently dropped from cache tracking with a UserWarning.
    Within a session repeated calls with the same resource still hit via
    the other (fingerprintable) args, but a different `InSceneVolumeResource`
    wrapping a different node will NOT invalidate the cache — potential
    stale hits. Don't wrap `Cache()` contexts around pipelines that
    involve scene-backed resources and expect correct invalidation.

    Metadata
    --------
    `get_metadata()` returns Slicer-specific identity keys
    (`slicer_node_id`, `slicer_node_name`). kwneuro's
    `update_volume_metadata` preserves custom keys through the pipeline
    so these origin markers flow through cleanly. They don't survive a
    NIfTI save (nibabel only writes real NIfTI fields), which is the
    only sensible behaviour — a scene node ID has no meaning outside the
    session that assigned it.
    """

    is_loaded: ClassVar[bool] = True

    node_id: str
    """The MRML ID of the backing scene node."""

    _node: Any = field(default=None, repr=False, compare=False)
    """Cached reference to the MRML node; resolved lazily from the scene."""

    def load(self) -> InSceneVolumeResource:
        return self

    def get_node(self) -> Any:
        """Resolve and cache the underlying MRML node."""
        if self._node is None:
            self._node = slicer.mrmlScene.GetNodeByID(self.node_id)
            if self._node is None:
                msg = f"MRML node {self.node_id!r} not found in the scene"
                raise ValueError(msg)
        return self._node

    def get_array(self) -> NDArray[np.number]:
        node = self.get_node()
        return vtk_image_to_numpy(node.GetImageData())

    def get_affine(self) -> NDArray[np.floating]:
        import vtk

        node = self.get_node()
        ijk_to_ras = vtk.vtkMatrix4x4()
        node.GetIJKToRASMatrix(ijk_to_ras)
        return ijk_to_ras_matrix_to_affine(ijk_to_ras)

    def get_metadata(self) -> dict[str, Any]:
        node = self.get_node()
        return {
            "slicer_node_id": self.node_id,
            "slicer_node_name": node.GetName(),
        }

    # --- Conversions / factories ---

    def to_in_memory(self) -> InMemoryVolumeResource:
        """Detach from the scene, returning a pure-Python kwneuro resource."""
        return InMemoryVolumeResource(
            array=self.get_array(),
            affine=self.get_affine(),
            metadata=self.get_metadata(),
        )

    @staticmethod
    def from_resource(
        vol: VolumeResource,
        name: str = "kwneuro_volume",
        show: bool = False,
    ) -> InSceneVolumeResource:
        """Push any kwneuro VolumeResource into Slicer's scene as a new node.

        Creates a `vtkMRMLScalarVolumeNode` for 3D arrays or a
        `vtkMRMLVectorVolumeNode` for 4D arrays. DWI data should go through
        `InSceneDwi.from_dwi` instead to get a first-class
        `vtkMRMLDiffusionWeightedVolumeNode`.
        """
        if not vol.is_loaded:
            vol = vol.load()
        array = vol.get_array()
        affine = vol.get_affine()

        if array.ndim == 4:
            cls_name = "vtkMRMLVectorVolumeNode"
        elif array.ndim == 3:
            cls_name = "vtkMRMLScalarVolumeNode"
        else:
            msg = f"Array must be 3D or 4D, got {array.ndim}D"
            raise ValueError(msg)

        node = slicer.mrmlScene.AddNewNodeByClass(cls_name, name)
        node.SetIJKToRASMatrix(affine_to_ijk_to_ras_matrix(affine))
        node.SetAndObserveImageData(numpy_to_vtk_image(array))

        if show:
            slicer.util.setSliceViewerLayers(background=node)

        return InSceneVolumeResource(node_id=node.GetID(), _node=node)

    @staticmethod
    def from_node(node: Any) -> InSceneVolumeResource:
        """Wrap an existing scene node as an InSceneVolumeResource."""
        node_id = node.GetID()
        if node_id is None:
            msg = "Node must be added to the scene before wrapping"
            raise ValueError(msg)
        return InSceneVolumeResource(node_id=node_id, _node=node)

    @staticmethod
    def from_scene_by_name(name: str) -> InSceneVolumeResource:
        """Look up a scene node by display name and wrap it."""
        node = slicer.util.getNode(name)
        if node is None:
            msg = f"No node named {name!r} found in the scene"
            raise ValueError(msg)
        return InSceneVolumeResource.from_node(node)
