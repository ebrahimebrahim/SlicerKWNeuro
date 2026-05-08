"""InSceneStructuralImage: a kwneuro.StructuralImage backed by a scalar node."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import slicer

from kwneuro.io import NiftiVolumeResource
from kwneuro.structural import StructuralImage

from kwneuro_slicer_bridge.volume import InSceneVolumeResource


@dataclass(init=False)
class InSceneStructuralImage(StructuralImage):
    """A kwneuro StructuralImage whose volume lives in Slicer's MRML scene."""

    node_id: str
    """MRML ID of the backing vtkMRMLScalarVolumeNode."""

    _node: Any = field(default=None, repr=False, compare=False)

    def __init__(self, node_id: str, _node: Any = None) -> None:
        self._node = _node if _node is not None else slicer.mrmlScene.GetNodeByID(node_id)
        if self._node is None:
            msg = f"MRML node {node_id!r} not found in the scene"
            raise ValueError(msg)
        if not self._node.IsA("vtkMRMLScalarVolumeNode"):
            msg = (
                "Structural images must be backed by vtkMRMLScalarVolumeNode, "
                f"got {self._node.GetClassName()}."
            )
            raise ValueError(msg)
        self.node_id = node_id
        super().__init__(
            volume=InSceneVolumeResource(node_id=node_id, _node=self._node),
        )

    def get_node(self) -> Any:
        return self._node

    def to_in_memory(self) -> StructuralImage:
        """Detach from the scene and return a plain kwneuro StructuralImage."""
        return StructuralImage(
            volume=self.volume.to_in_memory(),  # type: ignore[attr-defined]
        )

    @staticmethod
    def from_node(node: Any) -> InSceneStructuralImage:
        """Wrap an existing vtkMRMLScalarVolumeNode."""
        node_id = node.GetID()
        if node_id is None:
            msg = "Node must be added to the scene before wrapping"
            raise ValueError(msg)
        return InSceneStructuralImage(node_id=node_id, _node=node)

    @staticmethod
    def from_structural(
        structural: StructuralImage,
        name: str = "kwneuro_structural",
        show: bool = False,
    ) -> InSceneStructuralImage:
        """Push a kwneuro StructuralImage into the scene as a scalar volume."""
        loaded = structural.load() if not structural.volume.is_loaded else structural
        array = loaded.volume.get_array()
        if array.ndim != 3:
            msg = f"Structural image volume must be 3D, got shape {array.shape}"
            raise ValueError(msg)

        scene_volume = InSceneVolumeResource.from_resource(
            loaded.volume, name=name, show=show,
        )
        node = scene_volume.get_node()
        if not node.IsA("vtkMRMLScalarVolumeNode"):
            slicer.mrmlScene.RemoveNode(node)
            msg = "Structural image publishing unexpectedly created a non-scalar node."
            raise RuntimeError(msg)
        return InSceneStructuralImage(node_id=scene_volume.node_id, _node=node)

    @staticmethod
    def from_nifti_path(
        volume_path: Path,
        name: str = "kwneuro_structural",
        show: bool = False,
    ) -> InSceneStructuralImage:
        """Load a structural NIfTI image into the Slicer scene."""
        structural = StructuralImage(
            volume=NiftiVolumeResource(Path(volume_path)),
        ).load()
        return InSceneStructuralImage.from_structural(
            structural, name=name, show=show,
        )
