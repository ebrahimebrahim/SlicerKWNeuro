"""Helpers for publishing kwneuro labeled volumes as Slicer labelmaps."""
from __future__ import annotations

from typing import Any

import numpy as np

import slicer

from kwneuro_slicer_bridge.conversions import (
    affine_to_ijk_to_ras_matrix,
    numpy_to_vtk_image,
)


def _as_label_array(array: np.ndarray, *, binary: bool) -> np.ndarray:
    """Return an integer array suitable for vtkMRMLLabelMapVolumeNode."""
    raw = np.asarray(array)
    if binary:
        return (raw > 0).astype(np.uint8)

    if np.issubdtype(raw.dtype, np.integer):
        return np.ascontiguousarray(raw)

    rounded = np.rint(raw)
    if not np.allclose(raw, rounded):
        msg = (
            "Labelmap outputs must contain integer-valued labels. "
            "Refusing to publish non-integer values as a labelmap."
        )
        raise ValueError(msg)

    max_value = int(np.max(rounded)) if rounded.size else 0
    min_value = int(np.min(rounded)) if rounded.size else 0
    if min_value >= 0 and max_value <= np.iinfo(np.uint8).max:
        dtype = np.uint8
    elif min_value >= 0 and max_value <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.int16
    return rounded.astype(dtype)


def publish_labelmap_resource(
    resource: Any,
    name: str,
    *,
    binary: bool = False,
) -> str:
    """Publish a 3D kwneuro volume resource as a Slicer labelmap.

    ``binary=True`` is for masks and thresholds all positive voxels to
    1. Multi-label outputs keep their label values; integer-like float
    arrays are converted to an integer dtype without collapsing labels.
    """
    array = _as_label_array(resource.get_array(), binary=binary)
    if array.ndim != 3:
        msg = f"Labelmap output must be 3D, got shape {array.shape}"
        raise ValueError(msg)

    labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", name,
    )
    try:
        labelmap_node.SetIJKToRASMatrix(
            affine_to_ijk_to_ras_matrix(resource.get_affine()),
        )
        labelmap_node.SetAndObserveImageData(numpy_to_vtk_image(array))
        labelmap_node.CreateDefaultDisplayNodes()
        display_node = labelmap_node.GetDisplayNode()
        if display_node is not None:
            display_node.SetAndObserveColorNodeID("vtkMRMLColorTableNodeLabels")
    except BaseException:
        slicer.mrmlScene.RemoveNode(labelmap_node)
        raise
    return labelmap_node.GetID()
