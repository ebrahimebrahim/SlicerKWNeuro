"""kwneuro_slicer_bridge — expose kwneuro resources as 3D Slicer MRML nodes.

Intended to be imported inside 3D Slicer's bundled Python (Python interactor,
scripted modules, SlicerJupyter kernels). Importing from a regular Python
process without `slicer` and `vtk` on the path will fail at module load.
"""
from __future__ import annotations

from kwneuro_slicer_bridge.async_helpers import (
    ProgressDialog,
    ensure_extras_installed,
    run_in_worker,
    run_with_progress_dialog,
)
from kwneuro_slicer_bridge.dti import InSceneDti
from kwneuro_slicer_bridge.dwi import InSceneDwi
from kwneuro_slicer_bridge.transform import InSceneTransformResource
from kwneuro_slicer_bridge.volume import InSceneVolumeResource

__all__ = [
    "InSceneDti",
    "InSceneDwi",
    "InSceneTransformResource",
    "InSceneVolumeResource",
    "ProgressDialog",
    "ensure_extras_installed",
    "run_in_worker",
    "run_with_progress_dialog",
]
