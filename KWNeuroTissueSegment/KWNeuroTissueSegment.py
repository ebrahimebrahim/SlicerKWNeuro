"""KWNeuroTissueSegment - Atropos / Deep Atropos structural segmentation."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class KWNeuroTissueSegment(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Tissue Segment")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Segment structural images into tissue labels using ANTs "
            "Atropos or ANTsPyNet Deep Atropos."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroTissueSegmentLogic(ScriptedLoadableModuleLogic):
    SUPPORTED_METHODS = ("atropos", "deep_atropos")

    _DEEP_ATROPOS_SUBPROCESS_CODE = r"""
from __future__ import annotations

import sys
from pathlib import Path

from kwneuro.io import NiftiVolumeResource
from kwneuro.structural import StructuralImage


def main() -> int:
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    structural = StructuralImage(volume=NiftiVolumeResource(input_path))
    labels = structural.segment_tissues(method="deep_atropos")
    NiftiVolumeResource.save(labels, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        structural_node: Any,
        mask_node: Any | None,
        method: str,
    ) -> tuple[Any, Any, str, str]:
        """Materialise structural image and optional mask. **Main thread only.**"""
        from kwneuro_slicer_bridge import (
            InSceneStructuralImage,
            InSceneVolumeResource,
            ensure_extras_installed,
        )

        if structural_node is None:
            msg = "Input structural volume is required."
            raise ValueError(msg)
        if method not in self.SUPPORTED_METHODS:
            msg = f"Unsupported method {method!r}; expected one of {self.SUPPORTED_METHODS}."
            raise ValueError(msg)
        if method == "deep_atropos":
            ensure_extras_installed(["antspynet"])

        structural_name = structural_node.GetName() or "structural"
        structural = InSceneStructuralImage.from_node(structural_node).to_in_memory()
        mask = None
        if method == "atropos" and mask_node is not None:
            mask = InSceneVolumeResource.from_node(mask_node).to_in_memory()
            self._validate_mask_matches_structural(structural, mask)
        return structural, mask, structural_name, method

    def run_segmentation(self, structural: Any, mask: Any, method: str) -> Any:
        """Run structural tissue segmentation. **Thread-safe.**"""
        logging.info("KWNeuroTissueSegment: running %s", method)
        if method == "deep_atropos":
            return self._run_deep_atropos_subprocess(structural)
        return structural.segment_tissues(mask=mask, method=method)

    @staticmethod
    def _validate_mask_matches_structural(structural: Any, mask: Any) -> None:
        structural_volume = structural.volume
        structural_shape = structural_volume.get_array().shape
        mask_array = mask.get_array()
        if mask_array.shape != structural_shape:
            msg = (
                f"Mask shape {mask_array.shape} does not match structural "
                f"image shape {structural_shape}."
            )
            raise ValueError(msg)

        if not np.allclose(
            mask.get_affine(),
            structural_volume.get_affine(),
            rtol=1e-4,
            atol=1e-4,
        ):
            msg = "Mask geometry does not match the structural image geometry."
            raise ValueError(msg)

        if not np.any(mask_array > 0):
            msg = "Mask is empty; Atropos requires at least one foreground voxel."
            raise ValueError(msg)

    def _run_deep_atropos_subprocess(self, structural: Any) -> Any:
        """Run ANTsPyNet Deep Atropos outside SlicerApp.

        TensorFlow can segfault while importing inside SlicerApp after
        Slicer's Qt/VTK/native modules are loaded. The same ANTsPyNet
        import is stable in plain PythonSlicer, so isolate this method in
        a child process and shuttle NIfTI files across the boundary.
        """
        from kwneuro.io import NiftiVolumeResource

        python_slicer = Path(sys.executable)
        if not python_slicer.exists():
            msg = f"Could not locate PythonSlicer executable at {python_slicer!s}."
            raise RuntimeError(msg)

        with tempfile.TemporaryDirectory(prefix="kwneuro_deep_atropos_") as tmp:
            tmpdir = Path(tmp)
            input_path = tmpdir / "input_t1.nii.gz"
            output_path = tmpdir / "deep_atropos_labels.nii.gz"

            loaded = structural.load()
            NiftiVolumeResource.save(loaded.volume, input_path)

            env = dict(os.environ)
            env["MPLCONFIGDIR"] = str(tmpdir / "matplotlib")

            result = subprocess.run(
                [
                    str(python_slicer),
                    "-c",
                    textwrap.dedent(self._DEEP_ATROPOS_SUBPROCESS_CODE),
                    str(input_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            if result.returncode != 0:
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                details = "\n".join(
                    part for part in (stdout, stderr) if part
                )
                if len(details) > 4000:
                    details = details[-4000:]
                msg = (
                    "Deep Atropos failed in the isolated PythonSlicer "
                    f"process with exit code {result.returncode}."
                )
                if details:
                    msg = f"{msg}\n\n{details}"
                raise RuntimeError(msg)
            if not output_path.exists():
                msg = (
                    "Deep Atropos completed without creating the expected "
                    f"output file: {output_path!s}"
                )
                raise RuntimeError(msg)
            return NiftiVolumeResource(output_path).load()

    def publish_to_scene(self, labels: Any, base_name: str, method: str) -> str:
        """Publish tissue labels as a labelmap. **Main thread only.**"""
        from kwneuro_slicer_bridge import publish_labelmap_resource

        return publish_labelmap_resource(
            labels, f"{base_name}_tissue_{method}", binary=False,
        )

    def process(
        self,
        structural_node: Any,
        mask_node: Any | None = None,
        method: str = "atropos",
    ) -> str:
        structural, mask, name, resolved_method = self.prepare_inputs(
            structural_node, mask_node, method,
        )
        labels = self.run_segmentation(structural, mask, resolved_method)
        return self.publish_to_scene(labels, name, resolved_method)


class KWNeuroTissueSegmentWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroTissueSegment.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroTissueSegmentLogic()
        self.ui.inputStructuralSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
        )
        self.ui.methodComboBox.connect(
            "currentIndexChanged(int)", self._onMethodChanged,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self._onMethodChanged()
        self._updateApplyEnabled()

    def enter(self) -> None:
        self._onMethodChanged()
        self._updateApplyEnabled()

    def _selected_method(self) -> str:
        text = str(self.ui.methodComboBox.currentText)
        return "deep_atropos" if text == "Deep Atropos" else "atropos"

    def _onMethodChanged(self, *_args: Any) -> None:
        is_atropos = self._selected_method() == "atropos"
        self.ui.inputMaskLabel.visible = is_atropos
        self.ui.inputMaskSelector.visible = is_atropos

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputStructuralSelector.currentNode() is not None
        )

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        structural_node = self.ui.inputStructuralSelector.currentNode()
        mask_node = self.ui.inputMaskSelector.currentNode()
        method = self._selected_method()

        with slicer.util.tryWithErrorDisplay(
            _("Failed to segment tissues."), waitCursor=False,
        ):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                structural, mask, name, resolved_method = self.logic.prepare_inputs(
                    structural_node, mask_node, method,
                )
            finally:
                qt.QApplication.restoreOverrideCursor()

            labels = run_with_progress_dialog(
                lambda: self.logic.run_segmentation(
                    structural, mask, resolved_method,
                ),
                title=_("KWNeuroTissueSegment"),
                status=_("Segmenting tissues..."),
                capture_stdout=True,
            )
            node_id = self.logic.publish_to_scene(labels, name, resolved_method)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            self.ui.resultLabel.text = f"Created: {node.GetName() if node else node_id}"


class KWNeuroTissueSegmentTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroTissueSegment widget smoke test")
        module = slicer.util.getModule("KWNeuroTissueSegment")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
