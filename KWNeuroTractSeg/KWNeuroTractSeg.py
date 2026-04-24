"""KWNeuroTractSeg - run TractSeg on a DWI + brain mask.

Wraps ``kwneuro.tractseg.extract_tractseg``. Output depends on
``output_type`` — a 4D volume whose last dim is 72 (tract_segmentation),
144 (endings_segmentation), or 60 (TOM) components.

Requires kwneuro[tractseg]; also strongly benefits from a CUDA GPU.
"""
from __future__ import annotations

import logging
from typing import Any

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class KWNeuroTractSeg(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro TractSeg")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Segment white-matter tracts using TractSeg (kwneuro[tractseg]). "
            "Internally computes CSD peaks and feeds them to TractSeg's CNN. "
            "A CUDA GPU is strongly recommended."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroTractSegLogic(ScriptedLoadableModuleLogic):
    SUPPORTED_OUTPUT_TYPES = (
        "tract_segmentation",
        "endings_segmentation",
        "TOM",
    )

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self, dwi_node: Any, mask_node: Any,
    ) -> tuple[Any, Any, str]:
        """Materialise inputs + check tractseg extra. **Main thread only.**"""
        from kwneuro_slicer_bridge import (
            InSceneDwi, InSceneVolumeResource, ensure_extras_installed,
        )

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)
        if mask_node is None:
            msg = "Brain mask is required for TractSeg."
            raise ValueError(msg)

        ensure_extras_installed(["tractseg"])

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        mask = InSceneVolumeResource.from_node(mask_node).to_in_memory()
        return dwi, mask, dwi_name

    def run_tractseg(self, dwi: Any, mask: Any, output_type: str) -> Any:
        """Run TractSeg. **Thread-safe** (no MRML touches)."""
        from kwneuro.tractseg import extract_tractseg

        if output_type not in self.SUPPORTED_OUTPUT_TYPES:
            msg = (
                f"Unsupported output_type {output_type!r}; "
                f"must be one of {self.SUPPORTED_OUTPUT_TYPES}."
            )
            raise ValueError(msg)

        logging.info("KWNeuroTractSeg: running (output_type=%s)", output_type)
        return extract_tractseg(dwi=dwi, mask=mask, output_type=output_type)

    def publish_to_scene(
        self, tract_volume: Any, base_name: str, output_type: str,
    ) -> str:
        """Publish the 4D tract-seg volume. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneVolumeResource

        suffix = {
            "tract_segmentation": "tractseg",
            "endings_segmentation": "tractseg_endings",
            "TOM": "tractseg_tom",
        }[output_type]
        svr = InSceneVolumeResource.from_resource(
            tract_volume, name=f"{base_name}_{suffix}",
        )
        svr.get_node().CreateDefaultDisplayNodes()
        return svr.node_id

    def process(
        self,
        dwi_node: Any,
        mask_node: Any,
        output_type: str = "tract_segmentation",
    ) -> str:
        """Synchronous full pipeline."""
        dwi, mask, name = self.prepare_inputs(dwi_node, mask_node)
        tract_volume = self.run_tractseg(dwi, mask, output_type)
        return self.publish_to_scene(tract_volume, name, output_type)


class KWNeuroTractSegWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroTractSeg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroTractSegLogic()

        for sel in (self.ui.inputDwiSelector, self.ui.maskSelector):
            sel.connect("currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputDwiSelector.currentNode() is not None
            and self.ui.maskSelector.currentNode() is not None
        )

    def _selectedOutputType(self) -> str:
        if self.ui.endingsSegmentationRadio.checked:
            return "endings_segmentation"
        if self.ui.tomRadio.checked:
            return "TOM"
        return "tract_segmentation"

    @staticmethod
    def _cuda_available() -> bool:
        """True iff TractSeg would see a usable CUDA device.

        Isolated as a static method so tests can monkey-patch it to
        exercise both GPU / no-GPU branches without needing a real GPU.
        """
        try:
            import torch  # torch arrives with kwneuro[tractseg]
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()
        mask_node = self.ui.maskSelector.currentNode()
        output_type = self._selectedOutputType()

        with slicer.util.tryWithErrorDisplay(_("TractSeg failed."), waitCursor=False):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                dwi, mask, name = self.logic.prepare_inputs(dwi_node, mask_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            # GPU pre-flight. Without CUDA, TractSeg falls back to CPU
            # inference and takes 30+ min on realistic data. Warn
            # before committing the user to the progress dialog.
            if not self._cuda_available():
                if not slicer.util.confirmYesNoDisplay(
                    _(
                        "No CUDA GPU detected. TractSeg will run on CPU, "
                        "which can take 30+ minutes on realistic data. "
                        "Proceed anyway?",
                    ),
                    windowTitle=_("KWNeuroTractSeg - no GPU"),
                ):
                    return

            tract_volume = run_with_progress_dialog(
                lambda: self.logic.run_tractseg(dwi, mask, output_type),
                title=_("KWNeuroTractSeg"),
                status=_("Running TractSeg..."),
            )

            node_id = self.logic.publish_to_scene(tract_volume, name, output_type)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = f"Created: {node.GetName()}"


class KWNeuroTractSegTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroTractSeg widget smoke test")
        module = slicer.util.getModule("KWNeuroTractSeg")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
