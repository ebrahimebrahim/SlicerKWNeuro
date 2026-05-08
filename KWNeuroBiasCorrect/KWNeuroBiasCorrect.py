"""KWNeuroBiasCorrect - N4 bias correction for structural images."""
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


class KWNeuroBiasCorrect(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Bias Correct")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Apply ANTs N4 bias field correction to a structural scalar "
            "volume. Wraps StructuralImage.correct_bias."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroBiasCorrectLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(self, structural_node: Any) -> tuple[Any, str]:
        """Materialise structural input into memory. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneStructuralImage

        if structural_node is None:
            msg = "Input structural volume is required."
            raise ValueError(msg)
        structural_name = structural_node.GetName() or "structural"
        structural = InSceneStructuralImage.from_node(structural_node).to_in_memory()
        return structural, structural_name

    def run_bias_correct(self, structural: Any) -> Any:
        """Run N4 bias correction. **Thread-safe.**"""
        logging.info("KWNeuroBiasCorrect: running StructuralImage.correct_bias")
        return structural.correct_bias()

    def publish_to_scene(self, corrected: Any, base_name: str) -> str:
        """Publish the corrected structural image. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneStructuralImage

        scene_structural = InSceneStructuralImage.from_structural(
            corrected, name=f"{base_name}_bias_corrected",
        )
        try:
            scene_structural.get_node().CreateDefaultDisplayNodes()
        except BaseException:
            slicer.mrmlScene.RemoveNode(scene_structural.get_node())
            raise
        return scene_structural.node_id

    def process(self, structural_node: Any) -> str:
        structural, name = self.prepare_inputs(structural_node)
        corrected = self.run_bias_correct(structural)
        return self.publish_to_scene(corrected, name)


class KWNeuroBiasCorrectWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroBiasCorrect.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroBiasCorrectLogic()
        self.ui.inputStructuralSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputStructuralSelector.currentNode() is not None
        )

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        node = self.ui.inputStructuralSelector.currentNode()
        with slicer.util.tryWithErrorDisplay(
            _("Failed to correct bias."), waitCursor=False,
        ):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                structural, name = self.logic.prepare_inputs(node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            corrected = run_with_progress_dialog(
                lambda: self.logic.run_bias_correct(structural),
                title=_("KWNeuroBiasCorrect"),
                status=_("Running N4 bias correction..."),
                capture_stdout=True,
            )
            node_id = self.logic.publish_to_scene(corrected, name)
            result_node = slicer.mrmlScene.GetNodeByID(node_id)
            self.ui.resultLabel.text = (
                f"Created: {result_node.GetName() if result_node else node_id}"
            )


class KWNeuroBiasCorrectTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroBiasCorrect widget smoke test")
        module = slicer.util.getModule("KWNeuroBiasCorrect")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
