"""KWNeuroParcellate - DKT parcellation for structural images."""
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


class KWNeuroParcellate(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Parcellate")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Run ANTsPyNet Desikan-Killiany-Tourville parcellation on a "
            "structural scalar volume."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroParcellateLogic(ScriptedLoadableModuleLogic):
    SUPPORTED_METHODS = ("dkt",)

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(self, structural_node: Any, method: str = "dkt") -> tuple[Any, str, str]:
        """Materialise structural image and check ANTsPyNet. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneStructuralImage, ensure_extras_installed

        if structural_node is None:
            msg = "Input structural volume is required."
            raise ValueError(msg)
        if method not in self.SUPPORTED_METHODS:
            msg = f"Unsupported method {method!r}; expected one of {self.SUPPORTED_METHODS}."
            raise ValueError(msg)
        ensure_extras_installed(["antspynet"])

        name = structural_node.GetName() or "structural"
        structural = InSceneStructuralImage.from_node(structural_node).to_in_memory()
        return structural, name, method

    def run_parcellation(self, structural: Any, method: str) -> Any:
        """Run parcellation. **Thread-safe.**"""
        logging.info("KWNeuroParcellate: running %s", method)
        return structural.parcellate(method=method)

    def publish_to_scene(self, labels: Any, base_name: str, method: str) -> str:
        """Publish parcellation as a labelmap. **Main thread only.**"""
        from kwneuro_slicer_bridge import publish_labelmap_resource

        return publish_labelmap_resource(labels, f"{base_name}_{method}", binary=False)

    def process(self, structural_node: Any, method: str = "dkt") -> str:
        structural, name, resolved_method = self.prepare_inputs(structural_node, method)
        labels = self.run_parcellation(structural, resolved_method)
        return self.publish_to_scene(labels, name, resolved_method)


class KWNeuroParcellateWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroParcellate.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroParcellateLogic()
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
        method = "dkt"
        with slicer.util.tryWithErrorDisplay(
            _("Failed to parcellate structural image."), waitCursor=False,
        ):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                structural, name, resolved_method = self.logic.prepare_inputs(
                    node, method,
                )
            finally:
                qt.QApplication.restoreOverrideCursor()

            labels = run_with_progress_dialog(
                lambda: self.logic.run_parcellation(structural, resolved_method),
                title=_("KWNeuroParcellate"),
                status=_("Running DKT parcellation..."),
                capture_stdout=True,
            )
            node_id = self.logic.publish_to_scene(labels, name, resolved_method)
            result_node = slicer.mrmlScene.GetNodeByID(node_id)
            self.ui.resultLabel.text = (
                f"Created: {result_node.GetName() if result_node else node_id}"
            )


class KWNeuroParcellateTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroParcellate widget smoke test")
        module = slicer.util.getModule("KWNeuroParcellate")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
