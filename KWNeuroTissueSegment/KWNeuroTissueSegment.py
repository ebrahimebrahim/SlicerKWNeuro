"""KWNeuroTissueSegment - Atropos / Deep Atropos structural segmentation."""
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
        mask = (
            InSceneVolumeResource.from_node(mask_node).to_in_memory()
            if mask_node is not None else None
        )
        return structural, mask, structural_name, method

    def run_segmentation(self, structural: Any, mask: Any, method: str) -> Any:
        """Run structural tissue segmentation. **Thread-safe.**"""
        logging.info("KWNeuroTissueSegment: running %s", method)
        return structural.segment_tissues(mask=mask, method=method)

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
