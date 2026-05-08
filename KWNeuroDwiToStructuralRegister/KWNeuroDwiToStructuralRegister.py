"""KWNeuroDwiToStructuralRegister - register DWI mean-b0 to structural image."""
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


class KWNeuroDwiToStructuralRegister(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro DWI To Structural Register")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Register a DWI mean-b0 image to a structural volume and "
            "optionally inverse-warp structural labels into DWI space."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroDwiToStructuralRegisterLogic(ScriptedLoadableModuleLogic):
    SUPPORTED_TRANSFORM_TYPES = ("Rigid", "Affine", "SyN", "SyNRA")

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        dwi_node: Any,
        structural_node: Any,
        dwi_mask_node: Any | None,
        structural_mask_node: Any | None,
        labelmap_node: Any | None,
    ) -> tuple[Any, Any, Any, Any, Any, str, str | None]:
        """Materialise all scene inputs into memory. **Main thread only.**"""
        from kwneuro_slicer_bridge import (
            InSceneDwi,
            InSceneStructuralImage,
            InSceneVolumeResource,
        )

        if dwi_node is None or structural_node is None:
            msg = "Both DWI and structural inputs are required."
            raise ValueError(msg)

        dwi_name = dwi_node.GetName() or "dwi"
        label_name = labelmap_node.GetName() if labelmap_node is not None else None
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        structural = InSceneStructuralImage.from_node(structural_node).to_in_memory()
        dwi_mask = (
            InSceneVolumeResource.from_node(dwi_mask_node).to_in_memory()
            if dwi_mask_node is not None else None
        )
        structural_mask = (
            InSceneVolumeResource.from_node(structural_mask_node).to_in_memory()
            if structural_mask_node is not None else None
        )
        labelmap = (
            InSceneVolumeResource.from_node(labelmap_node).to_in_memory()
            if labelmap_node is not None else None
        )
        return (
            dwi,
            structural,
            dwi_mask,
            structural_mask,
            labelmap,
            dwi_name,
            label_name,
        )

    def run_registration(
        self,
        dwi: Any,
        structural: Any,
        transform_type: str,
        dwi_mask: Any,
        structural_mask: Any,
        labelmap: Any,
    ) -> tuple[Any, Any, Any]:
        """Run DWI-to-structural registration and optional label warp."""
        from kwneuro.reg import register_dwi_to_structural

        if transform_type not in self.SUPPORTED_TRANSFORM_TYPES:
            msg = (
                f"Unsupported transform type {transform_type!r}; "
                f"must be one of {self.SUPPORTED_TRANSFORM_TYPES}."
            )
            raise ValueError(msg)

        logging.info(
            "KWNeuroDwiToStructuralRegister: running register_dwi_to_structural "
            "(type=%s)",
            transform_type,
        )
        transform = register_dwi_to_structural(
            dwi=dwi,
            structural=structural,
            type_of_transform=transform_type,
            dwi_mask=dwi_mask,
            structural_mask=structural_mask,
        )

        mean_b0 = dwi.compute_mean_b0()
        warped_b0 = transform.apply(
            fixed=structural.volume,
            moving=mean_b0,
        )
        warped_labels = None
        if labelmap is not None:
            warped_labels = transform.apply(
                fixed=mean_b0,
                moving=labelmap,
                invert=True,
                interpolation="genericLabel",
            )
        return transform, warped_b0, warped_labels

    def publish_to_scene(
        self,
        transform: Any,
        warped_b0: Any,
        warped_labels: Any,
        dwi_name: str,
        label_name: str | None,
    ) -> dict[str, Any]:
        """Publish transform, QA warped b0, and optional DWI-space labels."""
        from kwneuro_slicer_bridge import (
            InSceneTransformResource,
            InSceneVolumeResource,
            publish_labelmap_resource,
        )

        warped_b0_svr = InSceneVolumeResource.from_resource(
            warped_b0, name=f"{dwi_name}_mean_b0_in_structural_space",
        )
        warped_b0_svr.get_node().CreateDefaultDisplayNodes()

        in_scene_tf = InSceneTransformResource.from_transform(
            transform, name_prefix=f"{dwi_name}_to_structural_transform",
        )
        warped_label_id = None
        if warped_labels is not None:
            base_label_name = label_name or "structural_labels"
            warped_label_id = publish_labelmap_resource(
                warped_labels,
                f"{base_label_name}_in_{dwi_name}_space",
                binary=False,
            )

        return {
            "warped_b0": warped_b0_svr.node_id,
            "transform_node_ids": list(in_scene_tf.node_ids),
            "warped_labels": warped_label_id,
        }

    def process(
        self,
        dwi_node: Any,
        structural_node: Any,
        transform_type: str = "Rigid",
        dwi_mask_node: Any | None = None,
        structural_mask_node: Any | None = None,
        labelmap_node: Any | None = None,
    ) -> dict[str, Any]:
        (
            dwi,
            structural,
            dwi_mask,
            structural_mask,
            labelmap,
            dwi_name,
            label_name,
        ) = self.prepare_inputs(
            dwi_node,
            structural_node,
            dwi_mask_node,
            structural_mask_node,
            labelmap_node,
        )
        transform, warped_b0, warped_labels = self.run_registration(
            dwi, structural, transform_type, dwi_mask, structural_mask, labelmap,
        )
        return self.publish_to_scene(
            transform, warped_b0, warped_labels, dwi_name, label_name,
        )


class KWNeuroDwiToStructuralRegisterWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(
            self.resourcePath("UI/KWNeuroDwiToStructuralRegister.ui"),
        )
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroDwiToStructuralRegisterLogic()
        for selector in (self.ui.dwiSelector, self.ui.structuralSelector):
            selector.connect(
                "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
            )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.dwiSelector.currentNode() is not None
            and self.ui.structuralSelector.currentNode() is not None
        )

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.dwiSelector.currentNode()
        structural_node = self.ui.structuralSelector.currentNode()
        dwi_mask_node = self.ui.dwiMaskSelector.currentNode()
        structural_mask_node = self.ui.structuralMaskSelector.currentNode()
        labelmap_node = self.ui.structuralLabelSelector.currentNode()
        transform_type = self.ui.transformTypeComboBox.currentText

        with slicer.util.tryWithErrorDisplay(
            _("DWI to structural registration failed."), waitCursor=False,
        ):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                prepared = self.logic.prepare_inputs(
                    dwi_node,
                    structural_node,
                    dwi_mask_node,
                    structural_mask_node,
                    labelmap_node,
                )
            finally:
                qt.QApplication.restoreOverrideCursor()

            (
                dwi,
                structural,
                dwi_mask,
                structural_mask,
                labelmap,
                dwi_name,
                label_name,
            ) = prepared
            transform, warped_b0, warped_labels = run_with_progress_dialog(
                lambda: self.logic.run_registration(
                    dwi,
                    structural,
                    transform_type,
                    dwi_mask,
                    structural_mask,
                    labelmap,
                ),
                title=_("KWNeuroDwiToStructuralRegister"),
                status=_("Registering DWI mean-b0 to structural image..."),
                capture_tqdm=True,
                capture_stdout=True,
            )
            ids = self.logic.publish_to_scene(
                transform, warped_b0, warped_labels, dwi_name, label_name,
            )
            warped_node = slicer.mrmlScene.GetNodeByID(ids["warped_b0"])
            n_tf = len(ids["transform_node_ids"])
            label_text = (
                " | labels warped" if ids.get("warped_labels") is not None else ""
            )
            self.ui.resultLabel.text = (
                f"Warped b0: {warped_node.GetName() if warped_node else ids['warped_b0']} "
                f"| {n_tf} transform node(s){label_text}"
            )


class KWNeuroDwiToStructuralRegisterTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroDwiToStructuralRegister widget smoke test")
        module = slicer.util.getModule("KWNeuroDwiToStructuralRegister")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
