"""KWNeuroRegister - register a moving volume to a fixed volume via ANTs.

Wraps ``kwneuro.reg.register_volumes``. Produces a warped moving
volume as a scalar volume node, plus one or more transform nodes
representing the forward transform (affine ``.mat`` and / or warp
``.nii.gz`` loaded as ``vtkMRMLLinearTransformNode`` /
``vtkMRMLGridTransformNode`` via the bridge).

Logic follows the three-phase split: scene nodes are materialised
into in-memory resources on the main thread, ANTs registration runs
on a worker thread, output nodes are added on the main thread.
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


class KWNeuroRegister(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Register")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Register a moving volume to a fixed volume via ANTs, "
            "with optional fixed and moving masks. Wraps "
            "kwneuro.reg.register_volumes. Produces the warped volume "
            "plus transform nodes."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroRegisterLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        fixed_node: Any,
        moving_node: Any,
        fixed_mask_node: Any | None,
        moving_mask_node: Any | None,
    ) -> tuple[Any, Any, Any, Any, str]:
        """Materialise all inputs into memory. **Main thread only.**

        Returns ``(fixed, moving, fixed_mask, moving_mask, moving_name)``
        where each resource is either an ``InMemoryVolumeResource`` or
        ``None``.
        """
        from kwneuro_slicer_bridge import InSceneVolumeResource

        if fixed_node is None or moving_node is None:
            msg = "Both fixed and moving volumes are required."
            raise ValueError(msg)

        moving_name = moving_node.GetName() or "moving"
        fixed = InSceneVolumeResource.from_node(fixed_node).to_in_memory()
        moving = InSceneVolumeResource.from_node(moving_node).to_in_memory()
        fixed_mask = (
            InSceneVolumeResource.from_node(fixed_mask_node).to_in_memory()
            if fixed_mask_node is not None else None
        )
        moving_mask = (
            InSceneVolumeResource.from_node(moving_mask_node).to_in_memory()
            if moving_mask_node is not None else None
        )
        return fixed, moving, fixed_mask, moving_mask, moving_name

    # ANTs transform types exposed in the UI. Validated explicitly
    # here so a UI typo surfaces as a ValueError with a useful message
    # rather than failing deep inside ANTs with a cryptic output.
    SUPPORTED_TRANSFORM_TYPES = ("SyN", "Affine", "Rigid", "SyNRA")

    def run_registration(
        self,
        fixed: Any,
        moving: Any,
        transform_type: str,
        fixed_mask: Any,
        moving_mask: Any,
    ) -> tuple[Any, Any]:
        """Run ANTs registration. **Thread-safe.**

        Returns ``(warped_moving, transform_resource)``.
        """
        from kwneuro.reg import register_volumes

        if transform_type not in self.SUPPORTED_TRANSFORM_TYPES:
            msg = (
                f"Unsupported transform type {transform_type!r}; "
                f"must be one of {self.SUPPORTED_TRANSFORM_TYPES}."
            )
            raise ValueError(msg)

        logging.info(
            "KWNeuroRegister: running register_volumes (type=%s)",
            transform_type,
        )
        return register_volumes(
            fixed=fixed,
            moving=moving,
            type_of_transform=transform_type,
            mask=fixed_mask,
            moving_mask=moving_mask,
        )

    def publish_to_scene(
        self,
        warped: Any,
        transform: Any,
        base_name: str,
    ) -> dict[str, Any]:
        """Publish warped volume + transform nodes. **Main thread only.**

        Returns ``{"warped": node_id, "transform_node_ids": [..]}``.
        """
        from kwneuro_slicer_bridge import (
            InSceneTransformResource,
            InSceneVolumeResource,
        )

        warped_svr = InSceneVolumeResource.from_resource(
            warped, name=f"{base_name}_warped",
        )
        warped_svr.get_node().CreateDefaultDisplayNodes()

        in_scene_tf = InSceneTransformResource.from_transform(
            transform, name_prefix=f"{base_name}_transform",
        )
        return {
            "warped": warped_svr.node_id,
            "transform_node_ids": list(in_scene_tf.node_ids),
        }

    def process(
        self,
        fixed_node: Any,
        moving_node: Any,
        transform_type: str = "SyN",
        fixed_mask_node: Any | None = None,
        moving_mask_node: Any | None = None,
    ) -> dict[str, Any]:
        """Synchronous full pipeline; composes the three phases."""
        fixed, moving, fm, mm, name = self.prepare_inputs(
            fixed_node, moving_node, fixed_mask_node, moving_mask_node,
        )
        warped, transform = self.run_registration(
            fixed, moving, transform_type, fm, mm,
        )
        return self.publish_to_scene(warped, transform, name)


class KWNeuroRegisterWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroRegister.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroRegisterLogic()

        for sel in (self.ui.fixedSelector, self.ui.movingSelector):
            sel.connect("currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)

        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.fixedSelector.currentNode() is not None
            and self.ui.movingSelector.currentNode() is not None
        )

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        fixed = self.ui.fixedSelector.currentNode()
        moving = self.ui.movingSelector.currentNode()
        fm = self.ui.fixedMaskSelector.currentNode()
        mm = self.ui.movingMaskSelector.currentNode()
        transform_type = self.ui.transformTypeComboBox.currentText

        with slicer.util.tryWithErrorDisplay(_("Registration failed."), waitCursor=False):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                fx, mv, fmr, mmr, name = self.logic.prepare_inputs(
                    fixed, moving, fm, mm,
                )
            finally:
                qt.QApplication.restoreOverrideCursor()

            warped, transform = run_with_progress_dialog(
                lambda: self.logic.run_registration(
                    fx, mv, transform_type, fmr, mmr,
                ),
                title=_("KWNeuroRegister"),
                status=_("Running ANTs registration..."),
            )

            ids = self.logic.publish_to_scene(warped, transform, name)
            warped_node = slicer.mrmlScene.GetNodeByID(ids["warped"])
            n_tf = len(ids["transform_node_ids"])
            self.ui.resultLabel.text = (
                f"Warped: {warped_node.GetName()}  |  "
                f"{n_tf} transform node(s) created"
            )


class KWNeuroRegisterTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroRegister widget smoke test")
        module = slicer.util.getModule("KWNeuroRegister")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
