"""KWNeuroDenoise - denoise a DWI via dipy's Patch2Self.

Uses the progress-dialog + tqdm-capture pattern: the worker's
per-gradient tqdm output flows into the dialog's Details log via
:class:`kwneuro_slicer_bridge.async_helpers.TqdmToProgressDialog`,
routed through the worker's ``progress_queue`` and drained on the
main thread.

Logic uses the three-phase split so MRML scene mutations stay on
the main Qt thread.
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


#
# KWNeuroDenoise (module)
#


class KWNeuroDenoise(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Denoise")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Denoise a DWI using dipy's Patch2Self. Wraps kwneuro's "
            "Dwi.denoise through the kwneuro_slicer_bridge scene "
            "resources. Per-gradient progress is routed into the "
            "progress dialog's Details log."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


#
# KWNeuroDenoiseLogic
#


class KWNeuroDenoiseLogic(ScriptedLoadableModuleLogic):
    """Denoise DWI via Patch2Self, split into main- and worker-thread phases."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(self, dwi_node: Any) -> tuple[Any, str]:
        """Resolve the DWI node. **Main thread only.**

        Returns ``(dwi_resource, dwi_name)``. The resource is a plain
        :class:`kwneuro.dwi.Dwi` whose volume has been fully materialised
        into main-thread memory, so the downstream worker thread never
        reads from ``node.GetImageData()``.
        """
        from kwneuro_slicer_bridge import InSceneDwi

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)
        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        return dwi, dwi_name

    def run_denoise(self, dwi: Any) -> Any:
        """Run Patch2Self on ``dwi``. **Thread-safe.**

        Pure numpy / dipy; no MRML writes. Returns a ``kwneuro.Dwi``.
        """
        logging.info("KWNeuroDenoise: running Patch2Self")
        return dwi.denoise()

    def publish_to_scene(self, denoised_dwi: Any, base_name: str) -> str:
        """Push the denoised DWI into the scene. **Main thread only.**

        Returns the MRML ID of the new DWI node.
        """
        from kwneuro_slicer_bridge import InSceneDwi

        scene_dwi = InSceneDwi.from_dwi(
            denoised_dwi, name=f"{base_name}_denoised",
        )
        scene_dwi.get_node().CreateDefaultDisplayNodes()
        return scene_dwi.node_id

    def process(self, dwi_node: Any) -> str:
        """Synchronous full pipeline; composes the three phases.

        Tests / headless callers use this; the widget calls the phases
        separately so it can wrap ``run_denoise`` in a progress dialog.
        """
        dwi, dwi_name = self.prepare_inputs(dwi_node)
        denoised = self.run_denoise(dwi)
        return self.publish_to_scene(denoised, dwi_name)


#
# KWNeuroDenoiseWidget
#


class KWNeuroDenoiseWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroDenoise.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroDenoiseLogic()

        self.ui.inputDwiSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)

        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputDwiSelector.currentNode() is not None
        )

    def onApplyClicked(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()

        with slicer.util.tryWithErrorDisplay(_("Failed to denoise DWI."), waitCursor=False):
            # Main thread: resolve inputs.
            dwi, dwi_name = self.logic.prepare_inputs(dwi_node)

            # Worker thread: run Patch2Self. capture_tqdm=True routes the
            # per-gradient progress lines dipy emits into the dialog log.
            denoised = run_with_progress_dialog(
                lambda: self.logic.run_denoise(dwi),
                title=_("KWNeuroDenoise"),
                status=_("Denoising..."),
                capture_tqdm=True,
            )

            # Main thread: publish result.
            node_id = self.logic.publish_to_scene(denoised, dwi_name)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = f"Created: {node.GetName()}"


#
# KWNeuroDenoiseTest
#


class KWNeuroDenoiseTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroDenoise widget smoke test")
        module = slicer.util.getModule("KWNeuroDenoise")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
