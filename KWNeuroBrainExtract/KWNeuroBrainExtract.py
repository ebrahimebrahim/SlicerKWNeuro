"""KWNeuroBrainExtract - run HD-BET on a DWI's mean b0 image.

Wraps ``kwneuro.masks.brain_extract_single`` and publishes the
resulting binary mask as a ``vtkMRMLLabelMapVolumeNode`` in the scene.

The hd_bet extra (``kwneuro[hdbet]``, which pulls in torch + nnunetv2)
is required; ``ensure_extras_installed`` checks this up front and
points the user at KWNeuroEnvironment if the extra isn't present.

Follows the three-phase split established by KWNeuroDTI: the DWI is
materialised into memory on the main Qt thread, HD-BET runs on a
worker thread, and the labelmap node is added on the main thread.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
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
# KWNeuroBrainExtract (module)
#


class KWNeuroBrainExtract(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Brain Extract")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Extract a brain mask from a DWI using HD-BET (deep-learning "
            "brain extractor). Wraps kwneuro.masks.brain_extract_single. "
            "Requires the kwneuro[hdbet] optional extra, managed from "
            "KWNeuroEnvironment."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


#
# KWNeuroBrainExtractLogic
#


class KWNeuroBrainExtractLogic(ScriptedLoadableModuleLogic):
    """HD-BET brain extraction, split into main- and worker-thread phases."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(self, dwi_node: Any) -> tuple[Any, str]:
        """Materialise DWI into memory and check for the hdbet extra.

        **Main thread only.** Returns ``(dwi_resource, dwi_name)``.
        Raises ``RuntimeError`` if the hdbet extra is not installed,
        with a message that points at KWNeuroEnvironment.
        """
        from kwneuro_slicer_bridge import InSceneDwi, ensure_extras_installed

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)

        ensure_extras_installed(["hdbet"])

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        return dwi, dwi_name

    def run_brain_extract(self, dwi: Any) -> Any:
        """Run HD-BET on the DWI's mean b0. **Thread-safe.**

        Calls ``kwneuro.masks.brain_extract_single`` with a temp
        output path and returns a loaded ``InMemoryVolumeResource``
        containing the binary mask.
        """
        from kwneuro.masks import brain_extract_single

        with tempfile.TemporaryDirectory(prefix="kwneuro_bet_") as tmp:
            output_path = Path(tmp) / "brainmask.nii.gz"
            logging.info("KWNeuroBrainExtract: running HD-BET -> %s", output_path)
            mask_resource = brain_extract_single(dwi, output_path)
            return mask_resource.load()

    def publish_to_scene(self, mask_resource: Any, base_name: str) -> str:
        """Add the mask as a labelmap volume. **Main thread only.**

        Returns the MRML ID of the new ``vtkMRMLLabelMapVolumeNode``.
        """
        from kwneuro_slicer_bridge.conversions import (
            affine_to_ijk_to_ras_matrix,
            numpy_to_vtk_image,
        )

        # InSceneVolumeResource.from_resource creates a scalar volume
        # node, but a brain mask belongs in a labelmap so Slicer's
        # segmentation tools can consume it. Build the labelmap node
        # directly; use the same conversion helpers the bridge uses.
        raw = mask_resource.get_array()
        # HD-BET returns a 0/1 mask with save_probabilities=False; treat
        # it as binary rather than casting — a silent astype would map
        # probability maps or >255 label values to the wrong thing.
        array = (raw > 0).astype("uint8")
        affine = mask_resource.get_affine()

        name = f"{base_name}_brainmask"
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", name,
        )
        try:
            labelmap_node.SetIJKToRASMatrix(affine_to_ijk_to_ras_matrix(affine))
            labelmap_node.SetAndObserveImageData(numpy_to_vtk_image(array))
            labelmap_node.CreateDefaultDisplayNodes()
            # CreateDefaultDisplayNodes on a labelmap creates a display
            # node but does not always attach a color table — users
            # then see an invisible / single-colour layer. Explicitly
            # assign Slicer's Labels table so the mask renders.
            display_node = labelmap_node.GetDisplayNode()
            if display_node is not None:
                display_node.SetAndObserveColorNodeID(
                    "vtkMRMLColorTableNodeLabels",
                )
        except BaseException:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise
        return labelmap_node.GetID()

    def process(self, dwi_node: Any) -> str:
        """Synchronous full pipeline; composes the three phases."""
        dwi, dwi_name = self.prepare_inputs(dwi_node)
        mask = self.run_brain_extract(dwi)
        return self.publish_to_scene(mask, dwi_name)


#
# KWNeuroBrainExtractWidget
#


class KWNeuroBrainExtractWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroBrainExtract.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroBrainExtractLogic()

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

        import qt

        with slicer.util.tryWithErrorDisplay(
            _("Failed to extract brain mask."), waitCursor=False,
        ):
            # Prepare happens on main thread — includes the extras
            # check so we surface a clean error without popping a
            # progress dialog on top of it. to_in_memory() can copy
            # ~100 MB for a real DWI; show a wait cursor briefly so
            # the UI doesn't appear hung.
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                dwi, dwi_name = self.logic.prepare_inputs(dwi_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            # HD-BET / nnunetv2 emit tqdm via bindings we don't
            # currently route through _TQDM_REBINDINGS, so capture_tqdm
            # wouldn't yield any lines. Leave it off rather than
            # pretending to capture.
            mask = run_with_progress_dialog(
                lambda: self.logic.run_brain_extract(dwi),
                title=_("KWNeuroBrainExtract"),
                status=_("Running HD-BET..."),
            )

            node_id = self.logic.publish_to_scene(mask, dwi_name)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = f"Created: {node.GetName()}"


#
# KWNeuroBrainExtractTest
#


class KWNeuroBrainExtractTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroBrainExtract widget smoke test")
        module = slicer.util.getModule("KWNeuroBrainExtract")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
