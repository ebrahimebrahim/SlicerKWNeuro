"""KWNeuroBrainExtract - run HD-BET on a DWI mean-b0 or structural image.

Wraps ``kwneuro.masks.brain_extract`` and publishes the resulting
binary mask as a ``vtkMRMLLabelMapVolumeNode`` in the scene.

The hd_bet extra (``kwneuro[hdbet]``, which pulls in torch + nnunetv2)
is required; ``ensure_extras_installed`` checks this up front and
points the user at KWNeuroEnvironment if the extra isn't present.

Uses the three-phase split: scene input is materialised into memory on
the main Qt thread, HD-BET runs on a worker thread, and the labelmap
node is added on the main thread.
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
            "Extract a brain mask from a DWI mean-b0 or structural image "
            "using HD-BET (deep-learning brain extractor). Wraps "
            "kwneuro.masks.brain_extract. Requires the kwneuro[hdbet] "
            "optional extra, managed from KWNeuroEnvironment."
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

    def prepare_inputs(self, input_node: Any) -> tuple[Any, str]:
        """Materialise input into a 3D volume and check for the hdbet extra.

        **Main thread only.** Returns ``(volume_resource, input_name)``.
        Raises ``RuntimeError`` if the hdbet extra is not installed,
        with a message that points at KWNeuroEnvironment.
        """
        from kwneuro_slicer_bridge import (
            InSceneDwi,
            InSceneVolumeResource,
            ensure_extras_installed,
        )

        if input_node is None:
            msg = "Input image node is required."
            raise ValueError(msg)

        ensure_extras_installed(["hdbet"])

        input_name = input_node.GetName() or "kwneuro_input"
        if input_node.IsA("vtkMRMLDiffusionWeightedVolumeNode"):
            dwi = InSceneDwi.from_node(input_node).to_in_memory()
            return dwi.compute_mean_b0(), input_name
        if input_node.IsA("vtkMRMLScalarVolumeNode"):
            return InSceneVolumeResource.from_node(input_node).to_in_memory(), input_name

        msg = (
            "Input must be a vtkMRMLDiffusionWeightedVolumeNode or "
            f"vtkMRMLScalarVolumeNode, got {input_node.GetClassName()}."
        )
        raise ValueError(msg)

    def run_brain_extract(self, volume: Any) -> Any:
        """Run HD-BET on a 3D volume. **Thread-safe.**

        Calls ``kwneuro.masks.brain_extract`` with a temp
        output path and returns a loaded ``InMemoryVolumeResource``
        containing the binary mask.

        Passes ``sequential=True`` so kwneuro routes nnunetv2 through
        its no-multiprocessing predictor. The default
        ``Pool``/``Manager`` path doesn't survive in embedded Pythons
        like Slicer's, where the spawned children try to re-run
        ``slicerqt.py`` and crash on names that only exist in the
        main Slicer process.
        """
        import kwneuro.masks as masks_mod

        with tempfile.TemporaryDirectory(prefix="kwneuro_bet_") as tmp:
            output_path = Path(tmp) / "brainmask.nii.gz"
            logging.info("KWNeuroBrainExtract: running HD-BET -> %s", output_path)
            brain_extract = getattr(masks_mod, "brain_extract", None)
            if brain_extract is not None:
                mask_resource = brain_extract(
                    volume=volume, output_path=output_path, sequential=True,
                )
            else:
                # Compatibility for older kwneuro installs that only
                # exposed brain_extract_single(dwi, ...). The adapter
                # supplies the one method that path needs.
                class _MeanB0Adapter:
                    def __init__(self, mean_b0: Any) -> None:
                        self._mean_b0 = mean_b0

                    def compute_mean_b0(self) -> Any:
                        return self._mean_b0

                mask_resource = masks_mod.brain_extract_single(
                    _MeanB0Adapter(volume), output_path, sequential=True,
                )
            return mask_resource.load()

    def publish_to_scene(self, mask_resource: Any, base_name: str) -> str:
        """Add the mask as a labelmap volume. **Main thread only.**

        Returns the MRML ID of the new ``vtkMRMLLabelMapVolumeNode``.
        """
        from kwneuro_slicer_bridge import publish_labelmap_resource

        name = f"{base_name}_brainmask"
        return publish_labelmap_resource(mask_resource, name, binary=True)

    def process(self, dwi_node: Any) -> str:
        """Synchronous full pipeline; composes the three phases."""
        volume, input_name = self.prepare_inputs(dwi_node)
        mask = self.run_brain_extract(volume)
        return self.publish_to_scene(mask, input_name)


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

        input_node = self.ui.inputDwiSelector.currentNode()

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
                volume, input_name = self.logic.prepare_inputs(input_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            # HD-BET / nnunetv2 emit tqdm via bindings we don't
            # currently route through _TQDM_REBINDINGS, so capture_tqdm
            # wouldn't yield any lines. Leave it off rather than
            # pretending to capture.
            mask = run_with_progress_dialog(
                lambda: self.logic.run_brain_extract(volume),
                title=_("KWNeuroBrainExtract"),
                status=_("Running HD-BET..."),
                capture_tqdm=True,
                capture_stdout=True,
            )

            node_id = self.logic.publish_to_scene(mask, input_name)
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
