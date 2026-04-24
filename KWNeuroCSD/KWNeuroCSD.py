"""KWNeuroCSD - compute Constrained Spherical Deconvolution peaks.

Wraps ``kwneuro.csd.compute_csd_peaks`` (which internally estimates a
single-shell single-tissue response function from low-b data) and
``combine_csd_peaks_to_vector_volume`` to produce a single 4D vector
volume in MRtrix3-style layout suitable for downstream tractography /
visualisation.

Output layout: a ``vtkMRMLVectorVolumeNode`` of shape
``(x, y, z, n_peaks * 3)`` where every triplet encodes one peak's
(x, y, z) direction scaled by its amplitude.
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


class KWNeuroCSD(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro CSD")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Compute CSD fiber-orientation peaks from a DWI + brain mask. "
            "Wraps kwneuro.csd.compute_csd_peaks + "
            "combine_csd_peaks_to_vector_volume to produce an MRtrix3-"
            "style peak vector volume."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroCSDLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self, dwi_node: Any, mask_node: Any,
    ) -> tuple[Any, Any, str]:
        """Materialise DWI + mask into memory. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)
        if mask_node is None:
            msg = "Brain mask is required for CSD."
            raise ValueError(msg)

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        mask = InSceneVolumeResource.from_node(mask_node).to_in_memory()
        return dwi, mask, dwi_name

    # kwneuro.csd hardcodes `sh_order_max=8` in
    # ConstrainedSphericalDeconvModel (csd.py:265). An 8th-order
    # spherical-harmonic fit requires at least (8+1)*(8+2)/2 = 45
    # independent diffusion-weighted directions. Below that, dipy
    # raises deep in its linalg solver with an unhelpful message, so
    # we surface it explicitly up front.
    _MIN_DWI_DIRECTIONS = 45

    def run_csd(
        self,
        dwi: Any,
        mask: Any,
        n_peaks: int,
        flip_bvecs_x: bool,
    ) -> Any:
        """Compute CSD peaks and combine into an MRtrix3-style vector volume.

        **Thread-safe.** Returns an ``InMemoryVolumeResource`` holding a
        4D array of shape ``(x, y, z, n_peaks * 3)``.
        """
        import numpy as np

        from kwneuro.csd import (
            combine_csd_peaks_to_vector_volume,
            compute_csd_peaks,
        )

        bvals = np.asarray(dwi.bval.get())
        n_dw = int((bvals > 50).sum())
        if n_dw < self._MIN_DWI_DIRECTIONS:
            msg = (
                f"CSD at spherical-harmonic order 8 requires at least "
                f"{self._MIN_DWI_DIRECTIONS} diffusion-weighted directions "
                f"(b > 50); this DWI has only {n_dw}. Acquire more "
                f"directions or run a lower-order CSD externally."
            )
            raise ValueError(msg)

        mask_voxels = int(np.sum(np.asarray(mask.get_array()) > 0))
        if mask_voxels < 100:
            msg = (
                f"Brain mask has only {mask_voxels} voxels; CSD response-"
                f"function estimation needs a reasonable interior region. "
                f"Expect unreliable peaks. Consider a larger mask."
            )
            logging.warning("KWNeuroCSD: %s", msg)

        logging.info(
            "KWNeuroCSD: computing peaks (n_peaks=%d, flip_bvecs_x=%s, n_dw=%d)",
            n_peaks, flip_bvecs_x, n_dw,
        )
        peak_dirs, peak_values = compute_csd_peaks(
            dwi=dwi,
            mask=mask,
            n_peaks=n_peaks,
            flip_bvecs_x=flip_bvecs_x,
        )
        return combine_csd_peaks_to_vector_volume(peak_dirs, peak_values)

    def publish_to_scene(self, vector_volume: Any, base_name: str) -> str:
        """Publish the peak vector volume. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneVolumeResource

        # 4D -> vtkMRMLVectorVolumeNode via the bridge's standard path.
        svr = InSceneVolumeResource.from_resource(
            vector_volume, name=f"{base_name}_csd_peaks",
        )
        svr.get_node().CreateDefaultDisplayNodes()
        return svr.node_id

    def process(
        self,
        dwi_node: Any,
        mask_node: Any,
        n_peaks: int = 5,
        flip_bvecs_x: bool = True,
    ) -> str:
        """Synchronous full pipeline."""
        dwi, mask, name = self.prepare_inputs(dwi_node, mask_node)
        peaks = self.run_csd(dwi, mask, n_peaks, flip_bvecs_x)
        return self.publish_to_scene(peaks, name)


class KWNeuroCSDWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroCSD.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroCSDLogic()

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

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()
        mask_node = self.ui.maskSelector.currentNode()
        n_peaks = int(self.ui.nPeaksSpinBox.value)
        flip = self.ui.flipBvecsXCheckBox.checked

        with slicer.util.tryWithErrorDisplay(_("CSD computation failed."), waitCursor=False):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                dwi, mask, name = self.logic.prepare_inputs(dwi_node, mask_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            peaks = run_with_progress_dialog(
                lambda: self.logic.run_csd(dwi, mask, n_peaks, flip),
                title=_("KWNeuroCSD"),
                status=_("Computing CSD peaks..."),
            )

            node_id = self.logic.publish_to_scene(peaks, name)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = f"Created: {node.GetName()}"


class KWNeuroCSDTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroCSD widget smoke test")
        module = slicer.util.getModule("KWNeuroCSD")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
