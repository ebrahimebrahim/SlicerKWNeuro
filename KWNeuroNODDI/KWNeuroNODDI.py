"""KWNeuroNODDI - fit the NODDI model on a DWI via AMICO.

Wraps ``kwneuro.noddi.Noddi.estimate_noddi`` and publishes NDI, ODI,
FWF (and optionally tissue-fraction-modulated NDI/ODI) as scalar
volume nodes. Requires the kwneuro[noddi] optional extra; the
prepare-inputs phase checks up front and points at KWNeuroEnvironment
if it's missing.
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


class KWNeuroNODDI(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro NODDI")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Fit NODDI microstructure parameters on a multi-shell DWI via "
            "AMICO (kwneuro[noddi]). Produces NDI (neurite density), ODI "
            "(orientation dispersion), and FWF (free-water fraction) "
            "scalar volumes."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroNODDILogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self, dwi_node: Any, mask_node: Any | None,
    ) -> tuple[Any, Any, str]:
        """Materialise inputs + check for the noddi extra. **Main thread only.**"""
        from kwneuro_slicer_bridge import (
            InSceneDwi, InSceneVolumeResource, ensure_extras_installed,
        )

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)

        ensure_extras_installed(["noddi"])

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        mask = (
            InSceneVolumeResource.from_node(mask_node).to_in_memory()
            if mask_node is not None else None
        )
        return dwi, mask, dwi_name

    def run_noddi(
        self,
        dwi: Any,
        mask: Any,
        dpar: float,
        n_kernel_dirs: int,
        create_modulated: bool,
    ) -> dict[str, Any]:
        """Fit NODDI. **Thread-safe** (no MRML touches).

        Returns a dict with ``"ndi"``, ``"odi"``, ``"fwf"`` and, if
        ``create_modulated``, ``"ndi_mod"`` and ``"odi_mod"``, each a
        ``VolumeResource``.
        """
        from kwneuro.noddi import Noddi

        logging.info(
            "KWNeuroNODDI: estimating NODDI (dpar=%.4g, n_kernel_dirs=%d)",
            dpar, n_kernel_dirs,
        )
        noddi = Noddi.estimate_noddi(
            dwi, mask=mask, dpar=dpar, n_kernel_dirs=n_kernel_dirs,
        )
        result: dict[str, Any] = {
            "ndi": noddi.ndi,
            "odi": noddi.odi,
            "fwf": noddi.fwf,
        }
        if create_modulated:
            ndi_mod, odi_mod = noddi.get_modulated_ndi_odi()
            result["ndi_mod"] = ndi_mod
            result["odi_mod"] = odi_mod
        return result

    def publish_to_scene(
        self, volumes: dict[str, Any], base_name: str,
    ) -> dict[str, str]:
        """Publish NODDI volumes. **Main thread only.**

        Returns ``{key: node_id}`` mirroring the keys of ``volumes``.
        """
        from kwneuro_slicer_bridge import InSceneVolumeResource

        ids: dict[str, str] = {}
        for key, vol in volumes.items():
            svr = InSceneVolumeResource.from_resource(
                vol, name=f"{base_name}_{key}",
            )
            svr.get_node().CreateDefaultDisplayNodes()
            ids[key] = svr.node_id
        return ids

    def process(
        self,
        dwi_node: Any,
        mask_node: Any | None = None,
        dpar: float = 1.7e-3,
        n_kernel_dirs: int = 500,
        create_modulated: bool = False,
    ) -> dict[str, str]:
        """Synchronous full pipeline."""
        dwi, mask, name = self.prepare_inputs(dwi_node, mask_node)
        volumes = self.run_noddi(dwi, mask, dpar, n_kernel_dirs, create_modulated)
        return self.publish_to_scene(volumes, name)


class KWNeuroNODDIWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroNODDI.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroNODDILogic()

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
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()
        mask_node = self.ui.maskSelector.currentNode()
        dpar = float(self.ui.dparSpinBox.value)
        n_kernel_dirs = int(self.ui.nKernelDirsSpinBox.value)
        create_modulated = self.ui.modulatedCheckBox.checked

        with slicer.util.tryWithErrorDisplay(_("NODDI fitting failed."), waitCursor=False):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                dwi, mask, name = self.logic.prepare_inputs(dwi_node, mask_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            volumes = run_with_progress_dialog(
                lambda: self.logic.run_noddi(
                    dwi, mask, dpar, n_kernel_dirs, create_modulated,
                ),
                title=_("KWNeuroNODDI"),
                status=_("Fitting NODDI (AMICO)..."),
            )

            ids = self.logic.publish_to_scene(volumes, name)
            created = ", ".join(
                slicer.mrmlScene.GetNodeByID(nid).GetName() for nid in ids.values()
            )
            self.ui.resultLabel.text = f"Created: {created}"


class KWNeuroNODDITest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroNODDI widget smoke test")
        module = slicer.util.getModule("KWNeuroNODDI")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
