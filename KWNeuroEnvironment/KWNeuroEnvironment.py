"""KWNeuroEnvironment — install status panel for the KWNeuro extension.

Provides a simple UI in 3D Slicer for inspecting and managing the Python
environment that the KWNeuro extension depends on: `kwneuro` itself and
the five kwneuro optional extras (`hdbet`, `noddi`, `tractseg`,
`combat`, `antspynet`).

Design notes:

* The `kwneuro_slicer_bridge` package ships *bundled with this extension*
  (it sits in the same `qt-scripted-modules/` directory as the modules),
  so no install step is needed for the bridge — it's importable from the
  moment Slicer loads the extension. The Apply environment changes button
  ensures `kwneuro` itself is on Slicer's Python path and applies the
  desired optional-extra state from the checkboxes.
* `kwneuro` is installed from a pinned PyPI release.
* Each optional extra is installed separately via `slicer.packaging.pip_install`,
  hard-coded with the package spec kwneuro's own pyproject.toml declares
  for that extra. TractSeg uses `skip_packages=["fury"]` to preserve
  Slicer's bundled VTK — installing fury would drag in a second,
  incompatible VTK alongside Slicer's and break rendering (see
  `CLAUDE.md` for the longer write-up).
* The Verify setup action imports kwneuro + the bridge, pushes a small
  synthetic volume into the scene via `InSceneVolumeResource`, verifies
  round-trip, and cleans up.
"""
from __future__ import annotations

import importlib.metadata
import logging
from typing import Callable

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


# Pip spec used by the Apply environment changes button.
KWNEURO_PINNED_VERSION = "1.0.0"
KWNEURO_PIP_SPEC = f"kwneuro=={KWNEURO_PINNED_VERSION}"


# Per-extra install specifications. Each "packages" list is the concrete
# PyPI (or PyPI-style) requirement(s) that mirror the corresponding
# optional dependency group in kwneuro's pyproject.toml. "skip_packages"
# is used only for tractseg to prune fury from the resolution so Slicer's
# bundled VTK is preserved.
EXTRAS_INSTALL_SPEC: dict[str, dict[str, object]] = {
    "hdbet": {
        "packages": ["hd-bet == 2.0.1"],
        "skip_packages": None,
        "import_probe": "HD_BET",
        "display_name": "HD-BET brain extraction",
    },
    "noddi": {
        "packages": ["dmri-amico == 2.1.1", "backports.tarfile"],
        "skip_packages": None,
        "import_probe": "amico",
        "display_name": "NODDI via AMICO",
    },
    "tractseg": {
        "packages": ["TractSeg"],
        "skip_packages": ["fury"],
        "import_probe": "tractseg",
        "display_name": "TractSeg white-matter tract segmentation",
    },
    "combat": {
        "packages": ["neuroCombat == 0.2.12"],
        "skip_packages": None,
        "import_probe": "neuroCombat",
        "display_name": "ComBat harmonisation",
    },
    "antspynet": {
        "packages": ["antspynet"],
        "skip_packages": None,
        "import_probe": "antspynet",
        "display_name": "ANTsPyNet structural segmentation / parcellation",
    },
}


#
# KWNeuroEnvironment (module)
#


class KWNeuroEnvironment(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Environment")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Install-status panel for the KWNeuro extension. Manages the "
            "kwneuro library and the five optional kwneuro extras "
            "(hdbet, noddi, tractseg, combat, antspynet). The "
            "kwneuro_slicer_bridge package is bundled with the extension "
            "and needs no install."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


#
# KWNeuroEnvironmentLogic
#


class KWNeuroEnvironmentLogic(ScriptedLoadableModuleLogic):
    """Install-status and verify-setup helpers for KWNeuroEnvironment."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    # --- Installed-version probes --------------------------------------------

    @staticmethod
    def installed_kwneuro_version() -> str | None:
        """Return the installed kwneuro version string, or None if absent."""
        try:
            return importlib.metadata.version("kwneuro")
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def extras_status() -> dict[str, bool]:
        """For each extra, probe-import its marker module and report presence."""
        import importlib.util
        status: dict[str, bool] = {}
        for name, spec in EXTRAS_INSTALL_SPEC.items():
            probe = spec["import_probe"]
            status[name] = importlib.util.find_spec(probe) is not None  # type: ignore[arg-type]
        return status

    # --- Install / uninstall -------------------------------------------------

    @staticmethod
    def ensure_kwneuro_installed(
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Pip-install (or upgrade) the kwneuro library into Slicer's Python.

        ``KWNEURO_PIP_SPEC`` pins the install source to the kwneuro
        release this extension targets.
        """
        import slicer.packaging

        msg = f"Installing / updating kwneuro: {KWNEURO_PIP_SPEC}"
        logging.info(msg)
        if log_callback is not None:
            log_callback(msg)

        slicer.packaging.pip_install(
            [KWNEURO_PIP_SPEC],
            requester="KWNeuroEnvironment",
        )

    @staticmethod
    def install_extra(name: str) -> None:
        """Install the named extra via slicer.packaging, preserving VTK for tractseg."""
        import slicer.packaging

        if name not in EXTRAS_INSTALL_SPEC:
            msg = f"Unknown extra {name!r}; must be one of {list(EXTRAS_INSTALL_SPEC)}"
            raise ValueError(msg)
        spec = EXTRAS_INSTALL_SPEC[name]
        logging.info("Installing kwneuro extra %r: %s", name, spec["packages"])
        slicer.packaging.pip_install(
            spec["packages"],  # type: ignore[arg-type]
            skip_packages=spec["skip_packages"],  # type: ignore[arg-type]
            requester=f"KWNeuroEnvironment / {name}",
        )

    @staticmethod
    def uninstall_extra(name: str) -> None:
        """Uninstall the named extra's top-level package(s)."""
        import slicer.packaging

        if name not in EXTRAS_INSTALL_SPEC:
            msg = f"Unknown extra {name!r}"
            raise ValueError(msg)
        spec = EXTRAS_INSTALL_SPEC[name]
        top_level = [
            str(pkg).split()[0].split("=")[0].strip()
            for pkg in spec["packages"]  # type: ignore[union-attr]
        ]
        slicer.packaging.pip_uninstall(top_level)

    # --- Verify setup -------------------------------------------------------

    @staticmethod
    def verify_setup() -> tuple[bool, str]:
        """Run a minimal bridge round-trip. Returns (passed, human-readable message).

        Checks: kwneuro imports, kwneuro_slicer_bridge imports (the bridge
        is bundled with the extension, so this confirms the
        qt-scripted-modules/ path is wired up), a synthetic 3D volume
        survives a round-trip through `InSceneVolumeResource`. A failure
        here means the environment is not ready for the rest of the
        KWNeuro extension to work; click Apply environment changes for
        the kwneuro half, or consult the error message.
        """
        try:
            import numpy as np

            import kwneuro
            from kwneuro.resource import InMemoryVolumeResource

            from kwneuro_slicer_bridge import InSceneVolumeResource
        except ImportError as exc:
            return False, f"import failed: {type(exc).__name__}: {exc}"

        try:
            arr = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})
            svr = InSceneVolumeResource.from_resource(
                mem, name="kwneuro_verify_setup", show=False,
            )
            if not np.allclose(svr.get_array(), arr):
                return False, "array round-trip mismatch"
            if not np.allclose(svr.get_affine(), affine):
                return False, "affine round-trip mismatch"
        finally:
            node = slicer.mrmlScene.GetFirstNodeByName("kwneuro_verify_setup")
            if node is not None:
                slicer.mrmlScene.RemoveNode(node)

        return (
            True,
            f"kwneuro {kwneuro.__version__}, bridge imports OK, 3D round-trip OK",
        )


#
# KWNeuroEnvironmentWidget
#


class KWNeuroEnvironmentWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroEnvironment.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroEnvironmentLogic()
        self._installed_extras_status: dict[str, bool] = {}
        self._desired_extras_status: dict[str, bool] = {}

        self.ui.installKwneuroButton.connect(
            "clicked(bool)", self.onApplyEnvironmentChangesClicked,
        )
        self.ui.verifySetupButton.connect("clicked(bool)", self.onVerifySetupClicked)

        for name in EXTRAS_INSTALL_SPEC:
            checkbox = getattr(self.ui, f"extra_{name}_CheckBox", None)
            if checkbox is None:
                logging.warning("UI file missing extra_%s_CheckBox; skipping binding", name)
                continue
            checkbox.connect(
                "toggled(bool)",
                lambda checked, n=name: self._onExtraDesiredStateChanged(n, checked),
            )

        self.refresh()

    def refresh(self) -> None:
        """Populate UI labels and checkboxes from current install state."""
        kwneuro_ver = self.logic.installed_kwneuro_version()
        self.ui.kwneuroVersionLabel.text = kwneuro_ver or "(not installed)"

        status = self.logic.extras_status()
        self._installed_extras_status = dict(status)
        self._desired_extras_status = dict(status)
        for name, installed in status.items():
            checkbox = getattr(self.ui, f"extra_{name}_CheckBox", None)
            if checkbox is None:
                continue
            was_blocking = checkbox.blockSignals(True)
            checkbox.checked = installed
            checkbox.blockSignals(was_blocking)

    def onApplyEnvironmentChangesClicked(self) -> None:
        with slicer.util.tryWithErrorDisplay(
            _("Failed to apply KWNeuro environment changes."), waitCursor=True,
        ):
            try:
                desired_status = self._current_desired_extras_status()
                self.logic.ensure_kwneuro_installed()
                installed_status = self.logic.extras_status()

                for name in EXTRAS_INSTALL_SPEC:
                    if desired_status[name] and not installed_status.get(name, False):
                        self.logic.install_extra(name)

                for name in EXTRAS_INSTALL_SPEC:
                    if not desired_status[name] and installed_status.get(name, False):
                        self.logic.uninstall_extra(name)
            finally:
                self.refresh()

    def onInstallKwneuroClicked(self) -> None:
        """Backward-compatible alias for older tests or scripted callers."""
        self.onApplyEnvironmentChangesClicked()

    def onVerifySetupClicked(self) -> None:
        passed, message = self.logic.verify_setup()
        self.ui.verifySetupResultLabel.text = ("PASS: " if passed else "FAIL: ") + message
        if passed:
            logging.info("KWNeuro verify setup: %s", message)
        else:
            logging.error("KWNeuro verify setup FAILED: %s", message)

    def _onExtraDesiredStateChanged(self, name: str, checked: bool) -> None:
        self._desired_extras_status[name] = bool(checked)

    def _current_desired_extras_status(self) -> dict[str, bool]:
        desired: dict[str, bool] = {}
        for name in EXTRAS_INSTALL_SPEC:
            checkbox = getattr(self.ui, f"extra_{name}_CheckBox", None)
            if checkbox is None:
                desired[name] = self._desired_extras_status.get(name, False)
            else:
                desired[name] = bool(checkbox.checked)
        self._desired_extras_status = dict(desired)
        return desired


#
# KWNeuroEnvironmentTest
#


class KWNeuroEnvironmentTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_LogicProbesWork()

    def test_LogicProbesWork(self):
        self.delayDisplay("Starting KWNeuroEnvironment logic smoke test")
        logic = KWNeuroEnvironmentLogic()
        _ = logic.installed_kwneuro_version()
        status = logic.extras_status()
        assert set(status) == set(EXTRAS_INSTALL_SPEC)
        for value in status.values():
            assert isinstance(value, bool)
        self.delayDisplay("Test passed")
