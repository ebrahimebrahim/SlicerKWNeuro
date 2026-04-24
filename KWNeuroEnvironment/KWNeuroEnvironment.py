"""KWNeuroEnvironment — install status panel for the KWNeuro extension.

Provides a simple UI in 3D Slicer for inspecting and managing the Python
environment that the KWNeuro extension depends on: `kwneuro` itself, the
`kwneuro_slicer_bridge` package, and the four kwneuro optional extras
(`hdbet`, `noddi`, `tractseg`, `combat`).

Design notes:

* The bridge package is pip-installed from its absolute location inside
  this extension repo. That call also pulls in `kwneuro` (via the
  `git+...@main` pin in the bridge's pyproject.toml) on first-time setup.
* Each extra is installed separately via `slicer.packaging.pip_install`,
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
from pathlib import Path
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


# Absolute path to the bridge package alongside this module. `pip_install`
# on this path triggers pip to build + install the bridge, pulling in
# `kwneuro` via the pyproject.toml `git+...@<ref>` pin.
BRIDGE_PATH = Path(__file__).resolve().parent.parent / "kwneuro_slicer_bridge"


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
            "kwneuro library, the kwneuro_slicer_bridge package, and the "
            "four optional kwneuro extras (hdbet, noddi, tractseg, combat)."
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
    def installed_bridge_version() -> str | None:
        """Return the installed kwneuro_slicer_bridge version, or None."""
        try:
            return importlib.metadata.version("kwneuro_slicer_bridge")
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
    def ensure_bridge_installed(
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Install the kwneuro_slicer_bridge package from its local path.

        Pip-installs `kwneuro_slicer_bridge` from `BRIDGE_PATH`. The
        package's pyproject.toml pins `kwneuro @ git+...` as a
        dependency, so pip will also fetch kwneuro from that ref on
        first-time setup (or skip it if kwneuro is already installed).
        """
        import slicer.packaging

        msg = f"Installing kwneuro_slicer_bridge from {BRIDGE_PATH}"
        logging.info(msg)
        if log_callback is not None:
            log_callback(msg)

        slicer.packaging.pip_install(
            [str(BRIDGE_PATH)],
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

        Checks: kwneuro imports, kwneuro_slicer_bridge imports, a synthetic
        3D volume survives a round-trip through `InSceneVolumeResource`.
        A failure here means the environment is not ready for the rest of
        the KWNeuro extension to work; re-run Install / Update, or consult
        the error message.
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

        self.ui.installBridgeButton.connect("clicked(bool)", self.onInstallBridgeClicked)
        self.ui.verifySetupButton.connect("clicked(bool)", self.onVerifySetupClicked)

        for name in EXTRAS_INSTALL_SPEC:
            checkbox = getattr(self.ui, f"extra_{name}_CheckBox", None)
            if checkbox is None:
                logging.warning("UI file missing extra_%s_CheckBox; skipping binding", name)
                continue
            checkbox.connect(
                "toggled(bool)",
                lambda checked, n=name: self._onExtraToggled(n, checked),
            )

        self.refresh()

    def refresh(self) -> None:
        """Populate UI labels and checkboxes from current install state."""
        kwneuro_ver = self.logic.installed_kwneuro_version()
        bridge_ver = self.logic.installed_bridge_version()
        self.ui.kwneuroVersionLabel.text = kwneuro_ver or "(not installed)"
        self.ui.bridgeVersionLabel.text = bridge_ver or "(not installed)"

        status = self.logic.extras_status()
        for name, installed in status.items():
            checkbox = getattr(self.ui, f"extra_{name}_CheckBox", None)
            if checkbox is None:
                continue
            was_blocking = checkbox.blockSignals(True)
            checkbox.checked = installed
            checkbox.blockSignals(was_blocking)

    def onInstallBridgeClicked(self) -> None:
        with slicer.util.tryWithErrorDisplay(
            _("Failed to install kwneuro_slicer_bridge."), waitCursor=True,
        ):
            self.logic.ensure_bridge_installed()
            self.refresh()

    def onVerifySetupClicked(self) -> None:
        passed, message = self.logic.verify_setup()
        self.ui.verifySetupResultLabel.text = ("PASS: " if passed else "FAIL: ") + message
        if passed:
            logging.info("KWNeuro verify setup: %s", message)
        else:
            logging.error("KWNeuro verify setup FAILED: %s", message)

    def _onExtraToggled(self, name: str, checked: bool) -> None:
        action_desc = f"install kwneuro[{name}]" if checked else f"uninstall kwneuro[{name}]"
        with slicer.util.tryWithErrorDisplay(
            _(f"Failed to {action_desc}."), waitCursor=True,
        ):
            if checked:
                self.logic.install_extra(name)
            else:
                self.logic.uninstall_extra(name)
            self.refresh()


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
        _ = logic.installed_bridge_version()
        status = logic.extras_status()
        assert set(status) == set(EXTRAS_INSTALL_SPEC)
        for value in status.values():
            assert isinstance(value, bool)
        self.delayDisplay("Test passed")
