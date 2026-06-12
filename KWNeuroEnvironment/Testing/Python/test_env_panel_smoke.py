"""Smoke test for the KWNeuroEnvironment module's logic.

Exercises the logic class's install-status probes and the bridge
verify-setup. Requires that `kwneuro` is already installed in Slicer's
Python via the env panel's Apply environment changes button. The
`kwneuro_slicer_bridge` package ships bundled with the extension.
"""
from __future__ import annotations

import unittest
from unittest import mock


def _extras_status(**overrides: bool) -> dict[str, bool]:
    import KWNeuroEnvironment

    status = {name: False for name in KWNeuroEnvironment.EXTRAS_INSTALL_SPEC}
    status.update(overrides)
    return status


class _FakeEnvironmentLogic:
    def __init__(self, extras_status: dict[str, bool]) -> None:
        self._extras_status = dict(extras_status)
        self.calls: list[tuple[str, str | None]] = []

    def installed_kwneuro_version(self) -> str | None:
        self.calls.append(("version", None))
        return "test-version"

    def extras_status(self) -> dict[str, bool]:
        self.calls.append(("status", None))
        return dict(self._extras_status)

    def ensure_kwneuro_installed(self) -> None:
        self.calls.append(("ensure", None))

    def ensure_compatible_pytorch_installed(self) -> None:
        self.calls.append(("pytorch", None))

    def install_extra(self, name: str) -> None:
        self.calls.append(("install", name))
        self._extras_status[name] = True

    def uninstall_extra(self, name: str) -> None:
        self.calls.append(("uninstall", name))
        self._extras_status[name] = False


class TestKWNeuroEnvironmentSmoke(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_logic_methods_do_not_raise(self) -> None:
        import KWNeuroEnvironment

        self.assertEqual(
            KWNeuroEnvironment.KWNEURO_PIP_SPEC,
            f"kwneuro=={KWNeuroEnvironment.KWNEURO_PINNED_VERSION}",
        )

        logic = KWNeuroEnvironment.KWNeuroEnvironmentLogic()
        kwneuro_ver = logic.installed_kwneuro_version()
        self.assertTrue(kwneuro_ver is None or isinstance(kwneuro_ver, str))

        status = logic.extras_status()
        self.assertEqual(set(status), set(KWNeuroEnvironment.EXTRAS_INSTALL_SPEC))
        for name, value in status.items():
            self.assertIsInstance(value, bool, f"extras_status[{name!r}] must be bool")

    def test_pytorch_extra_specs_are_explicit(self) -> None:
        """Torch-consuming extras must use the PyTorch-preserving install path."""
        import KWNeuroEnvironment

        torch_extras = {
            name
            for name, spec in KWNeuroEnvironment.EXTRAS_INSTALL_SPEC.items()
            if spec.get("requires_pytorch", False)
        }
        self.assertEqual(torch_extras, KWNeuroEnvironment.PYTORCH_EXTRA_NAMES)

        for name in KWNeuroEnvironment.PYTORCH_EXTRA_NAMES:
            skip_packages = KWNeuroEnvironment.EXTRAS_INSTALL_SPEC[name]["skip_packages"]
            for package in KWNeuroEnvironment.PYTORCH_SKIP_PACKAGES:
                self.assertIn(package, skip_packages)

        self.assertIn(
            "fury",
            KWNeuroEnvironment.EXTRAS_INSTALL_SPEC["tractseg"]["skip_packages"],
        )

    def test_install_extra_preserves_pytorch_for_torch_consuming_extras(self) -> None:
        """hdbet and tractseg should install PyTorch first and skip PyTorch deps."""
        import sys
        import types

        import slicer

        import KWNeuroEnvironment

        calls: list[tuple[str, object, object, object]] = []

        def fake_pip_install(packages, skip_packages=None, requester=None) -> None:
            calls.append(("pip", list(packages), skip_packages, requester))

        def fake_ensure_pytorch() -> None:
            calls.append(("torch", None, None, None))

        fake_packaging = types.SimpleNamespace(pip_install=fake_pip_install)
        original_packaging_attr = getattr(slicer, "packaging", None)
        original_packaging_module = sys.modules.get("slicer.packaging")
        original_ensure = (
            KWNeuroEnvironment.KWNeuroEnvironmentLogic
            .ensure_compatible_pytorch_installed
        )

        slicer.packaging = fake_packaging
        sys.modules["slicer.packaging"] = fake_packaging
        KWNeuroEnvironment.KWNeuroEnvironmentLogic.ensure_compatible_pytorch_installed = (
            staticmethod(fake_ensure_pytorch)
        )
        try:
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.install_extra("hdbet")
            self.assertEqual(calls[0], ("torch", None, None, None))
            self.assertEqual(calls[1][0], "pip")
            self.assertEqual(calls[1][1], ["hd-bet == 2.0.1"])
            self.assertEqual(
                calls[1][2],
                KWNeuroEnvironment.PYTORCH_SKIP_PACKAGES,
            )

            calls.clear()
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.install_extra("tractseg")
            self.assertEqual(calls[0], ("torch", None, None, None))
            self.assertEqual(calls[1][1], ["TractSeg"])
            self.assertIn("fury", calls[1][2])
            for package in KWNeuroEnvironment.PYTORCH_SKIP_PACKAGES:
                self.assertIn(package, calls[1][2])

            calls.clear()
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.install_extra("noddi")
            self.assertEqual(calls[0][0], "pip")
            self.assertIsNone(calls[0][2])
        finally:
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.ensure_compatible_pytorch_installed = (
                staticmethod(original_ensure)
            )
            if original_packaging_attr is None:
                delattr(slicer, "packaging")
            else:
                slicer.packaging = original_packaging_attr
            if original_packaging_module is None:
                sys.modules.pop("slicer.packaging", None)
            else:
                sys.modules["slicer.packaging"] = original_packaging_module

    def test_incompatible_cuda_pytorch_is_reinstalled_through_pytorchutils(self) -> None:
        """A CUDA wheel outside ltt's compatible backend set must be replaced."""
        import sys
        import types

        import KWNeuroEnvironment

        calls: list[object] = []

        class _FakePyTorchUtilsLogic:
            @staticmethod
            def getCompatibleComputationBackends():
                calls.append("compatible")
                return ["cpu", "cu129"]

            def uninstallTorch(self):
                calls.append("uninstall")

            def installTorch(self, askConfirmation=False):
                calls.append(("install", askConfirmation))
                return object()

        fake_pytorch_utils = types.SimpleNamespace(
            PyTorchUtilsLogic=_FakePyTorchUtilsLogic,
        )
        original_pytorch_utils = sys.modules.get("PyTorchUtils")
        sys.modules["PyTorchUtils"] = fake_pytorch_utils
        try:
            with mock.patch.object(
                KWNeuroEnvironment.KWNeuroEnvironmentLogic,
                "_installed_pytorch_backend",
                staticmethod(lambda: "cu130"),
            ):
                (
                    KWNeuroEnvironment.KWNeuroEnvironmentLogic
                    .ensure_compatible_pytorch_installed()
                )
        finally:
            if original_pytorch_utils is None:
                sys.modules.pop("PyTorchUtils", None)
            else:
                sys.modules["PyTorchUtils"] = original_pytorch_utils

        self.assertEqual(calls, ["compatible", "uninstall", ("install", True)])

    def test_verify_setup_passes_when_kwneuro_installed(self) -> None:
        """If kwneuro is installed, verify_setup should pass.

        The bridge ships bundled with the extension, so its presence is
        guaranteed at this point — no need to gate on it. Only the
        external kwneuro library needs an install check.
        """
        import KWNeuroEnvironment

        logic = KWNeuroEnvironment.KWNeuroEnvironmentLogic()
        if logic.installed_kwneuro_version() is None:
            self.skipTest("kwneuro not installed; skipping verify_setup")

        passed, message = logic.verify_setup()
        self.assertTrue(passed, f"verify_setup failed: {message}")


class TestKWNeuroEnvironmentWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def tearDown(self) -> None:
        widget = self._widget()
        import KWNeuroEnvironment

        widget.logic = KWNeuroEnvironment.KWNeuroEnvironmentLogic()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroEnvironment")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_button_text_describes_combined_environment_action(self) -> None:
        widget = self._widget()
        self.assertEqual(widget.ui.installKwneuroButton.text, "Apply environment changes")
        self.assertIn("checked optional extras", widget.ui.installKwneuroButton.toolTip)

    def test_checkbox_toggles_only_change_desired_state(self) -> None:
        widget = self._widget()
        fake_logic = _FakeEnvironmentLogic(_extras_status(combat=True))
        widget.logic = fake_logic
        widget.refresh()
        fake_logic.calls.clear()

        widget.ui.extra_hdbet_CheckBox.checked = True
        widget.ui.extra_combat_CheckBox.checked = False
        self._pump()

        self.assertEqual(fake_logic.calls, [])
        self.assertTrue(widget.ui.extra_hdbet_CheckBox.checked)
        self.assertFalse(widget.ui.extra_combat_CheckBox.checked)

    def test_apply_installs_and_uninstalls_checkbox_delta(self) -> None:
        widget = self._widget()
        fake_logic = _FakeEnvironmentLogic(_extras_status(combat=True))
        widget.logic = fake_logic
        widget.refresh()

        widget.ui.extra_hdbet_CheckBox.checked = True
        widget.ui.extra_combat_CheckBox.checked = False
        fake_logic.calls.clear()

        widget.ui.installKwneuroButton.click()
        self._pump()

        self.assertEqual(
            fake_logic.calls,
            [
                ("ensure", None),
                ("status", None),
                ("install", "hdbet"),
                ("uninstall", "combat"),
                ("version", None),
                ("status", None),
            ],
        )
        self.assertTrue(widget.ui.extra_hdbet_CheckBox.checked)
        self.assertFalse(widget.ui.extra_combat_CheckBox.checked)

    def test_apply_repairs_pytorch_for_installed_torch_extra(self) -> None:
        widget = self._widget()
        fake_logic = _FakeEnvironmentLogic(_extras_status(hdbet=True))
        widget.logic = fake_logic
        widget.refresh()
        fake_logic.calls.clear()

        widget.ui.installKwneuroButton.click()
        self._pump()

        self.assertEqual(
            fake_logic.calls,
            [
                ("ensure", None),
                ("status", None),
                ("pytorch", None),
                ("version", None),
                ("status", None),
            ],
        )

    def test_refresh_discards_pending_checkbox_edits(self) -> None:
        widget = self._widget()
        fake_logic = _FakeEnvironmentLogic(_extras_status(combat=True))
        widget.logic = fake_logic
        widget.refresh()

        widget.ui.extra_hdbet_CheckBox.checked = True
        widget.ui.extra_combat_CheckBox.checked = False
        self._pump()
        self.assertTrue(widget.ui.extra_hdbet_CheckBox.checked)
        self.assertFalse(widget.ui.extra_combat_CheckBox.checked)

        widget.refresh()

        self.assertFalse(widget.ui.extra_hdbet_CheckBox.checked)
        self.assertTrue(widget.ui.extra_combat_CheckBox.checked)


if __name__ == "__main__":
    unittest.main()
