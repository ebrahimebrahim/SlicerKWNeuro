"""Smoke test for the KWNeuroEnvironment module's logic.

Exercises the logic class's install-status probes and the bridge
verify-setup. Requires that `kwneuro` is already installed in Slicer's
Python via the env panel's Apply environment changes button. The
`kwneuro_slicer_bridge` package ships bundled with the extension.
"""
from __future__ import annotations

import unittest


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
