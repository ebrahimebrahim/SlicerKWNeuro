"""Smoke test for the KWNeuroEnvironment module's logic.

Exercises the logic class's install-status probes and the bridge
verify-setup. Requires that `kwneuro` and `kwneuro_slicer_bridge` are
already installed in Slicer's Python (via the env panel's Install /
Update button, or `PythonSlicer -m pip install --no-deps -e
slicer-extn/kwneuro_slicer_bridge` in development).
"""
from __future__ import annotations

import unittest


class TestKWNeuroEnvironmentSmoke(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_logic_methods_do_not_raise(self) -> None:
        import KWNeuroEnvironment

        self.assertEqual(KWNeuroEnvironment.KWNEURO_PINNED_VERSION, "1.0.0")
        self.assertEqual(KWNeuroEnvironment.KWNEURO_PIP_SPEC, "kwneuro==1.0.0")

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


if __name__ == "__main__":
    unittest.main()
