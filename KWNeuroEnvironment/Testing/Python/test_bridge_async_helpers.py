"""Smoke tests for ``kwneuro_slicer_bridge.async_helpers``.

Exercises the fire-and-forget worker and the progress-dialog wrapper
on trivially-fast closures — we're confirming the threading /
qt.QTimer.singleShot / dialog-loop plumbing works end-to-end, not the
behaviour of any real pipeline call.
"""
from __future__ import annotations

import time
import unittest


class TestRunInWorker(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_success_path_invokes_on_complete_with_result(self) -> None:
        import qt
        import slicer

        from kwneuro_slicer_bridge import run_in_worker

        captured: dict[str, object] = {}

        def on_complete(result: object, exc: BaseException | None) -> None:
            captured["result"] = result
            captured["exception"] = exc

        handle = run_in_worker(lambda: 7 * 6, on_complete=on_complete)

        # Wait for the worker thread itself to finish, plus pump the
        # Qt event loop so the marshalled on_complete actually fires.
        handle.done_event.wait(timeout=5.0)
        deadline = time.time() + 5.0
        while "result" not in captured and time.time() < deadline:
            slicer.app.processEvents()
            qt.QThread.msleep(5)

        self.assertEqual(captured.get("result"), 42)
        self.assertIsNone(captured.get("exception"))
        self.assertTrue(handle.done)

    def test_exception_path_invokes_on_complete_with_exception(self) -> None:
        import qt
        import slicer

        from kwneuro_slicer_bridge import run_in_worker

        captured: dict[str, object] = {}

        def boom() -> None:
            msg = "worker raised"
            raise RuntimeError(msg)

        def on_complete(result: object, exc: BaseException | None) -> None:
            captured["result"] = result
            captured["exception"] = exc

        handle = run_in_worker(boom, on_complete=on_complete)
        handle.done_event.wait(timeout=5.0)
        deadline = time.time() + 5.0
        while "exception" not in captured and time.time() < deadline:
            slicer.app.processEvents()
            qt.QThread.msleep(5)

        self.assertIsNone(captured.get("result"))
        self.assertIsInstance(captured.get("exception"), RuntimeError)


class TestRunWithProgressDialog(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_returns_worker_result(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        result = run_with_progress_dialog(
            lambda: "hello", title="Test", status="Running test...",
        )
        self.assertEqual(result, "hello")

    def test_reraises_worker_exception(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        def boom() -> None:
            msg = "worker raised"
            raise ValueError(msg)

        with self.assertRaises(ValueError):
            run_with_progress_dialog(boom, title="Test", status="Running test...")


class TestEnsureExtrasInstalled(unittest.TestCase):
    def test_ok_when_no_extras_requested(self) -> None:
        from kwneuro_slicer_bridge import ensure_extras_installed

        # Must not raise.
        ensure_extras_installed([])

    def test_raises_for_unknown_extra(self) -> None:
        from kwneuro_slicer_bridge import ensure_extras_installed

        # "not_a_real_extra" is not in KWNeuroEnvironment.EXTRAS_INSTALL_SPEC.
        # extras_status() returns False for it (via .get(name, False)),
        # so we should get a RuntimeError complaining about it.
        with self.assertRaises(RuntimeError) as ctx:
            ensure_extras_installed(["not_a_real_extra"])
        self.assertIn("not_a_real_extra", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
