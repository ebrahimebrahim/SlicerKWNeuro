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


class TestTqdmToProgressDialog(unittest.TestCase):
    """Tqdm capture is a monkey-patching context manager, so it has
    more fragility than most helpers. These tests pin down:
      * Patching actually redirects dipy's patch2self tqdm binding.
      * Queue receives at least one formatted progress line per refresh.
      * Original tqdm is restored on context exit (including on raise).
      * Known dipy module bindings in _TQDM_REBINDINGS still resolve.
    """

    def test_known_bindings_resolve(self) -> None:
        import importlib

        from kwneuro_slicer_bridge.async_helpers import _TQDM_REBINDINGS

        self.assertGreater(len(_TQDM_REBINDINGS), 0)
        for module_name, attr in _TQDM_REBINDINGS:
            mod = importlib.import_module(module_name)
            self.assertTrue(
                hasattr(mod, attr),
                f"Binding {module_name}.{attr} not found — dipy may have "
                f"renamed / moved tqdm and our capture list is stale.",
            )

    def test_context_manager_patches_and_restores(self) -> None:
        import dipy.denoise.patch2self as p2s_mod

        from kwneuro_slicer_bridge.async_helpers import (
            TqdmToProgressDialog,
        )

        import queue as _queue

        original = p2s_mod.tqdm
        q: _queue.Queue = _queue.Queue()

        with TqdmToProgressDialog(q):
            self.assertIsNot(p2s_mod.tqdm, original)
            # Exercise the patched tqdm like dipy would.
            for _ in p2s_mod.tqdm(range(3), desc="Test"):
                pass

        # Context exit must restore the original binding.
        self.assertIs(p2s_mod.tqdm, original)

        # At least one line should have landed in the queue. tqdm
        # typically refreshes once or twice for a short 3-tick iterator.
        collected = []
        while not q.empty():
            collected.append(q.get_nowait())
        self.assertGreater(len(collected), 0)
        self.assertTrue(any("Test" in line for line in collected))

    def test_context_manager_restores_on_exception(self) -> None:
        import dipy.denoise.patch2self as p2s_mod

        from kwneuro_slicer_bridge.async_helpers import (
            TqdmToProgressDialog,
        )

        import queue as _queue

        original = p2s_mod.tqdm

        class Boom(Exception):
            pass

        try:
            with TqdmToProgressDialog(_queue.Queue()):
                raise Boom
        except Boom:
            pass

        self.assertIs(p2s_mod.tqdm, original)

    def test_run_with_progress_dialog_capture_tqdm_forwards_lines(self) -> None:
        """End-to-end: capture_tqdm=True routes worker-side tqdm into the dialog log."""
        import dipy.denoise.patch2self as p2s_mod

        from kwneuro_slicer_bridge import async_helpers, run_with_progress_dialog

        def _worker() -> int:
            n = 0
            for _ in p2s_mod.tqdm(range(8), desc="DialogCapture"):
                n += 1
            return n

        # Intercept appendLog on the ProgressDialog so we can assert
        # lines actually make it through the queue -> main-thread
        # drain -> dialog pipeline. Wrapping the class method is the
        # narrowest seam.
        captured: list[str] = []
        real_append = async_helpers.ProgressDialog.appendLog

        def _recording_append(self, line: str) -> None:
            captured.append(line)
            real_append(self, line)

        async_helpers.ProgressDialog.appendLog = _recording_append
        try:
            result = run_with_progress_dialog(
                _worker,
                title="Test",
                status="Running capture test...",
                capture_tqdm=True,
            )
        finally:
            async_helpers.ProgressDialog.appendLog = real_append

        self.assertEqual(result, 8)
        self.assertTrue(
            any("DialogCapture" in line for line in captured),
            f"Expected a DialogCapture progress line in {captured!r}",
        )
        # Tqdm binding must be restored after the context exits.
        from tqdm import tqdm as real_tqdm
        self.assertIs(p2s_mod.tqdm, real_tqdm)

    def test_concurrent_tqdm_contexts_refused(self) -> None:
        """Second TqdmToProgressDialog __enter__ while another is active must raise."""
        import queue as _queue

        from kwneuro_slicer_bridge.async_helpers import TqdmToProgressDialog

        outer = TqdmToProgressDialog(_queue.Queue())
        outer.__enter__()
        try:
            inner = TqdmToProgressDialog(_queue.Queue())
            with self.assertRaises(RuntimeError):
                inner.__enter__()
        finally:
            outer.__exit__(None, None, None)

    def test_no_uncovered_tqdm_imports_in_dipy(self) -> None:
        """Scan dipy's site-packages for `from tqdm import tqdm` — every
        hit must be covered by ``_TQDM_REBINDINGS`` or whitelisted below.

        Low-cost safety net for the fragile monkey-patch: if a future
        dipy release adds a tqdm-using submodule that a kwneuro pipeline
        call will route through, this test flags it before the user
        notices missing progress lines in a future KWNeuro module.
        """
        import re
        from pathlib import Path

        import dipy

        from kwneuro_slicer_bridge.async_helpers import _TQDM_REBINDINGS

        dipy_root = Path(dipy.__file__).parent
        covered: set[str] = {module for module, _ in _TQDM_REBINDINGS}

        # Submodules dipy uses tqdm in that kwneuro does not currently
        # route through. Expand this list as kwneuro starts using more
        # of dipy; every addition forces a conscious decision about
        # whether to capture its progress.
        known_uncovered: set[str] = {
            # multi_voxel_fit's tqdm bars are disabled unless the caller
            # passes verbose=True. kwneuro's Dti.estimate_dti does not,
            # so TensorModel.fit produces no tqdm output in practice.
            "dipy.reconst.multi_voxel",
            # dipy.sims.force is the FORCE model simulator. kwneuro
            # doesn't call it, and its bars are verbose=True-gated.
            "dipy.sims.force",
            # dipy.utils.parallel's tqdm bars are all `disable=not
            # verbose`; kwneuro never passes verbose=True, so there's
            # no progress to capture.
            "dipy.utils.parallel",
        }

        # Catch both `from tqdm import tqdm` and `from tqdm.auto import
        # tqdm` / `from tqdm.std import tqdm`. Missing this was how a
        # dipy submodule (dipy.data.fetcher) slipped past the first
        # version of this test.
        pattern = re.compile(
            r"^\s*from\s+tqdm(?:\.\w+)*\s+import\s+tqdm\b", re.MULTILINE,
        )
        missing: list[str] = []
        for py_file in dipy_root.rglob("*.py"):
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not pattern.search(text):
                continue
            rel = py_file.relative_to(dipy_root.parent).with_suffix("")
            module_name = ".".join(rel.parts)
            if module_name in covered or module_name in known_uncovered:
                continue
            missing.append(module_name)

        self.assertEqual(
            missing, [],
            "Found dipy submodules that import `from tqdm import tqdm` but "
            "are not in _TQDM_REBINDINGS or known_uncovered. If a kwneuro "
            "pipeline routes through one of these, add it to "
            "_TQDM_REBINDINGS; otherwise add it to known_uncovered in this "
            f"test. Missing: {missing}",
        )


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
