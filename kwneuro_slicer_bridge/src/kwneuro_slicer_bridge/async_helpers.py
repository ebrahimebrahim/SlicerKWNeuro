"""Async / progress helpers for KWNeuro scripted modules.

Two public entry points, both modelled on Slicer's
``_pip_install_with_dialog`` / ``_pip_install_nonblocking`` pair
(``Base/Python/slicer/packaging.py``) and the background-process
output pattern in ``slicer.util._startAsyncProcessOutputHandling``:

- :func:`run_in_worker` — fire-and-forget background work. The worker
  thread writes its result / exception into a shared handle; a
  main-thread ``qt.QTimer`` polls the handle and dispatches
  ``on_complete`` on the main thread. Progress lines that the worker
  pushes into the returned ``progress_queue`` are likewise drained on
  the main thread and forwarded to ``on_progress``. No QObject is
  touched from the worker thread.
- :func:`run_with_progress_dialog` — blocking-from-caller-perspective
  wrapper that shows a modal :class:`ProgressDialog` and keeps the UI
  responsive by pumping ``slicer.app.processEvents`` until the worker
  completes.

Plus :func:`ensure_extras_installed`, which probes the
``KWNeuroEnvironment`` install-status surface and raises a clear
"open KWNeuroEnvironment and tick the <extra> checkbox" error if the
module under the cursor needs an extra that is not installed.

Invariant: ``run_in_worker`` must be called from the main Qt thread
(the thread that owns ``qt.QTimer`` scheduling). Calling it from
another thread is not supported and will produce Qt
"timers can only be used with threads started with QThread" warnings.
"""
from __future__ import annotations

import importlib
import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_POLL_INTERVAL_MS = 50

# Global lock guarding TqdmToProgressDialog's monkey-patch of dipy's
# per-submodule tqdm bindings. The patch mutates module-level state
# (sys.modules[...].tqdm), so two concurrent contexts would clobber
# each other's "original" bookkeeping. We enforce one-at-a-time and
# raise rather than silently leaking.
_TQDM_PATCH_LOCK = threading.Lock()

# Modules that rebind tqdm and that we want to capture progress from.
# Dipy's pattern is universal but bindings are per-submodule, so we have
# to list each one. Some do `from tqdm import tqdm`, others
# `from tqdm.auto import tqdm` — doesn't matter at patch time since we
# just rebind the name on each listed module. Extend here as future
# modules come online.
_TQDM_REBINDINGS: tuple[tuple[str, str], ...] = (
    ("dipy.denoise.patch2self", "tqdm"),  # patch2self (Denoise module)
    ("dipy.data.fetcher", "tqdm"),        # dataset fetches (Importer module)
)


@dataclass
class WorkerHandle:
    """Handle to a call dispatched via :func:`run_in_worker`.

    Not a cancellation token — the long-running callable runs to
    completion. ``done_event`` flips once the worker has finished
    (regardless of success / failure). The main-thread poller reads
    ``result`` / ``exception`` only after ``done_event`` is set.

    ``progress_queue`` is the worker-side sink for progress lines.
    Callers who want per-step progress updates push strings into it
    (directly, or via a tqdm-capture shim installed around ``fn``).
    The main-thread poller drains it and invokes ``on_progress``.
    """

    thread: threading.Thread
    done_event: threading.Event = field(default_factory=threading.Event)
    progress_queue: queue.Queue = field(default_factory=queue.Queue)
    result: Any = None
    exception: BaseException | None = None

    @property
    def done(self) -> bool:
        return self.done_event.is_set()


def run_in_worker(
    fn: Callable[[], T],
    *,
    on_complete: Callable[[T | None, BaseException | None], None],
    on_progress: Callable[[str], None] | None = None,
    progress_queue: queue.Queue | None = None,
) -> WorkerHandle:
    """Run ``fn`` on a background thread; marshal callbacks to the main Qt thread.

    ``fn`` is a zero-arg callable (use a lambda or functools.partial to
    bind arguments). ``on_complete(result, exception)`` fires exactly
    once after ``fn`` returns or raises; exactly one of ``result`` or
    ``exception`` is not None. ``on_progress(line)`` fires zero or more
    times as lines are pushed into the handle's ``progress_queue`` — the
    worker side is responsible for populating it (e.g. via a
    ``TqdmToProgressDialog`` shim).

    ``progress_queue`` allows the caller to pre-build the queue (so it
    can be referenced from inside ``fn`` without racing thread start).
    If omitted, a fresh queue is created and stored on the handle.

    Both callbacks run on the main Qt thread, driven by a
    ``qt.QTimer`` that this function schedules from whichever thread
    called it — that thread must be the main Qt thread.
    """
    import qt

    if progress_queue is None:
        handle = WorkerHandle(thread=None)  # type: ignore[arg-type]
    else:
        handle = WorkerHandle(thread=None, progress_queue=progress_queue)  # type: ignore[arg-type]

    def _worker() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 — surface everything to on_complete
            handle.exception = exc
        else:
            handle.result = result
        finally:
            handle.done_event.set()

    handle.thread = threading.Thread(  # type: ignore[misc]
        target=_worker, name="kwneuro-worker", daemon=True,
    )
    handle.thread.start()

    def _drain_progress() -> None:
        if on_progress is None:
            return
        while True:
            try:
                line = handle.progress_queue.get_nowait()
            except queue.Empty:
                return
            try:
                on_progress(line)
            except Exception:  # noqa: BLE001
                # A broken on_progress must not halt the poll loop —
                # that would hang run_with_progress_dialog's busy wait.
                logging.exception("on_progress callback raised; continuing")

    def _poll() -> None:
        _drain_progress()

        if not handle.done_event.is_set():
            qt.QTimer.singleShot(_POLL_INTERVAL_MS, _poll)
            return

        # The worker has finished; flush any lines it pushed between
        # our last drain and its done_event.set() so the final batch
        # (e.g. tqdm's last "193/193") isn't silently dropped.
        _drain_progress()

        if handle.exception is not None:
            on_complete(None, handle.exception)
        else:
            on_complete(handle.result, None)

    qt.QTimer.singleShot(0, _poll)
    return handle


class ProgressDialog:
    """Modal progress dialog with status label, indeterminate bar, collapsible log.

    Adapted from ``slicer.packaging._PipProgressDialog``. Kept
    deliberately thin: a label, an indeterminate progress bar, and a
    collapsed-by-default details panel with a monospace log view.
    """

    def __init__(
        self,
        title: str = "Working...",
        status: str = "Running...",
        parent: Any = None,
    ) -> None:
        import ctk
        import qt
        import slicer

        self._dialog = qt.QDialog(parent or slicer.util.mainWindow())
        self._dialog.setModal(True)
        self._dialog.setWindowTitle(title)
        self._dialog.setWindowFlags(
            self._dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint,
        )
        self._escapeShortcut = qt.QShortcut(
            qt.QKeySequence(qt.Qt.Key_Escape), self._dialog,
        )
        self._escapeShortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)

        layout = qt.QVBoxLayout(self._dialog)

        self.statusLabel = qt.QLabel(status)
        layout.addWidget(self.statusLabel)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progressBar)

        self.detailsButton = ctk.ctkCollapsibleButton()
        self.detailsButton.text = "Details"
        self.detailsButton.collapsed = True
        detailsLayout = qt.QVBoxLayout(self.detailsButton)

        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setMinimumHeight(150)
        self.logText.setMaximumHeight(300)
        font = qt.QFont("Monospace")
        font.setStyleHint(qt.QFont.TypeWriter)
        self.logText.setFont(font)
        detailsLayout.addWidget(self.logText)

        layout.addWidget(self.detailsButton)
        self._dialog.resize(500, 120)

    def show(self) -> None:
        self._dialog.show()

    def close(self) -> None:
        self._dialog.close()

    def setStatus(self, text: str) -> None:
        self.statusLabel.setText(text)

    def appendLog(self, line: str) -> None:
        self.logText.appendPlainText(line)
        scrollBar = self.logText.verticalScrollBar()
        scrollBar.setValue(scrollBar.maximum)


class TqdmToProgressDialog:
    """Context manager: route tqdm updates from known call sites into a queue.

    Dipy-style modules typically do ``from tqdm import tqdm`` at the top
    of a module, so they hold a local binding that's unaffected by
    patches to ``tqdm.tqdm`` itself. We therefore patch each rebinding
    listed in :data:`_TQDM_REBINDINGS` directly. On enter, each bound
    name is replaced with a queue-writing subclass of the original tqdm;
    on exit, originals are restored.

    The subclass overrides ``display`` (tqdm's refresh hook); every
    refresh emits a formatted line like ``"Fitting and Denoising:
    12/193"`` into the provided queue. The main-thread poller inside
    :func:`run_in_worker` drains the queue and forwards each line to
    the progress dialog's log.

    This is deliberately fragile: new dipy submodules won't be
    captured until they're added to :data:`_TQDM_REBINDINGS`. The
    smoke tests in ``test_bridge_async_helpers.py`` (including a
    regex scan of dipy's source for ``from tqdm ... import tqdm``)
    are the safety net that catches "we added a module but forgot
    the binding" regressions.
    """

    def __init__(self, progress_queue: queue.Queue) -> None:
        self._queue = progress_queue
        self._originals: list[tuple[Any, str, Any]] = []
        self._lock_held = False

    def __enter__(self) -> TqdmToProgressDialog:
        from tqdm import tqdm as real_tqdm

        # One-at-a-time enforcement. The monkey-patch mutates module
        # globals (dipy submodule `.tqdm`), so two concurrent contexts
        # would race on `setattr` and one would restore the other's
        # subclass as "original", permanently leaking the capture
        # machinery. Rather than silently corrupt, refuse.
        if not _TQDM_PATCH_LOCK.acquire(blocking=False):
            msg = (
                "Another TqdmToProgressDialog context is already active. "
                "This helper is intentionally single-use — don't run two "
                "kwneuro pipeline operations with capture_tqdm=True in "
                "parallel."
            )
            raise RuntimeError(msg)
        self._lock_held = True

        capture_queue = self._queue

        class _QueueingTqdm(real_tqdm):  # type: ignore[misc, valid-type]
            def display(self, msg: Any = None, pos: Any = None) -> Any:  # type: ignore[override]
                result = super().display(msg, pos)
                try:
                    line = self.format_meter(**self.format_dict)
                except Exception:  # noqa: BLE001 — best-effort
                    line = None
                if line:
                    capture_queue.put(line)
                return result

        try:
            for module_name, attr in _TQDM_REBINDINGS:
                try:
                    mod = importlib.import_module(module_name)
                except ImportError:
                    continue
                if hasattr(mod, attr):
                    self._originals.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, _QueueingTqdm)
        except BaseException:
            # If patching itself fails partway, restore what we've done
            # and release the lock before re-raising.
            for mod, attr, original in self._originals:
                setattr(mod, attr, original)
            self._originals = []
            _TQDM_PATCH_LOCK.release()
            self._lock_held = False
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            for mod, attr, original in self._originals:
                setattr(mod, attr, original)
            self._originals = []
        finally:
            if self._lock_held:
                _TQDM_PATCH_LOCK.release()
                self._lock_held = False


def run_with_progress_dialog(
    fn: Callable[[], T],
    *,
    title: str = "Working...",
    status: str = "Running...",
    parent: Any = None,
    capture_tqdm: bool = False,
) -> T:
    """Run ``fn`` on a background thread behind a modal progress dialog.

    Blocks from the caller's perspective; returns whatever ``fn``
    returned, or re-raises whatever ``fn`` raised. Progress lines
    pushed into the worker's ``progress_queue`` — either directly by
    ``fn`` or by a :class:`TqdmToProgressDialog` shim (enabled via
    ``capture_tqdm=True``) — are forwarded into the dialog's log area
    on the main thread.

    :param capture_tqdm: If True, wrap ``fn`` in a
        :class:`TqdmToProgressDialog` context manager so tqdm progress
        lines from dipy call sites are routed into the dialog log.
    """
    import qt
    import slicer

    dialog = ProgressDialog(title=title, status=status, parent=parent)
    dialog.show()
    slicer.app.processEvents()

    completed = threading.Event()
    result: dict[str, Any] = {"value": None, "exception": None}

    def _on_complete(value: T | None, exc: BaseException | None) -> None:
        result["value"] = value
        result["exception"] = exc
        completed.set()

    # The TqdmToProgressDialog context manager must wrap `fn` in such a
    # way that it activates on the worker thread (so monkey-patching
    # affects the dipy import the worker will use) and reverts after
    # fn completes. We build the queue up front and pass it both into
    # run_in_worker and into the context manager — capturing it in a
    # closure, not fetching it off the handle afterwards, to avoid a
    # race where the worker dispatches fn before we bind.
    if capture_tqdm:
        progress_queue: queue.Queue = queue.Queue()

        def _wrapped_with_capture() -> T:
            with TqdmToProgressDialog(progress_queue):
                return fn()

        run_in_worker(
            _wrapped_with_capture,
            on_complete=_on_complete,
            on_progress=dialog.appendLog,
            progress_queue=progress_queue,
        )
    else:
        run_in_worker(fn, on_complete=_on_complete, on_progress=dialog.appendLog)

    while not completed.is_set():
        slicer.app.processEvents()
        qt.QThread.msleep(10)

    dialog.close()

    if result["exception"] is not None:
        exc = result["exception"]
        # Preserve the worker traceback in the log AND on the raised
        # exception. Python 3 attaches `__traceback__` to the exception
        # object at raise-time in the worker thread, so re-raising here
        # naturally carries that frame; but we pass it through
        # `with_traceback` explicitly to defend against any interpreter
        # that might reset it on cross-thread re-raise.
        dialog.appendLog("".join(traceback.format_exception(exc)))
        raise exc.with_traceback(exc.__traceback__)
    return result["value"]  # type: ignore[no-any-return]


def ensure_extras_installed(names: list[str]) -> None:
    """Raise RuntimeError if any of the named kwneuro extras are not installed.

    ``names`` elements must match keys in
    ``KWNeuroEnvironment.EXTRAS_INSTALL_SPEC``
    (``hdbet``/``noddi``/``tractseg``/``combat``). The error message
    nudges the user at the KWNeuroEnvironment panel so they can fix
    it with a checkbox click rather than manual pip.
    """
    try:
        from KWNeuroEnvironment import KWNeuroEnvironmentLogic
    except ImportError as exc:  # pragma: no cover — Slicer-only import path
        msg = (
            "Could not import KWNeuroEnvironment to check optional extras. "
            "Is the KWNeuro extension loaded?"
        )
        raise RuntimeError(msg) from exc

    status = KWNeuroEnvironmentLogic.extras_status()
    missing = [n for n in names if not status.get(n, False)]
    if not missing:
        return
    pretty = ", ".join(f"kwneuro[{n}]" for n in missing)
    msg = (
        f"This module needs the following kwneuro extras that are not "
        f"currently installed: {pretty}. Open the KWNeuroEnvironment "
        f"module and tick the relevant checkbox(es) under 'Optional "
        f"extras' to install."
    )
    raise RuntimeError(msg)
