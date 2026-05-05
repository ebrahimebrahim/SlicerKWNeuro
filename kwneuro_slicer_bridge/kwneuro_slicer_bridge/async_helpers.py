"""Async / progress helpers for KWNeuro scripted modules.

The pattern is the same as Slicer's ``_pip_install_with_dialog``
(``Base/Python/slicer/packaging.py``) — show a modal dialog, run the
work on a background thread, busy-wait on the main thread while
pumping ``slicer.app.processEvents`` so the UI stays responsive —
but without an extra ``qt.QTimer`` indirection: the busy-wait polls
the worker's ``threading.Event`` directly.

Public entry points:

- :func:`run_in_worker` — fire-and-forget background work. Returns a
  handle whose ``done_event`` flips once the worker exits; the
  caller is responsible for waiting / draining its
  ``progress_queue``. No Qt objects are created.
- :func:`run_with_progress_dialog` — blocking-from-caller-perspective
  wrapper that shows a modal :class:`ProgressDialog`, dispatches the
  callable into a worker thread, and busy-pumps the main event loop
  until the worker is done. Drains the worker's ``progress_queue``
  inline so tqdm capture lines stream into the dialog's log.

Plus :func:`ensure_extras_installed`, which probes the
``KWNeuroEnvironment`` install-status surface and raises a clear
"open KWNeuroEnvironment and tick the <extra> checkbox" error if the
module under the cursor needs an extra that is not installed.
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
    (regardless of success / failure). Read ``result`` /
    ``exception`` only after ``done_event`` is set.

    ``progress_queue`` is the worker-side sink for progress lines.
    Callers who want per-step progress updates push strings into it
    (directly, or via a tqdm-capture shim installed around ``fn``)
    and drain it from the main thread.
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
    progress_queue: queue.Queue | None = None,
) -> WorkerHandle:
    """Run ``fn`` on a background thread and return a handle to it.

    ``fn`` is a zero-arg callable (use a lambda or functools.partial
    to bind arguments). The worker exits as soon as ``fn`` returns or
    raises; the result lands on ``handle.result`` and any exception
    on ``handle.exception``, and ``handle.done_event`` flips. Callers
    are responsible for waiting on the event and reading the fields.

    ``progress_queue`` lets the caller pre-build the queue (so it can
    be captured by reference inside ``fn`` without racing thread
    start). If omitted, a fresh queue is created on the handle. No Qt
    objects are created here, so this is safe to call from any
    thread.
    """
    if progress_queue is None:
        handle = WorkerHandle(thread=None)  # type: ignore[arg-type]
    else:
        handle = WorkerHandle(thread=None, progress_queue=progress_queue)  # type: ignore[arg-type]

    def _worker() -> None:
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 — surface everything via the handle
            handle.exception = exc
        else:
            handle.result = result
        finally:
            handle.done_event.set()

    handle.thread = threading.Thread(  # type: ignore[misc]
        target=_worker, name="kwneuro-worker", daemon=True,
    )
    handle.thread.start()
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
    progress_queue: queue.Queue | None = None,
) -> T:
    """Run ``fn`` on a background thread behind a modal progress dialog.

    Blocks from the caller's perspective; returns whatever ``fn``
    returned, or re-raises whatever ``fn`` raised. Progress lines
    pushed into the worker's ``progress_queue`` — either directly by
    ``fn`` or by a :class:`TqdmToProgressDialog` shim (enabled via
    ``capture_tqdm=True``) — are forwarded into the dialog's log area
    by the same loop that pumps ``slicer.app.processEvents``.

    :param capture_tqdm: If True, wrap ``fn`` in a
        :class:`TqdmToProgressDialog` context manager so tqdm progress
        lines from dipy call sites are routed into the dialog log.
    :param progress_queue: Optional caller-provided queue that ``fn``
        can write progress strings into directly (useful when ``fn``
        does its own non-tqdm I/O — e.g. plain ``urllib.urlretrieve``).
        If omitted, a fresh queue is built internally.
    """
    import qt
    import slicer

    dialog = ProgressDialog(title=title, status=status, parent=parent)
    dialog.show()
    slicer.app.processEvents()

    if progress_queue is None:
        progress_queue = queue.Queue()

    if capture_tqdm:
        def _runner() -> T:
            with TqdmToProgressDialog(progress_queue):
                return fn()
    else:
        _runner = fn  # type: ignore[assignment]

    handle = run_in_worker(_runner, progress_queue=progress_queue)

    def _drain() -> None:
        while True:
            try:
                line = progress_queue.get_nowait()
            except queue.Empty:
                return
            try:
                dialog.appendLog(line)
            except Exception:  # noqa: BLE001
                logging.exception("appendLog raised; continuing")

    # Busy-wait on done_event directly. `processEvents` keeps the dialog
    # responsive (Details toggle, redraws); the inline drain forwards
    # tqdm lines without a separate QTimer poller.
    #
    # Important: use ``done_event.wait(timeout=...)`` rather than
    # ``QThread.msleep``. ``threading.Event.wait`` is pure Python and
    # releases the GIL for the full timeout, letting a CPU-bound
    # Python worker run uninterrupted. ``QThread.msleep`` GIL-starved
    # nibabel + dipy work to a 28× slowdown in testing.
    while not handle.done_event.wait(timeout=0.05):
        _drain()
        slicer.app.processEvents()
    _drain()  # final flush after the worker set done_event

    dialog.close()
    slicer.app.processEvents()

    if handle.exception is not None:
        exc = handle.exception
        # Worker-thread tracebacks survive a cross-thread re-raise via
        # `with_traceback`; appending the formatted exception to the
        # dialog log is a no-op once the dialog is closed but useful
        # if the caller reopens it.
        dialog.appendLog("".join(traceback.format_exception(exc)))
        raise exc.with_traceback(exc.__traceback__)
    return handle.result  # type: ignore[no-any-return]


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
