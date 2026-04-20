# E6 — Long-op async UX probe

## What this experiment is (and isn't)

Phase 0 can measure **wall-clock** for representative kwneuro stages
running inside Slicer, and can inspect the shape of the available async
mechanisms in Slicer core. It cannot meaningfully probe **UI-freeze
behavior** because headless Slicer (`--no-main-window`) has no UI.
Responsiveness benchmarking belongs to Phase 1, when the first scripted
module with an Apply button exists.

## Wall-clock baseline (from E4)

Sherbrooke 3-shell HARDI193 (4D, `(128, 128, 60, 193)`, ~184 MB NIfTI):

| Step | Wall-clock | Source |
|------|-----------|--------|
| `NiftiVolumeResource(path).load()` (nibabel) | 1.86 s | E4 |
| `SlicerVolumeResource.from_resource` (4D → `vtkMRMLVectorVolumeNode`) | 0.33 s | E4 |
| `dwi.denoise()` (DIPY Patch2Self) | **159.9 s** | E4 |

Sherbrooke is a small dataset by research standards. Real-world subject
volumes are often 2-5× larger per dimension; denoise time scales roughly
linearly with number of voxels × number of gradients. A representative
non-sample volume could plausibly take 5-15 minutes for denoise.

## Per-stage classification (based on CLAUDE.md, source inspection, and E4)

| kwneuro stage | Algorithm | Expected wall-clock on typical data | Sync acceptable? |
|---------------|-----------|------------------------------------|------------------|
| `denoise_dwi` | DIPY Patch2Self | 1-10 min | **No** |
| `Dti.estimate_dti` | DIPY TensorModel | 5-30 s | Borderline |
| `estimate_response_function` | DIPY SSST | 5-20 s | Yes |
| `compute_csd_peaks` | DIPY CSD | 20-120 s | **No** |
| `harmonize_volumes` | neuroCombat | 5-30 s | Yes |
| `brain_extract_batch` | HD-BET (NN) | 1-3 min/subject + init | **No** |
| `Noddi.estimate_noddi` | AMICO | 2-10 min | **No** |
| `extract_tractseg` | TractSeg (NN) | 2-10 min | **No** |
| `register_volumes` (SyN) | ANTs SyN | 5-30 min | **No** |
| `build_template` | iterative groupwise | 20 min - 2 h | **No** |

Anything "**No**" freezes Slicer for the user if called synchronously from
a Widget's button handler. Anything "Borderline" is tolerable but
noticeable.

## Async mechanisms available in Slicer Python

### 1. `slicer.cli.run(module, node, parameters, wait_for_completion=False)`

Defined in `~/Slicer/Base/Python/slicer/cli.py`. Designed for modules
authored as SlicerExecutionModel CLI executables (XML descriptor + C or
Python entry point). Returns a `vtkMRMLCommandLineModuleNode` that can be
observed for status updates. The module runs in a separate process, so
the main thread is fully free. Progress and log reporting are built in.

**Wrapping a Python pipeline as a CLI module** requires:

- An XML parameter descriptor (SlicerExecutionModel format).
- A small Python entry script that reads parsed parameters, calls
  kwneuro, and writes outputs to disk paths supplied by the framework.
- Registering the module in the extension's `CMakeLists.txt`.

This is moderate overhead — about a day of work per stage to shim
correctly — but the benefits are large for the really heavy stages:
Slicer can cancel the job, shows progress natively, and there is zero UI
freeze regardless of the work done in Python.

Best fit: **`extract_tractseg`**, **`Noddi.estimate_noddi`**,
**`register_volumes`** (SyN), **`build_template`**, **`brain_extract_batch`**.
The wall-clock cost justifies the shim.

### 2. `qt.QThread` with signal marshalling

The idiomatic scripted-module pattern, with a working reference in
`~/Slicer/Base/Python/slicer/packaging.py` (the progress-dialog path).
Create a `QThread` subclass whose `run()` calls the kwneuro function,
emit a signal when done, connect it to a main-thread slot that consumes
the result. Use `slicer.app.processEvents()` in any polling loop.

Lighter to set up than a CLI shim but does not isolate the worker from
the main process. The kwneuro function runs in the same Python
interpreter — meaning any segfault or OOM takes Slicer down.

Best fit: **`denoise_dwi`**, **`compute_csd_peaks`**,
**`Dti.estimate_dti`** (if anyone ever wants responsiveness during
the 5-30 s fit). Medium-length ops where a CLI shim would be overkill.

### 3. Synchronous Python call inside the Widget button

Acceptable for stages under ~5 s. Wrap in
`slicer.util.tryWithErrorDisplay(...)` for clean error reporting. Show
`slicer.util.setOverrideCursor(qt.Qt.BusyCursor)` via a context manager
so the user sees a busy cursor.

Best fit: **`estimate_response_function`**, **`harmonize_volumes`**.

## Recommended Phase 2 per-stage async strategy

| kwneuro stage | Phase 2 shape | Notes |
|---------------|---------------|-------|
| `denoise_dwi` | QThread | Medium-length; emit Patch2Self tqdm progress via signal; the existing dipy progress bar is exactly what we want to route to Slicer's status bar. |
| `Dti.estimate_dti` | QThread or sync | Probably fast enough for sync; QThread if we want a uniform pattern across all pipeline modules. |
| `estimate_response_function` | Sync | <30 s, simple. |
| `compute_csd_peaks` | QThread | Medium-length. |
| `harmonize_volumes` | Sync with busy cursor | Quick, no I/O. |
| `brain_extract_batch` | CLI module | HD-BET init is expensive; batch amortises — expose as "run on N subjects" CLI with progress per subject. |
| `Noddi.estimate_noddi` | CLI module | AMICO writes to tmpdir; process isolation is valuable. |
| `extract_tractseg` | CLI module | Deep learning; long; benefits from cancellation. |
| `register_volumes` (SyN) | CLI module | Longest single-pair op; ANTs has its own progress. |
| `build_template` | CLI module | Multi-hour; must have cancel. |

The **threshold** between QThread and CLI shim is roughly *"does the
operation sometimes exceed 3-5 minutes on realistic data?"*. If yes, the
shim cost pays back in cancel + progress + segfault isolation. If no,
QThread is the lighter path.

## Progress-reporting pattern to reuse

Both QThread and CLI paths need to get progress updates back to the user.
The dipy Patch2Self `tqdm` bar (visible in E4 output) is the model: it
prints one line per gradient as it finishes. A small `TqdmToSlicerStatus`
adapter class that sets `slicer.util.showStatusMessage` on each tqdm
update gives users continuous feedback without changing the underlying
kwneuro code. Worth building once and reusing across modules.

## When real benchmarking should happen

**Phase 1.** Once the first scripted module (likely `KWNeuroEnvironment`)
extends to a pipeline module (e.g. a `KWNeuroDenoise` module with an
Apply button), benchmark sync vs QThread on real data in an actual
GUI session. Numbers from Phase 1 will refine the table above.

For Phase 0 purposes, the wall-clock numbers and the architectural
framing above are enough to plan Phase 2 without guessing.
