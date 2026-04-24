# CLAUDE.md

Working notes for anyone (human or AI) maintaining this 3D Slicer
extension. Consolidates the architectural decisions, coordinate-system
traps, and review-driven test patterns that aren't obvious from the
code alone.

The user-facing docs are in `README.md` and `docs/`. This file is
for contributors.

## Architectural decisions

### Three-phase split in every pipeline module

Every scripted module's Logic class exposes three phase methods:

1. **`prepare_inputs`** — runs on the main Qt thread. Materialises
   MRML nodes into in-memory `kwneuro` resources via
   `.to_in_memory()`, calls `ensure_extras_installed(...)` if the
   module needs an optional extra, and does cheap validation
   (shape, affine, row-count, etc.).
2. **`run_*`** — pure numpy / dipy / ANTs / AMICO / TractSeg compute.
   No MRML touches. Safe to dispatch to a background thread via
   `run_with_progress_dialog`.
3. **`publish_to_scene`** — runs on the main Qt thread. Creates the
   output MRML nodes. Always wraps node-creation in `try` /
   `mrmlScene.RemoveNode(...)` / `raise` so a display-setup failure
   doesn't leave dangling partial state.

A `process()` convenience composes the three synchronously for
headless / test callers; widgets call the phases separately so only
`run_*` is wrapped in the progress dialog.

**Why**: MRML scene mutations off the main Qt thread crash the
subject-hierarchy plugin. The failure mode is a chatty `vtkMRMLSub­
jectHierarchyNode::GetSubjectHierarchyNode: Invalid scene given`
followed by abnormal Slicer exit. Reads from VTK image data from
another thread can be OK *if* nothing concurrent writes; writes
never are.

### Bridge is a separately-installable pip package

`kwneuro_slicer_bridge/` lives alongside the scripted modules but
is shipped as its own `pyproject.toml`. `KWNeuroEnvironment` pip-
installs it into Slicer's Python at first-run time from its local
path; the module's `pyproject.toml` pulls `kwneuro` itself via a
git-ref pin (currently the head of the `allow-custom-metadata` PR
on kwneuro — flip to `@main` once that PR merges).

Scripted modules import with `from kwneuro_slicer_bridge import ...`.
The Python interactor and SlicerJupyter notebook see the same
imports without sys.path tricks.

### `InSceneDwi` / `InSceneDti` subclass kwneuro's `Dwi` / `Dti`

So anything that takes a `kwneuro.Dwi` as input accepts an
`InSceneDwi` directly — no conversion step in user code. Uses
`@dataclass(init=False)` to subclass kwneuro's dataclasses.

### Async helpers don't touch Qt from the worker

`run_in_worker` / `run_with_progress_dialog` run `fn` on a plain
`threading.Thread`; completion marshals back to the main thread via
a main-thread `qt.QTimer.singleShot` poll loop. `QTimer.singleShot`
*must not* be called from the worker — that produces
`Timers can only be used with threads started with QThread`
warnings and the timer silently never fires.

### `TqdmToProgressDialog` is deliberately fragile

Dipy-style modules do `from tqdm import tqdm` (or
`from tqdm.auto import tqdm`) at module top, so the rebinding is
per-submodule and `tqdm.tqdm` itself isn't a useful patch target.
`_TQDM_REBINDINGS` in `async_helpers.py` is a hand-maintained list
of dipy submodule + attr pairs we rebind to a queue-writing
subclass.

Safety net: `test_no_uncovered_tqdm_imports_in_dipy` regex-scans
dipy for `from tqdm[...] import tqdm` and flags any unlisted
match — forces a conscious decision when kwneuro starts routing
through new dipy submodules.

### Caching is off for scene-backed resources

`InSceneVolumeResource` holds a `_node` field (a live VTK object)
that kwneuro's fingerprinter can't hash — the resource is silently
dropped from cache tracking with a `UserWarning`. Two consequences:

1. A scene-backed argument changing between two calls in the same
   `Cache()` context **will not invalidate the cache**. Don't wrap
   `Cache()` around pipelines that take `InScene*` inputs and expect
   correct invalidation.
2. Scene-backed resources are detached via `.to_in_memory()`; the
   result is fingerprint-stable and cache-safe.

## Coordinate systems and data layout — known traps

### 4D NIfTI + `slicer.util.loadVolume` = silent data loss

`loadVolume` defaults to `vtkMRMLScalarVolumeNode` and drops the 4th
dimension of a DWI without warning. Always use
`InSceneDwi.from_nifti_path` (or the `KWNeuroImporter` module) to
load DWI files — it targets `vtkMRMLDiffusionWeightedVolumeNode`
with gradients + b-values attached.

### Slicer's `arrayFromVolume` returns KJI, kwneuro expects IJK

`slicer.util.arrayFromVolume` returns arrays in slice-row-column
(KJI) order. `kwneuro` and `nibabel` use (x, y, z) = IJK. Bridge
conversions in `kwneuro_slicer_bridge/src/kwneuro_slicer_bridge/conversions.py`
handle this — specifically `vtk_image_to_numpy` does
`reshape(..., order="F")` so axis 0 is `i`.

`slicer.util.arrayFromSegmentBinaryLabelmap` also returns KJI — the
segmentation-mask branch in `KWNeuroDTI._extract_mask_resource`
transposes before wrapping.

### `xyzt_units = 2` (mm) on detached resources

`InSceneVolumeResource.to_in_memory()` seeds metadata with
`{"xyzt_units": 2}` (spatial = mm, temporal = unknown). Without it,
`kwneuro.resource.InMemoryVolumeResource.to_ants_image()` raises
"Volume must have spatial units in 'mm'" because nibabel's default
header says "unknown". We intentionally do *not* set a temporal
unit — Slicer doesn't track one, and claiming "seconds" would be a
fabrication that leaks into downstream NIfTI saves.

### RAS+ vs LPS+ via ANTsImage

kwneuro's `InMemoryVolumeResource.to_ants_image` / `.from_ants_image`
converts between nibabel's RAS+ and ANTs' LPS+ automatically. Don't
hand-roll that conversion in our code.

### 4D DWI as `arrayFromSegmentBinaryLabelmap` reference

Works in practice — the reference's 3D geometry (IJKToRAS + spatial
extent) is what gets used, the 4th dimension is stored as scalar
components, not extent. If it ever stops working, build a throwaway
3D scalar reference node from the DWI's geometry and use that.

## Review-driven test patterns

These came out of reviewer passes and are worth keeping top-of-mind
when adding new modules or tests:

### Tests must assert on values, not just shapes

A shape-only test passes vacuously against a bug that returns
zeros, random noise, or the input unchanged. Every "real run" test
should assert on at least one numerical invariant:

- KWNeuroRegister: Pearson correlation warped↔fixed > 0.8, and
  warped↔fixed > moving↔fixed.
- KWNeuroTemplate: template ≠ simple arithmetic mean of inputs
  (proves the SyN/sharpen pipeline ran).
- KWNeuroHarmonize: per-batch mean *gap* shrinks from ~20 to <10.
- KWNeuroNODDI: NDI_mod ≈ NDI × (1 − FWF) to catch operand-swap
  bugs.

### Spy on `ensure_extras_installed` by argument

Modules that gate on an extra should have a test that replaces
`ensure_extras_installed` with a spy, runs `prepare_inputs`, and
asserts the argument list is *exactly* what you expect (e.g.
`[["noddi"]]`, not just "message contains noddi"). Catches
regressions that ask for the wrong extra, or ask for an extra plus
an unintended second one.

### Mock where the name is resolved, and assert it was called

Our production code does `from kwneuro.foo import bar` *inside* the
function that uses it — patching `kwneuro.foo.bar` takes effect at
call time. A hoisted `from kwneuro.foo import bar` at module top
would bypass the patch silently, and the test would no-op. Defend
against this with `call_count == 1` assertions on the mock.

### Widget-state checks, not just logic

`qMRMLNodeComboBox` updates its selection via the Qt event loop,
not synchronously on `setCurrentPath = x`. Widget tests that exercise
signal wiring must `slicer.app.processEvents()` between the setter
and the assertion, AND must cover at least one transition
(disabled → enabled) so a broken signal connection would fail
the test.

### `isHidden()` not `.visible` under `--no-main-window`

In the test harness, the top-level widget is never shown, so
`widget.isVisible()` always returns False regardless of
`setVisible()` calls. Use `widget.isHidden()` — that reflects the
explicit `setVisible(False)` state independently of parent
visibility.

### Radio-button exclusivity must be asserted explicitly

Qt groups auto-exclude grouped `QRadioButton`s, but PythonQt
sometimes doesn't propagate exclusivity on programmatic
`.checked = True` before the widget is shown. Tests that check
"each radio maps to the right output" should also assert that
setting one True drives the other two False.

### Fail loud, don't skip, when an extra is load-bearing for coverage

`py_test_kwneuroharmonize` raises rather than skipping if
`neuroCombat` is absent — silent skip would leave the module's only
real end-to-end assertion uncovered in CI. Opposite convention for
environmental dependencies (Sherbrooke cache, etc.): those skip
cleanly because their absence is a fixture-provisioning problem,
not a code problem.

### `mrmlSceneChanged` → `setMRMLScene` wiring lives in `.ui`

Every `.ui` file that contains a `qMRMLNodeComboBox` (or any
qMRML widget) needs a `<connections>` block wiring the top-level
`qMRMLWidget`'s `mrmlSceneChanged(vtkMRMLScene*)` signal to each
child widget's `setMRMLScene(vtkMRMLScene*)` slot. Without it the
child never sees the scene, never auto-selects nodes, and never
fires `currentNodeChanged`. Missing this connection was the
root cause of the first round of "Apply button never enables"
bugs.

## Optional extras — install notes

Four kwneuro optional extras managed by `KWNeuroEnvironment`:

| extra | PyPI package(s) | notes |
|---|---|---|
| `hdbet` | `hd-bet == 2.0.1` | Heavy (torch + nnunetv2 + batchgenerators, several GB). Strongly wants CUDA. |
| `noddi` | `dmri-amico == 2.1.1`, `backports.tarfile` | AMICO writes kernel caches to disk; kwneuro redirects to a tmpdir via `set_config("ATOMS_path", tmpdir)`. |
| `tractseg` | `TractSeg` | **MUST pass `skip_packages=["fury"]`** to `slicer.packaging.pip_install` — `fury` drags in `vtk<9.4` which would clobber Slicer's bundled VTK 9.6+ and break rendering. |
| `combat` | `neuroCombat == 0.2.12` | Pinned because `neuroCombat` is dormant; we want Dependabot to flag any new release. |

### The TractSeg install quirk

`slicer.packaging.pip_install(..., skip_packages=["fury"])` parses
its input through `packaging.requirements.Requirement`, which does
*not* accept filesystem paths or extras syntax like
`kwneuro[tractseg]`. Pass `["TractSeg"]` (the bare PyPI package
name) instead. `KWNeuroEnvironment.install_extra` already handles
this — don't duplicate.

## Known design questions / follow-ups

### CSD `flip_bvecs_x` default

`kwneuro.csd.compute_csd_peaks` defaults `flip_bvecs_x=True`, which
means "flip FSL convention → MRtrix3 convention". Sherbrooke
(from `dipy.data.fetch_sherbrooke_3shell`) is FSL-convention, as is
any data loaded via `InSceneDwi.from_nifti_path`. So the kwneuro
library default is correct for the typical Slicer flow.

**But**: a reviewer flagged that the default may be unintuitive for
users coming from MRtrix3 workflows. If that becomes a real user
complaint, the UI default in `KWNeuroCSD` can be flipped to `False`
without changing the library.

### TractSeg `vtkMRMLVectorVolumeNode` output

A 72-component `tract_segmentation` output renders as first-3-as-RGB
via Slicer's default vector-volume display, which is arbitrary. A
better future representation: a subject-hierarchy folder of 72
scalar volumes, one per named bundle. Defer until someone asks.

### Cancellation for heavy modules

`run_with_progress_dialog` has no Cancel button; dismissing the
dialog (if the user figures out how) leaves the worker thread
running to completion. Python-level cancellation can't interrupt
dipy/ANTs/TractSeg because they're in C-loop-heavy code with no
yield points. The real fix is the `slicer.cli.run` pattern — a
module-specific CLI-shim that can be killed at process level.
Template + TractSeg are the most likely candidates.

### GPU pre-flight

`KWNeuroTractSeg` probes `torch.cuda.is_available()` before
dispatching, confirms via yes/no dialog that the user really wants
CPU inference. `KWNeuroBrainExtract` has no such check; HD-BET is
nearly as slow on CPU and probably deserves the same treatment.
`KWNeuroNODDI` could also benefit on very large data.

## Repository layout

```
slicer-extn/
├── CMakeLists.txt                  # extension metadata, add_subdirectory() calls
├── CLAUDE.md                       # this file
├── README.md                       # user-facing docs
├── KWNeuroEnvironment/             # install-status panel + bridge tests
├── KWNeuroImporter/                # DWI loader + Sherbrooke fetch
├── KWNeuroBrainExtract/            # HD-BET (extra: hdbet)
├── KWNeuroDenoise/                 # Patch2Self
├── KWNeuroDTI/                     # tensor fit
├── KWNeuroCSD/                     # CSD peaks
├── KWNeuroNODDI/                   # AMICO (extra: noddi)
├── KWNeuroTractSeg/                # TractSeg (extra: tractseg)
├── KWNeuroRegister/                # ANTs registration
├── KWNeuroTemplate/                # ANTs template building
├── KWNeuroHarmonize/               # ComBat (extra: combat)
├── kwneuro_slicer_bridge/          # pip-installable bridge package
├── docs/                           # Sphinx site (user-facing)
└── notebooks/                      # SlicerJupyter walkthroughs
```

Each scripted-module directory follows the same shape:

```
KWNeuro<Name>/
├── CMakeLists.txt
├── KWNeuro<Name>.py                # Module + Logic + Widget + Test
├── Resources/
│   ├── Icons/KWNeuro<Name>.png
│   ├── UI/KWNeuro<Name>.ui
│   └── requirements.txt            # mostly empty; deps come via the bridge
└── Testing/
    ├── CMakeLists.txt              # add_subdirectory(Python)
    └── Python/
        ├── CMakeLists.txt          # slicer_add_python_unittest(SCRIPT ...)
        └── test_kwneuro<name>.py
```

When adding a new module, copy `KWNeuroDTI/` as the template.
