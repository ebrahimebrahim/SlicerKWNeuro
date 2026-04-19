# Phase 0 — Findings

Groundwork investigation for the KWNeuro 3D Slicer extension. Scope and
rationale are in `/home/thog/.claude/plans/joyful-soaring-harp.md`; each
experiment's raw notes are in `experiments/*.md`.

## Environment

- Slicer: build at `~/slicer-superbuild-v5.11`, launcher at
  `Slicer-build/Slicer`.
- Python: 3.12.10, from the superbuild's `python-install/bin/PythonSlicer`.
- OS: Linux (Ubuntu), kernel 6.8.0-106-generic.
- kwneuro: installed from `/home/thog/kwneuro` (local source tree),
  reported as `0.3.1.dev29+gf9f3e453f.d20260419`.
- VTK: Slicer's bundled 9.6.1, preserved through all installs.

## Experiment summary

| | Experiment | Outcome |
|---|------------|---------|
| E0 | Extension scaffold | **PASS** (headless test exits 0) |
| E1 | Bare kwneuro install | PyPI wheel is **broken**; local source install is clean |
| E2 | Optional extras | 3 of 4 clean via plain pip; tractseg requires `slicer.packaging` with `skip_packages=["fury"]` |
| E3 | `SlicerVolumeResource` round-trip | **PASS** on both synthetic in-memory and NIfTI-nibabel round-trips |
| E4 | End-to-end denoise on real DWI | **PASS** after two bridge fixes; 159.9 s wall-clock for Sherbrooke 3-shell |
| E5 | SlicerJupyter build probe | Fix is **weeks**-sized work; no volunteer branch available |
| E6 | Async UX | Architectural analysis + wall-clock baseline from E4; per-stage recommendations |

## E1 — Bare kwneuro install

Detail: `experiments/e1_bare_install.md`.

- **PyPI `kwneuro-0.3.0` is structurally incompatible with Slicer.** It
  declares `hd-bet`, `dmri-amico`, `TractSeg`, `backports.tarfile` as core
  dependencies instead of optional extras. `TractSeg → fury → vtk<9.4`
  therefore triggers on every bare install; pip tries to replace Slicer's
  bundled VTK 9.6.1 and fails with `uninstall-no-record-file`.
- **Local-source install is clean.** From `/home/thog/kwneuro`, pip
  installs just `antspyx`, `dipy`, `rich`, `typer`, `statsmodels`,
  `trx-python` on top of what Slicer already ships. No VTK touch.
- **Side effect:** Slicer's `scipy 1.16.3` was downgraded to `1.15.3`
  during the failed PyPI attempt. The downgrade stuck. Functional but
  noteworthy.

**Phase 1 implication:** the env panel's default install flow cannot
simply `pip_ensure(["kwneuro"])` against PyPI. Either a new kwneuro PyPI
release must be cut from the current source tree (refactor has already
happened locally; just needs a version bump and upload) before Phase 1
ships, or the env panel must install from a Git URL.

## E2 — Optional extras

Detail: `experiments/e2_extras.md`, script `e2_tractseg_via_slicer_packaging.py`.

Naive `pip install 'kwneuro[<extra>]'` results:

- `[combat]` — clean (adds `neuroCombat 0.2.12`).
- `[noddi]` — clean (adds `dmri-amico 2.1.1`).
- `[hdbet]` — clean but heavy (adds `torch 2.11.0`, `nnunetv2 2.7.0`,
  `timm`, `huggingface_hub`, batchgenerators, acvl-utils, etc.). Users
  will want a size warning before clicking.
- `[tractseg]` — **fails** via fury → vtk conflict, as expected.

`slicer.packaging.pip_install(..., skip_packages=["fury"])` resolves the
tractseg case cleanly, but **requires an important caveat**: it parses its
input through `packaging.requirements.Requirement`, which does *not*
accept filesystem paths or extras specifiers like
`kwneuro[tractseg]`. Passing such an input caused `Requirement(...)` to
raise internally, the input was silently skipped, and the call returned
an empty list with nothing installed.

**Canonical Phase 1 call for the tractseg extra:**

```python
import slicer.packaging
slicer.packaging.pip_install(
    ["TractSeg"],                       # published PyPI name, not kwneuro[tractseg]
    skip_packages=["fury"],
    requester="KWNeuroEnvironment",
    show_progress=False,
)
```

The Phase 1 env panel therefore has **two install code paths**, not one:

1. kwneuro itself + the three extras that install via plain pip.
2. tractseg installed as a standalone requirement via `slicer.packaging`
   with `skip_packages=["fury"]`.

## E3 — Bridge round-trip

Detail: `experiments/e3_roundtrip.py` (headless test), `experiments/_bridge.py`
(the adapted `SlicerVolumeResource`).

Two assertions:

- **A. In-memory round-trip.** `InMemoryVolumeResource → from_resource →
  get_array/get_affine` is exact (`np.allclose` passes).
- **B. NIfTI ↔ Slicer ↔ nibabel agreement.** Save synthetic volume to
  `.nii.gz`; load via `slicer.util.loadVolume` (→ scene node → bridge);
  load the same file via `nibabel.load`; assert both views agree
  element-wise on array *and* affine.

Both PASS on 3D data. **No coordinate-system corrections were needed
against the starter gist**: its IJK ordering via `order="F"` reshape and
its direct `IJKToRAS` matrix handling produce results that match nibabel's
RAS+ affine and IJK-ordered `get_fdata()` out of the box.

## E4 — End-to-end denoise on real DWI

Detail: `experiments/e4_denoise_interactor.md`, script
`experiments/e4_denoise_interactor.py`.

Data: Sherbrooke 3-shell HARDI193, 4D `(128, 128, 60, 193)`.

- Load via `kwneuro.io.NiftiVolumeResource`: **1.86 s**.
- Wrap into scene via `SlicerVolumeResource.from_resource`: **0.33 s**.
- `dwi.denoise()` (DIPY Patch2Self): **159.9 s**.
- Wrap result back into scene: near-instant.

**Two bridge bugs surfaced and were fixed in `experiments/_bridge.py`:**

1. **`slicer.util.loadVolume` silently drops the 4th dimension** of a
   NIfTI DWI, because it defaults to `vtkMRMLScalarVolumeNode`. Work-
   around for Phase 0: route 4D loads through `nibabel` +
   `SlicerVolumeResource.from_resource`, which picks
   `vtkMRMLVectorVolumeNode` for 4D arrays.
   **Phase 1 must provide a `SlicerDwiResource` that either forces a
   VectorVolumeNode load or — preferred — targets
   `vtkMRMLDiffusionWeightedVolumeNode` so bvec/bval become native scene
   attributes.**
2. **`SlicerVolumeResource.get_metadata()` returned Slicer-specific keys**
   (`slicer_node_name`, `slicer_node_id`) that collide with kwneuro's
   `update_volume_metadata` — the latter writes keys directly into a
   `nib.Nifti1Header` which rejects anything that isn't a valid NIfTI
   field. Phase 0 work-around: return `{}`. **Phase 1 needs a real fix
   on one of the two sides** (filter keys in `update_volume_metadata`, or
   populate the bridge's metadata dict from a Slicer storage node in a
   NIfTI-compatible way).

## E5 — SlicerJupyter build probe

Detail: `experiments/e5_slicerjupyter_build.md`.

- SlicerJupyter's external pins are ~3–6 years out of date (`xeus-python
  0.14.3`, `cppzmq` from a 2020 Slicer fork, `pybind11 v2.8.1`, `xeus
  2.4.1`).
- No commits since April 2024; no active PRs.
- benbennett's claimed July 2025 WIP fix (issue #78) was never pushed
  publicly. `gh` search confirms no fork, no branches.
- Real fix on Linux: bump 4–5 external deps simultaneously and adapt
  SlicerJupyter's C++ kernel bindings to xeus-python's evolved API.

**Effort bucket: weeks** (single dev, Linux-only). Cross-platform would be
much more.

## E6 — Async UX

Detail: `experiments/e6_async_probe.md`.

Phase 0 can't observe real UI freeze (`--no-main-window`), so E6 is an
architectural analysis against three Slicer mechanisms — `slicer.cli.run`
(separate process), `qt.QThread` (in-process background), and synchronous
— combined with wall-clock bounds from E4.

Phase 2 per-stage recommendation (short version):

| Stage | Recommended invocation |
|-------|------------------------|
| `denoise_dwi`, `compute_csd_peaks` | **QThread** |
| `estimate_response_function`, `harmonize_volumes`, `Dti.estimate_from_dwi` | **Sync with busy cursor** |
| `brain_extract_batch`, `Noddi.estimate_from_dwi`, `extract_tractseg`, `register_volumes` (SyN), `build_template` | **CLI module shim** — wall-clock and cancellation justify the shim cost |

A `TqdmToSlicerStatus` adapter that forwards dipy/kwneuro `tqdm` progress
to `slicer.util.showStatusMessage` should be built once and reused
everywhere.

---

# Decisions feeding Phase 1, Phase 1.5, Phase 2

## Phase 1 — env panel and bridge

- **Primary module**: `KWNeuroEnvironment` (the Phase 0 stub). Expand to
  the install-status-and-smoke-test panel described in the plan.
- **Two install code paths, four toggles**: combat / noddi / hdbet go
  through plain pip or `slicer.packaging.pip_ensure`; tractseg goes
  through `slicer.packaging.pip_install(["TractSeg"], skip_packages=["fury"])`.
  Toggle state reflects the current install status of each.
- **HD-BET needs a download-size / GPU advisory** before the user clicks.
- **`kwneuro` itself**: install-from-source via a Git URL or, if a new
  PyPI release is cut before Phase 1 ships, from PyPI. The PyPI wheel
  must have optional-deps refactored first — see Followups.
- **Bridge module surface**:
  - `SlicerVolumeResource` — promoted from `experiments/_bridge.py`, with
    the `get_metadata()` fix decided in Phase 1.
  - `SlicerDwiResource` — new; solves the 4D load issue properly.
    Preferred target: `vtkMRMLDiffusionWeightedVolumeNode` with native
    bvec/bval handling. Fallback: wrap a `VectorVolumeNode` for the
    volume and in-memory resources for bvec/bval.
  - `SlicerBvalResource`, `SlicerBvecResource` — probably not needed if
    `SlicerDwiResource` stores bvec/bval on the DWI node.
  - `SlicerTransformResource` — wraps `vtkMRMLLinearTransformNode` /
    `vtkMRMLGridTransformNode`, for the registration API.
  - `SlicerDtiResource` — targets `vtkMRMLDiffusionTensorVolumeNode` so
    Slicer's tensor-glyph tooling works on kwneuro output.
  - **Cache opt-out**: `SlicerVolumeResource` and siblings are not
    meaningfully fingerprintable (node IDs change per session). Their
    `_cache_files` / `_cache_load` / `_cache_save` should raise
    `NotImplementedError` or declare the class as uncacheable. There is
    no precedent in kwneuro; Phase 1 will need to extend the cache
    protocol, probably with a class-level sentinel or a dedicated
    `uncacheable: ClassVar[bool] = True`.
- **Example notebooks** (under `slicer-extn/notebooks/`): an interactor
  walkthrough demonstrating each of the bridge classes on real
  Sherbrooke data. Jupytext percent format to match kwneuro's existing
  notebook convention.

## Phase 1.5 — SlicerJupyter

- **Defer.** Effort bucket "weeks" is too much to bundle into a "Phase
  1.5". Phase 1 docs should target the Python interactor only. Revisit
  SlicerJupyter after Phase 1 ships and only if users actively ask for
  notebook support.
- If reconsidered, frame the work as a general-purpose contribution to
  SlicerJupyter upstream, not a KWNeuro-specific fork.

## Phase 2 — pipeline GUI modules

- **Per-module async choices** as in the table above.
- **Module layout**: one scripted module per pipeline stage, all under
  the `KWNeuro` category. Likely first-batch candidates:
  `KWNeuroDenoise`, `KWNeuroDTI`, `KWNeuroNODDI`, `KWNeuroTractSeg`,
  `KWNeuroRegister`. `KWNeuroEnvironment` stays as the install/status
  sibling.
- **Killer-demo framing** (from our pre-Phase-0 discussion): Phase 2's
  pitch for the Extension Index is "first NODDI / TractSeg GUI in
  Slicer," not "another dMRI plugin."

---

# Followups captured during Phase 0

These are items discovered during the investigation that need to live
somewhere stable after Phase 0 closes. Decide before Phase 1 starts.

1. **kwneuro PyPI release hygiene.** `0.3.0` on PyPI has optional extras
   baked in as core deps. A `0.3.1` or `0.4.0` release should be cut from
   the current source before Phase 1's env panel targets PyPI. File as a
   kwneuro issue.
2. **Coordinate-metadata contract between bridge and
   `update_volume_metadata`.** Either filter keys in kwneuro or populate
   NIfTI-compatible keys in the bridge. Phase 1 decides; file as a
   followup on the kwneuro side.
3. **4D NIfTI loading in Slicer.** Phase 1's `SlicerDwiResource` must
   handle this explicitly — `slicer.util.loadVolume` silently drops the
   4th dimension.
4. **Cache opt-out for non-fingerprintable resources.** The `@cacheable`
   protocol needs a way for classes whose instances aren't meaningfully
   hashable (anything that holds a MRML node ID) to declare themselves
   uncacheable without forcing every call site to branch. Phase 1
   design work; possibly a follow-up PR to kwneuro's `cache.py`.
5. **`scipy 1.16.3 → 1.15.3` downgrade** as a side effect of the failed
   PyPI install in E1. Phase 1's env panel should probe for this and
   offer a "reset to Slicer-bundled versions" button.
6. **`SlicerJupyter` Windows / macOS state** — Phase 0 was Linux-only.
   If Phase 1.5 ever happens, cross-platform work is additional.
7. **ExtensionsIndex naming validation** — `KWNeuro` was chosen
   specifically because Slicer-prefix names are now rejected. Keep in
   mind if the extension ever gets renamed.

---

# Post-Phase-0 decisions (user-directed)

These decisions were made during a Q&A pass after Phase 0 closed.
They supersede recommendations above where they differ, and feed
directly into Phase 1 planning.

## kwneuro version targeting

- **Target the `main` branch** of `kwneuro` for Phase 1, via a
  `requirements.txt` entry of the form
  `kwneuro @ git+https://github.com/brain-microstructure-exploration-tools/kwneuro@main`.
- PyPI `0.3.0` is known stale; no effort will be spent working around it
  until a new release is cut.
- The Phase 1 env panel's "upgrade available" indicator compares the
  *installed* kwneuro version against whatever the `requirements.txt`
  resolves to (effectively: latest `main` HEAD). If they differ, the
  install button offers to upgrade. **No PyPI polling.**

## SlicerDwiResource target

- `vtkMRMLDiffusionWeightedVolumeNode`. Use its native
  `SetDiffusionGradient` / `SetBValue` / `SetMeasurementFrameMatrix`
  accessors to represent bvec/bval as node-local attributes. This
  unlocks Slicer's existing DWI tooling for kwneuro output.

## Metadata contract fix

- **Upstream in kwneuro**: change `kwneuro.util.update_volume_metadata`
  to silently skip keys that aren't valid NIfTI header fields. Benefits
  any non-NIfTI-backed VolumeResource, not just the Slicer bridge. One
  small PR against kwneuro.

## Cache opt-out

- **No kwneuro change.** Rely on the current silent-untrack behaviour:
  `SlicerVolumeResource._node` holds a VTK object, fingerprinting
  returns `None`, and the caller emits a `UserWarning` and drops the
  arg. That's already "warn and bypass" in effect.
- **Document the limitation** in the bridge's docstring: Slicer-backed
  resources are not cache-correct; a user wrapping them in a `Cache()`
  context can see stale hits in the "different bridge resource, same
  other args" case. Don't rely on caching for Slicer workflows. If this
  fragility bites later, revisit with a class-level sentinel.

## SlicerJupyter (Phase 1.5)

- **Commit to a Linux-first fix**, starting with a bounded week-1 spike:
  bump the four external pins in `SuperBuild/External_*.cmake`
  (`xeus-python`, `xeus`, `cppzmq`, `pybind11`) and attempt a build
  against current Slicer. If the spike succeeds, proceed to adapt
  SlicerJupyter's kernel binding layer to any xeus-python API changes
  and submit PRs upstream. If the spike blocks on ABI or protocol, pause
  and re-evaluate.
- Phase 1 proceeds in parallel; bridge classes and env panel do not
  depend on Phase 1.5 landing.
- The user expects heavy use of an AI assistant through this, so
  calendar-time estimate is under the "weeks" bucket from the original
  analysis.

## Phase 1 — env panel MVP features

- Install status + four extras toggles (hdbet / noddi / tractseg /
  combat), each reflecting current install state. Toggle action runs
  the appropriate install/uninstall (tractseg via `slicer.packaging`
  with `skip_packages=["fury"]`).
- Smoke-test button: imports kwneuro, instantiates a tiny `Dwi`,
  round-trips through `SlicerVolumeResource`, reports pass/fail.
- Upgrade check **against the extension's `requirements.txt` git+ pin**
  (not PyPI). If the installed kwneuro version doesn't match the
  required one, indicate so and let the install button do the upgrade.
- *Not* in MVP: HD-BET size/GPU advisory dialog, PyPI upgrade poll,
  scipy-downgrade detection. These can land in a Phase 1.1 iteration.

## Phase 1 — bridge Resource types to ship

- `SlicerVolumeResource` (promoted from the Phase 0 gist).
- `SlicerDwiResource` (→ `vtkMRMLDiffusionWeightedVolumeNode`).
- `SlicerDtiResource` (→ `vtkMRMLDiffusionTensorVolumeNode`).
- `SlicerTransformResource` (→ `vtkMRMLLinearTransformNode` +
  `vtkMRMLGridTransformNode`).
- `SlicerVectorVolumeResource` for CSD peaks (→
  `vtkMRMLVectorVolumeNode`).
- All five ship in Phase 1.

## Phase 2 — first pipeline-stage module

- **`KWNeuroDTI`** first. It is the lightest computationally, good for
  nailing down the QThread + progress-reporting pattern without
  committing to the CLI-shim territory. Once that pattern is shaken
  out, the other modules (TractSeg, NODDI, Denoise, Registration,
  Template) proceed using the same scaffolding.

## Phase 1 — docs home

- **Inside the extension repo.** A `docs/` directory with a Sphinx
  site, plus `notebooks/` with jupytext-percent `.py` notebooks
  mirroring kwneuro's convention. The extension repo owns its
  user-facing story; no tangling with kwneuro's release cadence or
  doc-CI.

## Bridge Python layout

- **Separate pip-installable package** `kwneuro_slicer_bridge` (working
  name) living in the extension repo alongside the scripted modules.
  The env panel pip-installs it into Slicer's Python at first-run
  time. Scripted modules import with
  `from kwneuro_slicer_bridge import SlicerVolumeResource`, and the
  Python interactor / notebooks see the same import without sys.path
  tricks.
- Layout:
  ```
  slicer-extn/
  ├── CMakeLists.txt
  ├── KWNeuroEnvironment/             # scripted module
  ├── KWNeuroDTI/                     # Phase 2 scripted module
  └── kwneuro_slicer_bridge/
      ├── pyproject.toml
      └── src/kwneuro_slicer_bridge/
          ├── __init__.py             # SlicerVolumeResource et al
          ├── _conversions.py         # _numpy_to_vtk_image, ...
          └── py.typed
  ```
- **The kwneuro version requirement lives in
  `kwneuro_slicer_bridge/pyproject.toml`** under
  `[project.dependencies]`, as
  `kwneuro @ git+https://github.com/brain-microstructure-exploration-tools/kwneuro@main`.
  This supersedes the earlier idea of a top-level `requirements.txt`:
  putting the pin in the bridge's `pyproject.toml` is cleaner, keeps
  the bridge self-describing, and makes the env panel's "upgrade
  available" check a comparison against whatever the bridge package
  currently requires.

## Phase 1 work sequence

- **Env panel logic first**, so the bridge is exercised in its target
  environment from the start. A minimal env panel skeleton + the first
  bridge type (`SlicerVolumeResource`) is the first deliverable. The
  smoke-test button exercises the bridge end-to-end.
- Beyond that first milestone, order isn't critical. What matters is
  **testing every step as it lands**:
  - `PythonSlicer` directly when no Qt / main app is needed (pure
    bridge conversions, pure kwneuro invocations).
  - Headless Slicer (`--no-main-window --python-script` or
    `--python-code`) when Qt / scene / module loading is needed.
- Phase 1.5 (SlicerJupyter spike) runs in parallel with Phase 1's
  bridge + panel work.

## Extension Index submission timing

- **After Phase 2 is substantially complete.** Wait until 3-4 pipeline
  modules (starting with `KWNeuroDTI`) are shipped. Stronger first
  impression in the Extension Manager; avoids advertising a
  developer-only preview.
- Consequence: Phase 1 users find the extension by cloning the repo
  and building locally, not via the Extension Manager. Acceptable
  during the developer-focused window.

## CI

- **Manual testing throughout Phase 1 and Phase 2.** No GitHub Actions
  or other automated CI until we approach Extension Index
  submission.
- When CI does come in (around the Extension Index prep), it will need
  to run the scripted module tests + the bridge unit tests against a
  headless Slicer. Slicer's CDash dashboard covers post-submission
  nightly builds automatically.

## `SlicerDwiResource` measurement frame

- **Identity by default.** `vtkMRMLDiffusionWeightedVolumeNode`'s 3x3
  `MeasurementFrameMatrix` is set to identity unless the caller
  explicitly supplies one. kwneuro's `Dwi` assumes gradients are
  already in the scan frame (dipy convention); identity is the correct
  round-trip default.
- **Document the assumption** in the `SlicerDwiResource` class
  docstring so users with DICOM-origin DWI know to supply a non-
  identity frame manually if they care about the glyph-space
  orientation.

## `SlicerResponseFunctionResource`

- **Not shipping one.** kwneuro's `ResponseFunctionResource` (SH
  coefficients + average signal) has no natural MRML equivalent and
  the cost of wrapping it in a ScriptedModuleNode is higher than the
  value.
- Callers pass it as a pure-Python value between pipeline stages.
  Persistence stays via `kwneuro.io.JsonResponseFunctionResource`.

## Upstream kwneuro PR (metadata filter)

- The user opens a **draft PR** against kwneuro early in Phase 1
  implementing the `update_volume_metadata` filter. Phase 1 bridge
  work pins kwneuro to that PR's branch-head commit (via the
  `pyproject.toml` `git+...@<commit>` entry) until the PR merges.
- Once Phase 1 is complete, the user marks the draft PR ready for
  review and the bridge's pin flips back to `@main`.
- **Claude never runs `gh pr create`** — PR creation is manual user
  work (also saved as a persistent feedback memory).

## Repo split timing

- **Split when it becomes inconvenient.** Keep `slicer-extn/` nested
  inside `kwneuro/` for now with the existing `.gitignore` stopgap.
  Split to a standalone repo when the nested arrangement actively
  gets in the way (first collaborator, confusion over which repo
  holds which artifact, `pyproject.toml` name collisions, etc.).

## Phase 1.5 sequencing

- **After Phase 1, not in parallel.** Phase 1 (env panel + bridge +
  docs + example notebook) ships first. Phase 1.5 (SlicerJupyter fix
  on Linux) starts after Phase 1 is in a working state.
- All Phase 1.5 work happens collaboratively via Claude Code. No
  artificial timebox or "spike" structure — we iterate until the
  build works or until we decide to stop.

## Bridge package name

- **`kwneuro_slicer_bridge`.** The verbose name is worth the clarity.
  Import sites read unambiguously:
  `from kwneuro_slicer_bridge import SlicerVolumeResource`.

