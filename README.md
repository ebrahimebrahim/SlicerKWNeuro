# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/KitwareMedical/kwneuro)
diffusion-MRI library into Slicer — both as a scriptable bridge for
pipeline developers and as clickable GUI modules for researchers.

## Status: Phase 2 (pipeline GUI modules)

Eleven scripted modules (KWNeuroEnvironment plus ten pipeline-stage
wrappers), each wrapping a kwneuro pipeline stage as a Slicer module. Same three-phase architecture across all of them —
materialise inputs on the main Qt thread, run the heavy numpy /
dipy / ANTs / AMICO / TractSeg compute on a background worker,
publish outputs back on the main thread — so every module stays
responsive under a modal progress dialog without crashing the
subject-hierarchy plugin.

| Module | Role | kwneuro extra required |
|---|---|---|
| **KWNeuroEnvironment** | Install / manage kwneuro, bridge, and the four optional extras | — |
| **KWNeuroImporter** | Load DWI from NIfTI + FSL bval/bvec (preserves 4D); fetch Sherbrooke sample | — |
| **KWNeuroBrainExtract** | HD-BET brain mask from DWI mean b0 | `hdbet` |
| **KWNeuroDenoise** | Patch2Self denoising (dipy) | — |
| **KWNeuroDTI** | Tensor fit + optional FA / MD; mask accepts scalar / labelmap / segmentation | — |
| **KWNeuroCSD** | Constrained Spherical Deconvolution peaks (MRtrix3-style vector volume) | — |
| **KWNeuroNODDI** | NODDI via AMICO (NDI / ODI / FWF, optional modulated maps) | `noddi` |
| **KWNeuroTractSeg** | CNN-based tract segmentation (72 bundle masks, endings, or TOM) | `tractseg` |
| **KWNeuroRegister** | ANTs registration (Rigid / Affine / SyN / SyNRA) with optional masks | — |
| **KWNeuroTemplate** | Iterative unbiased group-wise template construction via ANTs | — |
| **KWNeuroHarmonize** | Cross-site ComBat harmonisation of scalar maps (group-level) | `combat` |

Plus **`kwneuro_slicer_bridge`** — a small pip-installable Python
package exposing:

- `InSceneVolumeResource`, `InSceneDwi`, `InSceneDti`,
  `InSceneTransformResource`: scene-backed wrappers. `InSceneDwi`
  and `InSceneDti` subclass kwneuro's own `Dwi` / `Dti` so they
  drop directly into any pipeline function that takes the parent
  type.
- `run_in_worker`, `run_with_progress_dialog`, `ProgressDialog`,
  `TqdmToProgressDialog`, `ensure_extras_installed`: the async +
  extras helpers that every pipeline module uses.

Phase summary:

- **Phase 0** (complete) — scaffold, install probes, coordinate-system
  correctness. Findings in `phase-0-findings.md`.
- **Phase 1** (complete) — `KWNeuroEnvironment`, the four bridge
  classes, docs.
- **Phase 1.5** (complete) — SlicerJupyter fixes for Linux + Python
  3.12 (landed upstream in Slicer/SlicerJupyter).
- **Phase 2** (complete) — the ten pipeline modules above (on top of
  KWNeuroEnvironment), async/progress infrastructure, review-driven
  test hardening.

Future: Extension Index submission, CI, cancellation story for the
heavy multi-minute modules (TractSeg / Template), NVIDIA-GPU
pre-flight improvements beyond the TractSeg warning dialog.

## Layout

- `CMakeLists.txt` — extension metadata.
- Eleven `KWNeuro*/` scripted-module directories (KWNeuroEnvironment
  plus ten pipeline modules), each with `*.py`, `Resources/UI/*.ui`,
  `Testing/Python/test_*.py`.
- `kwneuro_slicer_bridge/` — pip-installable Python package. Its
  `pyproject.toml` pins a specific `kwneuro` git ref.
- `docs/` — Sphinx site.
- `notebooks/` — SlicerJupyter-kernel walkthroughs (see below).
- `experiments/` — Phase 0 experiment scripts retained as reference.
- `phase-0-findings.md` — Phase 0 findings + user decisions feeding
  Phase 1.

## Using the modules

Launch Slicer with the extension (either via the Extension Manager
once released, or a build-tree launcher during development — see
*Development* below).

**Typical single-subject flow** (matches the notebook at
`notebooks/kwneuro-pipeline-walkthrough.py`):

1. **KWNeuro Environment**: click *Install / Update* to sync the
   bridge + kwneuro, tick any optional extras you need.
2. **KWNeuro Importer**: either load your own DWI (pick the NIfTI,
   `.bval`, `.bvec` files + a node name) or click *Load Sherbrooke
   3-shell* for sample data.
3. **KWNeuro Denoise** (optional): patch2self denoising.
4. **KWNeuro Brain Extract** (optional, needs `hdbet`): HD-BET mask.
5. **KWNeuro DTI**: fit the tensor. Accepts a scalar / labelmap /
   segmentation mask — pick the segment in the second dropdown that
   appears when a segmentation is selected.
6. **KWNeuro CSD** / **KWNeuro NODDI** / **KWNeuro TractSeg**: any
   of the model-fit modules.

**Multi-volume modules** (operate on two or more volumes at once):

- **KWNeuro Register** (pairwise): align a moving volume to a fixed
  volume via ANTs. Per-subject, not group-level.
- **KWNeuro Template** (group): build an unbiased template from ≥ 2
  volumes.
- **KWNeuro Harmonize** (group): ComBat-harmonise scalar maps across
  sites. Requires a CSV whose row order matches the volume list, plus
  a batch column; volumes must share an affine (enforced at
  validation).

### Demo notebook

`notebooks/kwneuro-pipeline-walkthrough.py` (jupytext percent format)
runs the single-subject pipeline end-to-end inside SlicerJupyter.
Convert to `.ipynb` with `jupytext --to ipynb` or execute cell-by-cell
via the Slicer Python console. Full prereqs are in the notebook
header.

## Development

### Configure + build the extension once

Tests are registered via `slicer_add_python_unittest(...)` in each
module's `Testing/Python/CMakeLists.txt` and run through CTest against
a Slicer build. The instructions below assume a Slicer superbuild at
`~/slicer-superbuild-v5.11/`; substitute your own path.

#### Prerequisites

Pip-install the bridge package into Slicer's Python so the bridge
tests and the modules' lazy imports have something to import:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install --no-deps -e kwneuro_slicer_bridge
```

The `--no-deps` flag preserves whatever `kwneuro` is already installed
(useful when you're iterating on both repos in lockstep). Drop
`--no-deps` the first time to let pip pull `kwneuro` from the git ref
pinned in `kwneuro_slicer_bridge/pyproject.toml`.

Several pipeline modules call `ensure_extras_installed(...)` on the
matching kwneuro optional extra and will raise a clear error if it's
missing — that's caught in the widget and surfaced as an error
dialog at Apply time. For tests, the ComBat extra is **required**:
`py_test_kwneuroharmonize` deliberately fails rather than skips if
`neuroCombat` isn't importable (see the "Run the test suite" section
below). To install it into Slicer's Python:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer \
    -m pip install "neuroCombat==0.2.12"
```

or via the **KWNeuro Environment** module UI by ticking the *combat*
checkbox after loading the extension in Slicer.

#### Configure + build

```sh
mkdir -p /tmp/kwneuro-extn-build && cd /tmp/kwneuro-extn-build
cmake -DSlicer_DIR=$HOME/slicer-superbuild-v5.11/Slicer-build $OLDPWD
cmake --build .
```

CMake copies the scripted-module sources into
`lib/Slicer-5.11/qt-scripted-modules/` where CTest picks them up.
Re-run `cmake --build .` after editing any module source.
`kwneuro_slicer_bridge/` changes don't need a rebuild (editable pip).

#### Launch Slicer with the extension

```sh
/tmp/kwneuro-extn-build/SlicerWithKWNeuro
```

This is a CMake-generated launcher that points Slicer at the build
tree's module paths — the KWNeuro modules appear under *Modules →
KWNeuro* without a permanent install.

### Run the test suite

```sh
cd /tmp/kwneuro-extn-build
ctest -j$(nproc) --output-on-failure --no-tests=error
```

Expected: **38 tests, all pass in ~2-3 min**. The test count is
stable regardless of which optional extras are installed — almost
every module's tests either use synthetic data or mock the optional
dependency (HD-BET, AMICO, TractSeg). The one exception is
`py_test_kwneuroharmonize`, which **fails rather than skips** if
`neuroCombat` is missing (see prerequisites above) — we deliberately
avoid silently skipping the only real end-to-end coverage of the
harmonisation module.

Two tests *skip cleanly* when the Sherbrooke 3-shell DWI hasn't been
cached locally (see the note below): `test_from_nifti_path_preserves_4d_shape`
and `test_load_sherbrooke_if_cached`. The fetch code path itself is
covered by a mocked test that doesn't require the data.

### Run one test by name

```sh
ctest -R py_test_kwneurodti --no-tests=error --output-on-failure
```

`--no-tests=error` is important: without it, a typo'd regex matching
zero tests prints "No tests were found!!!" but exits 0 — a silently-
passing typo. List available tests first:

```sh
ctest -N
```

### Notes

- **Sample-data prerequisite** for some tests: the Sherbrooke 3-shell
  DWI. Populate the DIPY cache once with:
  ```sh
  ~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer \
    -c "from dipy.data import fetch_sherbrooke_3shell; fetch_sherbrooke_3shell()"
  ```
  Tests that need it skip cleanly when absent; the mocked-Sherbrooke
  test in `KWNeuroImporter` covers the fetch code path regardless.

## Building the docs

Install the bridge with its `docs` extra (one-time setup; picks up
sphinx + sphinx-autoapi + myst_parser + sphinx_copybutton + furo),
then invoke sphinx:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install --no-deps -e './kwneuro_slicer_bridge[docs]'
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m sphinx -n -T docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the site.

## License

Apache-2.0 — matches `kwneuro`.
