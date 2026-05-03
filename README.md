# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/KitwareMedical/kwneuro)
diffusion-MRI library into Slicer — both as a scriptable bridge for
pipeline developers and as clickable GUI modules for researchers.

## What's here

Eleven scripted modules: **KWNeuroEnvironment** (manages install
state) plus ten pipeline-stage wrappers. Each pipeline module uses
the same three-phase architecture — materialise inputs on the main
Qt thread, run the heavy numpy / dipy / ANTs / AMICO / TractSeg
compute on a background worker, publish outputs back on the main
thread — so every module stays responsive under a modal progress
dialog without crashing the subject-hierarchy plugin.

| Module | Role | kwneuro extra required |
|---|---|---|
| **KWNeuroEnvironment** | Install / manage kwneuro and the four optional extras (the bridge ships bundled, no install needed) | — |
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

Plus **`kwneuro_slicer_bridge`** — a small Python package bundled
with the extension (no separate install) exposing:

- `InSceneVolumeResource`, `InSceneDwi`, `InSceneDti`,
  `InSceneTransformResource`: scene-backed wrappers. `InSceneDwi`
  and `InSceneDti` subclass kwneuro's own `Dwi` / `Dti` so they
  drop directly into any pipeline function that takes the parent
  type.
- `run_in_worker`, `run_with_progress_dialog`, `ProgressDialog`,
  `TqdmToProgressDialog`, `ensure_extras_installed`: the async +
  extras helpers that every pipeline module uses.

Known follow-up work: Extension Index submission, CI, cancellation
story for the heavy multi-minute modules (TractSeg / Template),
NVIDIA-GPU pre-flight improvements beyond the TractSeg warning
dialog.

## Layout

- `CMakeLists.txt` — extension metadata.
- Eleven `KWNeuro*/` scripted-module directories (KWNeuroEnvironment
  plus ten pipeline modules), each with `*.py`, `Resources/UI/*.ui`,
  `Testing/Python/test_*.py`.
- `kwneuro_slicer_bridge/` — bundled Python package (built via
  `slicerMacroBuildScriptedModule` like a library "module"; ends up
  alongside the scripted modules in the install layout).
- `docs/` — Sphinx site.
- `notebooks/` — SlicerJupyter-kernel walkthroughs (see below).
- `CLAUDE.md` — working notes for contributors: architectural
  decisions, coordinate-system traps, review-driven test patterns.

## Using the modules

Launch Slicer with the extension (either via the Extension Manager
once released, or a build-tree launcher during development — see
*Development* below).

**Typical single-subject flow** (matches the notebook at
`notebooks/kwneuro-pipeline-walkthrough.py`):

1. **KWNeuro Environment**: click *Install / Update* to install (or
   refresh) the `kwneuro` library into Slicer's Python; tick any
   optional extras you need. The `kwneuro_slicer_bridge` package
   ships with the extension and needs no install.
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

### 1. Configure + build

The extension is standard Slicer CMake — no scripted-module Python
install required up front. Point CMake at your Slicer build tree and
run the build:

```sh
mkdir -p /tmp/kwneuro-extn-build && cd /tmp/kwneuro-extn-build
cmake -DSlicer_DIR=/path/to/Slicer-build $OLDPWD
cmake --build .
```

Substitute your own Slicer build path for `/path/to/Slicer-build`.
Re-run `cmake --build .` after editing any scripted module or any
file inside `kwneuro_slicer_bridge/` (the bridge goes through the
same per-file copy pipeline as the modules — incremental builds
re-copy only the changed files).

### 2. Launch Slicer with the extension

```sh
/tmp/kwneuro-extn-build/SlicerWithKWNeuro
```

This is a CMake-generated launcher that points Slicer at the
build-tree's module paths — the KWNeuro modules appear under
*Modules → KWNeuro* without a permanent install.

### 3. Install `kwneuro` + any extras

Open **KWNeuro Environment** and click **Install / Update**. That
pip-installs the `kwneuro` library from its pinned git ref into
Slicer's Python. Then tick any optional-extra checkboxes you want
(`hdbet`, `noddi`, `tractseg`, `combat`); the panel drives
`slicer.packaging.pip_install` for each, including the
`skip_packages=["fury"]` dance TractSeg needs.

The `kwneuro_slicer_bridge` package ships bundled with the extension
(in the same `qt-scripted-modules/` directory as the modules), so
nothing needs to install it — `import kwneuro_slicer_bridge` just
works.

Click **Verify setup** to confirm the bridge round-trips a synthetic
volume through the scene.

For running the CTest suite (below), you need the **combat** extra
ticked — `py_test_kwneuroharmonize` fails rather than skips if
`neuroCombat` is absent.

You can now quit Slicer; the installs are persistent in Slicer's
bundled Python.

### 4. Run the test suite

```sh
cd /tmp/kwneuro-extn-build
ctest -j$(nproc) --output-on-failure --no-tests=error
```

Expected: **38 tests, all pass in ~2-3 min**. The test count is
stable regardless of which extras are installed — almost every
module's tests either use synthetic data or mock the optional
dependency (HD-BET, AMICO, TractSeg). The one exception is
`py_test_kwneuroharmonize`, which fails rather than skips without
the `combat` extra.

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

### Sample-data prerequisite

Two tests load the Sherbrooke 3-shell DWI from DIPY's cache and
skip when it's absent. The simplest way to populate the cache is
through the extension itself: in Slicer, open **KWNeuro Importer**
and click *Load Sherbrooke 3-shell (HARDI193)* once. That downloads
the dataset to `~/.dipy/sherbrooke_3shell/`, where the CTest run
will find it.

(If you'd rather populate the cache headlessly for CI, `dipy.data.fetch_sherbrooke_3shell()`
does the same thing from any Python that imports `dipy`.)

## Building the docs

The docs build runs outside Slicer:

```sh
python -m pip install sphinx sphinx-autoapi myst-parser sphinx-copybutton furo
python -m sphinx -n -T docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the site. The bridge
package is no longer pip-installable, but `sphinx-autoapi` reads
the `.py` source files directly so no install is needed for it.

## License

Apache-2.0 — matches `kwneuro`.
