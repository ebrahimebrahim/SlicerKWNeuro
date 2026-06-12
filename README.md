# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/KitwareMedical/kwneuro)
neuroimage processing toolkit into Slicer — both as a scriptable
bridge for pipeline developers and as clickable GUI modules for
researchers.

## What's here

Fifteen scripted modules: **KWNeuroEnvironment** (manages install
state) plus structural, diffusion, registration, and group-analysis
pipeline-stage wrappers. Each pipeline module uses
the same three-phase architecture — materialise inputs on the main
Qt thread, run the heavy numpy / dipy / ANTs / AMICO / TractSeg
compute on a background worker, publish outputs back on the main
thread — so every module stays responsive under a modal progress
dialog without crashing the subject-hierarchy plugin.

| Module | Role | kwneuro extra required |
|---|---|---|
| **KWNeuroEnvironment** | Install / manage kwneuro and the five optional extras (the bridge ships bundled, no install needed) | — |
| **KWNeuroImporter** | Load DWI from NIfTI + FSL bval/bvec (preserves 4D), load structural NIfTI, and fetch sample datasets | — |
| **KWNeuroBrainExtract** | HD-BET brain mask from DWI mean b0 or structural image | `hdbet` |
| **KWNeuroBiasCorrect** | N4 bias correction for structural images | — |
| **KWNeuroTissueSegment** | Atropos / Deep Atropos tissue labels | `antspynet` for Deep Atropos only |
| **KWNeuroParcellate** | DKT structural parcellation | `antspynet` |
| **KWNeuroDwiToStructuralRegister** | Register DWI mean-b0 to T1 and optionally warp structural labels into DWI space | — |
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
  `InSceneStructuralImage`,
  `InSceneTransformResource`: scene-backed wrappers. `InSceneDwi`
  `InSceneDti`, and `InSceneStructuralImage` subclass kwneuro's own
  `Dwi` / `Dti` / `StructuralImage` so they drop directly into any
  pipeline function that takes the parent type.
- `run_in_worker`, `run_with_progress_dialog`, `ProgressDialog`,
  `TqdmToProgressDialog`, `ensure_extras_installed`: the async +
  extras helpers that every pipeline module uses.

Known follow-up work: Extension Index submission, CI, cancellation
story for the heavy multi-minute modules (TractSeg / Template),
NVIDIA-GPU pre-flight improvements beyond the TractSeg warning
dialog.

## Layout

- `CMakeLists.txt` — extension metadata.
- Fifteen `KWNeuro*/` scripted-module directories (KWNeuroEnvironment
  plus pipeline modules), each with `*.py`, `Resources/UI/*.ui`,
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

**Typical diffusion flow** (matches the notebook at
`notebooks/kwneuro-pipeline-walkthrough.py`):

1. **KWNeuro Environment**: click *Install / Update* to install (or
   refresh) the `kwneuro` library into Slicer's Python; tick any
   optional extras you need. The `kwneuro_slicer_bridge` package
   ships with the extension and needs no install.
2. **KWNeuro Importer**: either load your own DWI (pick the NIfTI,
   `.bval`, `.bvec` files + a node name) or click a DWI sample-data
   button.
3. **KWNeuro Denoise** (optional): patch2self denoising.
4. **KWNeuro Brain Extract** (optional, needs `hdbet`): HD-BET mask.
5. **KWNeuro DTI**: fit the tensor. Accepts a scalar / labelmap /
   segmentation mask — pick the segment in the second dropdown that
   appears when a segmentation is selected.
6. **KWNeuro CSD** / **KWNeuro NODDI** / **KWNeuro TractSeg**: any
   of the model-fit modules.

**Typical structural + diffusion flow**:

1. **KWNeuro Importer**: load a T1/structural NIfTI and a DWI, or use
   the ds000221 multimodal sample button.
2. **KWNeuro Bias Correct**: apply N4 to the structural image.
3. **KWNeuro Brain Extract**: create a structural or DWI brain mask.
4. **KWNeuro Tissue Segment** / **KWNeuro Parcellate**: create
   labelmap outputs from the corrected structural image.
5. **KWNeuro DWI To Structural Register**: register mean-b0 to T1 and
   optionally inverse-warp structural labels into DWI space.

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
install required up front. From the repository root, point CMake at
your Slicer build tree and run the build. Set `BUILD_DIR` to any
writable directory you want to use for the extension build tree:

```sh
BUILD_DIR=/path/to/KWNeuro-build
cmake -S . -B "$BUILD_DIR" -DSlicer_DIR=/path/to/Slicer-build
cmake --build "$BUILD_DIR"
```

Substitute your own Slicer build path for `/path/to/Slicer-build`.
Re-run `cmake --build "$BUILD_DIR"` after editing any scripted module or any
file inside `kwneuro_slicer_bridge/` (the bridge goes through the
same per-file copy pipeline as the modules — incremental builds
re-copy only the changed files).

### 2. Launch Slicer with the extension

```sh
"${BUILD_DIR}/SlicerWithKWNeuro"
```

This is a CMake-generated launcher that points Slicer at the
build-tree's module paths — the KWNeuro modules appear under
*Modules → KWNeuro* without a permanent install.

### 3. Install `kwneuro` + any extras

Open **KWNeuro Environment** and click **Install / Update**. That
pip-installs the pinned `kwneuro==1.0.0` release into Slicer's
Python. Then tick any optional-extra checkboxes you want
(`hdbet`, `noddi`, `tractseg`, `combat`, `antspynet`); the panel drives
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
ctest --test-dir "$BUILD_DIR" \
  -j$(nproc) \
  --output-on-failure \
  --no-tests=error
```

Expected: all tests pass in a few minutes. List the exact count with
`ctest --test-dir "$BUILD_DIR" -N`; it changes as scripted modules
are added. Almost every module's tests either use synthetic data or
mock the optional dependency (HD-BET, AMICO, TractSeg, ANTsPyNet).
The one exception is `py_test_kwneuroharmonize`, which fails rather
than skips without the `combat` extra.

Two tests *skip cleanly* when the Sherbrooke 3-shell DWI hasn't been
cached locally (see the note below): `test_from_nifti_path_preserves_4d_shape`
and `test_load_sherbrooke_if_cached`. The fetch code path itself is
covered by a mocked test that doesn't require the data.

### Run one test by name

```sh
ctest --test-dir "$BUILD_DIR" \
  -R py_test_kwneurodti \
  --no-tests=error \
  --output-on-failure
```

`--no-tests=error` is important: without it, a typo'd regex matching
zero tests prints "No tests were found!!!" but exits 0 — a silently-
passing typo. List available tests first:

```sh
ctest --test-dir "$BUILD_DIR" -N
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
