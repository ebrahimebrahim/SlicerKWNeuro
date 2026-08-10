# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/KitwareMedical/kwneuro)
neuroimage processing toolkit into Slicer, both as a scriptable
bridge for pipeline developers and as clickable GUI modules for
researchers.

# Modules

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
| **KWNeuroTractSeg** | CNN-based tract segmentation (bundle / endpoint segmentation nodes, or TOM vector volume) | `tractseg` |
| **KWNeuroRegister** | ANTs registration (Rigid / Affine / SyN / SyNRA) with optional masks | — |
| **KWNeuroTemplate** | Iterative unbiased group-wise template construction via ANTs | — |
| **KWNeuroHarmonize** | Cross-site ComBat harmonisation of scalar maps (group-level) | `combat` |

Plus **`kwneuro_slicer_bridge`**, which is a small Python package bundled
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

## Installation

See the releases on this repository.

## Example usages

**Typical diffusion flow** (matches the notebook at
`notebooks/kwneuro-pipeline-walkthrough.py`):

1. **KWNeuro Environment**: check any optional extras you need, then
   click *Apply environment changes* to install (or refresh) the
   `kwneuro` library and apply the selected extras. The
   `kwneuro_slicer_bridge` package ships with the extension and needs
   no install.
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
runs the single-subject pipeline end-to-end inside SlicerJupyter (which
has questionable reliability at the time of writing this).
Convert to `.ipynb` with `jupytext --to ipynb` or execute cell-by-cell
via the Slicer Python console. Full prereqs are in the notebook
header.



## License

Apache-2.0
