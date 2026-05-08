# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the
[kwneuro](https://github.com/KitwareMedical/kwneuro) neuroimage
processing toolkit into Slicer. Ships:

- A bundled Python package `kwneuro_slicer_bridge` (ships with the
  extension; reachable via plain `import kwneuro_slicer_bridge` from
  any Slicer Python session) exposing scene-backed resource classes
  (`InSceneVolumeResource`, `InSceneDwi`, `InSceneDti`,
  `InSceneStructuralImage`, `InSceneTransformResource`) that let
  kwneuro values live as MRML nodes and, where meaningful, subclass
  kwneuro's own `Dwi` / `Dti` / `StructuralImage` /
  `VolumeResource` directly — so any kwneuro pipeline function
  accepts a scene-backed resource without a conversion step.
- Fifteen scripted modules covering structural and diffusion
  workflows:
  environment management (`KWNeuroEnvironment`), import
  (`KWNeuroImporter`), brain extraction, structural bias correction,
  tissue segmentation, parcellation, denoising, DTI, CSD, NODDI,
  TractSeg, DWI-to-structural registration, general registration,
  template building, and ComBat harmonisation.
- Shared async / progress / extras helpers in
  `kwneuro_slicer_bridge.async_helpers` so every module stays
  responsive during multi-minute compute without crashing Slicer's
  subject-hierarchy plugin.

```{toctree}
:maxdepth: 2
:hidden:

getting-started
bridge-reference
Tutorials <tutorials/index>
API Reference <autoapi/index>
```

## What's here

- **{doc}`getting-started`** — load the extension, install kwneuro
  via the environment panel, run Verify setup, use the bridge from
  the Slicer Python interactor.
- **{doc}`bridge-reference`** — architectural notes on the bridge
  classes and design decisions.
- **{doc}`Tutorials <tutorials/index>`** — hand-written walkthrough
  of end-to-end diffusion and structural workflows.
  A runnable SlicerJupyter notebook lives in the repo at
  `notebooks/kwneuro-pipeline-walkthrough.py`.
- **{doc}`API Reference <autoapi/index>`** — auto-generated reference
  for every bridge class.

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
