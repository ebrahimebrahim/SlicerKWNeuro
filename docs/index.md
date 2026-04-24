# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the
[kwneuro](https://github.com/KitwareMedical/kwneuro) diffusion-MRI
library into Slicer. Ships:

- A pip-installable Python package `kwneuro_slicer_bridge` with
  scene-backed resource classes (`InSceneVolumeResource`,
  `InSceneDwi`, `InSceneDti`, `InSceneTransformResource`) that let
  kwneuro values live as MRML nodes and, where meaningful, subclass
  kwneuro's own `Dwi` / `Dti` / `VolumeResource` directly — so any
  kwneuro pipeline function accepts a scene-backed resource without
  a conversion step.
- Eleven scripted modules covering the standard DWI workflow:
  environment management (`KWNeuroEnvironment`), import
  (`KWNeuroImporter`), brain extraction, denoising, DTI, CSD, NODDI,
  TractSeg, registration, template building, and ComBat
  harmonisation.
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

- **{doc}`getting-started`** — install the extension + bridge, run
  Verify setup, use the bridge from the Slicer Python interactor.
- **{doc}`bridge-reference`** — architectural notes on the bridge
  classes and design decisions.
- **{doc}`Tutorials <tutorials/index>`** — hand-written walkthrough
  of an end-to-end pipeline (denoise → DTI → SyN registration).
  A runnable SlicerJupyter notebook lives in the repo at
  `notebooks/kwneuro-pipeline-walkthrough.py`.
- **{doc}`API Reference <autoapi/index>`** — auto-generated reference
  for every bridge class.

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
