# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the
[kwneuro](https://github.com/KitwareMedical/kwneuro) diffusion-MRI
library into Slicer. Phase 1 ships two things:

- A pip-installable Python package `kwneuro_slicer_bridge` with four
  scene-backed classes (`InSceneVolumeResource`, `InSceneDwi`,
  `InSceneDti`, `InSceneTransformResource`) that let kwneuro values
  live as MRML nodes and, where meaningful, subclass kwneuro's own
  `Dwi` / `Dti` / `VolumeResource` directly.
- A scripted module, `KWNeuroEnvironment`, that manages the install
  status of kwneuro, the bridge, and the four kwneuro optional extras
  (hdbet, noddi, tractseg, combat).

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
- **{doc}`bridge-reference`** — architectural notes on the four bridge
  classes, design decisions, and Phase 0 / Phase 1 context.
- **{doc}`Tutorials <tutorials/index>`** — hand-written walkthrough of
  an end-to-end pipeline (denoise → DTI → SyN registration). A
  runnable notebook version lands once SlicerJupyter is fixed
  (Phase 1.5).
- **{doc}`API Reference <autoapi/index>`** — auto-generated reference
  for every bridge class.

## Current status

Phase 1 developer release. Shipping inside the `kwneuro` working tree
for now; will move to its own repo once convenient. No Extension Index
submission yet — install by cloning the extension and pointing Slicer
at it via `--additional-module-paths`.

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
