# Getting started

This page walks through installing the KWNeuro Slicer extension for
development, opening the environment panel, and doing a round-trip of
a volume through `InSceneVolumeResource` from Slicer's Python
interactor.

## 1. Build or install Slicer

Phase 1 targets Slicer with Python 3.12 (a recent nightly or
5.11-class superbuild). A fresh build or a pre-built installer both
work.

## 2. Clone the extension and point Slicer at it

The extension currently lives inside the `kwneuro` working tree at
`kwneuro/slicer-extn/`. Phase 1 users run Slicer with
`--additional-module-paths` pointed at the `KWNeuroEnvironment`
module:

```sh
path/to/Slicer \
  --additional-module-paths /absolute/path/to/kwneuro/slicer-extn/KWNeuroEnvironment
```

Alternatively, add the path via *Edit → Application Settings →
Modules → Additional module paths* in the Slicer GUI and restart.

## 3. Open the KWNeuro Environment module

Navigate to the **KWNeuro** category in the module selector and open
**KWNeuro Environment**. The panel has two collapsible sections:

- **Status.** Shows the installed `kwneuro` and `kwneuro_slicer_bridge`
  versions, an **Optional extras** groupbox with one checkbox per
  kwneuro extra (`hdbet`, `noddi`, `tractseg`, `combat`), and the
  **Install / Update** button. On first launch, both version fields
  read *(not installed)*. Click *Install / Update* — pip fetches the
  bridge package from this extension's local `kwneuro_slicer_bridge/`
  directory, and the bridge's pyproject.toml pulls `kwneuro` from its
  `git+...` pin. Ticking an extras checkbox installs that extra;
  unticking uninstalls it. TractSeg is the only one that needs special
  handling (`fury` is pruned to preserve Slicer's bundled VTK); the
  panel does that for you automatically.

- **Verification.** Click **Verify setup**. The button's tooltip
  describes what gets checked (imports + a synthetic round-trip).
  Output should be `PASS: kwneuro X.Y.Z, bridge imports OK, 3D
  round-trip OK`.

## 4. Use the bridge from the Python interactor

Open *View → Python Console* (or `Ctrl+3`). Try:

```python
from pathlib import Path

import numpy as np
from kwneuro.resource import InMemoryVolumeResource
from kwneuro_slicer_bridge import InSceneVolumeResource

arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
affine = np.diag([2.0, 3.0, 4.0, 1.0])
mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})

svr = InSceneVolumeResource.from_resource(mem, name="my_first_bridge", show=True)
```

A new volume node called `my_first_bridge` appears in the scene tree
and slice viewers. Reading back:

```python
svr.get_array().shape    # (3, 4, 5)
svr.get_affine()         # the 4x4 affine you passed in
svr.get_metadata()       # {'slicer_node_id': ..., 'slicer_node_name': 'my_first_bridge'}
```

Wrapping an existing scene node by name:

```python
svr = InSceneVolumeResource.from_scene_by_name("my_first_bridge")
```

## 5. Dwi / Dti classes IS-A kwneuro's Dwi / Dti

`InSceneDwi` and `InSceneDti` subclass `kwneuro.dwi.Dwi` and
`kwneuro.dti.Dti` respectively. That means any pipeline function that
takes a `Dwi` or `Dti` accepts the scene-backed version directly:

```python
from kwneuro_slicer_bridge import InSceneDwi
from pathlib import Path
from dipy.data import fetch_sherbrooke_3shell

_, data_dir = fetch_sherbrooke_3shell()
data_dir = Path(data_dir)

sdwi = InSceneDwi.from_nifti_path(
    data_dir / "HARDI193.nii.gz",
    data_dir / "HARDI193.bval",
    data_dir / "HARDI193.bvec",
    name="HARDI193", show=True,
)
denoised = sdwi.denoise()          # inherited Dwi method
dti      = sdwi.estimate_dti()     # inherited
```

No explicit conversion step needed.

See {doc}`bridge-reference` for the rest of the bridge API and
{doc}`Tutorials <tutorials/index>` for a fuller pipeline walkthrough.
