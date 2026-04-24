# Example pipeline: denoise → DTI → SyN registration

This walkthrough runs a small diffusion-MRI pipeline end-to-end inside
3D Slicer using the KWNeuro bridge. Every intermediate result becomes a
Slicer scene node via the appropriate bridge class, so you can inspect
each stage visually alongside running the code.

**Run this walkthrough interactively** from Slicer's Python console
(`Ctrl+3`). Paste each cell block one at a time and watch the scene
tree populate after each stage. A runnable SlicerJupyter notebook
version of the same workflow lives at
`slicer-extn/notebooks/kwneuro-pipeline-walkthrough.py`.

## Prerequisites

- The KWNeuro extension is on Slicer's additional module paths (see
  {doc}`../getting-started`).
- The `KWNeuroEnvironment` module's **Install / Update** button has
  been run; **Verify setup** passes.
- The Sherbrooke 3-shell sample is cached at
  `~/.dipy/sherbrooke_3shell/`. If not, run once from a terminal:
  ```sh
  ~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer \
    -c "from dipy.data import fetch_sherbrooke_3shell; fetch_sherbrooke_3shell()"
  ```

## 1. Load the DWI into Slicer as a diffusion-weighted volume node

`InSceneDwi.from_nifti_path` loads the 4D DWI via `nibabel` + kwneuro
(bypassing `slicer.util.loadVolume`, which silently drops the 4th
dimension), and creates a proper `vtkMRMLDiffusionWeightedVolumeNode`
with bval / bvec carried as node attributes.

```python
from pathlib import Path

from dipy.data import fetch_sherbrooke_3shell

from kwneuro_slicer_bridge import (
    InSceneDti,
    InSceneDwi,
    InSceneTransformResource,
    InSceneVolumeResource,
)

_, data_dir = fetch_sherbrooke_3shell()
data_dir = Path(data_dir)
nifti = data_dir / "HARDI193.nii.gz"
bval  = data_dir / "HARDI193.bval"
bvec  = data_dir / "HARDI193.bvec"

sdwi = InSceneDwi.from_nifti_path(nifti, bval, bvec, name="HARDI193", show=True)
print("DWI node:", sdwi.node_id)
print("volume shape:", sdwi.volume.get_array().shape)  # (128, 128, 60, 193)
print("bval / bvec:", sdwi.bval.get().shape, sdwi.bvec.get().shape)
```

**Inspect in Slicer.** The scene tree now has an `HARDI193` diffusion-
weighted volume node. Select it in the Volumes module; slice views
default to the b0 channel.

## 2. Compute a mean b0 and push it into the scene

Because `InSceneDwi` is-a `kwneuro.dwi.Dwi`, you can call every `Dwi`
method directly on it — no `to_dwi()` conversion needed.

```python
mean_b0 = sdwi.compute_mean_b0()  # returns an InMemoryVolumeResource

mean_b0_svr = InSceneVolumeResource.from_resource(
    mean_b0, name="HARDI193_mean_b0", show=True,
)
print("mean b0 node:", mean_b0_svr.node_id, "shape:", mean_b0_svr.get_array().shape)
```

## 3. Denoise the DWI (Patch2Self)

Takes a couple of minutes on the Sherbrooke sample and freezes
Slicer's UI while it runs. If you want a responsive UI during the
run, use the `KWNeuroDenoise` scripted module instead — it wraps the
same call in a progress dialog and runs the work on a background
thread.

```python
denoised_dwi = sdwi.denoise()

denoised_sdwi = InSceneDwi.from_dwi(denoised_dwi, name="HARDI193_denoised", show=True)
denoised_mean_b0 = denoised_dwi.compute_mean_b0()
denoised_mean_b0_svr = InSceneVolumeResource.from_resource(
    denoised_mean_b0, name="HARDI193_denoised_mean_b0", show=False,
)
```

## 4. Estimate DTI and show as a tensor volume

```python
from kwneuro.dti import Dti

dti = Dti.estimate_dti(denoised_dwi)
sdti = InSceneDti.from_dti(dti, name="HARDI193_dti", show=False)
print("DTI node:", sdti.node_id)
print("DTI volume shape (6 lower-triangular components):", sdti.volume.get_array().shape)
```

Slicer's **Diffusion Tensor Scalars** module can render FA / MD / tensor
glyphs on the new `HARDI193_dti` node.

## 5. Register the denoised mean b0 back to the original mean b0

Sanity-check registration: the two volumes are near-identical by
construction so the recovered transform is close to identity. The
point is to exercise both the affine and warp paths of the ANTs SyN
output and the corresponding `InSceneTransformResource` wrapping.

```python
from kwneuro.reg import register_volumes

warped, transform = register_volumes(
    fixed=mean_b0,                  # the original (un-denoised) mean b0
    moving=denoised_mean_b0,
    type_of_transform="SyN",
)

# Wrap the ANTs forward transform chain as Slicer transform nodes:
# one vtkMRMLLinearTransformNode for the .mat, one
# vtkMRMLGridTransformNode for the warp .nii.gz.
transform_svr = InSceneTransformResource.from_transform(
    transform, name_prefix="HARDI193_register",
)
for node in transform_svr.get_nodes():
    print(f"  {node.GetID()}: class={node.GetClassName()} name={node.GetName()}")

warped_svr = InSceneVolumeResource.from_resource(
    warped, name="HARDI193_denoised_mean_b0_registered", show=True,
)
```

## Where to look in Slicer

After running all cells the scene tree should contain:

| Scene node | Class | Source |
|---|---|---|
| `HARDI193` | `vtkMRMLDiffusionWeightedVolumeNode` | original 4D DWI |
| `HARDI193_mean_b0` | `vtkMRMLScalarVolumeNode` | mean b0 of original |
| `HARDI193_denoised` | `vtkMRMLDiffusionWeightedVolumeNode` | denoised 4D DWI |
| `HARDI193_denoised_mean_b0` | `vtkMRMLScalarVolumeNode` | mean b0 of denoised |
| `HARDI193_dti` | `vtkMRMLDiffusionTensorVolumeNode` | DTI (6 LT components, rendered as 3x3 symmetric) |
| `HARDI193_register_*` | `vtkMRMLLinearTransformNode` + `vtkMRMLGridTransformNode` | SyN forward transform |
| `HARDI193_denoised_mean_b0_registered` | `vtkMRMLScalarVolumeNode` | moving volume warped into fixed space |

The transform nodes in the subject-hierarchy view can be dragged onto
any other scene node to compose transforms downstream.
