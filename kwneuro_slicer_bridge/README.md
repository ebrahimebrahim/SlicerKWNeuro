# kwneuro_slicer_bridge

Bridge classes that expose [kwneuro](https://github.com/KitwareMedical/kwneuro)
neuroimage resources as 3D Slicer MRML nodes.

**Only usable inside 3D Slicer's bundled Python.** The package
imports `slicer` and `vtk` at module load; attempting to import it
from a regular Python process will fail.

This package is **bundled with the KWNeuro extension** — when the
extension is loaded by Slicer, the package is reachable from any
Slicer Python session (interactor, scripted module, SlicerJupyter
kernel) via `import kwneuro_slicer_bridge`. There is no separate
pip install step.

## What the bridge provides

The package is intentionally small: it adapts kwneuro storage and
domain objects to Slicer's MRML scene without adding pipeline logic of
its own.

| Bridge API | Backing Slicer node | kwneuro relationship |
|---|---|---|
| `InSceneVolumeResource` | `vtkMRMLScalarVolumeNode` or `vtkMRMLVectorVolumeNode` | subclass of `VolumeResource` |
| `InSceneDwi` | `vtkMRMLDiffusionWeightedVolumeNode` | subclass of `Dwi` |
| `InSceneDti` | `vtkMRMLDiffusionTensorVolumeNode` | subclass of `Dti` |
| `InSceneStructuralImage` | `vtkMRMLScalarVolumeNode` | subclass of `StructuralImage` |
| `InSceneTransformResource` | linear/grid transform nodes | scene wrapper around kwneuro transform files |
| `publish_labelmap_resource` | `vtkMRMLLabelMapVolumeNode` | helper for masks and multi-label outputs |

Every `InScene*` object stores the MRML node ID and exposes
`to_in_memory()` when a pipeline needs a plain kwneuro value detached
from the scene. This matters for background execution: copy scene
inputs on the main Qt thread, run numpy / ANTs / dipy work on a
worker thread, then publish results back on the main thread.

## Quick start in the Slicer Python interactor

```python
from pathlib import Path
from kwneuro_slicer_bridge import InSceneVolumeResource
from kwneuro.io import NiftiVolumeResource

# Load a NIfTI via kwneuro, push it into Slicer's scene as a visible node.
nifti = NiftiVolumeResource(Path("/path/to/volume.nii.gz")).load()
vol = InSceneVolumeResource.from_resource(nifti, name="my_volume", show=True)

# Wrap an existing scene node.
existing = InSceneVolumeResource.from_scene_by_name("my_volume")
arr = existing.get_array()

# Copy back to kwneuro's in-memory representation for further pipeline use.
mem = existing.to_in_memory()
```

`InSceneVolumeResource.from_resource` publishes 3D arrays as scalar
volume nodes and 4D arrays as vector volume nodes. DWI-shaped data
should use `InSceneDwi` instead so gradient metadata is preserved.

For DWI files specifically — Slicer's built-in *Add Data* dialog
silently drops the 4th dimension; use `InSceneDwi.from_nifti_path`
instead so gradients + b-values stay attached to the volume node:

```python
from kwneuro_slicer_bridge import InSceneDwi

sdwi = InSceneDwi.from_nifti_path(
    volume_path=Path("/path/to/dwi.nii.gz"),
    bval_path=Path("/path/to/dwi.bval"),
    bvec_path=Path("/path/to/dwi.bvec"),
    name="HARDI", show=True,
)
```

`InSceneDwi` is a subclass of `kwneuro.dwi.Dwi`, so any kwneuro
pipeline function that takes a `Dwi` accepts a scene-backed
instance directly:

```python
denoised = sdwi.denoise()    # inherited from kwneuro.dwi.Dwi
dti = sdwi.estimate_dti()    # ditto
```

Structural images follow the same scene-backed subclass pattern:

```python
from kwneuro_slicer_bridge import InSceneStructuralImage

st1 = InSceneStructuralImage.from_nifti_path(
    Path("/path/to/T1w.nii.gz"),
    name="T1w",
    show=True,
)

corrected = st1.correct_bias()          # inherited StructuralImage method
tissue = corrected.segment_tissues()    # returns a kwneuro VolumeResource
```

Publish masks, tissue labels, parcellations, or warped labels as
Slicer labelmaps with one helper:

```python
from kwneuro_slicer_bridge import publish_labelmap_resource

label_id = publish_labelmap_resource(tissue, "T1w_tissue_atropos")
```

For binary masks, pass `binary=True`; multi-label arrays keep their
integer label values.

Registration outputs can be pushed into the scene too:

```python
from kwneuro_slicer_bridge import InSceneTransformResource

in_scene_transform = InSceneTransformResource.from_transform(
    transform,
    name_prefix="dwi_to_t1",
)
```

The transform bridge is currently one-way (`kwneuro` transform files
to Slicer transform nodes). Going from arbitrary Slicer transform
nodes back to a kwneuro `TransformResource` would require saving
scene nodes to ANTs-compatible files first and is not implemented.

## Notes for module authors

- Do MRML node wrapping and `.to_in_memory()` on the main Qt thread.
- Use `run_with_progress_dialog` only around the pure compute phase.
- Publish labelmaps through `publish_labelmap_resource` so integer
  labels are preserved and Slicer's Labels color table is attached.
- Use `ensure_extras_installed(["extra_name"])` in `prepare_inputs`
  for optional kwneuro extras before starting a long worker task.

## License

Apache-2.0 (covered by the extension's top-level `LICENSE`).
