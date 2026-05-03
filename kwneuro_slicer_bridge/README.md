# kwneuro_slicer_bridge

Bridge classes that expose [kwneuro](https://github.com/KitwareMedical/kwneuro)
diffusion-MRI resources as 3D Slicer MRML nodes.

**Only usable inside 3D Slicer's bundled Python.** The package
imports `slicer` and `vtk` at module load; attempting to import it
from a regular Python process will fail.

This package is **bundled with the KWNeuro extension** — when the
extension is loaded by Slicer, the package is reachable from any
Slicer Python session (interactor, scripted module, SlicerJupyter
kernel) via `import kwneuro_slicer_bridge`. There is no separate
pip install step.

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

## License

Apache-2.0 (covered by the extension's top-level `LICENSE`).
