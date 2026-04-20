# kwneuro_slicer_bridge

Bridge classes that expose [kwneuro](https://github.com/KitwareMedical/kwneuro)
diffusion-MRI resources as 3D Slicer MRML nodes.

**Only usable inside 3D Slicer's bundled Python.** The package imports
`slicer` and `vtk` at module load; attempting to import it from a regular
Python process will fail.

This package is pip-installed into Slicer's Python by the KWNeuro extension's
`KWNeuroEnvironment` module. For end-user documentation, see the KWNeuro
extension's docs site.

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

## License

Apache-2.0.
