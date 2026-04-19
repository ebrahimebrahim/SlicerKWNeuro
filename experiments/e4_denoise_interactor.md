# E4 — End-to-end denoise on real Sherbrooke DWI

**Data:** HARDI193 from `dipy.data.fetch_sherbrooke_3shell`. Volume shape
`(128, 128, 60, 193)`, 4D multi-shell DWI, b-values 0 / 1000 / 2000 / 3500,
193 gradient directions.

## Result: PASS

```
[e4] data: /home/thog/.dipy/sherbrooke_3shell/HARDI193.nii.gz
[e4] kwneuro NiftiVolumeResource load: 1.86s, shape=(128, 128, 60, 193)
[e4] SlicerVolumeResource.from_resource: 0.33s, shape=(128, 128, 60, 193), node=vtkMRMLVectorVolumeNode1
[e4] bvals shape=(193,), bvecs shape=(193, 3)
[e4] Dwi constructed with SlicerVolumeResource as the volume
[e4] denoise: 159.9s
[e4] output node vtkMRMLVectorVolumeNode2, shape=(128, 128, 60, 193), min=-22.07, max=2894.24
[e4] PASS (denoise wall-clock 159.9s)
```

## Two real Phase 0 bugs surfaced

1. **`slicer.util.loadVolume` silently drops the 4th dimension.** The first
   attempt used `slicer.util.loadVolume("HARDI193.nii.gz")` directly, which
   created a `vtkMRMLScalarVolumeNode` holding only a `(128, 128, 60)` view.
   Patch2Self then correctly rejected the 3D array.

   **Phase 1 implication:** a dedicated DWI loader is needed. Either:
   - use `slicer.util.loadVolume(path, properties={'show': False, ...})`
     with a type hint that forces a vector or DWI volume node, or
   - bypass Slicer's loader entirely for DWI: load via nibabel, construct a
     `kwneuro` in-memory resource, then push into the scene via the bridge's
     `from_resource` path (which already picks `vtkMRMLVectorVolumeNode` for
     4D arrays).

   The ideal long-term target is
   `vtkMRMLDiffusionWeightedVolumeNode`, which has first-class
   `SetDiffusionGradient`/`SetBValue`/`SetMeasurementFrameMatrix` accessors
   matching kwneuro's `Dwi(volume, bval, bvec)` model. Phase 1 should define
   a `SlicerDwiResource` that targets this node type.

2. **`SlicerVolumeResource.get_metadata()` returned Slicer-specific keys**
   (`slicer_node_name`, `slicer_node_id`) that broke kwneuro's
   `update_volume_metadata` — the function writes keys directly into a
   `nibabel.Nifti1Header`, which rejects non-standard field names with
   `ValueError: no field of name slicer_node_name`. Fix for Phase 0: return
   `{}`. Phase 1 should either extract NIfTI-compatible metadata from the
   Slicer storage node, or change `update_volume_metadata` in kwneuro to
   tolerate extra keys.

## Timing and UX notes

- NIfTI load (kwneuro -> nibabel): **1.86s** — perceptually instant.
- Bridge wrap into scene: **0.33s** — instant.
- Patch2Self denoise: **159.9s** on CPU. Observable progress bar from dipy
  (per-gradient status) streams to stdout through kwneuro.
- Output is a new `vtkMRMLVectorVolumeNode` with same shape and realistic
  value range (`min=-22.07, max=2894.24`, vs input values normally in
  thousands — edge effects are expected).

**For Phase 2:** a ~160s synchronous call freezes Slicer's UI for 2.5
minutes. That is unacceptable for interactive use. The denoise module
almost certainly wants a QThread pattern with progress updates from the
dipy tqdm bar. See E6 for the formal async recommendation.
