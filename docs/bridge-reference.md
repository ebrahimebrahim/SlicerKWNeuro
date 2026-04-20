# Bridge reference

The `kwneuro_slicer_bridge` package provides four scene-backed classes
that let kwneuro domain objects live in Slicer's MRML scene. This page
is the high-level architectural reference; method-level documentation
lives under {doc}`autoapi/index`.

## Classes

| Bridge class | Backing MRML node type | Relationship to kwneuro |
|---|---|---|
| `InSceneVolumeResource` | `vtkMRMLScalarVolumeNode` (3D) or `vtkMRMLVectorVolumeNode` (4D) | subclass of `kwneuro.resource.VolumeResource` |
| `InSceneDwi` | `vtkMRMLDiffusionWeightedVolumeNode` | **subclass of `kwneuro.dwi.Dwi`** |
| `InSceneDti` | `vtkMRMLDiffusionTensorVolumeNode` | **subclass of `kwneuro.dti.Dti`** |
| `InSceneTransformResource` | list of `vtkMRMLLinearTransformNode` / `vtkMRMLGridTransformNode` | standalone wrapper; kwneuro's `TransformResource` is file-based and doesn't fit scene-node semantics cleanly |

The `InScene` prefix sits on the storage-location axis alongside
kwneuro's existing `InMemoryVolumeResource` / `NiftiVolumeResource`.
`VolumeResource` data can live in memory, on disk as NIfTI, or (with
this package) in a Slicer scene node.

## Why subclass `Dwi` / `Dti`?

So that `InSceneDwi` IS-A `kwneuro.dwi.Dwi`. Any pipeline function that
takes a `Dwi` accepts an `InSceneDwi` directly — no conversion step
needed:

```python
sdwi = InSceneDwi.from_nifti_path(...)
denoised = sdwi.denoise()             # inherited Dwi method; returns kwneuro.Dwi
dti      = sdwi.estimate_dti()        # inherited
mean_b0  = sdwi.compute_mean_b0()     # inherited
```

Same for `InSceneDti` / `Dti`. The inherited fields (`volume`, `bval`,
`bvec` on Dwi; `volume` on Dti) are populated from the scene node at
construction.

## Round-trips and coordinate conventions

Every bridge class holds an MRML `node_id` plus a cached `_node`
reference. Data accessors read through the cached node; factory
methods (`from_resource`, `from_dwi`, `from_dti`, …) push data into the
scene and return a wrapping resource.

Coordinate conventions match kwneuro's (nibabel-style IJK ordering
with a 4x4 RAS affine). Conversions to/from Slicer's internal KJI /
VTK-interleaved-tensor layout live in
`kwneuro_slicer_bridge.conversions`.

## InSceneDwi avoids the `loadVolume` 4D trap

`slicer.util.loadVolume` defaults to `vtkMRMLScalarVolumeNode`, which
silently drops the 4th dimension of a 4D NIfTI (surfaced in Phase 0
E4). `InSceneDwi.from_nifti_path` loads the DWI via `nibabel` +
`kwneuro.io.NiftiVolumeResource` and pushes it into the scene as a
`vtkMRMLDiffusionWeightedVolumeNode` — gradient dimension preserved,
bval / bvec stored as node attributes.

The `MeasurementFrameMatrix` defaults to identity; `kwneuro.dwi.Dwi`
doesn't model a gradient frame explicitly (dipy convention: gradients
are already in the scan frame). DICOM-origin users who care about
glyph-space orientation pass a non-identity matrix to `from_dwi`.

## InSceneDti expands 6-component to 9-component

kwneuro / dipy store DTI as 6 lower-triangular components per voxel
`(Dxx, Dxy, Dyy, Dxz, Dyz, Dzz)`. Slicer's DTI node stores the full
symmetric 3x3 tensor (9 row-major components) in VTK's dedicated
`PointData.Tensors` attribute. The bridge expands on push and
compresses on read.

Consequence: `InSceneDti.volume` is an `InMemoryVolumeResource`
snapshot (the 6-LT form), built at construction from the scene node's
9-component tensor data. It is not a live view — if the scene node's
tensor data changes afterwards, `.volume` does not refresh. Call
`InSceneDti.from_node(node)` again to rewrap.

## Caching

Scene-backed resources are not fingerprint-stable under kwneuro's
`@cacheable` mechanism. The `_node` field (a live VTK object) fails
kwneuro's fingerprinter; the resource is silently dropped from cache
tracking with a `UserWarning`. That means:

- Within-session repeated calls with the same resource usually hit the
  cache via the other (fingerprintable) args.
- A different scene-backed resource wrapping a different node will
  not invalidate the cache — stale hits are possible.

Don't wrap `Cache()` contexts around pipelines that involve
scene-backed resources and expect correct invalidation.

## Metadata flow

`InSceneVolumeResource.get_metadata()` returns `slicer_node_id` and
`slicer_node_name`. Those aren't NIfTI header fields, but kwneuro's
`update_volume_metadata` preserves custom keys through the pipeline
(as of the `allow-custom-metadata` change), so the Slicer origin
markers flow through pipeline stages intact. They don't survive a
`NiftiVolumeResource.save` (nibabel only writes real NIfTI fields),
which is the only sensible behaviour — a scene node ID is an ephemeral
per-session identifier and has no meaning off the scene it was
assigned in.

**Note on staleness.** `get_metadata()` reads the current node at call
time. Copying the metadata dict into a non-scene-backed resource (e.g.
via `to_in_memory()`) produces a snapshot. If you then recreate a
scene node via `from_resource`, the original `slicer_node_id` is an
*origin* marker — it points at the node the data came from, not the
new node.

## Conversions away from the scene

Each scene-backed class exposes `to_in_memory()` to return a plain
kwneuro value fully detached from the scene:

- `InSceneVolumeResource.to_in_memory() -> InMemoryVolumeResource`
- `InSceneDwi.to_in_memory() -> Dwi` (a plain `kwneuro.dwi.Dwi`)
- `InSceneDti.to_in_memory() -> Dti` (a plain `kwneuro.dti.Dti`)

For day-to-day pipeline use, you usually don't need `to_in_memory()` —
`InSceneDwi` / `InSceneDti` are already usable as `Dwi` / `Dti` via
subclassing. `to_in_memory()` matters when you want to serialize, hand
the data off to code that inspects types strictly, or free the scene
node without losing the data.

## Phase 1 transitional notes

- `InSceneTransformResource` supports one-way conversion
  (kwneuro → Slicer) only. Slicer → kwneuro (saving scene transforms to
  ANTs `.mat` / `.nii`) is deferred.
- `InSceneDti.volume` is a snapshot (see above). Phase 2 may replace
  it with a live view that reads the node's tensor data on each access
  if the snapshot becomes a practical limitation.
