# Tutorials

```{toctree}
:maxdepth: 1

dmri-pipeline-walkthrough
```

Phase 1 ships a single hand-written walkthrough of an end-to-end
pipeline (denoise → DTI → SyN registration). Each bridge `Resource`
class gets exercised; every intermediate result lands in the scene as
the appropriate MRML node.

The tutorial is written as a sequence of cells you paste into Slicer's
Python console. Once SlicerJupyter is fixed (Phase 1.5) the same
content will also ship as a runnable notebook under
`slicer-extn/notebooks/`.
