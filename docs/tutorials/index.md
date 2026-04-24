# Tutorials

```{toctree}
:maxdepth: 1

dmri-pipeline-walkthrough
```

A hand-written walkthrough of an end-to-end pipeline (denoise → DTI
→ SyN registration). Each bridge `Resource` class gets exercised;
every intermediate result lands in the scene as the appropriate
MRML node.

The tutorial is written as a sequence of cells you paste into
Slicer's Python console. A runnable SlicerJupyter notebook version
lives in the repo at `slicer-extn/notebooks/kwneuro-pipeline-walkthrough.py`
(jupytext percent-format — convert to `.ipynb` with
`jupytext --to ipynb`).
