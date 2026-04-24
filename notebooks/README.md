# Notebooks

Runnable notebooks for the KWNeuro extension. They execute inside
3D Slicer via the [SlicerJupyter](https://github.com/Slicer/SlicerJupyter)
kernel — SlicerJupyter must be installed and its kernel running
before the notebooks can import `slicer`, `vtk`, or the KWNeuro
pipeline modules.

Notebooks are stored in [jupytext](https://jupytext.readthedocs.io/)
percent-format `.py` files. Convert to runnable `.ipynb` with:

```sh
jupytext --to ipynb <file>.py
```

## Available

- **`kwneuro-pipeline-walkthrough.py`** — single-subject pipeline:
  load Sherbrooke sample data via `KWNeuroImporter`, denoise, fit
  DTI + FA/MD, extract brain mask (if `kwneuro[hdbet]` is installed),
  compute CSD peaks, list everything that landed in the scene. A
  good first run to verify the extension is wired up correctly.

## Prerequisites

1. A Slicer install with the KWNeuro extension loaded.
2. The `KWNeuroEnvironment` *Install / Update* button has been
   clicked at least once, so `kwneuro` and `kwneuro_slicer_bridge`
   are installed in Slicer's Python.
3. SlicerJupyter installed. Open its **JupyterKernel** module and
   start the kernel / server.
4. Run `jupyter notebook` from the shell; the `slicer` kernel will
   appear in the New Notebook dropdown.

If you don't want Jupyter at all, every cell in these notebooks also
runs verbatim in Slicer's built-in Python console (Ctrl+3).
