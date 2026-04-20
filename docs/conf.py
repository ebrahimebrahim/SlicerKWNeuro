from __future__ import annotations

import importlib.metadata

project = "KWNeuro Slicer extension"
copyright = "2026, Kitware"
author = "Ebrahim Ebrahim"
try:
    version = release = importlib.metadata.version("kwneuro_slicer_bridge")
except importlib.metadata.PackageNotFoundError:
    version = release = "0.0.1"

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

# The bridge package's top-level modules import `slicer` and `vtk`, which are
# not available outside of 3D Slicer's bundled Python. Mock them so
# sphinx-autoapi's introspection doesn't crash in a regular docs-build
# environment.
autodoc_mock_imports = ["slicer", "vtk", "vtkmodules", "vtk.util", "vtk.util.numpy_support"]

autoapi_dirs = ["../kwneuro_slicer_bridge/src/kwneuro_slicer_bridge"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpick_ignore_regex = [
    (r"py:.*", r"slicer\..*"),
    (r"py:.*", r"vtk\..*"),
    (r"py:.*", r"numpy\.typing\.NDArray"),
    (r"py:.*", r"kwneuro\..*"),
]
