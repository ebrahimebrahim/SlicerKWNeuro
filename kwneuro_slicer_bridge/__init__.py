# Empty source-tree-only marker. Not shipped.
#
# The real Python package lives one level deeper at
# ``kwneuro_slicer_bridge/kwneuro_slicer_bridge/`` and is the only
# thing CMake's `slicerMacroBuildScriptedModule` copies into the
# build / install tree (see CMakeLists.txt — only the nested
# `kwneuro_slicer_bridge/...` paths appear in MODULE_PYTHON_SCRIPTS).
#
# This file is here so that someone running Slicer with
# ``--additional-module-paths /path/to/slicer-extn/kwneuro_slicer_bridge``
# (i.e. directly against the source tree, not against a build)
# doesn't surprise Slicer's scripted-module loader with a directory
# that has no Python entry. After install, the importable
# ``kwneuro_slicer_bridge`` is the nested package's ``__init__.py``,
# not this one.
