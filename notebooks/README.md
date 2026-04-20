# Notebooks — placeholder

Deliberately empty in Phase 1. A runnable notebook requires a Jupyter
kernel that lives inside a running 3D Slicer (so `slicer.mrmlScene`,
the bridge classes, and MRML nodes all resolve). The only such kernel
is [SlicerJupyter](https://github.com/Slicer/SlicerJupyter), which is
currently broken against Slicer 5.9+ (known Python-3.12 /
xeus-python / cppzmq incompatibility — see the Phase 0 findings).

Phase 1.5 fixes SlicerJupyter on Linux, at which point runnable
notebooks for the bridge's workflows will land here. Until then, the
same content ships as a hand-written walkthrough at
`docs/tutorials/example-pipeline.md`.
