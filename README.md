# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/brain-microstructure-exploration-tools/kwneuro)
diffusion-MRI library into Slicer — first as a scriptable bridge for pipeline
developers, then as clickable GUI modules for researchers.

## Status: Phase 0 (groundwork)

This repo currently contains scaffolding and investigation output. No
user-facing features yet. See `phase-0-findings.md` for the current state of
the investigation and the decisions feeding later phases.

Phases:

- **Phase 0** (now) — scaffold, install probes, coordinate-system correctness
  checks, async UX probes, SlicerJupyter feasibility.
- **Phase 1** — `KWNeuroEnvironment` panel (install status + smoke test) plus
  bridge `Resource` subclasses usable from the Slicer Python interactor. Docs
  and example notebooks.
- **Phase 1.5** — conditional on Phase 0 findings: fix `SlicerJupyter` so the
  bridge works from a Slicer-backed Jupyter kernel.
- **Phase 2** — GUI modules exposing kwneuro pipeline stages (DTI, NODDI,
  TractSeg, CSD, registration), many of which have no Slicer UI today.

## Layout

- `CMakeLists.txt` — extension metadata.
- `KWNeuroEnvironment/` — stub scripted module that becomes the Phase 1
  environment panel.
- `experiments/` — scratch scripts used during Phase 0 experiments (E1–E6).
- `notebooks/` — placeholder; real example notebooks land in Phase 1.
- `phase-0-findings.md` — the Phase 0 findings document.

## License

Apache-2.0 — matches `kwneuro`.
