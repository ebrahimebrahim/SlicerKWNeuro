# KWNeuro — 3D Slicer extension bridging `kwneuro`

A 3D Slicer extension that brings the [kwneuro](https://github.com/KitwareMedical/kwneuro)
diffusion-MRI library into Slicer — first as a scriptable bridge for pipeline
developers, then as clickable GUI modules for researchers.

## Status: Phase 1 (developer release)

Installable developer release. The `KWNeuroEnvironment` module manages
kwneuro + bridge + four extras; the `kwneuro_slicer_bridge` package
exposes four scene-backed classes, two of which subclass kwneuro's own
`Dwi` / `Dti` (so they drop directly into any pipeline function).

Phases:

- **Phase 0** (complete) — scaffold, install probes, coordinate-system
  correctness checks. Findings in `phase-0-findings.md`.
- **Phase 1** (now) — `KWNeuroEnvironment` panel + four bridge classes
  + docs + hand-written example-pipeline tutorial.
- **Phase 1.5** (after Phase 1) — fix `SlicerJupyter` so the bridge
  works from a Slicer-backed Jupyter kernel. Runnable notebook
  versions of the tutorials land then.
- **Phase 2** — GUI modules exposing kwneuro pipeline stages (DTI first,
  then NODDI, TractSeg, CSD, registration), many with no Slicer UI today.

## Layout

- `CMakeLists.txt` — extension metadata.
- `KWNeuroEnvironment/` — the environment-panel scripted module; its
  `Testing/Python/` hosts the bridge round-trip tests.
- `kwneuro_slicer_bridge/` — pip-installable Python package exposing
  the four scene-backed classes. Pinned to a specific kwneuro git ref
  via its `pyproject.toml`.
- `docs/` — Sphinx site (build instructions below).
- `notebooks/` — placeholder; runnable notebooks land once
  SlicerJupyter is fixed in Phase 1.5.
- `experiments/` — Phase 0 experiment scripts retained as reference.
- `phase-0-findings.md` — Phase 0 findings + user decisions feeding
  Phase 1.

## Running tests

Prerequisites: a Slicer build (these instructions assume
`~/slicer-superbuild-v5.11/`; substitute your own path). The
`kwneuro_slicer_bridge` package must be installed into Slicer's Python
before the bridge tests can pass — open the `KWNeuroEnvironment`
module's env panel and click **Install / Update**, or do it from a
terminal:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install --no-deps -e kwneuro_slicer_bridge
```

The `--no-deps` flag preserves whatever `kwneuro` you have installed
locally (useful during development when you're iterating on both repos
in lockstep). Drop `--no-deps` the first time to let pip pull kwneuro
from the branch pinned in `kwneuro_slicer_bridge/pyproject.toml`.

### Run a single test

```sh
~/slicer-superbuild-v5.11/Slicer-build/Slicer \
  --no-main-window --no-splash --testing \
  --additional-module-paths $(pwd)/KWNeuroEnvironment \
  --python-code "import slicer.testing; \
    slicer.testing.runUnitTest(['$(pwd)/KWNeuroEnvironment/Testing/Python'], \
                               'test_bridge_volume_roundtrip')"
```

Swap in any of the test-file basenames from
`KWNeuroEnvironment/Testing/Python/`:

- `test_bridge_volume_roundtrip`
- `test_bridge_dwi_roundtrip`
- `test_bridge_dti_roundtrip`
- `test_bridge_transform_roundtrip`
- `test_env_panel_smoke`

### Run the whole suite

```sh
for t in test_bridge_volume_roundtrip test_bridge_dwi_roundtrip \
         test_bridge_dti_roundtrip test_bridge_transform_roundtrip \
         test_env_panel_smoke; do
  echo "=== $t ==="
  ~/slicer-superbuild-v5.11/Slicer-build/Slicer \
    --no-main-window --no-splash --testing \
    --additional-module-paths $(pwd)/KWNeuroEnvironment \
    --python-code "import slicer.testing; \
      slicer.testing.runUnitTest(['$(pwd)/KWNeuroEnvironment/Testing/Python'], '$t')"
done
```

### Run the in-module `KWNeuroEnvironmentTest`

```sh
~/slicer-superbuild-v5.11/Slicer-build/Slicer \
  --no-main-window --no-splash --testing \
  --additional-module-paths $(pwd)/KWNeuroEnvironment \
  --python-code "import slicer.testing; \
    slicer.testing.runUnitTest(['$(pwd)/KWNeuroEnvironment'], 'KWNeuroEnvironment')"
```

### Notes

- Headless Slicer does not auto-exit on exception; scripted tests use
  `slicer.app.exit(...)` explicitly. Hangs generally mean a test
  script raised before reaching the exit call.
- A few tests depend on sample data cached by DIPY:
  `test_bridge_dwi_roundtrip.py`'s 4D-shape check uses the Sherbrooke
  3-shell DWI. Populate the cache by running
  `PythonSlicer -c "from dipy.data import fetch_sherbrooke_3shell; fetch_sherbrooke_3shell()"`
  once; subsequent runs use the local copy.
- CI is not set up yet (per Phase 0 decision — CI comes around
  Extension Index submission). Run tests manually while iterating on
  Phase 1 / 2.

## Building the docs

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install \
  'sphinx>=7.0' 'sphinx-autoapi>=3.0' 'myst_parser>=0.13' sphinx_copybutton \
  'furo>=2023.08.17'
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m sphinx -n -T docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the site.

The tutorials are hand-written markdown in `docs/tutorials/`. Once
SlicerJupyter is fixed in Phase 1.5, runnable notebook versions of the
tutorials will land under `notebooks/` and a build-time step will
render them into `docs/tutorials/`.

## License

Apache-2.0 — matches `kwneuro`.
