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
  + docs + a hand-written DMRI pipeline walkthrough tutorial.
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

Tests are registered via `slicer_add_python_unittest(...)` in
`KWNeuroEnvironment/Testing/Python/CMakeLists.txt` and run through
CTest against a Slicer build. The instructions below assume a Slicer
superbuild at `~/slicer-superbuild-v5.11/`; substitute your own path.

### Prerequisites

Pip-install the bridge package into Slicer's Python so the bridge
tests have something to import:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install --no-deps -e kwneuro_slicer_bridge
```

The `--no-deps` flag preserves whatever `kwneuro` is already installed
(useful when you're iterating on both repos in lockstep). Drop
`--no-deps` the first time to let pip pull `kwneuro` from the git ref
pinned in `kwneuro_slicer_bridge/pyproject.toml`.

### Configure + build the extension once

```sh
mkdir -p /tmp/kwneuro-extn-build && cd /tmp/kwneuro-extn-build
cmake -DSlicer_DIR=$HOME/slicer-superbuild-v5.11/Slicer-build $OLDPWD
cmake --build .
```

The build step copies the scripted-module sources into a per-build
`lib/Slicer-5.11/qt-scripted-modules/` tree where CTest finds them. Re-
run `cmake --build .` after editing any source under
`KWNeuroEnvironment/`. Changes in `kwneuro_slicer_bridge/` don't need
a rebuild — the pip install is editable.

### Run the suite

```sh
cd /tmp/kwneuro-extn-build
ctest -j$(nproc) --output-on-failure --no-tests=error
```

Expected output: `100% tests passed, 0 tests failed out of 7` (five
bridge round-trip tests, the env-panel smoke, plus an automatically-
added generic module-loads test from the `WITH_GENERIC_TESTS` flag).

### Run one test by name

```sh
ctest -R py_test_bridge_volume_roundtrip --no-tests=error --output-on-failure
```

`--no-tests=error` is important: without it, a typo'd regex matching
zero tests prints "No tests were found!!!" but exits 0 — a silently-
passing typo. List available tests first:

```sh
ctest -N
```

### Notes

- Sample-data prerequisite: `test_bridge_dwi_roundtrip.py`'s
  4D-shape check uses the Sherbrooke 3-shell DWI. Populate the DIPY
  cache once with:
  ```sh
  ~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer \
    -c "from dipy.data import fetch_sherbrooke_3shell; fetch_sherbrooke_3shell()"
  ```
- No GitHub-Actions-style CI yet (per Phase 0 decision — CI lands
  around Extension Index submission). Run `ctest` manually while
  iterating on Phase 1 / 2.

## Building the docs

Install the bridge with its `docs` extra (one-time setup; picks up
sphinx + sphinx-autoapi + myst_parser + sphinx_copybutton + furo),
then invoke sphinx:

```sh
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install --no-deps -e './kwneuro_slicer_bridge[docs]'
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m sphinx -n -T docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the site.

The tutorials are hand-written markdown in `docs/tutorials/`. Once
SlicerJupyter is fixed in Phase 1.5, runnable notebook versions of the
tutorials will land under `notebooks/` and a build-time step will
render them into `docs/tutorials/`.

## License

Apache-2.0 — matches `kwneuro`.
