# E1 — Bare kwneuro install into Slicer's Python

**Environment:** `PythonSlicer` (3.12.10) from `~/slicer-superbuild-v5.11/python-install/bin/`.

## Result: FAIL from PyPI, PASS from local source

### Attempt 1: PyPI

```
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install kwneuro
```

Failed with:

```
Cannot uninstall vtk 9.6.1
The package's contents are unknown: no RECORD file was found for vtk.
```

**Root cause:** The current PyPI release `kwneuro-0.3.0` declares the optional
extras (`hd-bet`, `dmri-amico`, `TractSeg`, `backports.tarfile`) as **core
dependencies** rather than optional extras. This contradicts the current source
tree in `pyproject.toml` which has them correctly under
`[project.optional-dependencies]`.

Because hd-bet pulls in torch, and TractSeg pulls in fury (which pins
`vtk<9.4`), pip's resolver tries to replace Slicer's bundled VTK 9.6.1 with a
lower version. Slicer's VTK is installed by the superbuild (no pip RECORD
file), so the uninstall fails and the whole install aborts.

**Side effect**: before the abort, pip managed to swap out `scipy` — Slicer's
1.16.3 was downgraded to 1.15.3. Not blocking but worth knowing.

### Attempt 2: Local source tree

```
~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer -m pip install /home/thog/kwneuro
```

Succeeded. Resolved to `kwneuro-0.3.1.dev29+gf9f3e453f` (built from current
tree via setuptools_scm).

New packages installed for the bare case (via `antspyx>=0.6.2` and `dipy>=1.9`):

- `antspyx 0.6.3`
- `dipy 1.12.0`
- `rich 15.0.0`
- `typer 0.24.1`
- `statsmodels 0.14.6`
- `trx-python 0.4.0`
- plus the transitive `markdown-it-py`, `deepdiff`, `annotated-doc`, etc. that
  were already present.

**No VTK replacement; no fury; no TractSeg / hd-bet / dmri-amico.** Clean.

### Smoke test

```
PythonSlicer -c "import kwneuro; from kwneuro.resource import InMemoryVolumeResource; \
                  from kwneuro.dwi import Dwi; from kwneuro.dti import Dti; \
                  print('kwneuro', kwneuro.__version__); print('imports: OK')"
```

Output:

```
kwneuro 0.3.1.dev29+gf9f3e453f
imports: OK
```

## Phase 1 consequences

**Before Phase 1 can publish an env panel that just does
`pip_ensure(["kwneuro"])` against PyPI, the PyPI release needs to be cut
from the current source** — the 0.3.0 release is structurally incompatible
with Slicer-as-target. This is a followup on the kwneuro side, not the
extension side.

For Phase 1 dev setup, installing from source (`pip install <path>` or
`pip install git+https://github.com/.../kwneuro`) works cleanly.
