# E2 — Each kwneuro optional extra, one at a time

**Environment:** after E1, kwneuro installed from local source
(`/home/thog/kwneuro`) into Slicer's Python 3.12 at
`~/slicer-superbuild-v5.11/python-install/bin/PythonSlicer`.

## Results per extra

| Extra | Naive `pip install 'kwneuro[<extra>]'` | Added packages |
|-------|----------------------------------------|----------------|
| `[combat]` | ✅ clean | `neuroCombat 0.2.12` |
| `[noddi]` | ✅ clean | `dmri-amico 2.1.1` (wheel cached, no rebuild) |
| `[hdbet]` | ✅ clean | `hd-bet 2.0.1`, `nnunetv2 2.7.0`, `torch 2.11.0`, `torchvision`, `batchgenerators 0.25.1`, `batchgeneratorsv2 0.3.2`, `acvl-utils 0.2.6`, `dynamic-network-architectures 0.4.3`, `timm 1.0.22`, `huggingface_hub 1.11.0`, `seaborn`, `httpx`, `nvidia-cusolver`, `argparse` |
| `[tractseg]` | ❌ **fails** — `fury >=0.11.0` pulls `vtk <9.4`; pip tries to downgrade Slicer's bundled VTK 9.6.1 and hits `uninstall-no-record-file` |

## Tractseg workaround via `slicer.packaging.pip_install(..., skip_packages=["fury"])`

**Probe script:** `e2_tractseg_via_slicer_packaging.py`.

### Important finding about `skip_packages` input format

`skip_packages` internally parses each requirement via
`packaging.requirements.Requirement`, which does **not** accept filesystem
paths or `-e` editable directives. My first attempt passed
`"/home/thog/kwneuro[tractseg]"` — `Requirement(...)` raised, the input was
silently skipped (logged as a warning), and the call returned `[]` with
nothing installed.

**Phase 1 implication:** the env panel must separate the two installs:

1. Install `kwneuro` itself via whatever means is appropriate (PyPI once the
   release is refreshed — see E1; for dev, from a local path via a plain
   `pip install` or `pip_install` without `skip_packages`).
2. Independently call `slicer.packaging.pip_install(["TractSeg"],
   skip_packages=["fury"])` for the tractseg extra.

### Canonical Phase 1 call for the tractseg case

```python
import slicer.packaging

skipped = slicer.packaging.pip_install(
    ["TractSeg"],
    skip_packages=["fury"],
    requester="KWNeuroEnvironment",
    show_progress=False,  # True for the real panel, False in the probe script
)
# skipped == ["fury"]
```

### Probe result

```
Collecting TractSeg
Successfully installed TractSeg-2.10
[e2] skip_packages returned: ['fury']
[e2] tractseg: IMPORT OK
```

After the probe: `vtk 9.6.1` is intact (Slicer's bundled VTK preserved);
`import tractseg` works; `from kwneuro.tractseg import extract_tractseg`
works. No fury in the environment.

## Full-extras smoke test (post-install)

```
vtk 9.6.1
tractseg: OK
HD_BET: OK
amico: OK
neuroCombat: OK
all kwneuro extras modules: OK
```

## Phase 1 env panel consequences

- **Four extras, three install shapes.** combat / noddi / hdbet install with
  plain pip; tractseg alone needs the `skip_packages=["fury"]` path via
  `slicer.packaging`. The panel UI should offer four toggles but route the
  tractseg toggle through a different code path.
- **Persistent scipy downgrade from E1.** Slicer's originally-bundled
  `scipy 1.16.3` was replaced with `scipy 1.15.3` during E1's failed PyPI
  install. Subsequent extras did not touch it further. 1.15.3 is adequate for
  kwneuro operation but the Phase 1 env panel should probably probe for this
  and offer a "reset to Slicer-bundled versions" escape hatch.
- **HD-BET is heavy.** `torch 2.11.0` + CUDA stack ≈ several GB. The hdbet
  toggle should warn users about download size / GPU expectation before
  starting.
