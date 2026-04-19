"""E2: install TractSeg via slicer.packaging with fury skipped.

Installs TractSeg as a standalone requirement (not via kwneuro[tractseg]),
because `slicer.packaging.pip_install(..., skip_packages=...)` parses its
input through `packaging.Requirement`, which does not accept filesystem
paths. Phase 1's env panel will therefore install kwneuro (core) and the
optional tractseg dependency tree in two separate steps.

Run via:

  Slicer --no-main-window --no-splash --testing \\
    --python-script slicer-extn/experiments/e2_tractseg_via_slicer_packaging.py
"""

from __future__ import annotations

import sys
import traceback

import slicer

try:
    import slicer.packaging

    skipped = slicer.packaging.pip_install(
        ["TractSeg"],
        skip_packages=["fury"],
        requester="KWNeuro Phase 0 — E2 tractseg probe",
        show_progress=False,
    )
    print("[e2] skip_packages returned:", skipped)

    # Smoke: import the optional-dep module.
    import tractseg  # noqa: F401
    from kwneuro.tractseg import extract_tractseg  # noqa: F401

    print("[e2] tractseg: IMPORT OK")
    slicer.app.exit(0)
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    slicer.app.exit(1)
