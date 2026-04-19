# E5 — SlicerJupyter build probe

## Repo state (as of Phase 0)

- `github.com/Slicer/SlicerJupyter`, default branch `master`.
- **Last commit:** 2024-04-06 (`724809a` "Fix ZeroMQ build error"). Nothing
  since.
- **No tagged releases.**
- **Open issues:** #78 (Windows cppzmq build broken on VS2022); #77 (macOS
  won't start after install); #76 (update to xeus 3.0); #73 (Jupyter Hub
  integration); #49 (kernel dies repeatedly). Several of these have been
  open for >1 year with no author response.
- **PRs:** none open; none merged in 2025.

## Pinned external dependency versions

| Dependency | Pinned version | Age | Upstream current |
|------------|---------------|-----|------------------|
| `xeus-python` | `0.14.3` | ~3 years | ~0.17–0.18 line |
| `xeus` | `2.4.1` | ~3 years | 5.x line |
| `cppzmq` | `5ad14cbd` (Slicer fork, "slicer-v4.7.0-2020-04-25") | ~6 years | 4.11+ |
| `pybind11` | `v2.8.1` | ~4 years | 2.13+ |
| `xtl` | `0.7.4` | ~3 years | 0.8+ |
| `pybind11_json` | `0.2.12` | ~3 years | similar |
| `nlohmann_json` | set elsewhere | recent-ish | — |

All pins live in
`SuperBuild/External_<name>.cmake` under `${CMAKE_PROJECT_NAME}_<name>_GIT_TAG`.

## Root cause of breakage on current Slicer

Slicer 5.9+ ships Python 3.12 and an updated pybind11. `xeus-python 0.14.3`
depends on an older pybind11 ABI and on an older `xeus`. The `cppzmq` fork
used by SlicerJupyter is unmaintained and has known VS2022 compilation
failures (issue #78, with MSVC errors C3646/C4430 in xeus template
instantiations at `zmq.hpp` line 2506).

The Discourse thread
(`discourse.slicer.org/t/slicerjupyter-not-available-for-latest-version/41791`)
confirms users on Slicer ≥5.9 / Python 3.12 cannot install SlicerJupyter
from the extension manager.

## benbennett's WIP fix

In issue #78 (July 2025) user `benbennett` claimed to have a working build
after bumping cppzmq to upstream, but:

- `gh api search/repositories?q=SlicerJupyter+user:benbennett` → no
  public fork.
- `gh api search/issues?q=repo:Slicer/SlicerJupyter+author:benbennett` →
  no further comments or PRs.

Their fix was never pushed publicly. It is not available to build against.

## Scope of a real fix (Linux-only)

Realistically, getting SlicerJupyter building against a current Slicer on
Linux means:

1. Bump `xeus-python` to a version compatible with Slicer's pybind11.
2. Bump `xeus` to a version compatible with the new `xeus-python`.
3. Bump `cppzmq` away from the 2020-era Slicer fork to upstream current.
4. Reconcile any API changes in xeus-python 0.15+ (kernel protocol
   evolved) against SlicerJupyter's integration layer.
5. Rebuild and test end-to-end: Slicer launches as the kernel, a notebook
   can execute `slicer.util.arrayFromVolume`, a node appears in the scene
   tree, etc.

**No "benbennett branch" shortcut available**, contrary to the optimistic
assumption in the plan. Steps 4 and 5 are the variable cost: they may be
trivial bumps, or they may require modifying SlicerJupyter's C++ kernel
binding code to track xeus-python's newer API.

## Go / no-go estimate

**Bucket: weeks.** Linux-only, single developer, with prior experience in
Slicer extension superbuilds: 1–3 weeks of focused work, realistically.
Cross-platform (adding macOS and Windows) would be significantly more.

**Recommendation for Phase 1.5:** defer. Phase 1 should ship assuming the
Slicer Python interactor as the sole execution environment. A SlicerJupyter
fix — once it happens — becomes additive value for the bridge without
being load-bearing. If Phase 1 sees meaningful uptake and users ask for
notebooks, Phase 1.5 can be reconsidered as its own project (and framed to
the wider Slicer community as a general-purpose contribution rather than
KWNeuro-specific).
