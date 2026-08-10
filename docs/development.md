# Development

This page covers the local build, test, and documentation workflow for
KWNeuro contributors.

## 1. Configure and build

The extension is standard Slicer CMake, no scripted-module Python
install required up front. From the repository root, point CMake at
your Slicer build tree and run the build. Set `BUILD_DIR` to any
writable directory you want to use for the extension build tree:

```sh
BUILD_DIR=/path/to/KWNeuro-build
cmake -S . -B "$BUILD_DIR" -DSlicer_DIR=/path/to/Slicer-build
cmake --build "$BUILD_DIR"
```

Substitute your own Slicer build path for `/path/to/Slicer-build`.
Re-run `cmake --build "$BUILD_DIR"` after editing any scripted module
or any file inside `kwneuro_slicer_bridge/`. The bridge goes through
the same per-file copy pipeline as the modules, so incremental builds
re-copy only the changed files.

## 2. Launch Slicer with the extension

```sh
"${BUILD_DIR}/SlicerWithKWNeuro"
```

This is a CMake-generated launcher that points Slicer at the
build-tree's module paths — the KWNeuro modules appear under
*Modules → KWNeuro* without a permanent install.

## 3. Install `kwneuro` and any extras

Open **KWNeuro Environment**, tick any optional-extra checkboxes you
want (`hdbet`, `noddi`, `tractseg`, `combat`, `antspynet`), then click
**Apply environment changes**. That pip-installs the pinned
`kwneuro` release into Slicer's Python and applies the selected extra
state; the panel drives `slicer.packaging.pip_install` for each newly
checked extra. HD-BET and TractSeg install PyTorch through Slicer's
PyTorchUtils / light-the-torch path first, then prune PyTorch packages
from the extra dependency walk so pip cannot replace the compatible
wheel. TractSeg also keeps the `skip_packages=["fury"]` handling it
needs to preserve Slicer's bundled VTK.

The `kwneuro_slicer_bridge` package ships bundled with the extension
(in the same `qt-scripted-modules/` directory as the modules), so
nothing needs to install it — `import kwneuro_slicer_bridge` just
works.

Click **Verify setup** to confirm the bridge round-trips a synthetic
volume through the scene.

For running the CTest suite (below), you need the **combat** extra
ticked — `py_test_kwneuroharmonize` fails rather than skips if
`neuroCombat` is absent.

You can now quit Slicer; the installs are persistent in Slicer's
bundled Python.

## 4. Run the test suite

```sh
ctest --test-dir "$BUILD_DIR" \
  -j$(nproc) \
  --output-on-failure \
  --no-tests=error
```

Expected: all tests pass in a few minutes. List the exact count with
`ctest --test-dir "$BUILD_DIR" -N`; it changes as scripted modules
are added. Almost every module's tests either use synthetic data or
mock the optional dependency (HD-BET, AMICO, TractSeg, ANTsPyNet).
The one exception is `py_test_kwneuroharmonize`, which fails rather
than skips without the `combat` extra.

Two tests *skip cleanly* when the Sherbrooke 3-shell DWI hasn't been
cached locally (see the note below): `test_from_nifti_path_preserves_4d_shape`
and `test_load_sherbrooke_if_cached`. The fetch code path itself is
covered by a mocked test that doesn't require the data.

### Run one test by name

```sh
ctest --test-dir "$BUILD_DIR" \
  -R py_test_kwneurodti \
  --no-tests=error \
  --output-on-failure
```

`--no-tests=error` is important: without it, a typo'd regex matching
zero tests prints "No tests were found!!!" but exits 0 — a silently-
passing typo. List available tests first:

```sh
ctest --test-dir "$BUILD_DIR" -N
```

### Sample-data prerequisite

Two tests load the Sherbrooke 3-shell DWI from DIPY's cache and
skip when it's absent. The simplest way to populate the cache is
through the extension itself: in Slicer, open **KWNeuro Importer**
and click *Load Sherbrooke 3-shell (HARDI193)* once. That downloads
the dataset to `~/.dipy/sherbrooke_3shell/`, where the CTest run
will find it.

If you would rather populate the cache headlessly for CI,
`dipy.data.fetch_sherbrooke_3shell()` does the same thing from any
Python environment that provides `dipy`.

## 5. Build the docs

The docs build runs outside Slicer:

```sh
python -m pip install sphinx sphinx-autoapi myst-parser sphinx-copybutton furo
python -m sphinx -n -T docs docs/_build/html
```

Open `docs/_build/html/index.html` to view the site. The bridge
package is no longer pip-installable, but `sphinx-autoapi` reads
the `.py` source files directly so no install is needed for it.
