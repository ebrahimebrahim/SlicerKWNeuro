"""E3: validate SlicerVolumeResource round-trip correctness.

Two round-trips are exercised:

A. In-memory array -> SlicerVolumeResource (via from_resource) -> get_array/get_affine
   — confirms `_numpy_to_vtk_image` and `_vtk_image_to_numpy` are inverses
   and that `_affine_to_ijk_to_ras_matrix` preserves the affine.

B. InMemoryVolumeResource -> on-disk NIfTI (via kwneuro.io.NiftiVolumeResource.save)
   -> slicer.util.loadVolume -> SlicerVolumeResource (via from_node)
   -> compared against nibabel.load of the same file.
   — confirms the bridge's view of a Slicer-loaded volume matches nibabel's
   view of the same NIfTI on disk.

Run via:

  Slicer --no-main-window --no-splash --testing \\
    --python-script slicer-extn/experiments/e3_roundtrip.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import slicer  # noqa: F401

# Ensure experiments/ is on sys.path so `_bridge` imports cleanly.
sys.path.insert(0, str(Path(__file__).parent))

from _bridge import InMemoryVolumeResource, SlicerVolumeResource  # noqa: E402

from kwneuro.io import NiftiVolumeResource  # noqa: E402


def _synthetic_array() -> NDArray[np.float32]:
    nx, ny, nz = 5, 7, 9
    arr = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
    arr += 100.0 * np.arange(nx)[:, None, None]
    arr += 10.0 * np.arange(ny)[None, :, None]
    arr += 1.0 * np.arange(nz)[None, None, :]
    return arr


def _synthetic_affine() -> NDArray[np.float64]:
    return np.array(
        [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, -30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_in_memory_roundtrip() -> None:
    arr = _synthetic_array()
    affine = _synthetic_affine()
    mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})
    svr = SlicerVolumeResource.from_resource(mem, name="e3_in_memory")
    got_arr = svr.get_array()
    got_affine = svr.get_affine()
    assert got_arr.shape == arr.shape, f"shape mismatch: got {got_arr.shape}, want {arr.shape}"
    assert np.allclose(got_arr, arr), "array content mismatch after round-trip"
    assert np.allclose(got_affine, affine), f"affine mismatch:\n got\n{got_affine}\n want\n{affine}"
    print("[e3/A] in-memory round-trip: PASS")


def test_nifti_nibabel_agreement(tmpdir: Path) -> None:
    import nibabel as nib

    arr = _synthetic_array()
    affine = _synthetic_affine()
    mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})

    nifti_path = tmpdir / "e3_synthetic.nii.gz"
    NiftiVolumeResource.save(mem, nifti_path)

    node = slicer.util.loadVolume(str(nifti_path))
    svr = SlicerVolumeResource.from_node(node)
    s_arr = svr.get_array()
    s_affine = svr.get_affine()

    nib_img = nib.load(str(nifti_path))
    n_arr = np.asarray(nib_img.get_fdata(), dtype=np.float32)
    n_affine = np.asarray(nib_img.affine, dtype=np.float64)

    s_arr_f = np.asarray(s_arr, dtype=np.float32)
    if s_arr_f.shape != n_arr.shape:
        raise AssertionError(f"shape mismatch: Slicer {s_arr_f.shape}, nibabel {n_arr.shape}")
    if not np.allclose(s_arr_f, n_arr):
        max_diff = float(np.max(np.abs(s_arr_f - n_arr)))
        raise AssertionError(f"Slicer vs nibabel array mismatch (max abs diff = {max_diff})")
    if not np.allclose(s_affine, n_affine):
        raise AssertionError(f"affine mismatch:\n Slicer\n{s_affine}\n nibabel\n{n_affine}")

    print("[e3/B] NIfTI <-> Slicer <-> nibabel agreement: PASS")


def main() -> int:
    try:
        test_in_memory_roundtrip()
        with tempfile.TemporaryDirectory(prefix="e3_roundtrip_") as tmp:
            test_nifti_nibabel_agreement(Path(tmp))
        print("[e3] ALL TESTS PASSED")
        return 0
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        return 1


slicer.app.exit(main())
