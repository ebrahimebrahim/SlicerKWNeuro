"""E4: End-to-end denoise using real Sherbrooke 3-shell DWI.

Flow:
  1. Fetch (cached) HARDI193 via dipy.data.fetch_sherbrooke_3shell.
  2. slicer.util.loadVolume on the 4D DWI NIfTI.
  3. Wrap the scene node as SlicerVolumeResource (Phase 0 gist-derived).
  4. Build a kwneuro.dwi.Dwi from the SlicerVolumeResource + in-memory
     FslBval/FslBvec (loaded from disk).
  5. Call dwi.denoise() synchronously. Time it.
  6. Wrap the returned InMemoryVolumeResource back as a Slicer scene node
     via SlicerVolumeResource.from_resource.

Run via:

  Slicer --no-main-window --no-splash --testing \\
    --python-script slicer-extn/experiments/e4_denoise_interactor.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import slicer  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from _bridge import SlicerVolumeResource  # noqa: E402


def main() -> int:
    try:
        from dipy.data import fetch_sherbrooke_3shell
        from kwneuro.dwi import Dwi
        from kwneuro.io import FslBvalResource, FslBvecResource

        from kwneuro.io import NiftiVolumeResource

        _, data_dir = fetch_sherbrooke_3shell()
        data_dir = Path(data_dir)
        nifti = data_dir / "HARDI193.nii.gz"
        bval_path = data_dir / "HARDI193.bval"
        bvec_path = data_dir / "HARDI193.bvec"
        print(f"[e4] data: {nifti}")

        # NOTE: slicer.util.loadVolume defaults to vtkMRMLScalarVolumeNode,
        # which silently drops the 4th dimension of a DWI. For 4D data we
        # go through nibabel/kwneuro so from_resource can create a
        # vtkMRMLVectorVolumeNode. Phase 1 will need a dedicated loader.
        t0 = time.perf_counter()
        in_mem = NiftiVolumeResource(nifti).load()
        t_load = time.perf_counter() - t0
        print(f"[e4] kwneuro NiftiVolumeResource load: {t_load:.2f}s, "
              f"shape={in_mem.array.shape}")

        t0 = time.perf_counter()
        svr = SlicerVolumeResource.from_resource(in_mem, name="HARDI193", show=False)
        t_wrap = time.perf_counter() - t0
        in_shape = svr.get_array().shape
        print(f"[e4] SlicerVolumeResource.from_resource: {t_wrap:.2f}s, "
              f"shape={in_shape}, node={svr.node_id}")

        bval = FslBvalResource(bval_path).load()
        bvec = FslBvecResource(bvec_path).load()
        print(f"[e4] bvals shape={bval.array.shape}, bvecs shape={bvec.array.shape}")

        dwi = Dwi(volume=svr, bval=bval, bvec=bvec)
        print("[e4] Dwi constructed with SlicerVolumeResource as the volume")

        t0 = time.perf_counter()
        denoised = dwi.denoise()
        t_denoise = time.perf_counter() - t0
        print(f"[e4] denoise: {t_denoise:.1f}s")

        # dwi.denoise() returns a new Dwi; we want its .volume (the VolumeResource).
        out_svr = SlicerVolumeResource.from_resource(
            denoised.volume, name="HARDI193_denoised", show=False,
        )
        arr = out_svr.get_array()
        out_shape = arr.shape
        print(f"[e4] output node {out_svr.node_id}, shape={out_shape}, "
              f"min={arr.min():.2f}, max={arr.max():.2f}")

        assert in_shape == out_shape, f"shape mismatch: in {in_shape}, out {out_shape}"
        print(f"[e4] PASS (denoise wall-clock {t_denoise:.1f}s)")
        return 0
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        return 1


slicer.app.exit(main())
