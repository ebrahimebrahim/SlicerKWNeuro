"""Round-trip tests for `InSceneVolumeResource`.

Two checks:

A. In-memory array -> InSceneVolumeResource (via from_resource) -> get_array/get_affine
   — confirms `conversions.numpy_to_vtk_image` and `vtk_image_to_numpy`
   are inverses on both 3D and 4D arrays, and that
   `affine_to_ijk_to_ras_matrix` preserves the 4x4 affine.

B. NIfTI -> InSceneVolumeResource (via Slicer's loader) vs. nibabel.load
   agreement. Verifies the bridge's view of a Slicer-loaded volume
   matches nibabel's view of the same file on disk.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class TestBridgeVolumeRoundtrip(unittest.TestCase):
    def _synthetic_3d(self) -> NDArray[np.float32]:
        nx, ny, nz = 5, 7, 9
        arr = np.arange(nx * ny * nz, dtype=np.float32).reshape(nx, ny, nz)
        arr += 100.0 * np.arange(nx)[:, None, None]
        arr += 10.0 * np.arange(ny)[None, :, None]
        arr += 1.0 * np.arange(nz)[None, None, :]
        return arr

    def _synthetic_4d(self) -> NDArray[np.float32]:
        nx, ny, nz, nt = 4, 5, 6, 3
        arr = np.arange(nx * ny * nz * nt, dtype=np.float32).reshape(nx, ny, nz, nt)
        return arr

    def _synthetic_affine(self) -> NDArray[np.float64]:
        return np.array(
            [
                [2.0, 0.0, 0.0, -10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, -30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_in_memory_roundtrip_3d(self) -> None:
        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneVolumeResource

        arr = self._synthetic_3d()
        affine = self._synthetic_affine()
        mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})
        svr = InSceneVolumeResource.from_resource(mem, name="bridge_test_3d")

        np.testing.assert_allclose(svr.get_array(), arr)
        np.testing.assert_allclose(svr.get_affine(), affine)

    def test_in_memory_roundtrip_4d(self) -> None:
        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneVolumeResource

        arr = self._synthetic_4d()
        affine = self._synthetic_affine()
        mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})
        svr = InSceneVolumeResource.from_resource(mem, name="bridge_test_4d")

        np.testing.assert_allclose(svr.get_array(), arr)
        np.testing.assert_allclose(svr.get_affine(), affine)

    def test_nifti_nibabel_agreement(self) -> None:
        import nibabel as nib
        import slicer

        from kwneuro.io import NiftiVolumeResource
        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneVolumeResource

        arr = self._synthetic_3d()
        affine = self._synthetic_affine()
        mem = InMemoryVolumeResource(array=arr, affine=affine, metadata={})

        with tempfile.TemporaryDirectory(prefix="test_bridge_volume_") as tmp:
            path = Path(tmp) / "synthetic.nii.gz"
            NiftiVolumeResource.save(mem, path)

            node = slicer.util.loadVolume(str(path))
            svr = InSceneVolumeResource.from_node(node)
            slicer_arr = np.asarray(svr.get_array(), dtype=np.float32)
            slicer_affine = np.asarray(svr.get_affine(), dtype=np.float64)

            nib_img = nib.load(str(path))
            nib_arr = np.asarray(nib_img.get_fdata(), dtype=np.float32)
            nib_affine = np.asarray(nib_img.affine, dtype=np.float64)

            np.testing.assert_allclose(slicer_arr, nib_arr)
            np.testing.assert_allclose(slicer_affine, nib_affine)

    def test_to_in_memory(self) -> None:
        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneVolumeResource

        arr = self._synthetic_3d()
        affine = self._synthetic_affine()
        svr = InSceneVolumeResource.from_resource(
            InMemoryVolumeResource(array=arr, affine=affine, metadata={}),
            name="bridge_test_to_in_memory",
        )

        mem = svr.to_in_memory()
        self.assertIsInstance(mem, InMemoryVolumeResource)
        np.testing.assert_allclose(mem.get_array(), arr)
        np.testing.assert_allclose(mem.get_affine(), affine)


if __name__ == "__main__":
    unittest.main()
