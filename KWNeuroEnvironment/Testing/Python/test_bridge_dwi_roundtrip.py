"""Round-trip tests for `InSceneDwi`.

`InSceneDwi` is a `kwneuro.dwi.Dwi` subclass backed by a
`vtkMRMLDiffusionWeightedVolumeNode`. Tests exercise:

* Synthetic DWI pushed via `from_dwi`, detached via `to_in_memory()`,
  round-trip preserves voxel data, affine, bval, bvec.
* The node class is truly `vtkMRMLDiffusionWeightedVolumeNode`.
* `InSceneDwi` is a `Dwi` (isinstance check passes).
* `from_nifti_path` on the cached Sherbrooke 3-shell HARDI193 dataset
  preserves the 4th dimension — the whole reason this bridge method
  exists, since `slicer.util.loadVolume` silently drops it.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


def _synthetic_dwi():
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 3, 4, 5, 6
    rng = np.random.default_rng(seed=42)
    volume_array = rng.uniform(100.0, 1000.0, size=(nx, ny, nz, n_grad)).astype(np.float32)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -5.0],
            [0.0, 3.0, 0.0, 10.0],
            [0.0, 0.0, 4.0, -15.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    bvals = np.array([0.0, 1000.0, 1000.0, 2000.0, 2000.0, 3000.0], dtype=np.float64)
    unit_vectors = rng.normal(size=(n_grad - 1, 3))
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), unit_vectors]).astype(np.float64)

    volume = InMemoryVolumeResource(array=volume_array, affine=affine, metadata={})
    return Dwi(
        volume=volume,
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


class TestBridgeDwiRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_synthetic_dwi_roundtrip(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        dwi = _synthetic_dwi()
        sdwi = InSceneDwi.from_dwi(dwi, name="bridge_test_dwi")

        back = sdwi.to_in_memory()

        np.testing.assert_allclose(back.volume.get_array(), dwi.volume.get_array())
        np.testing.assert_allclose(back.volume.get_affine(), dwi.volume.get_affine())
        np.testing.assert_allclose(back.bval.get(), dwi.bval.get())
        np.testing.assert_allclose(back.bvec.get(), dwi.bvec.get())

    def test_is_a_dwi(self) -> None:
        """InSceneDwi drops directly into code that expects a kwneuro.Dwi."""
        from kwneuro.dwi import Dwi

        from kwneuro_slicer_bridge import InSceneDwi

        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="bridge_test_dwi_isinstance")
        self.assertIsInstance(sdwi, Dwi)
        # And the inherited fields are populated as expected.
        self.assertEqual(sdwi.bval.get().shape, (6,))
        self.assertEqual(sdwi.bvec.get().shape, (6, 3))

    def test_node_type_is_dwi_volume(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="bridge_test_dwi_type")
        node = sdwi.get_node()
        self.assertEqual(node.GetClassName(), "vtkMRMLDiffusionWeightedVolumeNode")
        self.assertEqual(node.GetNumberOfGradients(), sdwi.bval.get().shape[0])

    def test_from_nifti_path_preserves_4d_shape(self) -> None:
        """Sherbrooke 3-shell must load as 4D (128,128,60,193), not truncated to 3D."""
        data_dir = Path.home() / ".dipy" / "sherbrooke_3shell"
        nifti = data_dir / "HARDI193.nii.gz"
        bval = data_dir / "HARDI193.bval"
        bvec = data_dir / "HARDI193.bvec"
        if not nifti.exists():
            self.skipTest(
                f"Sherbrooke data not cached at {data_dir}; run "
                "`PythonSlicer -c 'from dipy.data import fetch_sherbrooke_3shell; "
                "fetch_sherbrooke_3shell()'` to populate.",
            )

        from kwneuro_slicer_bridge import InSceneDwi

        sdwi = InSceneDwi.from_nifti_path(
            volume_path=nifti, bval_path=bval, bvec_path=bvec, name="sherbrooke_test",
        )

        self.assertEqual(sdwi.volume.get_array().shape, (128, 128, 60, 193))
        self.assertEqual(sdwi.bval.get().shape, (193,))
        self.assertEqual(sdwi.bvec.get().shape, (193, 3))


if __name__ == "__main__":
    unittest.main()
