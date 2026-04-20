"""Round-trip tests for `InSceneDti`.

kwneuro Dti volumes carry 6 lower-triangular components per voxel in
the dipy ordering (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz). Slicer's DTI node
holds the full 3x3 symmetric tensor (9 components row-major). The
bridge expands 6 -> 3x3 on push and compresses 3x3 -> 6 on read.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_dti():
    from kwneuro.dti import Dti
    from kwneuro.resource import InMemoryVolumeResource

    nx, ny, nz = 3, 4, 5
    rng = np.random.default_rng(seed=123)
    base = rng.normal(scale=5e-5, size=(nx, ny, nz, 6)).astype(np.float32)
    base[..., 0] += 1.5e-3  # Dxx
    base[..., 2] += 0.8e-3  # Dyy
    base[..., 5] += 0.6e-3  # Dzz
    affine = np.diag([2.0, 3.0, 4.0, 1.0]).astype(np.float64)
    volume = InMemoryVolumeResource(array=base, affine=affine, metadata={})
    return Dti(volume=volume)


class TestBridgeDtiRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_roundtrip_preserves_six_components(self) -> None:
        from kwneuro_slicer_bridge import InSceneDti

        dti = _synthetic_dti()
        sdti = InSceneDti.from_dti(dti, name="bridge_test_dti")
        back = sdti.to_in_memory()

        np.testing.assert_allclose(back.volume.get_array(), dti.volume.get_array())
        np.testing.assert_allclose(back.volume.get_affine(), dti.volume.get_affine())

    def test_is_a_dti(self) -> None:
        """InSceneDti drops directly into code that expects a kwneuro.Dti."""
        from kwneuro.dti import Dti

        from kwneuro_slicer_bridge import InSceneDti

        sdti = InSceneDti.from_dti(_synthetic_dti(), name="bridge_test_dti_isinstance")
        self.assertIsInstance(sdti, Dti)
        # Inherited .volume is populated with the LT representation.
        self.assertEqual(sdti.volume.get_array().shape[-1], 6)

    def test_node_type_and_tensor_attribute(self) -> None:
        from kwneuro_slicer_bridge import InSceneDti

        sdti = InSceneDti.from_dti(_synthetic_dti(), name="bridge_test_dti_type")
        node = sdti.get_node()
        self.assertEqual(node.GetClassName(), "vtkMRMLDiffusionTensorVolumeNode")
        tensors = node.GetImageData().GetPointData().GetTensors()
        self.assertIsNotNone(tensors, "DTI node must have PointData.Tensors attached")
        self.assertEqual(tensors.GetNumberOfComponents(), 9)

    def test_full_symmetry_preserved(self) -> None:
        from kwneuro_slicer_bridge import InSceneDti

        sdti = InSceneDti.from_dti(_synthetic_dti(), name="bridge_test_dti_sym")
        full = sdti.get_tensor_array()
        np.testing.assert_allclose(full, np.swapaxes(full, -1, -2))

    def test_arrayfromvolume_agreement(self) -> None:
        """`slicer.util.arrayFromVolume` on our DTI node should read the same data."""
        import slicer

        from kwneuro_slicer_bridge import InSceneDti

        sdti = InSceneDti.from_dti(_synthetic_dti(), name="bridge_test_dti_afv")
        afv = slicer.util.arrayFromVolume(sdti.get_node())  # (nz, ny, nx, 3, 3)
        mine = sdti.get_tensor_array()  # (nx, ny, nz, 3, 3)
        np.testing.assert_allclose(afv, mine.transpose(2, 1, 0, 3, 4))


if __name__ == "__main__":
    unittest.main()
