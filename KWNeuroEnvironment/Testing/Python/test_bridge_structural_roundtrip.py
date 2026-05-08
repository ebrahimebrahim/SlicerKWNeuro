"""Round-trip tests for InSceneStructuralImage."""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np


def _require_structural_api() -> None:
    if importlib.util.find_spec("kwneuro.structural") is None:
        raise unittest.SkipTest("kwneuro structural API is not installed")


class TestBridgeStructuralRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_structural_roundtrip_preserves_values_and_affine(self) -> None:
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro.structural import StructuralImage
        from kwneuro_slicer_bridge import InSceneStructuralImage

        arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
        affine = np.array(
            [
                [1.1, 0.0, 0.0, -5.0],
                [0.0, 1.2, 0.0, 10.0],
                [0.0, 0.0, 1.3, -15.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        structural = StructuralImage(
            volume=InMemoryVolumeResource(
                array=arr,
                affine=affine,
                metadata={"xyzt_units": 2},
            ),
        )

        in_scene = InSceneStructuralImage.from_structural(
            structural, name="bridge_structural",
        )
        self.assertIsInstance(in_scene, StructuralImage)
        node = slicer.mrmlScene.GetNodeByID(in_scene.node_id)
        self.assertEqual(node.GetClassName(), "vtkMRMLScalarVolumeNode")

        detached = in_scene.to_in_memory()
        self.assertIsInstance(detached, StructuralImage)
        np.testing.assert_allclose(detached.volume.get_array(), arr)
        np.testing.assert_allclose(detached.volume.get_affine(), affine)
        self.assertEqual(detached.volume.get_metadata().get("xyzt_units"), 2)


if __name__ == "__main__":
    unittest.main()
