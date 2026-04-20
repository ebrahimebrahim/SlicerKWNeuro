"""Round-trip test for `InSceneTransformResource` (linear-affine path).

Exercises `from_affine_matrix` -> `get_linear_matrices` identity. The
grid-transform path (ANTs displacement fields) is exercised end-to-end
in the tutorial rather than here, since generating a realistic warp
field requires running ANTs registration.
"""
from __future__ import annotations

import unittest

import numpy as np


class TestBridgeTransformRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_affine_matrix_roundtrip(self) -> None:
        from kwneuro_slicer_bridge import InSceneTransformResource

        affine = np.array(
            [
                [0.98, -0.17, 0.05, 1.5],
                [0.17, 0.98, -0.02, -2.3],
                [-0.05, 0.02, 0.99, 0.7],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        svr = InSceneTransformResource.from_affine_matrix(affine, name="bridge_test_affine")
        matrices = svr.get_linear_matrices()

        self.assertEqual(len(matrices), 1)
        np.testing.assert_allclose(matrices[0], affine)

    def test_node_class_is_linear_transform(self) -> None:
        from kwneuro_slicer_bridge import InSceneTransformResource

        svr = InSceneTransformResource.from_affine_matrix(np.eye(4), name="bridge_test_linear_cls")
        [node] = svr.get_nodes()
        self.assertEqual(node.GetClassName(), "vtkMRMLLinearTransformNode")


if __name__ == "__main__":
    unittest.main()
