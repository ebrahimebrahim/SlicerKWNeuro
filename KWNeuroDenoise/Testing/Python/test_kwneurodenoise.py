"""End-to-end test for ``KWNeuroDenoiseLogic``.

Builds a small synthetic DWI, pushes it to the scene, runs the logic
through ``process()`` (which chains the three phase methods), and
asserts the result node class + shape.

Patch2Self on a 5x5x5x13 synthetic with default patch_radius=(0,0,0)
runs in a few hundred milliseconds — small enough for ctest.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_dwi_for_denoise():
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 5, 5, 5, 13
    rng = np.random.default_rng(seed=0)
    bvals = np.concatenate(
        [np.zeros(1), np.full(n_grad - 1, 1000.0)],
    ).astype(np.float64)
    directions = rng.normal(size=(n_grad - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions]).astype(np.float64)

    volume = np.empty((nx, ny, nz, n_grad), dtype=np.float32)
    volume[..., 0] = rng.uniform(700.0, 900.0, size=(nx, ny, nz))
    volume[..., 1:] = rng.uniform(100.0, 400.0, size=(nx, ny, nz, n_grad - 1))

    affine = np.array(
        [
            [2.0, 0.0, 0.0, -5.0],
            [0.0, 3.0, 0.0, 10.0],
            [0.0, 0.0, 4.0, -15.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Dwi(
        volume=InMemoryVolumeResource(array=volume, affine=affine, metadata={}),
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


class TestKWNeuroDenoiseLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _push_dwi(self, name: str):
        from kwneuro_slicer_bridge import InSceneDwi

        return InSceneDwi.from_dwi(_synthetic_dwi_for_denoise(), name=name)

    def test_process_creates_denoised_dwi(self) -> None:
        import slicer

        from KWNeuroDenoise import KWNeuroDenoiseLogic

        sdwi = self._push_dwi("denoise_test_dwi")
        logic = KWNeuroDenoiseLogic()

        node_id = logic.process(sdwi.get_node())

        self.assertIsNotNone(node_id)
        denoised_node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(
            denoised_node.GetClassName(), "vtkMRMLDiffusionWeightedVolumeNode",
        )
        self.assertEqual(denoised_node.GetName(), "denoise_test_dwi_denoised")

        # Output shape must match input spatial shape + gradient count.
        self.assertEqual(denoised_node.GetImageData().GetDimensions(), (5, 5, 5))
        self.assertEqual(
            denoised_node.GetNumberOfGradients(),
            sdwi.bval.get().shape[0],
        )

    def test_process_raises_on_missing_dwi(self) -> None:
        from KWNeuroDenoise import KWNeuroDenoiseLogic

        logic = KWNeuroDenoiseLogic()
        with self.assertRaises(ValueError):
            logic.process(None)


class TestKWNeuroDenoiseWidget(unittest.TestCase):
    """Apply-button-tracks-scene-state checks (mirrors KWNeuroDTI widget tests)."""

    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroDenoise")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_disabled_when_no_dwi_in_scene(self) -> None:
        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

    def test_apply_enables_when_dwi_added(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        InSceneDwi.from_dwi(_synthetic_dwi_for_denoise(), name="widget_denoise_dwi")
        self._pump()

        self.assertIsNotNone(widget.ui.inputDwiSelector.currentNode())
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
