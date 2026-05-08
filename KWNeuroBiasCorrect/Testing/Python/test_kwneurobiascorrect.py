"""Tests for KWNeuroBiasCorrect."""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np


def _require_structural_api() -> None:
    if importlib.util.find_spec("kwneuro.structural") is None:
        raise unittest.SkipTest("kwneuro structural API is not installed")


def _synthetic_structural():
    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro.structural import StructuralImage

    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    return StructuralImage(
        volume=InMemoryVolumeResource(
            array=arr,
            affine=affine,
            metadata={"xyzt_units": 2},
        ),
    )


def _scene_structural(name: str = "bias_t1"):
    from kwneuro_slicer_bridge import InSceneStructuralImage

    return InSceneStructuralImage.from_structural(_synthetic_structural(), name=name)


class TestKWNeuroBiasCorrectLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_run_bias_correct_calls_n4_once(self) -> None:
        import ants

        from KWNeuroBiasCorrect import KWNeuroBiasCorrectLogic

        structural = _synthetic_structural()
        original = ants.n4_bias_field_correction
        call_count = [0]

        def fake_n4(image, *args, **kwargs):
            call_count[0] += 1
            return image.clone()

        ants.n4_bias_field_correction = fake_n4
        try:
            corrected = KWNeuroBiasCorrectLogic().run_bias_correct(structural)
        finally:
            ants.n4_bias_field_correction = original

        self.assertEqual(call_count[0], 1)
        np.testing.assert_allclose(
            corrected.volume.get_array(), structural.volume.get_array(),
        )

    def test_publish_to_scene_creates_scalar_volume(self) -> None:
        import slicer

        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroBiasCorrect import KWNeuroBiasCorrectLogic

        node_id = KWNeuroBiasCorrectLogic().publish_to_scene(
            _synthetic_structural(), "bias_input",
        )
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetClassName(), "vtkMRMLScalarVolumeNode")
        self.assertEqual(node.GetName(), "bias_input_bias_corrected")
        arr = InSceneVolumeResource.from_node(node).get_array()
        np.testing.assert_allclose(arr, _synthetic_structural().volume.get_array())


class TestKWNeuroBiasCorrectWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def _widget(self):
        import slicer
        return slicer.util.getModule("KWNeuroBiasCorrect").widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_enables_when_structural_added(self) -> None:
        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        _scene_structural("bias_widget_t1")
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
