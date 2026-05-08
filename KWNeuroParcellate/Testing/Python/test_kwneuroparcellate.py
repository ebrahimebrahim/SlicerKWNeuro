"""Tests for KWNeuroParcellate."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import numpy as np


def _require_structural_api() -> None:
    if importlib.util.find_spec("kwneuro.structural") is None:
        raise unittest.SkipTest("kwneuro structural API is not installed")


def _synthetic_structural():
    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro.structural import StructuralImage

    arr = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    return StructuralImage(
        volume=InMemoryVolumeResource(
            array=arr,
            affine=np.eye(4),
            metadata={"xyzt_units": 2},
        ),
    )


def _scene_structural(name: str):
    from kwneuro_slicer_bridge import InSceneStructuralImage

    return InSceneStructuralImage.from_structural(_synthetic_structural(), name=name)


class TestKWNeuroParcellateLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_dkt_calls_antspynet_once(self) -> None:
        from KWNeuroParcellate import KWNeuroParcellateLogic

        call_count = [0]

        def fake_dkt(t1, do_preprocessing=True):
            call_count[0] += 1
            labels = (np.asarray(t1.numpy()) % 84 + 1).astype(np.uint16)
            return {"parcellation_segmentation": t1.new_image_like(labels)}

        original_module = sys.modules.get("antspynet")
        sys.modules["antspynet"] = types.SimpleNamespace(
            desikan_killiany_tourville_labeling=fake_dkt,
        )
        try:
            labels = KWNeuroParcellateLogic().run_parcellation(
                _synthetic_structural(), "dkt",
            )
        finally:
            if original_module is None:
                sys.modules.pop("antspynet", None)
            else:
                sys.modules["antspynet"] = original_module

        self.assertEqual(call_count[0], 1)
        self.assertGreater(np.max(labels.get_array()), 1)

    def test_prepare_inputs_checks_antspynet_extra(self) -> None:
        import kwneuro_slicer_bridge

        from KWNeuroParcellate import KWNeuroParcellateLogic

        scene_structural = _scene_structural("parcel_extra_t1")
        calls: list[list[str]] = []
        original = kwneuro_slicer_bridge.ensure_extras_installed

        def fake_ensure(extra_names):
            calls.append(list(extra_names))

        kwneuro_slicer_bridge.ensure_extras_installed = fake_ensure
        try:
            KWNeuroParcellateLogic().prepare_inputs(
                scene_structural.get_node(), "dkt",
            )
        finally:
            kwneuro_slicer_bridge.ensure_extras_installed = original

        self.assertEqual(calls, [["antspynet"]])

    def test_publish_preserves_dkt_label_values(self) -> None:
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroParcellate import KWNeuroParcellateLogic

        labels = np.zeros((3, 3, 3), dtype=np.uint16)
        labels[0, 0, 0] = 17
        labels[1, 1, 1] = 84
        resource = InMemoryVolumeResource(
            array=labels,
            affine=np.eye(4),
            metadata={"xyzt_units": 2},
        )
        node_id = KWNeuroParcellateLogic().publish_to_scene(
            resource, "parcel_input", "dkt",
        )
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetClassName(), "vtkMRMLLabelMapVolumeNode")
        roundtrip = InSceneVolumeResource.from_node(node).get_array()
        self.assertEqual(set(np.unique(roundtrip).tolist()), {0, 17, 84})


class TestKWNeuroParcellateWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_apply_enables_when_structural_added(self) -> None:
        import slicer

        widget = slicer.util.getModule("KWNeuroParcellate").widgetRepresentation().self()
        slicer.app.processEvents()
        self.assertFalse(widget.ui.applyButton.enabled)
        _scene_structural("parcel_widget_t1")
        slicer.app.processEvents()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
