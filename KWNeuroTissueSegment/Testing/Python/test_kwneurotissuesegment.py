"""Tests for KWNeuroTissueSegment."""
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
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            metadata={"xyzt_units": 2},
        ),
    )


def _mask_resource():
    from kwneuro.resource import InMemoryVolumeResource

    return InMemoryVolumeResource(
        array=np.ones((4, 5, 6), dtype=np.uint8),
        affine=np.diag([1.0, 1.0, 1.0, 1.0]),
        metadata={"xyzt_units": 2},
    )


def _scene_structural(name: str):
    from kwneuro_slicer_bridge import InSceneStructuralImage

    return InSceneStructuralImage.from_structural(_synthetic_structural(), name=name)


class TestKWNeuroTissueSegmentLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_atropos_calls_ants_once(self) -> None:
        import ants

        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        original = ants.atropos
        call_count = [0]

        def fake_atropos(a, *args, **kwargs):
            call_count[0] += 1
            labels = (np.asarray(a.numpy()) % 3 + 1).astype(np.uint8)
            return {"segmentation": a.new_image_like(labels)}

        ants.atropos = fake_atropos
        try:
            labels = KWNeuroTissueSegmentLogic().run_segmentation(
                _synthetic_structural(), _mask_resource(), "atropos",
            )
        finally:
            ants.atropos = original

        self.assertEqual(call_count[0], 1)
        self.assertEqual(set(np.unique(labels.get_array()).tolist()), {1, 2, 3})

    def test_deep_atropos_calls_antspynet_once(self) -> None:
        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        call_count = [0]

        def fake_deep_atropos(t1, do_preprocessing=True):
            call_count[0] += 1
            labels = (np.asarray(t1.numpy()) % 6 + 1).astype(np.uint8)
            return {"segmentation_image": t1.new_image_like(labels)}

        original_module = sys.modules.get("antspynet")
        sys.modules["antspynet"] = types.SimpleNamespace(
            deep_atropos=fake_deep_atropos,
        )
        try:
            labels = KWNeuroTissueSegmentLogic().run_segmentation(
                _synthetic_structural(), None, "deep_atropos",
            )
        finally:
            if original_module is None:
                sys.modules.pop("antspynet", None)
            else:
                sys.modules["antspynet"] = original_module

        self.assertEqual(call_count[0], 1)
        self.assertEqual(set(np.unique(labels.get_array()).tolist()), {1, 2, 3, 4, 5, 6})

    def test_prepare_inputs_checks_only_deep_atropos_extra(self) -> None:
        import kwneuro_slicer_bridge

        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        scene_structural = _scene_structural("tissue_extra_t1")
        calls: list[list[str]] = []
        original = kwneuro_slicer_bridge.ensure_extras_installed

        def fake_ensure(extra_names):
            calls.append(list(extra_names))

        kwneuro_slicer_bridge.ensure_extras_installed = fake_ensure
        try:
            logic = KWNeuroTissueSegmentLogic()
            logic.prepare_inputs(scene_structural.get_node(), None, "atropos")
            self.assertEqual(calls, [])
            logic.prepare_inputs(scene_structural.get_node(), None, "deep_atropos")
            self.assertEqual(calls, [["antspynet"]])
        finally:
            kwneuro_slicer_bridge.ensure_extras_installed = original

    def test_publish_preserves_multilabel_values(self) -> None:
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        labels = np.zeros((3, 3, 3), dtype=np.uint16)
        labels[0, 0, 0] = 1
        labels[1, 1, 1] = 2
        labels[2, 2, 2] = 42
        resource = InMemoryVolumeResource(
            array=labels,
            affine=np.eye(4),
            metadata={"xyzt_units": 2},
        )
        node_id = KWNeuroTissueSegmentLogic().publish_to_scene(
            resource, "tissue_input", "atropos",
        )
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetClassName(), "vtkMRMLLabelMapVolumeNode")
        roundtrip = InSceneVolumeResource.from_node(node).get_array()
        self.assertEqual(set(np.unique(roundtrip).tolist()), {0, 1, 2, 42})


if __name__ == "__main__":
    unittest.main()
