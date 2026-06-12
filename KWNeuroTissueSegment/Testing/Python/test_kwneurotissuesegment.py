"""Tests for KWNeuroTissueSegment."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

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

    def test_deep_atropos_runs_in_python_slicer_subprocess(self) -> None:
        from kwneuro.io import NiftiVolumeResource
        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        calls = []
        original_run = subprocess.run

        def fake_run(cmd, **kwargs):
            calls.append((list(cmd), dict(kwargs)))
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1], "-c")
            self.assertIn("deep_atropos", cmd[2])
            input_path = Path(cmd[3])
            output_path = Path(cmd[4])
            self.assertTrue(input_path.exists())
            loaded = NiftiVolumeResource(input_path).load()
            labels = (np.asarray(loaded.get_array()) % 6 + 1).astype(np.uint8)
            NiftiVolumeResource.save(
                InMemoryVolumeResource(
                    array=labels,
                    affine=loaded.get_affine(),
                    metadata={"xyzt_units": 2},
                ),
                output_path,
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        subprocess.run = fake_run
        try:
            labels = KWNeuroTissueSegmentLogic().run_segmentation(
                _synthetic_structural(), None, "deep_atropos",
            )
        finally:
            subprocess.run = original_run

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1]["capture_output"])
        self.assertTrue(calls[0][1]["text"])
        self.assertFalse(calls[0][1]["check"])
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

    def test_prepare_inputs_rejects_mismatched_atropos_mask_shape(self) -> None:
        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        scene_structural = _scene_structural("tissue_mask_t1")
        bad_mask = InMemoryVolumeResource(
            array=np.ones((3, 5, 6), dtype=np.uint8),
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            metadata={"xyzt_units": 2},
        )
        bad_mask_node = InSceneVolumeResource.from_resource(
            bad_mask, name="bad_tissue_mask",
        ).get_node()

        with self.assertRaisesRegex(ValueError, "Mask shape"):
            KWNeuroTissueSegmentLogic().prepare_inputs(
                scene_structural.get_node(), bad_mask_node, "atropos",
            )

    def test_prepare_inputs_ignores_mask_for_deep_atropos(self) -> None:
        import kwneuro_slicer_bridge

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroTissueSegment import KWNeuroTissueSegmentLogic

        scene_structural = _scene_structural("tissue_deep_ignores_mask_t1")
        bad_mask = InMemoryVolumeResource(
            array=np.ones((3, 5, 6), dtype=np.uint8),
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
            metadata={"xyzt_units": 2},
        )
        bad_mask_node = InSceneVolumeResource.from_resource(
            bad_mask, name="deep_ignored_bad_mask",
        ).get_node()
        calls: list[list[str]] = []
        original = kwneuro_slicer_bridge.ensure_extras_installed

        def fake_ensure(extra_names):
            calls.append(list(extra_names))

        kwneuro_slicer_bridge.ensure_extras_installed = fake_ensure
        try:
            _structural, mask, _name, method = (
                KWNeuroTissueSegmentLogic().prepare_inputs(
                    scene_structural.get_node(), bad_mask_node, "deep_atropos",
                )
            )
        finally:
            kwneuro_slicer_bridge.ensure_extras_installed = original

        self.assertEqual(method, "deep_atropos")
        self.assertIsNone(mask)
        self.assertEqual(calls, [["antspynet"]])

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
