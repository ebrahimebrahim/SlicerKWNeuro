"""Tests for KWNeuroDwiToStructuralRegister."""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np


def _require_structural_api() -> None:
    if importlib.util.find_spec("kwneuro.structural") is None:
        raise unittest.SkipTest("kwneuro structural API is not installed")


def _synthetic_dwi():
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    arr = np.zeros((4, 5, 6, 3), dtype=np.float32)
    arr[..., 0] = 100.0
    arr[..., 1] = 200.0
    arr[..., 2] = 300.0
    bvals = np.array([0.0, 1000.0, 1000.0])
    bvecs = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return Dwi(
        volume=InMemoryVolumeResource(
            array=arr,
            affine=np.eye(4),
            metadata={"xyzt_units": 2},
        ),
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


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


def _scene_inputs():
    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import (
        InSceneDwi,
        InSceneStructuralImage,
        InSceneVolumeResource,
    )

    dwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="reg_dwi")
    structural = InSceneStructuralImage.from_structural(
        _synthetic_structural(), name="reg_t1",
    )
    mask_resource = InMemoryVolumeResource(
        array=np.ones((4, 5, 6), dtype=np.uint8),
        affine=np.eye(4),
        metadata={"xyzt_units": 2},
    )
    dwi_mask = InSceneVolumeResource.from_resource(mask_resource, name="reg_dwi_mask")
    structural_mask = InSceneVolumeResource.from_resource(
        mask_resource, name="reg_t1_mask",
    )
    labels = InMemoryVolumeResource(
        array=np.arange(4 * 5 * 6, dtype=np.uint16).reshape(4, 5, 6) % 4,
        affine=np.eye(4),
        metadata={"xyzt_units": 2},
    )
    labelmap = InSceneVolumeResource.from_resource(labels, name="reg_t1_labels")
    return dwi, structural, dwi_mask, structural_mask, labelmap


class _FakeTransform:
    def __init__(self) -> None:
        self.apply_calls = []

    def apply(self, **kwargs):
        self.apply_calls.append(kwargs)
        return kwargs["moving"]


class TestKWNeuroDwiToStructuralRegisterLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_run_registration_maps_arguments_and_inverse_label_warp(self) -> None:
        import kwneuro.reg as reg_mod

        from KWNeuroDwiToStructuralRegister import (
            KWNeuroDwiToStructuralRegisterLogic,
        )

        fake_transform = _FakeTransform()
        captured = {}

        def fake_register_dwi_to_structural(**kwargs):
            captured.update(kwargs)
            return fake_transform

        original = getattr(reg_mod, "register_dwi_to_structural", None)
        reg_mod.register_dwi_to_structural = fake_register_dwi_to_structural
        try:
            logic = KWNeuroDwiToStructuralRegisterLogic()
            dwi, structural, dwi_mask, structural_mask, labelmap, _, _ = (
                logic.prepare_inputs(*[node.get_node() for node in _scene_inputs()])
            )
            transform, warped_b0, warped_labels = logic.run_registration(
                dwi,
                structural,
                "Rigid",
                dwi_mask,
                structural_mask,
                labelmap,
            )
        finally:
            if original is None:
                delattr(reg_mod, "register_dwi_to_structural")
            else:
                reg_mod.register_dwi_to_structural = original

        self.assertIs(transform, fake_transform)
        self.assertIs(captured["dwi"], dwi)
        self.assertIs(captured["structural"], structural)
        self.assertEqual(captured["type_of_transform"], "Rigid")
        self.assertIs(captured["dwi_mask"], dwi_mask)
        self.assertIs(captured["structural_mask"], structural_mask)
        self.assertEqual(len(fake_transform.apply_calls), 2)
        np.testing.assert_allclose(
            warped_b0.get_array(), dwi.compute_mean_b0().get_array(),
        )
        np.testing.assert_allclose(warped_labels.get_array(), labelmap.get_array())
        label_call = fake_transform.apply_calls[1]
        self.assertTrue(label_call["invert"])
        self.assertEqual(label_call["interpolation"], "genericLabel")

    def test_run_registration_rejects_unknown_transform_type(self) -> None:
        from KWNeuroDwiToStructuralRegister import (
            KWNeuroDwiToStructuralRegisterLogic,
        )

        logic = KWNeuroDwiToStructuralRegisterLogic()
        dwi = _synthetic_dwi()
        structural = _synthetic_structural()
        with self.assertRaises(ValueError):
            logic.run_registration(
                dwi, structural, "BadTransform", None, None, None,
            )


class TestKWNeuroDwiToStructuralRegisterWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        _require_structural_api()

    def test_apply_requires_dwi_and_structural(self) -> None:
        import slicer

        widget = (
            slicer.util.getModule("KWNeuroDwiToStructuralRegister")
            .widgetRepresentation()
            .self()
        )
        widget.ui.dwiSelector.setCurrentNode(None)
        widget.ui.structuralSelector.setCurrentNode(None)
        slicer.app.processEvents()
        self.assertFalse(widget.ui.applyButton.enabled)

        dwi, structural, _, _, _ = _scene_inputs()
        widget.ui.dwiSelector.setCurrentNode(dwi.get_node())
        widget.ui.structuralSelector.setCurrentNode(None)
        slicer.app.processEvents()
        self.assertFalse(widget.ui.applyButton.enabled)

        widget.ui.structuralSelector.setCurrentNode(structural.get_node())
        slicer.app.processEvents()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
