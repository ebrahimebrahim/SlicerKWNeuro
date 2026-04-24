"""Tests for KWNeuroRegister.

The ANTs registration itself is real (fast on tiny synthetic volumes).
For ``Rigid``/``Affine`` transform types with identity-equals-answer
inputs, ANTs reliably converges to ~identity, which gives us a
deterministic assertion target.

Tests cover:
  * Rigid registration of a volume against itself → warped == input
    (within tolerance) and one linear transform node is created.
  * Missing fixed or moving raises ValueError.
  * Widget apply-enabled tracks fixed/moving selectors.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_volume(name: str, seed: int = 0):
    """A small 3D volume for ANTs registration tests."""
    import slicer

    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import InSceneVolumeResource

    rng = np.random.default_rng(seed=seed)
    nx, ny, nz = 12, 12, 12
    array = np.zeros((nx, ny, nz), dtype=np.float32)
    # A blob so ANTs has something to align to.
    array[3:9, 3:9, 3:9] = 100.0
    array += rng.normal(0.0, 5.0, size=array.shape).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    return InSceneVolumeResource.from_resource(
        InMemoryVolumeResource(array=array, affine=affine, metadata={}),
        name=name,
    ).get_node()


class TestKWNeuroRegisterLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_register_volume_to_itself_rigid(self) -> None:
        """Rigid-registering similar volumes converges to high correlation.

        A bug that returned the moving volume unmodified, zeros of the
        right shape, or some other garbage output would all pass a
        shape-only assertion. Instead, assert Pearson correlation
        between the warped output and the fixed volume exceeds a
        threshold — since our synthetic blobs overlap substantially
        and ANTs Rigid should converge to near-identity, the warped
        result should track the fixed closely.
        """
        import slicer

        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroRegister import KWNeuroRegisterLogic

        fixed_node = _synthetic_volume("reg_fixed")
        moving_node = _synthetic_volume("reg_moving", seed=1)

        logic = KWNeuroRegisterLogic()
        ids = logic.process(
            fixed_node=fixed_node,
            moving_node=moving_node,
            transform_type="Rigid",
        )

        self.assertIn("warped", ids)
        self.assertIn("transform_node_ids", ids)
        warped_node = slicer.mrmlScene.GetNodeByID(ids["warped"])
        self.assertEqual(warped_node.GetClassName(), "vtkMRMLScalarVolumeNode")
        self.assertEqual(warped_node.GetName(), "reg_moving_warped")

        self.assertGreater(len(ids["transform_node_ids"]), 0)
        for tf_id in ids["transform_node_ids"]:
            tf_node = slicer.mrmlScene.GetNodeByID(tf_id)
            self.assertIsNotNone(tf_node)
            self.assertTrue(
                tf_node.IsA("vtkMRMLTransformNode"),
                f"Transform node {tf_id} is not a vtkMRMLTransformNode",
            )

        warped_arr = InSceneVolumeResource.from_node(warped_node).get_array()
        fixed_arr = InSceneVolumeResource.from_node(fixed_node).get_array()
        moving_arr = InSceneVolumeResource.from_node(moving_node).get_array()
        self.assertEqual(warped_arr.shape, fixed_arr.shape)

        # Meaningful assertion: the warped output should correlate
        # strongly with the fixed volume.
        corr = np.corrcoef(warped_arr.ravel(), fixed_arr.ravel())[0, 1]
        self.assertGreater(
            corr, 0.8,
            f"Expected warped∼fixed correlation > 0.8 after Rigid "
            f"registration of similar blobs, got {corr:.3f}. This "
            f"failing indicates the registration didn't converge — or "
            f"returned a garbage output that happens to match shape.",
        )

        # And the registration should have MOVED the moving volume,
        # not returned it unchanged. The moving has an extra seed
        # (noise + blob shift), so warped should be closer to fixed
        # than moving was.
        moving_vs_fixed = np.corrcoef(moving_arr.ravel(), fixed_arr.ravel())[0, 1]
        self.assertGreater(
            corr, moving_vs_fixed,
            f"Warped ({corr:.3f}) should correlate with fixed at least "
            f"as well as the unregistered moving ({moving_vs_fixed:.3f}) "
            f"— otherwise registration did nothing useful.",
        )

    def test_run_registration_rejects_unknown_transform_type(self) -> None:
        from KWNeuroRegister import KWNeuroRegisterLogic

        logic = KWNeuroRegisterLogic()
        fixed = _synthetic_volume("reject_fixed")
        moving = _synthetic_volume("reject_moving")
        fx, mv, _, _, _ = logic.prepare_inputs(fixed, moving, None, None)
        with self.assertRaises(ValueError):
            logic.run_registration(fx, mv, "syn_lowercase_typo", None, None)

    def test_prepare_inputs_raises_on_missing_fixed(self) -> None:
        from KWNeuroRegister import KWNeuroRegisterLogic

        logic = KWNeuroRegisterLogic()
        moving = _synthetic_volume("reg_missing_fixed")
        with self.assertRaises(ValueError):
            logic.prepare_inputs(None, moving, None, None)

    def test_prepare_inputs_raises_on_missing_moving(self) -> None:
        from KWNeuroRegister import KWNeuroRegisterLogic

        logic = KWNeuroRegisterLogic()
        fixed = _synthetic_volume("reg_missing_moving")
        with self.assertRaises(ValueError):
            logic.prepare_inputs(fixed, None, None, None)

    def test_prepare_inputs_raises_on_both_missing(self) -> None:
        from KWNeuroRegister import KWNeuroRegisterLogic

        logic = KWNeuroRegisterLogic()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(None, None, None, None)


class TestKWNeuroRegisterWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        # Force the qMRMLNodeComboBoxes to reprocess NodeRemoved events
        # before we read applyButton.enabled — otherwise a stale
        # currentNode from a previous test can keep the button enabled.
        slicer.app.processEvents()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroRegister")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_requires_both_selectors(self) -> None:
        """Apply enables only once BOTH selectors have non-None current nodes.

        We explicitly clear the selectors at the start rather than
        rely on setUp's scene-clear to propagate to the persistent
        qMRMLNodeComboBoxes — the widget is module-scoped, not test-
        scoped, so stale selections from earlier tests can linger even
        though the backing nodes are gone.
        """
        widget = self._widget()
        widget.ui.fixedSelector.setCurrentNode(None)
        widget.ui.movingSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        fixed = _synthetic_volume("widget_fixed")
        widget.ui.fixedSelector.setCurrentNode(fixed)
        widget.ui.movingSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(
            widget.ui.applyButton.enabled,
            "Fixed set, moving unset — button must still be disabled.",
        )

        moving = _synthetic_volume("widget_moving")
        widget.ui.movingSelector.setCurrentNode(moving)
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)

        widget.ui.fixedSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(
            widget.ui.applyButton.enabled,
            "Clearing fixed must re-disable the button.",
        )


if __name__ == "__main__":
    unittest.main()
