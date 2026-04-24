"""Tests for KWNeuroBrainExtract.

HD-BET's real deep-learning model is heavy and not exercised in ctest.
Instead:

* ``run_brain_extract`` is covered by monkey-patching
  ``kwneuro.masks.brain_extract_single`` to write a synthetic mask
  NIfTI at the output path — this verifies our wrapping / error
  handling around HD-BET without the GPU / model weights cost.
* ``prepare_inputs`` is covered with an explicit
  ``ensure_extras_installed`` check — if the hdbet extra is absent
  (common in test environments), the error should be the
  RuntimeError pointing at KWNeuroEnvironment, and if present, it
  should proceed.
* ``publish_to_scene`` is covered with a hand-built mask array to
  verify the labelmap node is created with the right class, affine,
  and array.

The widget test covers Apply-button-tracks-input.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


def _synthetic_dwi():
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 5, 5, 5, 7
    rng = np.random.default_rng(seed=0)
    bvals = np.concatenate(
        [np.zeros(1), np.full(n_grad - 1, 1000.0)],
    ).astype(np.float64)
    directions = rng.normal(size=(n_grad - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions]).astype(np.float64)
    volume = rng.uniform(100.0, 900.0, size=(nx, ny, nz, n_grad)).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    return Dwi(
        volume=InMemoryVolumeResource(array=volume, affine=affine, metadata={}),
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


class TestKWNeuroBrainExtractLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_publish_to_scene_creates_labelmap(self) -> None:
        """publish_to_scene produces a labelmap with the right shape + affine."""
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroBrainExtract import KWNeuroBrainExtractLogic

        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        mask_resource = InMemoryVolumeResource(
            array=mask, affine=affine, metadata={},
        )

        logic = KWNeuroBrainExtractLogic()
        node_id = logic.publish_to_scene(mask_resource, "bet_test_dwi")
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetClassName(), "vtkMRMLLabelMapVolumeNode")
        self.assertEqual(node.GetName(), "bet_test_dwi_brainmask")

        # Round-trip the voxel values.
        from kwneuro_slicer_bridge import InSceneVolumeResource
        in_scene = InSceneVolumeResource.from_node(node).to_in_memory()
        np.testing.assert_array_equal(in_scene.get_array(), mask)
        np.testing.assert_allclose(in_scene.get_affine(), affine)

    def test_run_brain_extract_with_mocked_hdbet(self) -> None:
        """Monkey-patch brain_extract_single to verify wrapping is correct.

        The real HD-BET requires GPU + model weights and is not suitable
        for ctest; we stub it to a function that just writes a synthetic
        mask NIfTI, so the test still covers our tmpdir + load-result
        plumbing.

        We also assert the mock was actually invoked — if
        ``run_brain_extract`` ever stops routing through
        ``kwneuro.masks.brain_extract_single`` (e.g. someone refactors
        to a hoisted top-level import bound to a different symbol), the
        mock would silently miss and the test would start hitting real
        HD-BET. The ``call_count`` assertion catches that.
        """
        import kwneuro.masks as masks_mod

        from kwneuro.io import NiftiVolumeResource
        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroBrainExtract import KWNeuroBrainExtractLogic

        dwi = _synthetic_dwi()
        mock_mask = (dwi.volume.get_array() > 400).any(axis=-1).astype("uint8")
        call_count = [0]

        def fake_brain_extract_single(dwi_arg, output_path):
            call_count[0] += 1
            output_path = Path(output_path)
            resource = NiftiVolumeResource.save(
                InMemoryVolumeResource(
                    array=mock_mask,
                    affine=dwi_arg.volume.get_affine(),
                    metadata={},
                ),
                output_path,
            )
            return resource

        original = masks_mod.brain_extract_single
        masks_mod.brain_extract_single = fake_brain_extract_single
        try:
            logic = KWNeuroBrainExtractLogic()
            mask_resource = logic.run_brain_extract(dwi)
        finally:
            masks_mod.brain_extract_single = original

        self.assertEqual(
            call_count[0], 1,
            "brain_extract_single must be invoked exactly once via the "
            "patched kwneuro.masks symbol. If this fails, the module "
            "has a different import path that skipped the mock — which "
            "would cause real HD-BET to run in CI.",
        )
        self.assertTrue(mask_resource.is_loaded)
        np.testing.assert_array_equal(mask_resource.get_array(), mock_mask)
        np.testing.assert_allclose(
            mask_resource.get_affine(), dwi.volume.get_affine(),
        )

    def test_prepare_inputs_checks_hdbet_extra_explicitly(self) -> None:
        """prepare_inputs must query the 'hdbet' extra specifically.

        Monkey-patch ``KWNeuroEnvironmentLogic.extras_status`` to
        always report hdbet as missing, then verify the error message
        references ``hdbet`` and KWNeuroEnvironment. Monkey-patch with
        hdbet present, verify it proceeds. This catches a regression
        where the code asks for the wrong extra name (e.g. ``"hd_bet"``
        or ``"brainextract"``) because the ``extras_status()`` call
        would report the requested key as absent regardless of the
        real HD-BET install state, making the behaviour directly
        observable.
        """
        import KWNeuroEnvironment
        from kwneuro_slicer_bridge import InSceneDwi
        from KWNeuroBrainExtract import KWNeuroBrainExtractLogic

        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="bet_extras_check")
        logic = KWNeuroBrainExtractLogic()

        # Record the requested keys by wrapping extras_status to
        # observe what ensure_extras_installed asks for.
        requested_keys: list[str] = []
        original_status = KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status

        def absent_status():
            # Return a dict that marks every real extra as False. A
            # module asking for any extra key will see it missing.
            status = original_status()
            return {k: False for k in status}

        def present_status():
            status = original_status()
            return {k: True for k in status}

        # --- absent branch: prepare_inputs must raise with 'hdbet' ---
        KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = staticmethod(
            absent_status,
        )
        try:
            with self.assertRaises(RuntimeError) as ctx:
                logic.prepare_inputs(sdwi.get_node())
            err = str(ctx.exception)
            self.assertIn("hdbet", err)
            self.assertIn("KWNeuroEnvironment", err)
            # Negative check: the error must NOT claim a different
            # extra is missing, which would indicate the code asked
            # for the wrong one.
            for wrong in ("noddi", "tractseg", "combat"):
                self.assertNotIn(
                    f"[{wrong}]", err,
                    f"Error mentions kwneuro[{wrong}] — the module is "
                    f"asking for the wrong extra.",
                )
        finally:
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = (
                staticmethod(original_status)
            )

        # --- present branch: prepare_inputs must proceed ---
        KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = staticmethod(
            present_status,
        )
        try:
            dwi, name = logic.prepare_inputs(sdwi.get_node())
            self.assertIsNotNone(dwi)
            self.assertEqual(name, "bet_extras_check")
        finally:
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = (
                staticmethod(original_status)
            )

    def test_prepare_inputs_raises_on_missing_dwi(self) -> None:
        from KWNeuroBrainExtract import KWNeuroBrainExtractLogic

        logic = KWNeuroBrainExtractLogic()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(None)


class TestKWNeuroBrainExtractWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroBrainExtract")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_disabled_when_no_dwi(self) -> None:
        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

    def test_apply_enables_when_dwi_added(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        InSceneDwi.from_dwi(_synthetic_dwi(), name="bet_widget_dwi")
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)

    def test_apply_disables_when_dwi_removed(self) -> None:
        import slicer

        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="bet_widget_remove")
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)

        slicer.mrmlScene.RemoveNode(sdwi.get_node())
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

    def test_enter_resyncs_button_state(self) -> None:
        """If the widget is revisited after a scene mutation, the apply
        button state must be correct on re-entry regardless of signal
        state history.
        """
        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        # Start with nothing: button disabled.
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        # Add a DWI, then simulate a module re-entry.
        InSceneDwi.from_dwi(_synthetic_dwi(), name="bet_widget_enter")
        widget.enter()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
