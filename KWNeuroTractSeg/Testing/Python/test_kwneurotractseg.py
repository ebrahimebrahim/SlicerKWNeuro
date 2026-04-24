"""Tests for KWNeuroTractSeg.

TractSeg requires a CUDA GPU + model weights + the tractseg extra.
Tests mock ``kwneuro.tractseg.extract_tractseg`` with a stub returning
a synthetic output volume, so the bridge / wrapping is exercised
without the heavy dependencies.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_dwi():
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 5, 5, 5, 13
    rng = np.random.default_rng(0)
    bvals = np.concatenate([np.zeros(1), np.full(n_grad - 1, 1000.0)]).astype(np.float64)
    dirs = rng.normal(size=(n_grad - 1, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), dirs])
    volume = rng.uniform(100, 900, size=(nx, ny, nz, n_grad)).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    return Dwi(
        volume=InMemoryVolumeResource(volume, affine, {}),
        bval=InMemoryBvalResource(bvals),
        bvec=InMemoryBvecResource(bvecs),
    )


def _synthetic_mask():
    from kwneuro.resource import InMemoryVolumeResource
    return InMemoryVolumeResource(
        array=np.ones((5, 5, 5), dtype=np.uint8),
        affine=np.diag([2.0, 3.0, 4.0, 1.0]),
        metadata={},
    )


class TestKWNeuroTractSegLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_prepare_inputs_requires_dwi(self) -> None:
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        logic = KWNeuroTractSegLogic()
        from kwneuro_slicer_bridge import InSceneVolumeResource
        mask = InSceneVolumeResource.from_resource(_synthetic_mask(), name="m").get_node()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(None, mask)

    def test_prepare_inputs_requires_mask(self) -> None:
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        from kwneuro_slicer_bridge import InSceneDwi
        logic = KWNeuroTractSegLogic()
        dwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="d").get_node()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(dwi, None)

    def test_prepare_inputs_asks_for_tractseg_extra_by_name(self) -> None:
        """Spy on ensure_extras_installed — the load-bearing check."""
        import kwneuro_slicer_bridge as bridge

        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        dwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="ts_spy_dwi").get_node()
        mask = InSceneVolumeResource.from_resource(
            _synthetic_mask(), name="ts_spy_mask",
        ).get_node()

        requested: list[list[str]] = []
        original = bridge.ensure_extras_installed

        def spy(names: list[str]) -> None:
            requested.append(list(names))

        bridge.ensure_extras_installed = spy
        try:
            KWNeuroTractSegLogic().prepare_inputs(dwi, mask)
        finally:
            bridge.ensure_extras_installed = original

        self.assertEqual(requested, [["tractseg"]])

    def test_run_tractseg_rejects_unknown_output_type(self) -> None:
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        logic = KWNeuroTractSegLogic()
        with self.assertRaises(ValueError):
            logic.run_tractseg(
                dwi=_synthetic_dwi(),
                mask=_synthetic_mask(),
                output_type="not_a_valid_type",
            )

    def test_run_tractseg_with_mocked_tractseg(self) -> None:
        """Mock extract_tractseg; verify wrapping + publish-to-scene.

        Checks that:
          * Our code calls the mock exactly once.
          * The returned synthetic array propagates through to
            publish_to_scene and the resulting node has the right
            name, class, and voxel values.

        A bug that replaced our wrapping with a different tract-seg
        library call would fail call_count == 1. A bug in
        publish_to_scene that swapped volumes would fail the value
        check.
        """
        import kwneuro.tractseg as tractseg_mod
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        nx, ny, nz = 5, 5, 5
        n_bundles = 72
        fake_array = np.zeros((nx, ny, nz, n_bundles), dtype=np.float32)
        # Mark bundle 0 with a distinct value so we can verify propagation.
        fake_array[..., 0] = 1.0
        fake_array[..., 1] = 2.0
        fake_volume = InMemoryVolumeResource(
            fake_array, np.diag([2.0, 3.0, 4.0, 1.0]), {},
        )

        call_count = [0]

        def fake_extract_tractseg(**kwargs):
            call_count[0] += 1
            return fake_volume

        original = tractseg_mod.extract_tractseg
        tractseg_mod.extract_tractseg = fake_extract_tractseg
        try:
            logic = KWNeuroTractSegLogic()
            result = logic.run_tractseg(
                dwi=_synthetic_dwi(),
                mask=_synthetic_mask(),
                output_type="tract_segmentation",
            )
            node_id = logic.publish_to_scene(
                result, "tractseg_test", "tract_segmentation",
            )
        finally:
            tractseg_mod.extract_tractseg = original

        self.assertEqual(
            call_count[0], 1,
            "extract_tractseg must be invoked exactly once — failure "
            "means our import path missed the mock and the real "
            "TractSeg would have run.",
        )

        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetName(), "tractseg_test_tractseg")
        # 4D array -> vtkMRMLVectorVolumeNode via the bridge.
        self.assertEqual(node.GetClassName(), "vtkMRMLVectorVolumeNode")

        # Value propagation: bundle 0 should be all 1.0, bundle 1 all 2.0.
        scene_arr = InSceneVolumeResource.from_node(node).get_array()
        np.testing.assert_allclose(scene_arr[..., 0], 1.0)
        np.testing.assert_allclose(scene_arr[..., 1], 2.0)

    def test_publish_to_scene_names_bind_output_type_to_suffix(self) -> None:
        """Each output_type must produce its *specific* expected node name.

        A pure length-only check (`len({names}) == 3`) would pass a
        bug that swapped endings_segmentation <-> TOM suffixes.
        Pinning the exact mapping makes swap regressions loud.
        """
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        logic = KWNeuroTractSegLogic()
        fake = InMemoryVolumeResource(
            array=np.zeros((5, 5, 5, 8), dtype=np.float32),
            affine=np.diag([2.0, 3.0, 4.0, 1.0]),
            metadata={},
        )
        expected = {
            "tract_segmentation": "name_test_dwi_tractseg",
            "endings_segmentation": "name_test_dwi_tractseg_endings",
            "TOM": "name_test_dwi_tractseg_tom",
        }
        for output_type, expected_name in expected.items():
            nid = logic.publish_to_scene(fake, "name_test_dwi", output_type)
            actual = slicer.mrmlScene.GetNodeByID(nid).GetName()
            self.assertEqual(
                actual, expected_name,
                f"publish_to_scene({output_type!r}) produced node name "
                f"{actual!r}; expected {expected_name!r}.",
            )


class TestKWNeuroTractSegWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroTractSeg")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_requires_both_selectors(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource

        widget = self._widget()
        widget.ui.inputDwiSelector.setCurrentNode(None)
        widget.ui.maskSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        dwi_node = InSceneDwi.from_dwi(_synthetic_dwi(), name="ts_widget_dwi").get_node()
        widget.ui.inputDwiSelector.setCurrentNode(dwi_node)
        widget.ui.maskSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        mask_node = InSceneVolumeResource.from_resource(
            _synthetic_mask(), name="ts_widget_mask",
        ).get_node()
        widget.ui.maskSelector.setCurrentNode(mask_node)
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)

    def test_selected_output_type_tracks_radios(self) -> None:
        """Each radio maps to the right output_type AND is exclusive.

        Sets each radio .checked = True and asserts (a) _selectedOutputType
        returns the corresponding string, and (b) the other two radios
        are unchecked. Auto-exclusivity is a Qt feature of grouped
        QRadioButtons, but PythonQt sometimes doesn't propagate when
        the widget is programmatically driven — an exclusivity break
        would cause _selectedOutputType to silently prefer whichever
        radio the if-cascade hits first.
        """
        widget = self._widget()
        radios = {
            "tract_segmentation": widget.ui.tractSegmentationRadio,
            "endings_segmentation": widget.ui.endingsSegmentationRadio,
            "TOM": widget.ui.tomRadio,
        }
        for selected_type, selected_radio in radios.items():
            selected_radio.checked = True
            self.assertEqual(
                widget._selectedOutputType(), selected_type,
                f"_selectedOutputType should be {selected_type!r} when "
                f"{selected_radio.objectName} is checked.",
            )
            for other_type, other_radio in radios.items():
                if other_type == selected_type:
                    continue
                self.assertFalse(
                    other_radio.checked,
                    f"Selecting {selected_type} left {other_type} "
                    f"ALSO checked — radio-group exclusivity broken, "
                    f"_selectedOutputType will then depend on "
                    f"if-cascade ordering rather than user intent.",
                )


if __name__ == "__main__":
    unittest.main()
