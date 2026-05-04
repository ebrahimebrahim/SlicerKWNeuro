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


def _segment_voxel_count(segment) -> int:
    """Count positive voxels in a segment's binary labelmap representation.

    Empty segments have a degenerate or zero-valued labelmap; this
    distinguishes "channel had positive voxels" from "channel was all
    zero" in the tract-seg test.
    """
    import vtkSegmentationCorePython as vtkSegmentationCore

    labelmap_repr_name = (
        vtkSegmentationCore.vtkSegmentationConverter.
        GetSegmentationBinaryLabelmapRepresentationName()
    )
    labelmap = segment.GetRepresentation(labelmap_repr_name)
    if labelmap is None:
        return 0
    extent = labelmap.GetExtent()
    nx = extent[1] - extent[0] + 1
    ny = extent[3] - extent[2] + 1
    nz = extent[5] - extent[4] + 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        return 0
    from vtk.util import numpy_support
    arr = numpy_support.vtk_to_numpy(labelmap.GetPointData().GetScalars())
    return int((arr > 0).sum())


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
        """Mock extract_tractseg; verify wrapping + segmentation publishing.

        Checks that:
          * Our code calls the mock exactly once (any rebinding bug
            would fail this).
          * The 72-channel binary-mask output is published as a
            vtkMRMLSegmentationNode with 72 segments, each named
            after its TractSeg bundle.
          * Channel values propagate to the right segment: bundle 0
            ("AF_left") has its binary mask all-positive, bundle 71
            ("ST_OCC_right") similarly, and a marked-empty channel
            stays empty.
        """
        import kwneuro.tractseg as tractseg_mod
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic, _bundle_names_72

        nx, ny, nz = 5, 5, 5
        n_bundles = 72
        fake_array = np.zeros((nx, ny, nz, n_bundles), dtype=np.float32)
        # Mark a few channels with distinct, recognisable patterns so
        # we can verify they land on the right segments.
        fake_array[..., 0] = 1.0  # AF_left — full-volume positive
        fake_array[..., 71] = 1.0  # ST_OCC_right — full-volume positive
        # Channel 1 (AF_right) stays all-zero so we can verify empty
        # segments survive intact.

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
        self.assertEqual(node.GetClassName(), "vtkMRMLSegmentationNode")

        seg = node.GetSegmentation()
        self.assertEqual(
            seg.GetNumberOfSegments(), 72,
            "tract_segmentation should produce exactly 72 segments — "
            "one per TractSeg bundle.",
        )

        # Segment names must match the TractSeg bundle ordering.
        actual_names = [
            seg.GetSegment(seg.GetNthSegmentID(i)).GetName()
            for i in range(seg.GetNumberOfSegments())
        ]
        self.assertEqual(
            actual_names, list(_bundle_names_72()),
            "Segment names don't match the TractSeg 'All' bundle list "
            "in order — channel-to-name mapping is wrong.",
        )

        # Value propagation: the AF_left channel was all-positive,
        # so its segment's binary labelmap must be non-empty. The
        # AF_right channel was all-zero, so its segment must be empty.
        af_left_seg = seg.GetSegment(seg.GetSegmentIdBySegmentName("AF_left"))
        af_right_seg = seg.GetSegment(seg.GetSegmentIdBySegmentName("AF_right"))
        self.assertIsNotNone(af_left_seg)
        self.assertIsNotNone(af_right_seg)
        # GetBinaryLabelmapInternalRepresentation -> vtkOrientedImageData;
        # a segment with no positive voxels has empty extent.
        self.assertGreater(
            _segment_voxel_count(af_left_seg), 0,
            "AF_left was a full-positive channel; its segment must "
            "have voxels.",
        )
        self.assertEqual(
            _segment_voxel_count(af_right_seg), 0,
            "AF_right was all-zero; its segment must be empty.",
        )

    def test_publish_to_scene_node_names_per_output_type(self) -> None:
        """Each output_type produces a distinctly-named node and the
        right MRML class.

        Pins the output-name suffix so a regression that swapped two
        suffixes would fail loudly. Also asserts that
        ``tract_segmentation`` / ``endings_segmentation`` go to
        Segmentation nodes (rather than vector volumes), and that
        ``TOM`` stays as a vector volume.
        """
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        logic = KWNeuroTractSegLogic()
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

        cases = [
            (
                "tract_segmentation",
                "name_test_dwi_tractseg",
                "vtkMRMLSegmentationNode",
                72,
            ),
            (
                "endings_segmentation",
                "name_test_dwi_tractseg_endings",
                "vtkMRMLSegmentationNode",
                144,
            ),
            (
                "TOM",
                "name_test_dwi_tractseg_tom",
                "vtkMRMLVectorVolumeNode",
                60,
            ),
        ]
        for output_type, expected_name, expected_class, n_channels in cases:
            fake = InMemoryVolumeResource(
                array=np.zeros((5, 5, 5, n_channels), dtype=np.float32),
                affine=affine,
                metadata={},
            )
            nid = logic.publish_to_scene(fake, "name_test_dwi", output_type)
            node = slicer.mrmlScene.GetNodeByID(nid)
            self.assertEqual(
                node.GetName(), expected_name,
                f"publish_to_scene({output_type!r}) produced node "
                f"{node.GetName()!r}; expected {expected_name!r}.",
            )
            self.assertEqual(
                node.GetClassName(), expected_class,
                f"publish_to_scene({output_type!r}) produced a "
                f"{node.GetClassName()}; expected {expected_class}.",
            )

    def test_publish_segmentation_rejects_channel_count_mismatch(self) -> None:
        """If TractSeg ever changes its bundle count, publish_to_scene
        should fail loudly rather than silently mis-name segments."""
        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroTractSeg import KWNeuroTractSegLogic

        logic = KWNeuroTractSegLogic()
        # 8 channels but tract_segmentation expects 72 — mismatch.
        fake = InMemoryVolumeResource(
            array=np.zeros((5, 5, 5, 8), dtype=np.float32),
            affine=np.diag([2.0, 3.0, 4.0, 1.0]),
            metadata={},
        )
        with self.assertRaises(ValueError) as ctx:
            logic.publish_to_scene(fake, "mismatch", "tract_segmentation")
        msg = str(ctx.exception).lower()
        self.assertIn("channel", msg)
        self.assertIn("72", msg)


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
