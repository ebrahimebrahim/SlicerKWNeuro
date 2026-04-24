"""End-to-end test for ``KWNeuroDTILogic.process``.

Builds a small synthetic DWI, pushes it into the scene as a
``vtkMRMLDiffusionWeightedVolumeNode`` via the bridge, calls the
logic, and asserts that the returned node IDs resolve to the right
MRML classes with the expected shapes.

The tensor fit here is not physically meaningful — the synthetic
signal is random — but dipy's ``TensorModel.fit`` succeeds on any 4D
array with ~13 gradients, so the output node shape / class can still
be asserted.
"""
from __future__ import annotations

import unittest

import numpy as np


def _mask_first_slice_out(shape: tuple[int, int, int]) -> "np.ndarray":
    """Return a binary mask that zeroes out the first slice along axis 0."""
    arr = np.ones(shape, dtype=np.uint8)
    arr[0, :, :] = 0
    return arr


def _build_segmentation_from_array(
    mask_array: "np.ndarray",
    affine: "np.ndarray",
    segmentation_name: str,
):
    """Build a `vtkMRMLSegmentationNode` containing a single segment.

    Pipeline: construct a `vtkMRMLLabelMapVolumeNode` from ``mask_array``
    (using the bridge's numpy-to-VTK converters for axis consistency),
    then import it to a fresh segmentation node via the segmentations
    module logic. Returns the segmentation node and the ID of the
    single imported segment.
    """
    import slicer

    from kwneuro_slicer_bridge.conversions import (
        affine_to_ijk_to_ras_matrix,
        numpy_to_vtk_image,
    )

    labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode", f"{segmentation_name}_labelmap",
    )
    labelmap_node.SetIJKToRASMatrix(affine_to_ijk_to_ras_matrix(affine))
    labelmap_node.SetAndObserveImageData(
        numpy_to_vtk_image(mask_array.astype(np.uint8)),
    )

    seg_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", segmentation_name,
    )
    seg_node.CreateDefaultDisplayNodes()
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
        labelmap_node, seg_node,
    )
    segment_id = seg_node.GetSegmentation().GetNthSegmentID(0)
    return seg_node, segment_id


def _synthetic_dwi_for_dti():
    """Build a synthetic kwneuro.Dwi with enough gradients for a tensor fit.

    DTI requires >= 7 gradients (1 b=0 + 6 non-collinear DW
    directions); we use 13 to give TensorModel some headroom. Signal
    is random, but positive, so the fit itself won't blow up on
    negative log-ratios.
    """
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 5, 5, 5, 13
    rng = np.random.default_rng(seed=0)

    # One b=0 + twelve b=1000 gradients
    bvals = np.concatenate(
        [np.zeros(1), np.full(n_grad - 1, 1000.0)],
    ).astype(np.float64)

    # Directions: unit vectors uniformly distributed on the sphere.
    # First gradient is the zero vector (for b=0).
    directions = rng.normal(size=(n_grad - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions]).astype(np.float64)

    # Signal: b=0 high (say 800), DW lower and noisy.
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


class TestKWNeuroDTILogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _push_dwi(self, name: str):
        from kwneuro_slicer_bridge import InSceneDwi

        return InSceneDwi.from_dwi(_synthetic_dwi_for_dti(), name=name)

    def test_process_creates_dti_fa_md(self) -> None:
        import slicer

        from KWNeuroDTI import KWNeuroDTILogic

        sdwi = self._push_dwi("test_dwi")
        logic = KWNeuroDTILogic()

        ids = logic.process(
            dwi_node=sdwi.get_node(),
            mask_node=None,
            create_fa_md=True,
        )

        self.assertIn("dti", ids)
        self.assertIsNotNone(ids["dti"])
        self.assertIsNotNone(ids["fa"])
        self.assertIsNotNone(ids["md"])

        dti_node = slicer.mrmlScene.GetNodeByID(ids["dti"])
        fa_node = slicer.mrmlScene.GetNodeByID(ids["fa"])
        md_node = slicer.mrmlScene.GetNodeByID(ids["md"])

        self.assertEqual(dti_node.GetClassName(), "vtkMRMLDiffusionTensorVolumeNode")
        self.assertEqual(fa_node.GetClassName(), "vtkMRMLScalarVolumeNode")
        self.assertEqual(md_node.GetClassName(), "vtkMRMLScalarVolumeNode")

        self.assertEqual(dti_node.GetName(), "test_dwi_dti")
        self.assertEqual(fa_node.GetName(), "test_dwi_fa")
        self.assertEqual(md_node.GetName(), "test_dwi_md")

        # Tensor node shape: (kji, 3, 3) via arrayFromVolume.
        dti_arr = slicer.util.arrayFromVolume(dti_node)
        self.assertEqual(dti_arr.shape, (5, 5, 5, 3, 3))

        fa_arr = slicer.util.arrayFromVolume(fa_node)
        md_arr = slicer.util.arrayFromVolume(md_node)
        # Scalar volumes are indexed (k, j, i).
        self.assertEqual(fa_arr.shape, (5, 5, 5))
        self.assertEqual(md_arr.shape, (5, 5, 5))

    def test_process_without_fa_md(self) -> None:
        import slicer

        from KWNeuroDTI import KWNeuroDTILogic

        sdwi = self._push_dwi("test_dwi_no_fa_md")
        logic = KWNeuroDTILogic()

        ids = logic.process(
            dwi_node=sdwi.get_node(),
            mask_node=None,
            create_fa_md=False,
        )

        self.assertIsNotNone(ids["dti"])
        self.assertIsNone(ids["fa"])
        self.assertIsNone(ids["md"])

        dti_node = slicer.mrmlScene.GetNodeByID(ids["dti"])
        self.assertEqual(dti_node.GetClassName(), "vtkMRMLDiffusionTensorVolumeNode")

    def test_process_with_mask(self) -> None:
        import slicer

        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroDTI import KWNeuroDTILogic

        sdwi = self._push_dwi("test_dwi_masked")

        # Mask out the first slice along x; affine must match the DWI.
        mask_arr = np.ones((5, 5, 5), dtype=np.float32)
        mask_arr[0, :, :] = 0
        mask_resource = InMemoryVolumeResource(
            array=mask_arr, affine=sdwi.volume.get_affine(), metadata={},
        )
        mask_svr = InSceneVolumeResource.from_resource(mask_resource, name="test_mask")

        logic = KWNeuroDTILogic()
        ids = logic.process(
            dwi_node=sdwi.get_node(),
            mask_node=mask_svr.get_node(),
            create_fa_md=False,
        )

        self.assertIsNotNone(ids["dti"])
        dti_node = slicer.mrmlScene.GetNodeByID(ids["dti"])
        self.assertEqual(dti_node.GetClassName(), "vtkMRMLDiffusionTensorVolumeNode")

    def test_process_with_segmentation_mask(self) -> None:
        import slicer

        from KWNeuroDTI import KWNeuroDTILogic

        sdwi = self._push_dwi("test_dwi_seg_mask")
        seg_node, segment_id = _build_segmentation_from_array(
            mask_array=_mask_first_slice_out(shape=(5, 5, 5)),
            affine=sdwi.volume.get_affine(),
            segmentation_name="test_seg",
        )

        logic = KWNeuroDTILogic()
        ids = logic.process(
            dwi_node=sdwi.get_node(),
            mask_node=seg_node,
            create_fa_md=False,
            segment_id=segment_id,
        )

        self.assertIsNotNone(ids["dti"])
        dti_node = slicer.mrmlScene.GetNodeByID(ids["dti"])
        self.assertEqual(dti_node.GetClassName(), "vtkMRMLDiffusionTensorVolumeNode")

    def test_process_raises_on_segmentation_without_segment_id(self) -> None:
        from KWNeuroDTI import KWNeuroDTILogic

        sdwi = self._push_dwi("test_dwi_seg_no_id")
        seg_node, _ = _build_segmentation_from_array(
            mask_array=_mask_first_slice_out(shape=(5, 5, 5)),
            affine=sdwi.volume.get_affine(),
            segmentation_name="test_seg_no_id",
        )

        logic = KWNeuroDTILogic()
        with self.assertRaises(ValueError):
            logic.process(
                dwi_node=sdwi.get_node(),
                mask_node=seg_node,
                create_fa_md=False,
                segment_id=None,
            )

    def test_process_raises_on_missing_dwi(self) -> None:
        from KWNeuroDTI import KWNeuroDTILogic

        logic = KWNeuroDTILogic()
        with self.assertRaises(ValueError):
            logic.process(dwi_node=None, mask_node=None, create_fa_md=False)


class TestKWNeuroDTIWidget(unittest.TestCase):
    """Widget-state checks. Guards against regressions where the Apply
    button fails to track scene changes — the ``.ui`` must wire
    ``mrmlSceneChanged`` to every child ``qMRMLNodeComboBox``'s
    ``setMRMLScene`` or the combo never sees scene-added DWIs and the
    Apply button stays disabled.
    """

    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroDTI")
        widgetRep = module.widgetRepresentation()
        return widgetRep.self()

    def _pump(self) -> None:
        # qMRMLNodeComboBox processes NodeAddedEvent via Qt's event
        # loop: the dropdown model updates, then a deferred update
        # auto-selects the first matching node and emits
        # currentNodeChanged. Pump events so those handlers fire before
        # we assert on the resulting widget state.
        import slicer
        slicer.app.processEvents()

    def test_apply_disabled_when_no_dwi_in_scene(self) -> None:
        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

    def test_apply_enables_when_dwi_added_after_open(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        # Load a DWI into the scene (simulating a Python-console load
        # while the user already has the module open).
        InSceneDwi.from_dwi(_synthetic_dwi_for_dti(), name="widget_test_dwi")
        self._pump()

        self.assertIsNotNone(widget.ui.inputDwiSelector.currentNode())
        self.assertTrue(widget.ui.applyButton.enabled)

    def test_apply_disables_when_dwi_removed(self) -> None:
        import slicer

        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        sdwi = InSceneDwi.from_dwi(_synthetic_dwi_for_dti(), name="widget_test_dwi_remove")
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)

        slicer.mrmlScene.RemoveNode(sdwi.get_node())
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

    def test_segment_selector_hidden_without_segmentation_mask(self) -> None:
        # .isHidden() reflects the explicit setVisible() state, not the
        # effective on-screen visibility — useful in the no-main-window
        # test harness where the top-level widget is never shown so
        # isVisible() always returns False.
        widget = self._widget()
        self._pump()
        self.assertTrue(widget.ui.segmentIdSelector.isHidden())
        self.assertTrue(widget.ui.segmentIdLabel.isHidden())

    def test_segment_selector_shown_and_populated_with_segmentation(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi

        widget = self._widget()
        sdwi = InSceneDwi.from_dwi(_synthetic_dwi_for_dti(), name="widget_test_dwi_seg")
        seg_node, segment_id = _build_segmentation_from_array(
            mask_array=_mask_first_slice_out(shape=(5, 5, 5)),
            affine=sdwi.volume.get_affine(),
            segmentation_name="widget_test_seg",
        )
        self._pump()
        widget.ui.inputMaskSelector.setCurrentNode(seg_node)
        self._pump()

        self.assertFalse(widget.ui.segmentIdSelector.isHidden())
        self.assertFalse(widget.ui.segmentIdLabel.isHidden())
        self.assertEqual(widget.ui.segmentIdSelector.count, 1)
        self.assertEqual(
            widget.ui.segmentIdSelector.itemData(0), segment_id,
        )

    def test_segment_selector_hides_on_switch_back_to_labelmap(self) -> None:
        from kwneuro.resource import InMemoryVolumeResource

        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource

        widget = self._widget()
        sdwi = InSceneDwi.from_dwi(
            _synthetic_dwi_for_dti(), name="widget_test_dwi_switch",
        )
        seg_node, _ = _build_segmentation_from_array(
            mask_array=_mask_first_slice_out(shape=(5, 5, 5)),
            affine=sdwi.volume.get_affine(),
            segmentation_name="widget_test_seg_switch",
        )
        mask_svr = InSceneVolumeResource.from_resource(
            InMemoryVolumeResource(
                array=_mask_first_slice_out(shape=(5, 5, 5)),
                affine=sdwi.volume.get_affine(),
                metadata={},
            ),
            name="widget_test_lm_switch",
        )
        self._pump()

        widget.ui.inputMaskSelector.setCurrentNode(seg_node)
        self._pump()
        self.assertFalse(widget.ui.segmentIdSelector.isHidden())

        widget.ui.inputMaskSelector.setCurrentNode(mask_svr.get_node())
        self._pump()
        self.assertTrue(widget.ui.segmentIdSelector.isHidden())


if __name__ == "__main__":
    unittest.main()
