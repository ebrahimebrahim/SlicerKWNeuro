"""Tests for KWNeuroTemplate.

``build_template`` is real ANTs + SyN per iteration, so tests use 3
tiny 8³ volumes with iterations=1 to keep runtime under ~10s.

Tests cover:
  * End-to-end: three synthetic volumes -> template node of matching
    shape, published with the requested name.
  * prepare_inputs requires >= 2 volumes.
  * Widget: Add / Remove / dedupe / apply-enable threshold.
"""
from __future__ import annotations

import unittest

import numpy as np


def _push_volume(name: str, seed: int = 0, shape=(8, 8, 8)):
    import slicer

    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import InSceneVolumeResource

    rng = np.random.default_rng(seed=seed)
    arr = np.zeros(shape, dtype=np.float32)
    # Insert a blob at a slight offset per seed so ANTs has something
    # to register; otherwise the warps are degenerate.
    offset = seed % 3
    arr[2 + offset:6 + offset, 2:6, 2:6] = 100.0
    arr += rng.normal(0.0, 3.0, size=shape).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    svr = InSceneVolumeResource.from_resource(
        InMemoryVolumeResource(array=arr, affine=affine, metadata={}),
        name=name,
    )
    return svr.get_node()


class TestKWNeuroTemplateLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_process_builds_template_different_from_simple_average(self) -> None:
        """End-to-end: 3 volumes, 1 iteration.

        Asserts the resulting template is:
          * a scene node with the right class, name, and spatial shape;
          * numerically distinguishable from the simple arithmetic mean
            of the inputs (so the SyN-register + sharpen pipeline in
            ``build_template`` actually ran — a bug that short-circuited
            to the initial simple average would pass a shape-only check).
        """
        import slicer

        from kwneuro.build_template import average_volumes
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroTemplate import KWNeuroTemplateLogic

        nodes = [
            _push_volume(f"template_input_{i}", seed=i) for i in range(3)
        ]
        inputs_in_memory = [
            InSceneVolumeResource.from_node(n).to_in_memory() for n in nodes
        ]

        logic = KWNeuroTemplateLogic()
        node_id = logic.process(nodes, iterations=1, name="test_template")

        template = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(template.GetClassName(), "vtkMRMLScalarVolumeNode")
        self.assertEqual(template.GetName(), "test_template")

        arr = InSceneVolumeResource.from_node(template).get_array()
        self.assertEqual(arr.shape, (8, 8, 8))

        # Compute the simple arithmetic mean of the inputs — what
        # build_template starts from. The final template goes through
        # SyN registration + sharpening on top, so it should be
        # distinguishable from this baseline.
        simple_average = average_volumes(inputs_in_memory).get_array()
        diff = float(np.mean(np.abs(arr - simple_average)))
        self.assertGreater(
            diff, 0.0,
            "Template identical to the simple arithmetic mean — the "
            "SyN registration / sharpening pipeline appears to have "
            "short-circuited. Expected non-zero mean absolute "
            "difference.",
        )

    def test_prepare_inputs_requires_at_least_two(self) -> None:
        from KWNeuroTemplate import KWNeuroTemplateLogic

        logic = KWNeuroTemplateLogic()
        with self.assertRaises(ValueError):
            logic.prepare_inputs([])

        with self.assertRaises(ValueError):
            logic.prepare_inputs([_push_volume("only_one")])

    def test_prepare_inputs_accepts_two(self) -> None:
        from KWNeuroTemplate import KWNeuroTemplateLogic

        logic = KWNeuroTemplateLogic()
        nodes = [_push_volume(f"two_input_{i}", seed=i) for i in range(2)]
        resources = logic.prepare_inputs(nodes)
        self.assertEqual(len(resources), 2)
        for resource in resources:
            self.assertEqual(resource.get_array().shape, (8, 8, 8))


class TestKWNeuroTemplateWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroTemplate")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_add_and_remove_volumes(self) -> None:
        import slicer

        widget = self._widget()
        # Start empty.
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        # takeItem() does not fire itemSelectionChanged, so our
        # _updateApplyEnabled wiring doesn't re-run on empty → stale
        # button state could leak from the prior test. Refresh
        # explicitly.
        widget._updateApplyEnabled()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        node_a = _push_volume("add_test_a", seed=0)
        node_b = _push_volume("add_test_b", seed=1)

        # Add first volume: button still disabled (<2 volumes).
        widget.ui.volumeToAddSelector.setCurrentNode(node_a)
        widget.onAddVolumeClicked()
        self._pump()
        self.assertEqual(widget.ui.volumesListWidget.count, 1)
        self.assertFalse(widget.ui.applyButton.enabled)

        # Add second: apply enables.
        widget.ui.volumeToAddSelector.setCurrentNode(node_b)
        widget.onAddVolumeClicked()
        self._pump()
        self.assertEqual(widget.ui.volumesListWidget.count, 2)
        self.assertTrue(widget.ui.applyButton.enabled)

    def test_add_dedupes_same_node(self) -> None:
        widget = self._widget()
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        # takeItem() does not fire itemSelectionChanged, so our
        # _updateApplyEnabled wiring doesn't re-run on empty → stale
        # button state could leak from the prior test. Refresh
        # explicitly.
        widget._updateApplyEnabled()

        node = _push_volume("dedupe_test")
        widget.ui.volumeToAddSelector.setCurrentNode(node)
        widget.onAddVolumeClicked()
        widget.onAddVolumeClicked()
        widget.onAddVolumeClicked()

        self.assertEqual(
            widget.ui.volumesListWidget.count, 1,
            "Adding the same node three times must produce only one entry.",
        )

    def test_remove_selected_drops_items(self) -> None:
        widget = self._widget()
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        # takeItem() does not fire itemSelectionChanged, so our
        # _updateApplyEnabled wiring doesn't re-run on empty → stale
        # button state could leak from the prior test. Refresh
        # explicitly.
        widget._updateApplyEnabled()

        for i in range(3):
            node = _push_volume(f"remove_test_{i}", seed=i)
            widget.ui.volumeToAddSelector.setCurrentNode(node)
            widget.onAddVolumeClicked()
        self.assertEqual(widget.ui.volumesListWidget.count, 3)

        # Select the middle row and remove.
        widget.ui.volumesListWidget.setCurrentRow(1)
        widget.onRemoveSelectedClicked()
        self.assertEqual(widget.ui.volumesListWidget.count, 2)

    def test_enter_prunes_deleted_nodes(self) -> None:
        """If a volume listed in the widget is removed from the scene,
        re-entering the module drops it from the list — otherwise
        apply would hit a node-not-found error.
        """
        import slicer

        widget = self._widget()
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        # takeItem() does not fire itemSelectionChanged, so our
        # _updateApplyEnabled wiring doesn't re-run on empty → stale
        # button state could leak from the prior test. Refresh
        # explicitly.
        widget._updateApplyEnabled()

        node_a = _push_volume("prune_a", seed=0)
        node_b = _push_volume("prune_b", seed=1)
        widget.ui.volumeToAddSelector.setCurrentNode(node_a)
        widget.onAddVolumeClicked()
        widget.ui.volumeToAddSelector.setCurrentNode(node_b)
        widget.onAddVolumeClicked()
        self.assertEqual(widget.ui.volumesListWidget.count, 2)

        slicer.mrmlScene.RemoveNode(node_a)
        widget.enter()
        # Only node_b should remain.
        self.assertEqual(widget.ui.volumesListWidget.count, 1)

    def test_enter_refreshes_renamed_nodes(self) -> None:
        """Listed volume names refresh when the user renames the backing
        node in the scene. Previously the list showed the snapshotted
        name from Add time, which lied to the user even though the
        node-ID binding still worked.
        """
        widget = self._widget()
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        widget._updateApplyEnabled()

        node_a = _push_volume("original_a", seed=0)
        node_b = _push_volume("original_b", seed=1)
        for n in (node_a, node_b):
            widget.ui.volumeToAddSelector.setCurrentNode(n)
            widget.onAddVolumeClicked()

        node_a.SetName("renamed_a")
        node_b.SetName("renamed_b")
        widget.enter()

        texts = [
            widget.ui.volumesListWidget.item(i).text()
            for i in range(widget.ui.volumesListWidget.count)
        ]
        self.assertIn("renamed_a", texts)
        self.assertIn("renamed_b", texts)
        self.assertNotIn("original_a", texts)
        self.assertNotIn("original_b", texts)


if __name__ == "__main__":
    unittest.main()
