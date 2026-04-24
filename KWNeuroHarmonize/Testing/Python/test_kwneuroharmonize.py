"""Tests for KWNeuroHarmonize.

Exercises:
  * prepare_inputs validation (shape mismatch, mismatched row counts,
    missing batch column, fewer than 2 batches, missing mask).
  * Real neuroCombat end-to-end: four synthetic volumes across two
    batches on a small 4^3 grid. Verifies the output is a non-trivial
    transform (not identical to the inputs) and row-order is preserved.
  * The extras-install spy — ensure_extras_installed(["combat"]).
  * Widget: Add / Remove volumes, Apply enables only when all four
    inputs (>=2 volumes, mask, CSV path, batch col) are present.
"""
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _push_volume(name: str, seed: int, shape=(4, 4, 4)):
    import slicer

    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import InSceneVolumeResource

    rng = np.random.default_rng(seed=seed)
    arr = rng.normal(loc=100.0 + seed * 5.0, scale=2.0, size=shape).astype(np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    return InSceneVolumeResource.from_resource(
        InMemoryVolumeResource(array=arr, affine=affine, metadata={}),
        name=name,
    ).get_node()


def _push_mask(name: str, shape=(4, 4, 4)):
    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import InSceneVolumeResource

    arr = np.ones(shape, dtype=np.uint8)
    return InSceneVolumeResource.from_resource(
        InMemoryVolumeResource(
            array=arr, affine=np.diag([2.0, 2.0, 2.0, 1.0]), metadata={},
        ),
        name=name,
    ).get_node()


def _write_covars_csv(path: Path, rows: list[dict]) -> Path:
    """Write a small CSV. `rows` is a list of dicts with identical keys."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _combat_installed() -> bool:
    return importlib.util.find_spec("neuroCombat") is not None


class TestKWNeuroHarmonizeLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_prepare_inputs_requires_at_least_two_volumes(self) -> None:
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}],
            )
            mask = _push_mask("mask_one")
            v0 = _push_volume("v0", seed=0)
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0], mask, csv_path, "site")
            self.assertIn("at least 2", str(ctx.exception))

    def test_prepare_inputs_rejects_shape_mismatch(self) -> None:
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}],
            )
            mask = _push_mask("mask_shape")
            v0 = _push_volume("v0_shape", seed=0, shape=(4, 4, 4))
            v1 = _push_volume("v1_shape", seed=1, shape=(3, 4, 4))
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            self.assertIn("shape", str(ctx.exception).lower())

    def test_prepare_inputs_rejects_missing_batch_column(self) -> None:
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"other_col": "A"}, {"other_col": "B"}],
            )
            mask = _push_mask("mask_bc")
            v0 = _push_volume("v0_bc", seed=0)
            v1 = _push_volume("v1_bc", seed=1)
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            err = str(ctx.exception)
            self.assertIn("site", err)
            self.assertIn("other_col", err)  # tells user what IS there

    def test_prepare_inputs_rejects_row_count_mismatch(self) -> None:
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}, {"site": "A"}],
            )
            mask = _push_mask("mask_rc")
            v0 = _push_volume("v0_rc", seed=0)
            v1 = _push_volume("v1_rc", seed=1)
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            err = str(ctx.exception)
            self.assertIn("3 rows", err)
            self.assertIn("2 input volumes", err)

    def test_prepare_inputs_rejects_single_batch(self) -> None:
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "A"}],
            )
            mask = _push_mask("mask_single")
            v0 = _push_volume("v0_single", seed=0)
            v1 = _push_volume("v1_single", seed=1)
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            self.assertIn("at least 2 batches", str(ctx.exception))

    def test_asks_for_combat_extra_by_name(self) -> None:
        """Spy on ensure_extras_installed — must be exactly ["combat"]."""
        import kwneuro_slicer_bridge as bridge

        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}],
            )
            mask = _push_mask("mask_spy")
            v0 = _push_volume("v0_spy", seed=0)
            v1 = _push_volume("v1_spy", seed=1)

            requested: list[list[str]] = []
            original = bridge.ensure_extras_installed
            bridge.ensure_extras_installed = lambda names: requested.append(list(names))
            try:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            finally:
                bridge.ensure_extras_installed = original

        self.assertEqual(requested, [["combat"]])

    def test_run_harmonize_real_end_to_end(self) -> None:
        """Real neuroCombat run on 4 volumes across 2 batches.

        Asserts the actual ComBat contract: the per-batch mean GAP
        shrinks substantially after harmonisation. A bug that returned
        the input unchanged, or random noise, or some other garbage,
        would all fail this assertion — whereas a shape / count / name
        check would silently accept them.

        Fails loudly if neuroCombat isn't installed, rather than
        skipping: if we let CI skip, the only end-to-end coverage of
        this module is a no-op and users would ship regressions blind.
        """
        self.assertTrue(
            _combat_installed(),
            "neuroCombat is not installed. The Harmonize module requires "
            "kwneuro[combat]. Install it (or update CI config) rather "
            "than letting this test silently skip its only real coverage.",
        )

        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        # Two batches with deliberately different per-batch means.
        inputs_by_batch: dict[str, list[np.ndarray]] = {"A": [], "B": []}
        nodes = []
        for i, (batch, mean) in enumerate(
            [("A", 100.0), ("A", 102.0), ("B", 120.0), ("B", 118.0)],
        ):
            rng = np.random.default_rng(seed=i)
            arr = rng.normal(loc=mean, scale=2.0, size=(4, 4, 4)).astype(np.float32)
            inputs_by_batch[batch].append(arr)
            node = InSceneVolumeResource.from_resource(
                InMemoryVolumeResource(
                    array=arr, affine=np.diag([2.0, 2.0, 2.0, 1.0]), metadata={},
                ),
                name=f"harm_input_{i}",
            ).get_node()
            nodes.append(node)

        mask = _push_mask("harm_mask")

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "covars.csv",
                [{"site": "A"}, {"site": "A"}, {"site": "B"}, {"site": "B"}],
            )
            logic = KWNeuroHarmonizeLogic()
            ids = logic.process(
                nodes, mask, csv_path, "site", preserve_out_of_mask=False,
            )

        self.assertEqual(len(ids), 4)
        for nid, src_node in zip(ids, nodes):
            out_node = slicer.mrmlScene.GetNodeByID(nid)
            self.assertEqual(out_node.GetName(), f"{src_node.GetName()}_harmonized")

        # The core check: batch means should converge.
        out_a = [
            InSceneVolumeResource.from_node(
                slicer.mrmlScene.GetNodeByID(ids[i]),
            ).get_array()
            for i in (0, 1)
        ]
        out_b = [
            InSceneVolumeResource.from_node(
                slicer.mrmlScene.GetNodeByID(ids[i]),
            ).get_array()
            for i in (2, 3)
        ]
        in_gap = abs(
            float(np.mean([a.mean() for a in inputs_by_batch["A"]]))
            - float(np.mean([a.mean() for a in inputs_by_batch["B"]]))
        )
        out_gap = abs(
            float(np.mean([a.mean() for a in out_a]))
            - float(np.mean([a.mean() for a in out_b]))
        )
        self.assertGreater(in_gap, 15.0, "Setup check: expected ~20 gap.")
        self.assertLess(
            out_gap, in_gap / 2.0,
            f"Batch-mean gap should shrink substantially after "
            f"harmonisation: input {in_gap:.1f}, output {out_gap:.1f}. "
            f"A bug that returned the input unchanged or random noise "
            f"would fail this assertion.",
        )

    def test_prepare_inputs_rejects_affine_mismatch(self) -> None:
        """Two volumes with matching shape but different affines must
        be rejected — their voxel grids are in different physical
        spaces and voxel-wise harmonisation would produce nonsense.
        """
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        rng = np.random.default_rng(0)
        shape = (4, 4, 4)
        v0 = InSceneVolumeResource.from_resource(
            InMemoryVolumeResource(
                array=rng.normal(size=shape).astype(np.float32),
                affine=np.diag([2.0, 2.0, 2.0, 1.0]),
                metadata={},
            ),
            name="aff_v0",
        ).get_node()
        v1 = InSceneVolumeResource.from_resource(
            InMemoryVolumeResource(
                array=rng.normal(size=shape).astype(np.float32),
                # Different affine: shifted origin.
                affine=np.array(
                    [[2.0, 0, 0, 10], [0, 2.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1]]
                ),
                metadata={},
            ),
            name="aff_v1",
        ).get_node()
        mask = _push_mask("aff_mask", shape=shape)

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}],
            )
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1], mask, csv_path, "site")
            self.assertIn("affine", str(ctx.exception).lower())

    def test_prepare_inputs_rejects_nan_in_batch_column(self) -> None:
        """A NaN in the batch column means an unlabeled subject —
        caught with a targeted error rather than letting it become a
        silent 'one fewer batch' bug.
        """
        from KWNeuroHarmonize import KWNeuroHarmonizeLogic

        logic = KWNeuroHarmonizeLogic()
        with tempfile.TemporaryDirectory() as tmp:
            # Write with an explicitly empty field in row 2.
            csv_path = Path(tmp) / "nan.csv"
            csv_path.write_text("site,subject\nA,one\n,two\nB,three\n")
            mask = _push_mask("nan_mask")
            v0 = _push_volume("v0_nan", seed=0)
            v1 = _push_volume("v1_nan", seed=1)
            v2 = _push_volume("v2_nan", seed=2)
            with self.assertRaises(ValueError) as ctx:
                logic.prepare_inputs([v0, v1, v2], mask, csv_path, "site")
            err = str(ctx.exception)
            self.assertIn("NaN", err)
            self.assertIn("site", err)


class TestKWNeuroHarmonizeWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroHarmonize")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_enables_only_when_all_four_inputs_are_set(self) -> None:
        widget = self._widget()
        while widget.ui.volumesListWidget.count > 0:
            widget.ui.volumesListWidget.takeItem(0)
        widget._updateApplyEnabled()
        widget.ui.maskSelector.setCurrentNode(None)
        widget.ui.covarsPathLineEdit.currentPath = ""
        widget.ui.batchColLineEdit.text = ""
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        v0 = _push_volume("w_v0", seed=0)
        v1 = _push_volume("w_v1", seed=1)
        for v in (v0, v1):
            widget.ui.volumeToAddSelector.setCurrentNode(v)
            widget.onAddVolumeClicked()
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)  # mask + csv + batch still missing

        mask = _push_mask("w_mask")
        widget.ui.maskSelector.setCurrentNode(mask)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)  # csv + batch still missing

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_covars_csv(
                Path(tmp) / "c.csv",
                [{"site": "A"}, {"site": "B"}],
            )
            widget.ui.covarsPathLineEdit.currentPath = str(csv_path)
            self._pump()
            # batch col still empty-stripped? It defaults to "site" in
            # the .ui, so empty only if we reset above.
            self.assertFalse(widget.ui.applyButton.enabled)

            widget.ui.batchColLineEdit.text = "site"
            self._pump()
            self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
