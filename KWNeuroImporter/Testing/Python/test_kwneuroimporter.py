"""Tests for KWNeuroImporter.

Covers:
  * Disk-round-trip: save a synthetic Dwi to NIfTI+bval+bvec via
    kwneuro.io's save helpers, load it back through
    ``KWNeuroImporterLogic.load_from_paths``, and verify that the
    MRML node that lands in the scene has the expected class + shape
    + gradients.
  * Error behavior when any of the three paths is missing.
  * Widget: load button enabled-state tracking path + name fields.
  * Sherbrooke-fetch path runs successfully when the dataset is
    already cached (skipped if not).

The synthetic round-trip test is the meaningful one — it exercises
load_dwi_from_disk + publish_to_scene as a unit, asserts the node's
class is vtkMRMLDiffusionWeightedVolumeNode (NOT vtkMRMLScalarVolumeNode,
which is what Slicer's Add Data would have produced), and checks both
volume shape and gradient metadata.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_synthetic_dwi_to_disk(
    dest: Path,
    name: str = "synth_dwi",
) -> tuple[Path, Path, Path]:
    """Save a tiny DWI to NIfTI + FSL bval/bvec and return the three paths.

    kwneuro's on-disk resource ``save()`` helpers are the fixture — the
    test is only meaningful if reads round-trip through exactly the
    on-disk format the importer expects.
    """
    from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 3, 4, 5, 7
    rng = np.random.default_rng(seed=1)
    bvals = np.concatenate(
        [np.zeros(1), np.full(n_grad - 1, 1000.0)],
    ).astype(np.float64)
    directions = rng.normal(size=(n_grad - 1, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), directions]).astype(np.float64)
    volume = rng.uniform(100.0, 900.0, size=(nx, ny, nz, n_grad)).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    nii_path = dest / f"{name}.nii.gz"
    bval_path = dest / f"{name}.bval"
    bvec_path = dest / f"{name}.bvec"

    NiftiVolumeResource.save(
        InMemoryVolumeResource(array=volume, affine=affine, metadata={}), nii_path,
    )
    FslBvalResource.save(InMemoryBvalResource(array=bvals), bval_path)
    FslBvecResource.save(InMemoryBvecResource(array=bvecs), bvec_path)

    return nii_path, bval_path, bvec_path


class TestKWNeuroImporterLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_load_from_paths_creates_dwi_node(self) -> None:
        """End-to-end: disk -> DWI node, preserving 4D shape + gradients."""
        import slicer

        from KWNeuroImporter import KWNeuroImporterLogic

        with tempfile.TemporaryDirectory() as tmp:
            nii, bval, bvec = _write_synthetic_dwi_to_disk(Path(tmp), "test_import")
            logic = KWNeuroImporterLogic()
            node_id = logic.load_from_paths(nii, bval, bvec, name="imported_dwi")

            node = slicer.mrmlScene.GetNodeByID(node_id)
            self.assertEqual(
                node.GetClassName(), "vtkMRMLDiffusionWeightedVolumeNode",
                "Slicer's built-in loadVolume would have produced "
                "vtkMRMLScalarVolumeNode; the importer must use the bridge.",
            )
            self.assertEqual(node.GetName(), "imported_dwi")
            # 3D spatial extent of the image data.
            self.assertEqual(node.GetImageData().GetDimensions(), (3, 4, 5))
            # Gradients + b-values must be attached to the node.
            self.assertEqual(node.GetNumberOfGradients(), 7)

    def test_load_from_paths_round_trips_data(self) -> None:
        """Voxel values + bval / bvec must survive the save-then-import trip."""
        import slicer

        from kwneuro_slicer_bridge import InSceneDwi
        from KWNeuroImporter import KWNeuroImporterLogic

        with tempfile.TemporaryDirectory() as tmp:
            nii, bval, bvec = _write_synthetic_dwi_to_disk(Path(tmp), "rt_import")
            # Read the original back into a plain kwneuro.Dwi for comparison.
            from kwneuro.dwi import Dwi
            from kwneuro.io import (
                FslBvalResource, FslBvecResource, NiftiVolumeResource,
            )
            original = Dwi(
                NiftiVolumeResource(nii),
                FslBvalResource(bval),
                FslBvecResource(bvec),
            ).load()

            logic = KWNeuroImporterLogic()
            node_id = logic.load_from_paths(nii, bval, bvec, name="roundtrip_dwi")
            node = slicer.mrmlScene.GetNodeByID(node_id)
            in_scene = InSceneDwi.from_node(node).to_in_memory()

            np.testing.assert_allclose(
                in_scene.volume.get_array(), original.volume.get_array(),
            )
            np.testing.assert_allclose(
                in_scene.volume.get_affine(), original.volume.get_affine(),
            )
            np.testing.assert_allclose(in_scene.bval.get(), original.bval.get())
            np.testing.assert_allclose(in_scene.bvec.get(), original.bvec.get())

    def test_load_from_paths_raises_on_missing_file(self) -> None:
        """Missing any one of the three required files must raise.

        Parametrised so a typo that swapped the existence check for
        one path would be caught — not just the volume branch.
        """
        from KWNeuroImporter import KWNeuroImporterLogic

        for missing in ("volume", "bval", "bvec"):
            with self.subTest(missing=missing):
                with tempfile.TemporaryDirectory() as tmp:
                    nii, bval, bvec = _write_synthetic_dwi_to_disk(
                        Path(tmp), f"missing_{missing}",
                    )
                    logic = KWNeuroImporterLogic()
                    target = {"volume": nii, "bval": bval, "bvec": bvec}[missing]
                    target.unlink()
                    with self.assertRaises(FileNotFoundError):
                        logic.load_from_paths(
                            nii, bval, bvec, name="should_not_load",
                        )

    def test_load_sherbrooke_if_cached(self) -> None:
        """If Sherbrooke is already cached, fetch-and-load should succeed.

        Skipped (not failed) when the dataset isn't present locally; a
        CI that doesn't pre-download it shouldn't flap on this.
        """
        import slicer

        from KWNeuroImporter import KWNeuroImporterLogic

        cache = Path.home() / ".dipy" / "sherbrooke_3shell" / "HARDI193.nii.gz"
        if not cache.exists():
            self.skipTest(f"Sherbrooke not cached at {cache}")

        logic = KWNeuroImporterLogic()
        node_id = logic.load_sherbrooke(name="TestSherbrooke")
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(
            node.GetClassName(), "vtkMRMLDiffusionWeightedVolumeNode",
        )
        self.assertEqual(node.GetNumberOfGradients(), 193)

    def test_load_sherbrooke_path_with_mocked_fetch(self) -> None:
        """Fetch-path coverage that doesn't require the real dataset.

        The "cached" test above skips in CI; this monkey-patches
        `dipy.data.fetch_sherbrooke_3shell` with a no-op and points
        ``~/.dipy/sherbrooke_3shell/`` at a staged tempdir containing
        synthetic HARDI193.* files. Proves the call graph from
        ``load_sherbrooke`` through ``fetch_sherbrooke_paths`` and on
        into ``publish_to_scene`` still works under any environment.
        """
        import os
        import shutil
        import slicer

        import dipy.data

        from KWNeuroImporter import KWNeuroImporterLogic

        original_fetch = dipy.data.fetch_sherbrooke_3shell
        original_home = os.environ.get("HOME")

        with tempfile.TemporaryDirectory() as fake_home:
            staged = Path(fake_home) / ".dipy" / "sherbrooke_3shell"
            staged.mkdir(parents=True)
            nii, bval, bvec = _write_synthetic_dwi_to_disk(
                staged, "synthetic_sherbrooke",
            )
            # Rename into the expected HARDI193.* layout.
            shutil.move(str(nii), staged / "HARDI193.nii.gz")
            shutil.move(str(bval), staged / "HARDI193.bval")
            shutil.move(str(bvec), staged / "HARDI193.bvec")

            call_count = [0]

            def fake_fetch(*_args, **_kwargs) -> Any:
                call_count[0] += 1
                return None

            dipy.data.fetch_sherbrooke_3shell = fake_fetch
            os.environ["HOME"] = fake_home
            try:
                logic = KWNeuroImporterLogic()
                node_id = logic.load_sherbrooke(name="MockSherbrooke")
                node = slicer.mrmlScene.GetNodeByID(node_id)
                self.assertEqual(
                    node.GetClassName(), "vtkMRMLDiffusionWeightedVolumeNode",
                )
                self.assertEqual(node.GetNumberOfGradients(), 7)
                self.assertEqual(
                    call_count[0], 1,
                    "fetch_sherbrooke_3shell should be called exactly once",
                )
            finally:
                dipy.data.fetch_sherbrooke_3shell = original_fetch
                if original_home is not None:
                    os.environ["HOME"] = original_home
                else:
                    os.environ.pop("HOME", None)


class TestKWNeuroImporterWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroImporter")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_load_button_initially_disabled(self) -> None:
        widget = self._widget()
        # Start clean — clear any currentPath the widget may have persisted.
        widget.ui.volumePathLineEdit.currentPath = ""
        widget.ui.bvalPathLineEdit.currentPath = ""
        widget.ui.bvecPathLineEdit.currentPath = ""
        self._pump()
        self.assertFalse(widget.ui.loadButton.enabled)

    def test_load_button_enables_only_when_all_paths_and_name_set(self) -> None:
        widget = self._widget()

        with tempfile.TemporaryDirectory() as tmp:
            nii, bval, bvec = _write_synthetic_dwi_to_disk(
                Path(tmp), "widget_enable_test",
            )
            widget.ui.nameLineEdit.text = "some_name"

            # Volume alone: not enough.
            widget.ui.volumePathLineEdit.currentPath = str(nii)
            widget.ui.bvalPathLineEdit.currentPath = ""
            widget.ui.bvecPathLineEdit.currentPath = ""
            self._pump()
            self.assertFalse(widget.ui.loadButton.enabled)

            widget.ui.bvalPathLineEdit.currentPath = str(bval)
            self._pump()
            self.assertFalse(widget.ui.loadButton.enabled)

            widget.ui.bvecPathLineEdit.currentPath = str(bvec)
            self._pump()
            self.assertTrue(widget.ui.loadButton.enabled)

            # Emptying the name disables again.
            widget.ui.nameLineEdit.text = "   "
            self._pump()
            self.assertFalse(widget.ui.loadButton.enabled)

    def test_signal_connection_is_load_bearing(self) -> None:
        """Deleting a currentPathChanged connection breaks button updates.

        The main ``test_load_button_enables_only_when_all_paths_and_name_set``
        exercises the signal pipeline end-to-end via ``processEvents()``
        (ctkPathLineEdit defers emission to the event loop), but a
        future refactor that replaces the signal wiring with a direct
        poll call would pass that test vacuously. This test disconnects
        one of the path selectors' signals and confirms the button
        state no longer tracks that path's changes.
        """
        widget = self._widget()

        with tempfile.TemporaryDirectory() as tmp:
            nii, bval, bvec = _write_synthetic_dwi_to_disk(
                Path(tmp), "disconnect_test",
            )
            widget.ui.volumePathLineEdit.currentPath = ""
            widget.ui.bvalPathLineEdit.currentPath = str(bval)
            widget.ui.bvecPathLineEdit.currentPath = str(bvec)
            widget.ui.nameLineEdit.text = "test"
            self._pump()
            self.assertFalse(widget.ui.loadButton.enabled)

            # Drop the volume selector's currentPathChanged -> _updateLoadEnabled
            # connection. Setting the path now should NOT re-enable the
            # button (since no other mechanism drives _updateLoadEnabled
            # on path change).
            widget.ui.volumePathLineEdit.disconnect(
                "currentPathChanged(QString)", widget._updateLoadEnabled,
            )
            widget.ui.volumePathLineEdit.currentPath = str(nii)
            self._pump()
            self.assertFalse(
                widget.ui.loadButton.enabled,
                "With the volume selector's signal disconnected, the "
                "button must not update when its path changes — if it "
                "does, a non-signal mechanism is driving _updateLoadEnabled "
                "and the signal wiring is not actually load-bearing.",
            )

            # Reconnect so we don't poison the widget for downstream tests.
            widget.ui.volumePathLineEdit.connect(
                "currentPathChanged(QString)", widget._updateLoadEnabled,
            )


if __name__ == "__main__":
    unittest.main()
