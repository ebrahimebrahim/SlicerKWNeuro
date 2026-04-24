"""KWNeuroHarmonize - cross-site harmonisation via ComBat.

Wraps ``kwneuro.harmonize.harmonize_volumes``. Unlike the other
pipeline modules this is a GROUP-level operation: one apply runs
across N subjects and produces N harmonised volumes.

User gathers the input volumes via an Add-from-selector + QListWidget
pattern (same as KWNeuroTemplate), picks a brain mask node, points
at a covariates CSV whose row order matches the volume list, and
names the batch column. neuroCombat is invoked on all of it at once.

Requires kwneuro[combat]. Only the ComBat fit runs on the worker
thread (it's CPU-bound numpy / statsmodels); scene writes are on the
main thread per the three-phase pattern.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class KWNeuroHarmonize(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Harmonize")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Cross-site ComBat harmonisation of 3D scalar volumes in a "
            "common space. Wraps kwneuro.harmonize.harmonize_volumes. "
            "Requires kwneuro[combat]."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroHarmonizeLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        volume_nodes: list[Any],
        mask_node: Any,
        covars_csv_path: Path,
        batch_col: str,
    ) -> tuple[list[Any], Any, Any, str, list[str]]:
        """Materialise volumes + mask, load the CSV. **Main thread only.**

        Returns ``(resources, mask_resource, covars_df, batch_col, names)``.
        Validates:
          * at least 2 input volumes,
          * all volumes share a spatial shape,
          * CSV exists, loads as a DataFrame, contains ``batch_col``,
          * row count matches volume count,
          * at least 2 distinct batch values (otherwise ComBat
            silently produces a near-copy).
        """
        import numpy as np
        import pandas as pd

        from kwneuro_slicer_bridge import InSceneVolumeResource, ensure_extras_installed

        if len(volume_nodes) < 2:
            msg = (
                "Harmonisation needs at least 2 input volumes "
                f"(got {len(volume_nodes)})."
            )
            raise ValueError(msg)
        if mask_node is None:
            msg = "A mask is required for harmonisation."
            raise ValueError(msg)
        if not covars_csv_path or not Path(covars_csv_path).exists():
            msg = f"Covariates CSV not found at {covars_csv_path!r}."
            raise ValueError(msg)
        if not batch_col or not batch_col.strip():
            msg = "Batch column name cannot be empty."
            raise ValueError(msg)

        ensure_extras_installed(["combat"])

        resources = [
            InSceneVolumeResource.from_node(n).to_in_memory() for n in volume_nodes
        ]
        names = [n.GetName() for n in volume_nodes]

        # Identical shape is the minimum required by ComBat (voxel-wise
        # features). Identical affine is ALSO required in practice:
        # two volumes with the same shape but different IJK-to-RAS
        # matrices are in different physical coordinate systems, and
        # harmonising them voxel-wise would produce nonsense.
        ref_shape = np.asarray(resources[0].get_array()).shape
        ref_affine = np.asarray(resources[0].get_affine())
        ref_name = volume_nodes[0].GetName()
        for node, res in zip(volume_nodes, resources):
            if np.asarray(res.get_array()).shape != ref_shape:
                msg = (
                    f"Volume {node.GetName()!r} has shape "
                    f"{np.asarray(res.get_array()).shape}, "
                    f"but expected {ref_shape} (from {ref_name!r})."
                )
                raise ValueError(msg)
            if not np.allclose(
                np.asarray(res.get_affine()), ref_affine, atol=1e-4,
            ):
                msg = (
                    f"Volume {node.GetName()!r} has a different "
                    f"IJK-to-RAS affine than {ref_name!r}. All inputs "
                    f"must share a voxel space — typically you should "
                    f"resample every input into a common template "
                    f"(e.g. via KWNeuroTemplate or KWNeuroRegister) "
                    f"before harmonising."
                )
                raise ValueError(msg)

        mask_resource = InSceneVolumeResource.from_node(mask_node).to_in_memory()
        if np.asarray(mask_resource.get_array()).shape != ref_shape:
            msg = (
                f"Mask shape {np.asarray(mask_resource.get_array()).shape} "
                f"does not match volume shape {ref_shape}."
            )
            raise ValueError(msg)
        if not np.allclose(
            np.asarray(mask_resource.get_affine()), ref_affine, atol=1e-4,
        ):
            msg = (
                f"Mask {mask_node.GetName()!r} has a different "
                f"IJK-to-RAS affine than the volumes — its voxel grid "
                f"does not line up with the input volumes."
            )
            raise ValueError(msg)

        # Excel / Windows-exported CSVs sometimes carry a UTF-8 BOM
        # that otherwise sneaks into the first column's header.
        covars_df = pd.read_csv(covars_csv_path, encoding="utf-8-sig")
        # Drop all-NaN rows (Excel often leaves trailing blanks).
        covars_df = covars_df.dropna(how="all").reset_index(drop=True)
        if batch_col not in covars_df.columns:
            msg = (
                f"Batch column {batch_col!r} not found in CSV. "
                f"Available columns: {list(covars_df.columns)}"
            )
            raise ValueError(msg)
        if len(covars_df) != len(volume_nodes):
            msg = (
                f"Covariates CSV has {len(covars_df)} rows, but there are "
                f"{len(volume_nodes)} input volumes. Row order must match."
            )
            raise ValueError(msg)
        if covars_df[batch_col].isna().any():
            missing_rows = covars_df.index[covars_df[batch_col].isna()].tolist()
            msg = (
                f"Batch column {batch_col!r} has missing / NaN values in "
                f"row(s) {missing_rows}. Every row must have a batch label."
            )
            raise ValueError(msg)
        n_batches = covars_df[batch_col].nunique()
        if n_batches < 2:
            msg = (
                f"Batch column {batch_col!r} has only {n_batches} distinct "
                f"value(s); ComBat needs at least 2 batches to harmonise."
            )
            raise ValueError(msg)

        # Log the volume-name / batch-label pairing at INFO so users
        # (and reviewers, and our manual-verification checklists) can
        # scan the log to confirm row order is what they expected.
        # The in-scene-node / CSV-row alignment is the user's to get
        # right; this makes it auditable.
        pairs = ", ".join(
            f"{name}->{batch}"
            for name, batch in zip(names, covars_df[batch_col])
        )
        logging.info("KWNeuroHarmonize: volume->batch pairing: %s", pairs)

        return resources, mask_resource, covars_df, batch_col, names

    def run_harmonize(
        self,
        volumes: list[Any],
        mask: Any,
        covars_df: Any,
        batch_col: str,
        preserve_out_of_mask: bool,
    ) -> list[Any]:
        """Run ComBat. **Thread-safe.**

        Returns a list of ``InMemoryVolumeResource`` in the same order
        as ``volumes``.
        """
        from kwneuro.harmonize import harmonize_volumes

        logging.info(
            "KWNeuroHarmonize: running ComBat on %d volumes, batch_col=%r",
            len(volumes), batch_col,
        )
        harmonised, _estimates = harmonize_volumes(
            volumes=volumes,
            covars=covars_df,
            batch_col=batch_col,
            mask=mask,
            preserve_out_of_mask=preserve_out_of_mask,
        )
        return harmonised

    def publish_to_scene(
        self, harmonised: list[Any], names: list[str],
    ) -> list[str]:
        """Publish each harmonised volume. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneVolumeResource

        ids: list[str] = []
        for resource, name in zip(harmonised, names):
            svr = InSceneVolumeResource.from_resource(
                resource, name=f"{name}_harmonized",
            )
            svr.get_node().CreateDefaultDisplayNodes()
            ids.append(svr.node_id)
        return ids

    def process(
        self,
        volume_nodes: list[Any],
        mask_node: Any,
        covars_csv_path: Path,
        batch_col: str,
        preserve_out_of_mask: bool = True,
    ) -> list[str]:
        """Synchronous full pipeline."""
        resources, mask, df, batch, names = self.prepare_inputs(
            volume_nodes, mask_node, covars_csv_path, batch_col,
        )
        harmonised = self.run_harmonize(
            resources, mask, df, batch, preserve_out_of_mask,
        )
        return self.publish_to_scene(harmonised, names)


class KWNeuroHarmonizeWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        import qt
        self._node_id_role = qt.Qt.UserRole

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroHarmonize.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroHarmonizeLogic()

        self.ui.addVolumeButton.connect("clicked(bool)", self.onAddVolumeClicked)
        self.ui.removeSelectedButton.connect(
            "clicked(bool)", self.onRemoveSelectedClicked,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self.ui.maskSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
        )
        self.ui.covarsPathLineEdit.connect(
            "currentPathChanged(QString)", self._updateApplyEnabled,
        )
        self.ui.batchColLineEdit.connect(
            "textChanged(QString)", self._updateApplyEnabled,
        )

        self._updateApplyEnabled()

    def enter(self) -> None:
        self._sync_listed_nodes()
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.volumesListWidget.count >= 2
            and self.ui.maskSelector.currentNode() is not None
            and bool(self.ui.covarsPathLineEdit.currentPath)
            and bool(self.ui.batchColLineEdit.text.strip())
        )

    def onAddVolumeClicked(self) -> None:
        import qt

        node = self.ui.volumeToAddSelector.currentNode()
        if node is None:
            return
        node_id = node.GetID()
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            if item.data(self._node_id_role) == node_id:
                return
        item = qt.QListWidgetItem(node.GetName())
        item.setData(self._node_id_role, node_id)
        self.ui.volumesListWidget.addItem(item)
        self._updateApplyEnabled()

    def onRemoveSelectedClicked(self) -> None:
        rows = sorted(
            [idx.row() for idx in self.ui.volumesListWidget.selectedIndexes()],
            reverse=True,
        )
        for row in rows:
            self.ui.volumesListWidget.takeItem(row)
        self._updateApplyEnabled()

    def _sync_listed_nodes(self) -> None:
        to_remove = []
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            node_id = item.data(self._node_id_role)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                to_remove.append(i)
            elif item.text() != node.GetName():
                item.setText(node.GetName())
        for row in reversed(to_remove):
            self.ui.volumesListWidget.takeItem(row)

    def _listed_nodes(self) -> list[Any]:
        nodes: list[Any] = []
        missing: list[str] = []
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            node = slicer.mrmlScene.GetNodeByID(item.data(self._node_id_role))
            if node is None:
                missing.append(item.text())
            else:
                nodes.append(node)
        if missing:
            msg = (
                "Some listed volumes are no longer in the scene: "
                + ", ".join(missing)
            )
            raise ValueError(msg)
        return nodes

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        covars_path = Path(self.ui.covarsPathLineEdit.currentPath)
        batch_col = self.ui.batchColLineEdit.text.strip()
        mask_node = self.ui.maskSelector.currentNode()
        preserve = self.ui.preserveOutOfMaskCheckBox.checked

        with slicer.util.tryWithErrorDisplay(_("Harmonisation failed."), waitCursor=False):
            # Drop stale entries before reading — if the user deleted a
            # listed node via Data without switching modules, the list
            # would otherwise keep raising the same "no longer in the
            # scene" error every time they click Apply.
            self._sync_listed_nodes()
            volume_nodes = self._listed_nodes()

            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                resources, mask, df, bc, names = self.logic.prepare_inputs(
                    volume_nodes, mask_node, covars_path, batch_col,
                )
            finally:
                qt.QApplication.restoreOverrideCursor()

            harmonised = run_with_progress_dialog(
                lambda: self.logic.run_harmonize(
                    resources, mask, df, bc, preserve,
                ),
                title=_("KWNeuroHarmonize"),
                status=_("Running ComBat..."),
            )

            ids = self.logic.publish_to_scene(harmonised, names)
            self.ui.resultLabel.text = f"Harmonised {len(ids)} volume(s)."


class KWNeuroHarmonizeTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroHarmonize widget smoke test")
        module = slicer.util.getModule("KWNeuroHarmonize")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
