"""KWNeuroTemplate - build an unbiased group-wise template via ANTs.

Wraps ``kwneuro.build_template.build_template``. User assembles a
list of input 3D volumes (via an Add-from-selector + list widget
pattern), picks an iteration count, and clicks Apply. Output is a
single template volume published to the scene.

Logic follows the three-phase split: inputs are materialised into
in-memory resources on the main thread, ``build_template`` runs on a
worker thread, the output node is added on the main thread.
"""
from __future__ import annotations

import logging
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


class KWNeuroTemplate(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Template")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Build an unbiased group-wise template from multiple 3D "
            "volumes via iterative ANTs SyN registration + sharpening. "
            "Wraps kwneuro.build_template.build_template."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroTemplateLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        volume_nodes: list[Any],
    ) -> list[Any]:
        """Materialise volume nodes into in-memory resources. **Main thread only.**

        Raises ``ValueError`` if fewer than two volumes are provided —
        a "template" of one subject is degenerate. Also raises if
        volumes have non-3D arrays (``build_template`` rejects 4D+).

        NOTE: we deliberately do NOT enforce identical shapes or
        affines here: ``build_template`` resamples each input against
        the current template on every iteration, so inputs can have
        different grids. We only surface the cases that would fail
        hard inside ANTs with opaque errors.
        """
        import numpy as np

        from kwneuro_slicer_bridge import InSceneVolumeResource

        if len(volume_nodes) < 2:
            msg = (
                "Template building requires at least 2 input volumes; "
                f"got {len(volume_nodes)}."
            )
            raise ValueError(msg)
        resources = [
            InSceneVolumeResource.from_node(n).to_in_memory() for n in volume_nodes
        ]
        for node, res in zip(volume_nodes, resources):
            ndim = np.asarray(res.get_array()).ndim
            if ndim not in (2, 3):
                msg = (
                    f"Volume {node.GetName()!r} is {ndim}D; "
                    f"build_template requires 2D or 3D inputs."
                )
                raise ValueError(msg)
        return resources

    def run_build_template(
        self,
        volume_list: list[Any],
        iterations: int,
    ) -> Any:
        """Run ``build_template`` on the in-memory volumes. **Thread-safe.**

        Returns the resulting template as an ``InMemoryVolumeResource``.
        """
        from kwneuro.build_template import build_template

        logging.info(
            "KWNeuroTemplate: running build_template (n_volumes=%d, iterations=%d)",
            len(volume_list), iterations,
        )
        return build_template(volume_list=volume_list, iterations=iterations)

    def publish_to_scene(self, template: Any, name: str) -> str:
        """Publish the template volume. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneVolumeResource

        svr = InSceneVolumeResource.from_resource(template, name=name)
        svr.get_node().CreateDefaultDisplayNodes()
        return svr.node_id

    def process(
        self,
        volume_nodes: list[Any],
        iterations: int = 3,
        name: str = "kwneuro_template",
    ) -> str:
        """Synchronous full pipeline."""
        resources = self.prepare_inputs(volume_nodes)
        template = self.run_build_template(resources, iterations)
        return self.publish_to_scene(template, name)


class KWNeuroTemplateWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        import qt
        # Role under which each QListWidgetItem stores the MRML node ID
        # of the volume it represents. Using node ID rather than name
        # keeps our list resilient to renames and uniquely identifies
        # the node. Resolve qt.Qt.UserRole at runtime rather than
        # hardcoding 32 — survives any hypothetical Qt constant change.
        self._node_id_role = qt.Qt.UserRole

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroTemplate.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroTemplateLogic()

        self.ui.addVolumeButton.connect("clicked(bool)", self.onAddVolumeClicked)
        self.ui.removeSelectedButton.connect(
            "clicked(bool)", self.onRemoveSelectedClicked,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        # The list changes when items are added / removed; recompute
        # the apply-enabled state on any change.
        self.ui.volumesListWidget.connect(
            "itemSelectionChanged()", self._updateApplyEnabled,
        )

        self._updateApplyEnabled()

    def enter(self) -> None:
        # Scene could have been edited while we were away — prune any
        # list items whose backing node vanished, and refresh the
        # displayed names in case the user renamed any.
        self._sync_listed_nodes()
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = self.ui.volumesListWidget.count >= 2

    def onAddVolumeClicked(self) -> None:
        import qt

        node = self.ui.volumeToAddSelector.currentNode()
        if node is None:
            return
        node_id = node.GetID()
        # Don't add duplicates (same node ID).
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            if item.data(self._node_id_role) == node_id:
                return
        item = qt.QListWidgetItem(node.GetName())
        item.setData(self._node_id_role, node_id)
        self.ui.volumesListWidget.addItem(item)
        self._updateApplyEnabled()

    def onRemoveSelectedClicked(self) -> None:
        # Collect rows first; removing while iterating shifts indices.
        rows = sorted(
            [idx.row() for idx in self.ui.volumesListWidget.selectedIndexes()],
            reverse=True,
        )
        for row in rows:
            self.ui.volumesListWidget.takeItem(row)
        self._updateApplyEnabled()

    def _sync_listed_nodes(self) -> None:
        """Drop items whose backing node vanished; refresh names of survivors.

        If the user renamed a volume in the Data module, the list
        previously kept showing the stale name (the node ID binding
        was still correct so processing worked, but the UI lied).
        """
        to_remove = []
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            node_id = item.data(self._node_id_role)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                to_remove.append(i)
            else:
                # Refresh in case the node was renamed while we were away.
                if item.text() != node.GetName():
                    item.setText(node.GetName())
        for row in reversed(to_remove):
            self.ui.volumesListWidget.takeItem(row)

    def _listed_nodes(self) -> list[Any]:
        """Resolve the list items back into MRML nodes, in order."""
        nodes: list[Any] = []
        missing: list[str] = []
        for i in range(self.ui.volumesListWidget.count):
            item = self.ui.volumesListWidget.item(i)
            node_id = item.data(self._node_id_role)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is None:
                missing.append(item.text())
            else:
                nodes.append(node)
        if missing:
            msg = (
                "Some listed volumes are no longer in the scene: "
                + ", ".join(missing)
                + ". Remove them from the list or re-add."
            )
            raise ValueError(msg)
        return nodes

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        iterations = int(self.ui.iterationsSpinBox.value)

        with slicer.util.tryWithErrorDisplay(
            _("Template building failed."), waitCursor=False,
        ):
            nodes = self._listed_nodes()

            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                resources = self.logic.prepare_inputs(nodes)
            finally:
                qt.QApplication.restoreOverrideCursor()

            template = run_with_progress_dialog(
                lambda: self.logic.run_build_template(resources, iterations),
                title=_("KWNeuroTemplate"),
                status=_("Building template..."),
            )

            node_id = self.logic.publish_to_scene(template, "kwneuro_template")
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = (
                    f"Created: {node.GetName()} ({len(nodes)} subjects, "
                    f"{iterations} iteration(s))"
                )


class KWNeuroTemplateTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroTemplate widget smoke test")
        module = slicer.util.getModule("KWNeuroTemplate")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
