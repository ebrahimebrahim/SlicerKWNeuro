"""KWNeuroDTI — fit a diffusion tensor to a DWI from the scene.

Apply fires ``Dwi.estimate_dti`` via :class:`kwneuro_slicer_bridge.InSceneDwi`
and publishes the returned :class:`kwneuro.dti.Dti` as a
``vtkMRMLDiffusionTensorVolumeNode``. If the "Also create FA and MD"
option is on, ``Dti.get_fa_md`` is evaluated and the two scalar maps
are added as ``vtkMRMLScalarVolumeNode``\\s with ``_fa`` / ``_md``
suffixes.

Reference for the module-structure pattern shared across all
pipeline modules: scripted module under ``slicer-extn/KWNeuro<name>/``,
Module + Logic + Widget + Test in the main ``.py``, ``.ui`` under
``Resources/UI/``, test under ``Testing/Python/``.
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


#
# KWNeuroDTI (module)
#


class KWNeuroDTI(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro DTI")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Fit a diffusion tensor to a DWI from the scene. Optionally "
            "derive FA and MD scalar maps. Wraps kwneuro's "
            "Dwi.estimate_dti and Dti.get_fa_md through the "
            "kwneuro_slicer_bridge scene resources."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


#
# Mask-node helper
#


def _warn_if_segmentation_geometry_differs(
    seg_node: Any,
    reference_dwi_node: Any,
) -> None:
    """Log a warning if the segmentation's source geometry doesn't match the DWI.

    ``slicer.util.arrayFromSegmentBinaryLabelmap`` silently resamples
    the segment into the reference geometry. If the segmentation was
    drawn on a completely different volume (e.g. a T1 in a different
    session) and the user passes an unrelated DWI as the reference,
    the returned mask is a resampled approximation that may have no
    spatial correspondence with the DWI. Detecting this at least
    surfaces it in the application log so the user doesn't chase a
    silent correctness bug.
    """
    import numpy as np

    import slicer
    import vtk

    ref_role = slicer.vtkMRMLSegmentationNode.GetReferenceImageGeometryReferenceRole()
    seg_source = seg_node.GetNodeReference(ref_role)
    if seg_source is None:
        return  # Segmentation has no source geometry to compare against.

    dwi_matrix = vtk.vtkMatrix4x4()
    reference_dwi_node.GetIJKToRASMatrix(dwi_matrix)
    src_matrix = vtk.vtkMatrix4x4()
    seg_source.GetIJKToRASMatrix(src_matrix)

    dwi_affine = np.array(
        [[dwi_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
    )
    src_affine = np.array(
        [[src_matrix.GetElement(i, j) for j in range(4)] for i in range(4)],
    )

    if not np.allclose(dwi_affine, src_affine, atol=1e-4):
        logging.warning(
            "KWNeuroDTI: segmentation %r was drawn on %r whose IJK-to-RAS "
            "differs from the DWI %r. The segment will be resampled onto "
            "the DWI grid — check that they share a coordinate frame.",
            seg_node.GetName(), seg_source.GetName(), reference_dwi_node.GetName(),
        )


def _extract_mask_resource(
    mask_node: Any | None,
    segment_id: str | None,
    reference_dwi_node: Any,
) -> Any:
    """Turn an arbitrary mask node into a kwneuro VolumeResource.

    Handles three input shapes:

    * ``None`` -> ``None`` (no mask; fit runs over the whole volume).
    * Scalar / labelmap volume -> wrap via ``InSceneVolumeResource``.
    * Segmentation -> extract the named segment into a numpy mask via
      ``slicer.util.arrayFromSegmentBinaryLabelmap`` (sampled onto
      ``reference_dwi_node``'s grid), transpose KJI -> IJK so the axis
      ordering matches kwneuro / nibabel convention, and wrap as an
      ``InMemoryVolumeResource`` with the DWI's affine.

    Must run on the main Qt thread — the segmentation path internally
    adds and removes a temporary labelmap node, and MRML scene mutations
    off-thread crash the subject-hierarchy plugin.
    """
    import numpy as np

    import slicer

    from kwneuro.resource import InMemoryVolumeResource

    from kwneuro_slicer_bridge import InSceneVolumeResource
    from kwneuro_slicer_bridge.conversions import ijk_to_ras_matrix_to_affine

    if mask_node is None:
        return None

    if mask_node.IsA("vtkMRMLSegmentationNode"):
        if not segment_id:
            msg = (
                "mask_node is a segmentation but segment_id was not provided. "
                "Pick a segment in the module UI, or pass segment_id explicitly."
            )
            raise ValueError(msg)

        _warn_if_segmentation_geometry_differs(mask_node, reference_dwi_node)

        kji = slicer.util.arrayFromSegmentBinaryLabelmap(
            mask_node, segment_id, reference_dwi_node,
        )
        if kji is None:
            msg = (
                f"Could not extract segment {segment_id!r} from "
                f"{mask_node.GetName()!r} (returned None)."
            )
            raise ValueError(msg)
        # arrayFromSegmentBinaryLabelmap returns (k, j, i); kwneuro / the
        # bridge speak IJK, so transpose.
        ijk = np.ascontiguousarray(np.asarray(kji).transpose(2, 1, 0))

        import vtk

        ijk_to_ras = vtk.vtkMatrix4x4()
        reference_dwi_node.GetIJKToRASMatrix(ijk_to_ras)
        affine = ijk_to_ras_matrix_to_affine(ijk_to_ras)

        return InMemoryVolumeResource(array=ijk, affine=affine, metadata={})

    return InSceneVolumeResource.from_node(mask_node)


#
# KWNeuroDTILogic
#


class KWNeuroDTILogic(ScriptedLoadableModuleLogic):
    """Headless API for KWNeuroDTI.

    Split into three methods so the widget can keep scene I/O on the
    main Qt thread while dispatching the heavy numpy work to a worker:

    * :meth:`prepare_inputs` (main thread) — resolve MRML nodes into
      pure-Python kwneuro resources. For a segmentation mask this
      creates + removes a temporary reference node, which must not
      happen off-thread.
    * :meth:`run_estimation` (worker thread OK) — pure numpy / dipy.
      Returns a ``Dti`` plus an optional ``(fa, md)`` tuple.
    * :meth:`publish_to_scene` (main thread) — create the output MRML
      nodes.

    :meth:`process` chains the three synchronously and exists as the
    single-call convenience for tests and headless use; the widget
    calls the three separately so it can slip
    ``run_with_progress_dialog`` around ``run_estimation`` only.
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self,
        dwi_node: Any,
        mask_node: Any | None,
        segment_id: str | None,
    ) -> tuple[Any, Any, str]:
        """Resolve MRML nodes into kwneuro resources. **Main thread only.**

        Returns ``(dwi_resource, mask_resource, dwi_name)``. Both are
        fully materialised in main-thread memory — the DWI is a plain
        ``kwneuro.Dwi`` (not an :class:`InSceneDwi`, whose live
        ``InSceneVolumeResource`` would read ``node.GetImageData()`` on
        whichever thread later calls ``get_array()``), and the mask is
        either ``None`` or an ``InMemoryVolumeResource``. This guarantees
        the downstream worker thread touches only pure-numpy data.
        """
        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()

        mask = _extract_mask_resource(mask_node, segment_id, dwi_node)
        if isinstance(mask, InSceneVolumeResource):
            mask = mask.to_in_memory()

        return dwi, mask, dwi_name

    def run_estimation(
        self,
        dwi: Any,
        mask: Any,
        create_fa_md: bool,
    ) -> tuple[Any, tuple[Any, Any] | None]:
        """Fit the tensor and optionally derive FA/MD. **Thread-safe.**

        Pure numpy + dipy; no MRML writes.
        """
        logging.info("KWNeuroDTI: fitting tensor")
        dti = dwi.estimate_dti(mask=mask)
        fa_md: tuple[Any, Any] | None = None
        if create_fa_md:
            logging.info("KWNeuroDTI: deriving FA / MD from DTI")
            fa_md = dti.get_fa_md()
        return dti, fa_md

    def publish_to_scene(
        self,
        dti: Any,
        fa_md: tuple[Any, Any] | None,
        base_name: str,
    ) -> dict[str, str | None]:
        """Create MRML nodes for the fitted DTI and optional FA/MD. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneDti, InSceneVolumeResource

        scene_dti = InSceneDti.from_dti(dti, name=f"{base_name}_dti")
        scene_dti.get_node().CreateDefaultDisplayNodes()

        result: dict[str, str | None] = {
            "dti": scene_dti.node_id,
            "fa": None,
            "md": None,
        }

        if fa_md is not None:
            fa_res, md_res = fa_md
            scene_fa = InSceneVolumeResource.from_resource(
                fa_res, name=f"{base_name}_fa",
            )
            scene_md = InSceneVolumeResource.from_resource(
                md_res, name=f"{base_name}_md",
            )
            scene_fa.get_node().CreateDefaultDisplayNodes()
            scene_md.get_node().CreateDefaultDisplayNodes()
            result["fa"] = scene_fa.node_id
            result["md"] = scene_md.node_id

        return result

    def process(
        self,
        dwi_node: Any,
        mask_node: Any | None,
        create_fa_md: bool,
        segment_id: str | None = None,
    ) -> dict[str, str | None]:
        """Synchronous full pipeline; composes the three phase methods.

        The widget doesn't call this (it splits the phases around a
        worker thread); tests and headless callers do.
        """
        dwi, mask, dwi_name = self.prepare_inputs(dwi_node, mask_node, segment_id)
        dti, fa_md = self.run_estimation(dwi, mask, create_fa_md)
        return self.publish_to_scene(dti, fa_md, dwi_name)


#
# KWNeuroDTIWidget
#


class KWNeuroDTIWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroDTI.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        # mrmlSceneChanged -> setMRMLScene on each child combo box is
        # wired in the .ui file, so this call propagates the scene to
        # every qMRMLNodeComboBox and they auto-observe node adds /
        # removes from here on.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroDTILogic()

        self.ui.inputDwiSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled,
        )
        self.ui.inputMaskSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)", self._onMaskChanged,
        )
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)

        self._onMaskChanged(self.ui.inputMaskSelector.currentNode())
        self._updateApplyEnabled()

    def enter(self) -> None:
        # Safety net for the case where the user toggled back to this
        # module after scene changes.
        self._updateApplyEnabled()
        self._onMaskChanged(self.ui.inputMaskSelector.currentNode())

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputDwiSelector.currentNode() is not None
        )

    def _onMaskChanged(self, node: Any) -> None:
        is_segmentation = node is not None and node.IsA("vtkMRMLSegmentationNode")
        self.ui.segmentIdLabel.visible = is_segmentation
        self.ui.segmentIdSelector.visible = is_segmentation
        self.ui.segmentIdSelector.clear()
        if not is_segmentation:
            return
        segmentation = node.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):
            seg_id = segmentation.GetNthSegmentID(i)
            segment = segmentation.GetSegment(seg_id)
            self.ui.segmentIdSelector.addItem(segment.GetName(), seg_id)

    def onApplyClicked(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()
        mask_node = self.ui.inputMaskSelector.currentNode()
        create_fa_md = self.ui.createFaMdCheckBox.checked

        segment_id: str | None = None
        if mask_node is not None and mask_node.IsA("vtkMRMLSegmentationNode"):
            idx = self.ui.segmentIdSelector.currentIndex
            if idx < 0:
                slicer.util.errorDisplay(_("Pick a segment to use as the mask."))
                return
            # PythonQt occasionally returns itemData as a QVariant
            # wrapper on some builds; force to str so the downstream
            # arrayFromSegmentBinaryLabelmap call gets a plain string.
            segment_id = str(self.ui.segmentIdSelector.itemData(idx))

        with slicer.util.tryWithErrorDisplay(_("Failed to fit DTI."), waitCursor=False):
            # Main thread: resolve inputs (includes any temp-node
            # scene mutations for segmentation masks).
            dwi, mask, dwi_name = self.logic.prepare_inputs(
                dwi_node, mask_node, segment_id,
            )

            # Worker thread: the numpy / dipy fit. Wrapping scene
            # mutations here would crash the subject hierarchy plugin,
            # so we keep it strictly to pure computation.
            dti, fa_md = run_with_progress_dialog(
                lambda: self.logic.run_estimation(dwi, mask, create_fa_md),
                title=_("KWNeuroDTI"),
                status=_("Fitting tensor..."),
            )

            # Main thread: publish results to the scene.
            ids = self.logic.publish_to_scene(dti, fa_md, dwi_name)
            self._updateResultLabel(ids)

    def _updateResultLabel(self, ids: dict[str, str | None]) -> None:
        names: list[str] = []
        for key in ("dti", "fa", "md"):
            node_id = ids.get(key)
            if node_id is None:
                continue
            node = slicer.mrmlScene.GetNodeByID(node_id)
            names.append(node.GetName() if node is not None else node_id)
        self.ui.resultLabel.text = "Created: " + ", ".join(names)


#
# KWNeuroDTITest
#


class KWNeuroDTITest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroDTI widget smoke test")
        # The real functional tests live in Testing/Python; this just
        # exercises the module's widget construction path so the
        # generic ctest from `slicer_add_python_unittest(SCRIPT
        # KWNeuroDTI.py)` has something to run.
        module = slicer.util.getModule("KWNeuroDTI")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
