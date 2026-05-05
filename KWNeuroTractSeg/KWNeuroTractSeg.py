"""KWNeuroTractSeg - run TractSeg on a DWI + brain mask.

Wraps ``kwneuro.tractseg.extract_tractseg``. Output depends on
``output_type``:

* ``tract_segmentation`` (default): 72 binary bundle masks published
  as a single ``vtkMRMLSegmentationNode`` with one named segment per
  bundle (e.g. ``AF_left``, ``CST_right``, ...).
* ``endings_segmentation``: 144 binary masks (each of the 72 bundles
  has a ``_b`` begin and ``_e`` end region), published as a single
  ``vtkMRMLSegmentationNode``.
* ``TOM``: 60-component vector volume (20 tracts × xyz orientation)
  published as a single ``vtkMRMLVectorVolumeNode``. Vector data
  doesn't fit a segmentation; this stays as a multi-component volume.

Requires kwneuro[tractseg]; also strongly benefits from a CUDA GPU.
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


def _bundle_names_72() -> tuple[str, ...]:
    """The 72 TractSeg "All" bundle names in component order.

    Defers the ``tractseg`` import so opening this module doesn't
    require the heavy ``kwneuro[tractseg]`` extra. Called at
    publish-to-scene time, when the extra is guaranteed present
    (``prepare_inputs`` has already gated on it).
    """
    from tractseg.data.dataset_specific_utils import get_bundle_names
    # get_bundle_names returns ``("background", *72 bundles)``; drop the
    # background entry so indexes line up with TractSeg's per-channel
    # output (channel i corresponds to bundle i).
    return tuple(get_bundle_names("All")[1:])


def _endpoint_names_144() -> tuple[str, ...]:
    """The 144 TractSeg "All_endpoints" names in component order."""
    from tractseg.data.dataset_specific_utils import get_bundle_names
    return tuple(get_bundle_names("All_endpoints")[1:])


class KWNeuroTractSeg(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro TractSeg")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Segment white-matter tracts using TractSeg (kwneuro[tractseg]). "
            "Internally computes CSD peaks and feeds them to TractSeg's CNN. "
            "A CUDA GPU is strongly recommended.\n\n"
            "Denoising the input DWI first (KWNeuro Denoise -> Patch2Self) "
            "is a critical preprocessing step. TractSeg's CSD peaks are "
            "very sensitive to noise: on a noisy DWI most of the 72 "
            "bundles come back empty or fragmented, while on the same DWI "
            "after Patch2Self denoising you typically get all 72 bundles "
            "with anatomically plausible shapes. Run KWNeuro Denoise "
            "first and feed the denoised DWI here."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroTractSegLogic(ScriptedLoadableModuleLogic):
    SUPPORTED_OUTPUT_TYPES = (
        "tract_segmentation",
        "endings_segmentation",
        "TOM",
    )

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def prepare_inputs(
        self, dwi_node: Any, mask_node: Any,
    ) -> tuple[Any, Any, str]:
        """Materialise inputs + check tractseg extra. **Main thread only.**"""
        from kwneuro_slicer_bridge import (
            InSceneDwi, InSceneVolumeResource, ensure_extras_installed,
        )

        if dwi_node is None:
            msg = "Input DWI node is required."
            raise ValueError(msg)
        if mask_node is None:
            msg = "Brain mask is required for TractSeg."
            raise ValueError(msg)

        ensure_extras_installed(["tractseg"])

        dwi_name = dwi_node.GetName() or "kwneuro_dwi"
        dwi = InSceneDwi.from_node(dwi_node).to_in_memory()
        mask = InSceneVolumeResource.from_node(mask_node).to_in_memory()
        return dwi, mask, dwi_name

    def run_tractseg(self, dwi: Any, mask: Any, output_type: str) -> Any:
        """Run TractSeg. **Thread-safe** (no MRML touches)."""
        from kwneuro.tractseg import extract_tractseg

        if output_type not in self.SUPPORTED_OUTPUT_TYPES:
            msg = (
                f"Unsupported output_type {output_type!r}; "
                f"must be one of {self.SUPPORTED_OUTPUT_TYPES}."
            )
            raise ValueError(msg)

        logging.info("KWNeuroTractSeg: running (output_type=%s)", output_type)
        return extract_tractseg(dwi=dwi, mask=mask, output_type=output_type)

    def publish_to_scene(
        self, tract_volume: Any, base_name: str, output_type: str,
    ) -> str:
        """Publish the TractSeg result to the scene. **Main thread only.**

        For binary-mask outputs (``tract_segmentation`` and
        ``endings_segmentation``) returns the ID of a
        ``vtkMRMLSegmentationNode`` whose segments are named after the
        TractSeg bundle list. For the ``TOM`` vector-field output
        returns the ID of a ``vtkMRMLVectorVolumeNode`` (vector data
        doesn't map cleanly to a segmentation).
        """
        if output_type == "tract_segmentation":
            return self._publish_segmentation(
                tract_volume,
                node_name=f"{base_name}_tractseg",
                segment_names=_bundle_names_72(),
            )
        if output_type == "endings_segmentation":
            return self._publish_segmentation(
                tract_volume,
                node_name=f"{base_name}_tractseg_endings",
                segment_names=_endpoint_names_144(),
            )
        if output_type == "TOM":
            return self._publish_vector_volume(
                tract_volume, node_name=f"{base_name}_tractseg_tom",
            )
        msg = f"Unsupported output_type {output_type!r} in publish_to_scene"
        raise ValueError(msg)

    def _publish_vector_volume(self, tract_volume: Any, node_name: str) -> str:
        from kwneuro_slicer_bridge import InSceneVolumeResource

        svr = InSceneVolumeResource.from_resource(tract_volume, name=node_name)
        svr.get_node().CreateDefaultDisplayNodes()
        return svr.node_id

    @staticmethod
    def _publish_segmentation(
        tract_volume: Any,
        *,
        node_name: str,
        segment_names: tuple[str, ...],
    ) -> str:
        """Build a vtkMRMLSegmentationNode with one segment per channel.

        ``tract_volume`` is a kwneuro VolumeResource holding a 4D
        ``(x, y, z, n)`` array of binary masks. We add ``n`` named
        segments to a single segmentation node, sharing the input's
        IJK-to-RAS geometry. The names map positionally onto channels
        so the caller is responsible for passing the right list for
        the output_type at hand.
        """
        import vtkSegmentationCorePython as vtkSegmentationCore

        from kwneuro_slicer_bridge.conversions import (
            affine_to_ijk_to_ras_matrix,
            numpy_to_vtk_image,
        )

        arr = tract_volume.get_array()
        if arr.ndim != 4:
            msg = (
                f"Expected 4D array for segmentation publishing, "
                f"got shape {arr.shape}"
            )
            raise ValueError(msg)
        n_channels = arr.shape[-1]
        if n_channels != len(segment_names):
            msg = (
                f"Channel count mismatch: array has {n_channels} "
                f"channels but {len(segment_names)} segment names "
                f"were provided."
            )
            raise ValueError(msg)
        affine = tract_volume.get_affine()
        ijk_to_ras = affine_to_ijk_to_ras_matrix(affine)
        labelmap_repr_name = (
            vtkSegmentationCore.vtkSegmentationConverter.
            GetSegmentationBinaryLabelmapRepresentationName()
        )

        seg_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSegmentationNode", node_name,
        )
        try:
            seg_node.CreateDefaultDisplayNodes()
            # The segmentation needs a labelmap representation declared
            # before we add segments, otherwise AddSegment silently
            # drops the labelmap data we attach.
            seg_node.GetSegmentation().SetMasterRepresentationName(labelmap_repr_name)

            for i, name in enumerate(segment_names):
                binary = (arr[..., i] > 0).astype("uint8")
                # Build a vtkOrientedImageData carrying the binary
                # mask AND the IJK-to-RAS geometry. Adding via
                # vtkSegment.AddRepresentation guarantees one
                # segment per channel, even when the channel is
                # all-zero (which Slicer's labelmap-import path
                # would otherwise drop).
                oriented = vtkSegmentationCore.vtkOrientedImageData()
                oriented.DeepCopy(numpy_to_vtk_image(binary))
                oriented.SetGeometryFromImageToWorldMatrix(ijk_to_ras)

                segment = vtkSegmentationCore.vtkSegment()
                segment.SetName(name)
                segment.AddRepresentation(labelmap_repr_name, oriented)
                seg_node.GetSegmentation().AddSegment(segment, name)
        except BaseException:
            slicer.mrmlScene.RemoveNode(seg_node)
            raise
        return seg_node.GetID()

    def process(
        self,
        dwi_node: Any,
        mask_node: Any,
        output_type: str = "tract_segmentation",
    ) -> str:
        """Synchronous full pipeline."""
        dwi, mask, name = self.prepare_inputs(dwi_node, mask_node)
        tract_volume = self.run_tractseg(dwi, mask, output_type)
        return self.publish_to_scene(tract_volume, name, output_type)


class KWNeuroTractSegWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroTractSeg.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroTractSegLogic()

        for sel in (self.ui.inputDwiSelector, self.ui.maskSelector):
            sel.connect("currentNodeChanged(vtkMRMLNode*)", self._updateApplyEnabled)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyClicked)
        self._updateApplyEnabled()

    def enter(self) -> None:
        self._updateApplyEnabled()

    def _updateApplyEnabled(self, *_args: Any) -> None:
        self.ui.applyButton.enabled = (
            self.ui.inputDwiSelector.currentNode() is not None
            and self.ui.maskSelector.currentNode() is not None
        )

    def _selectedOutputType(self) -> str:
        if self.ui.endingsSegmentationRadio.checked:
            return "endings_segmentation"
        if self.ui.tomRadio.checked:
            return "TOM"
        return "tract_segmentation"

    @staticmethod
    def _cuda_available() -> bool:
        """True iff TractSeg would see a usable CUDA device.

        Isolated as a static method so tests can monkey-patch it to
        exercise both GPU / no-GPU branches without needing a real GPU.
        """
        try:
            import torch  # torch arrives with kwneuro[tractseg]
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def onApplyClicked(self) -> None:
        import qt

        from kwneuro_slicer_bridge import run_with_progress_dialog

        dwi_node = self.ui.inputDwiSelector.currentNode()
        mask_node = self.ui.maskSelector.currentNode()
        output_type = self._selectedOutputType()

        with slicer.util.tryWithErrorDisplay(_("TractSeg failed."), waitCursor=False):
            qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
            try:
                dwi, mask, name = self.logic.prepare_inputs(dwi_node, mask_node)
            finally:
                qt.QApplication.restoreOverrideCursor()

            # GPU pre-flight. Without CUDA, TractSeg falls back to CPU
            # inference and takes 30+ min on realistic data. Warn
            # before committing the user to the progress dialog.
            if not self._cuda_available():
                if not slicer.util.confirmYesNoDisplay(
                    _(
                        "No CUDA GPU detected. TractSeg will run on CPU, "
                        "which can take 30+ minutes on realistic data. "
                        "Proceed anyway?",
                    ),
                    windowTitle=_("KWNeuroTractSeg - no GPU"),
                ):
                    return

            tract_volume = run_with_progress_dialog(
                lambda: self.logic.run_tractseg(dwi, mask, output_type),
                title=_("KWNeuroTractSeg"),
                status=_("Running TractSeg..."),
                # nnunetv2 / TractSeg use plain print() and tqdm for
                # status — capture both so the dialog's Details log
                # mirrors what would otherwise only show in the
                # Python console.
                capture_tqdm=True,
                capture_stdout=True,
            )

            node_id = self.logic.publish_to_scene(tract_volume, name, output_type)
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                self.ui.resultLabel.text = f"Created: {node.GetName()}"


class KWNeuroTractSegTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroTractSeg widget smoke test")
        module = slicer.util.getModule("KWNeuroTractSeg")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
