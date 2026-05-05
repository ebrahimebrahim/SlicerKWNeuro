"""KWNeuroImporter - load a DWI into the scene from disk, or fetch dipy sample data.

Why this module exists: Slicer's built-in Add Data dialog loads 4D
NIfTI volumes as ``vtkMRMLScalarVolumeNode`` and silently drops the
4th dimension. For DWI data, gradients + b-values must stay attached
to the volume node. The bridge already provides
:meth:`InSceneDwi.from_nifti_path` that does the right thing via
``vtkMRMLDiffusionWeightedVolumeNode`` — this module exposes that
one-liner as a GUI.

Structure:

- "Import DWI from disk" section: three file pickers (volume / bval /
  bvec) + node-name field + Load button. Read-from-disk happens on a
  worker thread behind a progress dialog; scene-node creation happens
  on the main thread.
- "Sample data" section: two one-click fetch buttons:

    * Sherbrooke 3-shell HARDI (~30 MB) — small, fast download. Good
      for DTI / NODDI demos but TractSeg gives sparse output on it.
    * CENIR multi-b ("HCP-like", ~500 MB) — three shells at b=1000 /
      2000 / 3000 with hundreds of directions. Richer, what TractSeg
      was designed for.

  Both also register with Slicer's standard *Sample Data* module so
  they appear there alongside MRHead etc.

Logic uses the three-phase split so MRML scene writes stay on the
main Qt thread.
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


#
# KWNeuroImporter (module)
#


class KWNeuroImporter(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Importer")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Load a DWI from disk into the Slicer scene via the kwneuro "
            "bridge (preserving the 4th dimension and attaching "
            "gradients + b-values). Also exposes one-click fetch of two "
            "dipy DWI sample datasets (Sherbrooke 3-shell and CENIR "
            "multi-b)."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )

        # Make the DWI sample datasets discoverable in Slicer's standard
        # Sample Data module. Registration is idempotent and cheap so
        # doing it on module init is fine.
        slicer.app.connect(
            "startupCompleted()",
            _register_kwneuro_sample_data,
        )


def _register_kwneuro_sample_data() -> None:
    """Add KWNeuro DWI sample sets to Slicer's SampleData module.

    Uses ``customDownloader`` so we keep dipy's well-tested fetch +
    cache logic for the actual data, but get the SampleData module's
    GUI surface (a button per dataset, listed under a "KWNeuro DWI"
    category) for free.

    SampleData's built-in deduplication (``isSampleDataSourceRegistered``)
    uses object identity — fresh ``SampleDataSource`` instances always
    look new — so we guard by ``sampleName`` ourselves to keep this
    function idempotent across module reloads.
    """
    try:
        import SampleData
    except ImportError:
        logging.warning("KWNeuroImporter: SampleData module unavailable; "
                        "skipping sample-data registration")
        return

    category = "KWNeuro DWI"
    existing_names = {
        src.sampleName
        for src in SampleData.SampleDataLogic.sampleDataSourcesByCategory(category)
    }

    def _register(name: str, downloader: Any) -> None:
        if name in existing_names:
            return
        SampleData.SampleDataLogic.registerCustomSampleDataSource(
            category=category,
            sampleName=name,
            uris=[],
            fileNames=[],
            nodeNames=[],
            customDownloader=downloader,
        )
        existing_names.add(name)

    def _download_sherbrooke(_source: Any) -> None:
        # Resolve the logic class lazily — at module-import time
        # KWNeuroImporterLogic is not yet defined.
        KWNeuroImporterLogic().load_sherbrooke()

    def _download_cenir(_source: Any) -> None:
        KWNeuroImporterLogic().load_cenir()

    _register("Sherbrooke 3-shell HARDI", _download_sherbrooke)
    _register("CENIR multi-b (HCP-like, ~1.7 GB)", _download_cenir)


#
# KWNeuroImporterLogic
#


class KWNeuroImporterLogic(ScriptedLoadableModuleLogic):
    """Load DWIs into the scene via the kwneuro bridge.

    Three-phase split so MRML scene writes stay on the main thread:

    * :meth:`load_dwi_from_disk` (thread-safe) — read NIfTI + FSL
      bval/bvec into a plain ``kwneuro.Dwi``. Disk I/O only.
    * :meth:`fetch_sherbrooke_paths` (thread-safe) — calls
      ``dipy.data.fetch_sherbrooke_3shell()`` and returns the paths.
    * :meth:`publish_to_scene` (main thread) — push a ``kwneuro.Dwi``
      into the scene as a ``vtkMRMLDiffusionWeightedVolumeNode``.
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    @staticmethod
    def load_dwi_from_disk(
        volume_path: Path,
        bval_path: Path,
        bvec_path: Path,
    ) -> Any:
        """Read NIfTI + FSL bval/bvec from disk. **Thread-safe.**

        Returns a fully-loaded ``kwneuro.Dwi`` (in-memory). Does not
        touch the MRML scene.
        """
        from kwneuro.dwi import Dwi
        from kwneuro.io import FslBvalResource, FslBvecResource, NiftiVolumeResource

        for label, path in (
            ("volume", volume_path),
            ("bval", bval_path),
            ("bvec", bvec_path),
        ):
            if not Path(path).exists():
                msg = f"{label} file not found at {path!r}"
                raise FileNotFoundError(msg)

        logging.info(
            "KWNeuroImporter: loading DWI (volume=%s, bval=%s, bvec=%s)",
            volume_path, bval_path, bvec_path,
        )
        return Dwi(
            NiftiVolumeResource(Path(volume_path)),
            FslBvalResource(Path(bval_path)),
            FslBvecResource(Path(bvec_path)),
        ).load()

    @staticmethod
    def fetch_sherbrooke_paths() -> tuple[Path, Path, Path]:
        """Trigger dipy's Sherbrooke 3-shell fetch. **Thread-safe.**

        Returns ``(volume_path, bval_path, bvec_path)``.
        ``dipy.data.fetch_sherbrooke_3shell`` caches the dataset under
        ``~/.dipy/sherbrooke_3shell/`` and is a no-op after the first
        successful download.
        """
        import dipy.data

        logging.info(
            "KWNeuroImporter: invoking dipy.data.fetch_sherbrooke_3shell "
            "(will download to ~/.dipy/sherbrooke_3shell/ on first use)",
        )
        dipy.data.fetch_sherbrooke_3shell()

        data_dir = Path.home() / ".dipy" / "sherbrooke_3shell"
        volume = data_dir / "HARDI193.nii.gz"
        bval = data_dir / "HARDI193.bval"
        bvec = data_dir / "HARDI193.bvec"
        for label, path in (("volume", volume), ("bval", bval), ("bvec", bvec)):
            if not path.exists():
                msg = (
                    f"Sherbrooke {label} file not found at {path!r} after "
                    f"fetch_sherbrooke_3shell(); cache may be corrupted."
                )
                raise RuntimeError(msg)
        return volume, bval, bvec

    @staticmethod
    def fetch_cenir_dwi() -> Any:
        """Fetch CENIR multi-b and return a kwneuro.Dwi for the b=1000/2000/3000 shells.

        **Thread-safe.**

        First-time use: ``dipy.data.fetch_cenir_multib(with_raw=False)``
        downloads the per-shell eddy-corrected files (~1.7 GB total
        across three shells) to ``~/.dipy/cenir_multib/``.

        Each call (cached or not): reads the three shells via dipy's
        ``read_cenir_multib`` and assembles a ``kwneuro.Dwi`` in
        memory. We don't cache the concatenated output as an extra
        on-disk file — duplicating the ~1.7 GB of float32 DWI data
        isn't worth saving the ~30 seconds of disk-read + concat
        work, especially for a sample-data button.

        The HCP-like triple-shell layout matches what TractSeg was
        trained on, so downstream CSD / TractSeg gives noticeably
        denser bundles than Sherbrooke can.
        """
        import dipy.data
        import numpy as np

        from kwneuro.dwi import Dwi
        from kwneuro.resource import (
            InMemoryBvalResource,
            InMemoryBvecResource,
            InMemoryVolumeResource,
        )

        logging.info(
            "KWNeuroImporter: invoking dipy.data.fetch_cenir_multib "
            "(will download ~1.7 GB to ~/.dipy/cenir_multib/ on first use)",
        )
        dipy.data.fetch_cenir_multib(with_raw=False)

        logging.info(
            "KWNeuroImporter: assembling CENIR b=1000/2000/3000 in memory",
        )
        img, gtab = dipy.data.read_cenir_multib(bvals=[1000, 2000, 3000])

        # Renormalize bvecs to exactly unit (modulo float-64
        # precision). dipy's ``gtab.bvecs`` are near-unit but off by
        # ~1e-6 here and there, and ``vtkMRMLDiffusionWeightedVolumeNode``'s
        # ``SetDiffusionGradients`` strictly rejects gradient vectors
        # whose length is anything other than 0 or 1.
        bvecs = np.asarray(gtab.bvecs, dtype=np.float64)
        norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # leave the b=0 (0,0,0) rows alone
        bvecs = bvecs / norms

        # Cast volume to float32 (CENIR's per-shell files are already
        # float32; ``get_fdata`` upcasts to float64). Keeps the
        # in-memory size and downstream processing predictable.
        return Dwi(
            volume=InMemoryVolumeResource(
                array=np.asarray(img.get_fdata(), dtype=np.float32),
                affine=np.asarray(img.affine),
                metadata={},
            ),
            bval=InMemoryBvalResource(np.asarray(gtab.bvals, dtype=np.float64)),
            bvec=InMemoryBvecResource(bvecs),
        )

    def publish_to_scene(self, dwi: Any, name: str) -> str:
        """Push a kwneuro.Dwi into the scene. **Main thread only.**

        Returns the MRML ID of the new DWI node. If display-node
        creation or slice-viewer setup raises after the node is added,
        remove the partial node before re-raising — leaving dangling
        state in the scene is worse than a clean failure.
        """
        from kwneuro_slicer_bridge import InSceneDwi

        scene_dwi = InSceneDwi.from_dwi(dwi, name=name)
        try:
            # The node is a vtkMRMLDiffusionWeightedVolumeNode; the
            # default display node handles per-gradient component
            # selection so showing it as the slice background renders
            # one gradient at a time rather than failing.
            scene_dwi.get_node().CreateDefaultDisplayNodes()
            slicer.util.setSliceViewerLayers(background=scene_dwi.get_node())
        except BaseException:
            slicer.mrmlScene.RemoveNode(scene_dwi.get_node())
            raise
        return scene_dwi.node_id

    def load_from_paths(
        self,
        volume_path: Path,
        bval_path: Path,
        bvec_path: Path,
        name: str,
    ) -> str:
        """Synchronous full pipeline; composes the two phases.

        Tests / headless callers use this; the widget calls phases
        separately so it can wrap the disk-read in a progress dialog.
        """
        dwi = self.load_dwi_from_disk(volume_path, bval_path, bvec_path)
        return self.publish_to_scene(dwi, name)

    def load_sherbrooke(self, name: str = "HARDI193") -> str:
        """Synchronous Sherbrooke fetch + load; composes the three phases."""
        volume, bval, bvec = self.fetch_sherbrooke_paths()
        dwi = self.load_dwi_from_disk(volume, bval, bvec)
        return self.publish_to_scene(dwi, name)

    def load_cenir(self, name: str = "CENIR_HCP_like") -> str:
        """Synchronous CENIR multi-b fetch + load."""
        dwi = self.fetch_cenir_dwi()
        return self.publish_to_scene(dwi, name)


#
# KWNeuroImporterWidget
#


class KWNeuroImporterWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroImporter.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroImporterLogic()

        for line_edit in (
            self.ui.volumePathLineEdit,
            self.ui.bvalPathLineEdit,
            self.ui.bvecPathLineEdit,
        ):
            line_edit.connect(
                "currentPathChanged(QString)", self._updateLoadEnabled,
            )
        self.ui.nameLineEdit.connect("textChanged(QString)", self._updateLoadEnabled)
        self.ui.loadButton.connect("clicked(bool)", self.onLoadClicked)
        self.ui.loadSherbrookeButton.connect(
            "clicked(bool)", self.onLoadSherbrookeClicked,
        )
        self.ui.loadCenirButton.connect(
            "clicked(bool)", self.onLoadCenirClicked,
        )

        self._updateLoadEnabled()

    def enter(self) -> None:
        self._updateLoadEnabled()

    def _updateLoadEnabled(self, *_args: Any) -> None:
        paths_set = all(
            bool(line_edit.currentPath) for line_edit in (
                self.ui.volumePathLineEdit,
                self.ui.bvalPathLineEdit,
                self.ui.bvecPathLineEdit,
            )
        )
        name_set = bool(self.ui.nameLineEdit.text.strip())
        self.ui.loadButton.enabled = paths_set and name_set

    def onLoadClicked(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        volume = Path(self.ui.volumePathLineEdit.currentPath)
        bval = Path(self.ui.bvalPathLineEdit.currentPath)
        bvec = Path(self.ui.bvecPathLineEdit.currentPath)
        name = self.ui.nameLineEdit.text.strip()

        with slicer.util.tryWithErrorDisplay(_("Failed to load DWI."), waitCursor=False):
            # Worker: disk I/O. Main thread: scene add.
            dwi = run_with_progress_dialog(
                lambda: self.logic.load_dwi_from_disk(volume, bval, bvec),
                title=_("KWNeuroImporter"),
                status=_("Reading DWI from disk..."),
            )
            node_id = self.logic.publish_to_scene(dwi, name)
            self._updateResultLabel(node_id)

    def onLoadSherbrookeClicked(self) -> None:
        def _worker() -> Any:
            paths = self.logic.fetch_sherbrooke_paths()
            return self.logic.load_dwi_from_disk(*paths)
        self._run_sample_load(
            worker=_worker,
            node_name="HARDI193",
            status=_("Fetching Sherbrooke 3-shell..."),
            error_msg=_("Failed to fetch / load Sherbrooke sample data."),
        )

    def onLoadCenirClicked(self) -> None:
        self._run_sample_load(
            worker=self.logic.fetch_cenir_dwi,
            node_name="CENIR_HCP_like",
            status=_("Fetching + concatenating CENIR multi-b..."),
            error_msg=_("Failed to fetch / load CENIR sample data."),
        )

    def _run_sample_load(
        self,
        worker: Any,
        node_name: str,
        status: str,
        error_msg: str,
    ) -> None:
        """Shared fetch + load + publish flow for sample-data buttons.

        ``worker`` is a thread-safe callable returning a kwneuro.Dwi.
        capture_tqdm=True so dipy's per-chunk download progress flows
        into the dialog's Details log — otherwise the user stares at
        an indeterminate bar for the multi-MB download.
        """
        from kwneuro_slicer_bridge import run_with_progress_dialog

        with slicer.util.tryWithErrorDisplay(error_msg, waitCursor=False):
            dwi = run_with_progress_dialog(
                worker,
                title=_("KWNeuroImporter"),
                status=status,
                capture_tqdm=True,
            )
            node_id = self.logic.publish_to_scene(dwi, node_name)
            self._updateResultLabel(node_id)

    def _updateResultLabel(self, node_id: str) -> None:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node is not None:
            self.ui.resultLabel.text = f"Loaded: {node.GetName()} (ID {node_id})"
        else:
            self.ui.resultLabel.text = f"Loaded (ID {node_id})"


#
# KWNeuroImporterTest
#


class KWNeuroImporterTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_WidgetLoads()

    def test_WidgetLoads(self):
        self.delayDisplay("KWNeuroImporter widget smoke test")
        module = slicer.util.getModule("KWNeuroImporter")
        widget = module.widgetRepresentation()
        assert widget is not None
        self.delayDisplay("Test passed")
