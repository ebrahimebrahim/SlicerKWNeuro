"""KWNeuroImporter - load DWI and structural images into the scene.

Why this module exists: Slicer's built-in Add Data dialog loads 4D
NIfTI volumes as ``vtkMRMLScalarVolumeNode`` and silently drops the
4th dimension. For DWI data, gradients + b-values must stay attached
to the volume node. The bridge already provides
:meth:`InSceneDwi.from_nifti_path` that does the right thing via
``vtkMRMLDiffusionWeightedVolumeNode`` — this module exposes that
one-liner as a GUI. Structural NIfTI import is intentionally simpler:
it loads one 3D NIfTI and publishes a scalar volume node that also
round-trips as ``kwneuro.structural.StructuralImage``.

Structure:

- "Import DWI from disk" section: three file pickers (volume / bval /
  bvec) + node-name field + Load button. Read-from-disk happens on a
  worker thread behind a progress dialog; scene-node creation happens
  on the main thread.
- "Sample data" section: two one-click fetch buttons:

    * Sherbrooke 3-shell HARDI (~30 MB) — small, fast download. Good
      for DTI / NODDI demos but TractSeg gives sparse output on it.
    * CENIR multi-shell HARDI (~1.7 GB) — three shells at b=1000 /
      2000 / 3000 with hundreds of directions. Richer than Sherbrooke
      but TractSeg's training distribution is HCP-acquired data;
      results on CENIR are denser but still don't fully match what
      TractSeg can produce on real HCP data.
    * OpenNeuro ds000221 sub-010002 ses-01 multimodal T1 + DWI —
      cached under ``~/.kwneuro_sample_data/`` and useful for testing
      the structural + diffusion registration workflow.

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
            "Load DWI and structural NIfTI images into the Slicer scene "
            "via the kwneuro bridge. DWI import preserves the 4th "
            "dimension and attaches gradients + b-values. Sample-data "
            "buttons cover diffusion-only and multimodal structural + "
            "diffusion workflows."
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

    category = "KWNeuro Sample Data"
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

    def _download_edden(_source: Any) -> None:
        KWNeuroImporterLogic().load_edden()

    def _download_ds000221(_source: Any) -> None:
        KWNeuroImporterLogic().load_ds000221_multimodal()

    _register("Sherbrooke 3-shell HARDI", _download_sherbrooke)
    _register("CENIR multi-shell HARDI (b=1000/2000/3000, ~1.7 GB)", _download_cenir)
    _register(
        "EDDEN HCP-protocol DWI (OpenNeuro ds004666 sub-01, ~1.8 GB)",
        _download_edden,
    )
    _register(
        "LEMON multimodal T1 + DWI (OpenNeuro ds000221 sub-010002 ses-01)",
        _download_ds000221,
    )


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
    def load_structural_from_disk(volume_path: Path) -> Any:
        """Read a structural NIfTI from disk. **Thread-safe.**"""
        from kwneuro.io import NiftiVolumeResource
        from kwneuro.structural import StructuralImage

        if not Path(volume_path).exists():
            msg = f"structural volume file not found at {volume_path!r}"
            raise FileNotFoundError(msg)

        logging.info("KWNeuroImporter: loading structural NIfTI %s", volume_path)
        return StructuralImage(
            volume=NiftiVolumeResource(Path(volume_path)),
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
    def fetch_edden_paths(progress_callback: Any = None) -> tuple[Path, Path, Path]:
        """Fetch EDDEN sub-01 ses-1p5mm AVG_complex DWI from OpenNeuro.

        **Thread-safe.** Returns ``(volume_path, bval_path, bvec_path)``.

        EDDEN (OpenNeuro ds004666) is a single-subject CC0 dataset
        whose ses-1p5mm scan uses an HCP-style q-space schema (270
        diffusion directions: 90 each at b=1000, 2000, 3000; 27 b=0;
        1.5 mm isotropic) — the closest publicly redistributable
        match to what TractSeg was trained on. We pull the
        complex-averaged + preprocessed derivative for clean SNR.

        Files are cached at
        ``~/.kwneuro_sample_data/edden_ds004666_ses-1p5mm/``. Total
        download is ~1.8 GB (NIfTI dominates; bval/bvec are tiny).

        ``progress_callback`` is called with a one-line status string
        once per ~5% of the download.
        """
        cache_dir = (
            Path.home() / ".kwneuro_sample_data" / "edden_ds004666_ses-1p5mm"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        s3_root = "https://s3.amazonaws.com/openneuro.org/ds004666"
        files = {
            "volume": (
                f"{s3_root}/derivatives/ses-1p5mm/dwi/AVG_complex/"
                "sub-01_ses-1p5mm_dir-AP_AVG_complex_dwi_processed.nii.gz",
                cache_dir / "sub-01_ses-1p5mm_AVG_complex_dwi.nii.gz",
            ),
            "bval": (
                f"{s3_root}/sub-01/ses-1p5mm/dwi/sub-01_ses-1p5mm_dir-AP_dwi.bval",
                cache_dir / "sub-01_ses-1p5mm_dwi.bval",
            ),
            "bvec": (
                f"{s3_root}/sub-01/ses-1p5mm/dwi/sub-01_ses-1p5mm_dir-AP_dwi.bvec",
                cache_dir / "sub-01_ses-1p5mm_dwi.bvec",
            ),
        }

        for label, (url, dest) in files.items():
            if dest.exists() and dest.stat().st_size > 0:
                continue
            logging.info("KWNeuroImporter: downloading EDDEN %s from %s", label, url)
            if progress_callback is not None:
                progress_callback(f"Downloading EDDEN {label} ({url})...")
            KWNeuroImporterLogic._download_with_progress(
                url, dest, label, progress_callback,
            )

        return files["volume"][1], files["bval"][1], files["bvec"][1]

    @staticmethod
    def fetch_ds000221_multimodal_paths(
        progress_callback: Any = None,
    ) -> tuple[Path, Path, Path, Path]:
        """Fetch OpenNeuro ds000221 sub-010002 ses-01 T1w + DWI files.

        **Thread-safe.** Returns ``(t1_path, dwi_path, bval_path, bvec_path)``.
        Files are cached under
        ``~/.kwneuro_sample_data/ds000221_sub-010002_ses-01/``.
        """
        cache_dir = (
            Path.home()
            / ".kwneuro_sample_data"
            / "ds000221_sub-010002_ses-01"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        s3_root = "https://s3.amazonaws.com/openneuro.org/ds000221"
        files = {
            "t1": (
                f"{s3_root}/sub-010002/ses-01/anat/"
                "sub-010002_ses-01_acq-mp2rage_T1w.nii.gz",
                cache_dir / "sub-010002_ses-01_acq-mp2rage_T1w.nii.gz",
            ),
            "dwi": (
                f"{s3_root}/sub-010002/ses-01/dwi/"
                "sub-010002_ses-01_dwi.nii.gz",
                cache_dir / "sub-010002_ses-01_dwi.nii.gz",
            ),
            "bval": (
                f"{s3_root}/sub-010002/ses-01/dwi/"
                "sub-010002_ses-01_dwi.bval",
                cache_dir / "sub-010002_ses-01_dwi.bval",
            ),
            "bvec": (
                f"{s3_root}/sub-010002/ses-01/dwi/"
                "sub-010002_ses-01_dwi.bvec",
                cache_dir / "sub-010002_ses-01_dwi.bvec",
            ),
        }

        for label, (url, dest) in files.items():
            if dest.exists() and dest.stat().st_size > 0:
                continue
            logging.info(
                "KWNeuroImporter: downloading ds000221 %s from %s", label, url,
            )
            if progress_callback is not None:
                progress_callback(f"Downloading ds000221 {label} ({url})...")
            KWNeuroImporterLogic._download_with_progress(
                url, dest, label, progress_callback,
            )

        return (
            files["t1"][1],
            files["dwi"][1],
            files["bval"][1],
            files["bvec"][1],
        )

    @staticmethod
    def _download_with_progress(
        url: str,
        dest: Path,
        label: str,
        progress_callback: Any,
    ) -> None:
        """Download ``url`` to ``dest``, posting a status line every ~5%.

        Uses ``urllib.request.urlretrieve`` with a ``reporthook`` to
        emit progress lines compatible with our ProgressDialog (the
        ``progress_callback`` is the same kind of "push a string"
        function tqdm-capture uses).

        Writes through a ``.partial`` temp file and atomically
        renames on success, so an interrupted (or test-stubbed)
        download never leaves a half-written file at ``dest`` that
        the next ``fetch_edden_paths`` call would accept as cached.
        """
        import urllib.request

        last_pct = -1

        def _hook(block_idx: int, block_size: int, total: int) -> None:
            nonlocal last_pct
            if total <= 0 or progress_callback is None:
                return
            downloaded = block_idx * block_size
            pct = min(100, int(100 * downloaded / total))
            if pct - last_pct >= 5:
                last_pct = pct
                progress_callback(
                    f"{label}: {downloaded / 1e6:.0f} / "
                    f"{total / 1e6:.0f} MB ({pct}%)",
                )

        partial = dest.with_suffix(dest.suffix + ".partial")
        try:
            urllib.request.urlretrieve(url, partial, reporthook=_hook)
        except BaseException:
            # Clean up the partial so we don't trip the cache-hit
            # check on the next attempt.
            partial.unlink(missing_ok=True)
            raise
        partial.replace(dest)

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

    def publish_structural_to_scene(self, structural: Any, name: str) -> str:
        """Push a kwneuro StructuralImage into the scene. **Main thread only.**"""
        from kwneuro_slicer_bridge import InSceneStructuralImage

        scene_structural = InSceneStructuralImage.from_structural(
            structural, name=name,
        )
        try:
            scene_structural.get_node().CreateDefaultDisplayNodes()
            slicer.util.setSliceViewerLayers(background=scene_structural.get_node())
        except BaseException:
            slicer.mrmlScene.RemoveNode(scene_structural.get_node())
            raise
        return scene_structural.node_id

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

    def load_structural_from_path(self, volume_path: Path, name: str) -> str:
        """Synchronous structural import; composes load + publish."""
        structural = self.load_structural_from_disk(volume_path)
        return self.publish_structural_to_scene(structural, name)

    def load_sherbrooke(self, name: str = "HARDI193") -> str:
        """Synchronous Sherbrooke fetch + load; composes the three phases."""
        volume, bval, bvec = self.fetch_sherbrooke_paths()
        dwi = self.load_dwi_from_disk(volume, bval, bvec)
        return self.publish_to_scene(dwi, name)

    def load_cenir(self, name: str = "CENIR_HCP_like") -> str:
        """Synchronous CENIR multi-b fetch + load."""
        dwi = self.fetch_cenir_dwi()
        return self.publish_to_scene(dwi, name)

    def load_edden(self, name: str = "EDDEN_HCP_protocol") -> str:
        """Synchronous EDDEN sub-01 fetch + load."""
        volume, bval, bvec = self.fetch_edden_paths()
        dwi = self.load_dwi_from_disk(volume, bval, bvec)
        return self.publish_to_scene(dwi, name)

    def load_ds000221_multimodal(
        self,
        structural_name: str = "ds000221_sub-010002_ses-01_T1w",
        dwi_name: str = "ds000221_sub-010002_ses-01_DWI",
    ) -> dict[str, str]:
        """Synchronous OpenNeuro ds000221 fetch + structural/DWI load."""
        t1, dwi_path, bval, bvec = self.fetch_ds000221_multimodal_paths()
        structural = self.load_structural_from_disk(t1)
        dwi = self.load_dwi_from_disk(dwi_path, bval, bvec)
        return {
            "structural": self.publish_structural_to_scene(
                structural, structural_name,
            ),
            "dwi": self.publish_to_scene(dwi, dwi_name),
        }


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
        self.ui.structuralPathLineEdit.connect(
            "currentPathChanged(QString)", self._updateStructuralLoadEnabled,
        )
        self.ui.structuralNameLineEdit.connect(
            "textChanged(QString)", self._updateStructuralLoadEnabled,
        )
        self.ui.loadStructuralButton.connect(
            "clicked(bool)", self.onLoadStructuralClicked,
        )
        self.ui.loadSherbrookeButton.connect(
            "clicked(bool)", self.onLoadSherbrookeClicked,
        )
        self.ui.loadCenirButton.connect(
            "clicked(bool)", self.onLoadCenirClicked,
        )
        self.ui.loadEddenButton.connect(
            "clicked(bool)", self.onLoadEddenClicked,
        )
        self.ui.loadDs000221Button.connect(
            "clicked(bool)", self.onLoadDs000221Clicked,
        )

        self._updateLoadEnabled()
        self._updateStructuralLoadEnabled()

    def enter(self) -> None:
        self._updateLoadEnabled()
        self._updateStructuralLoadEnabled()

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

    def _updateStructuralLoadEnabled(self, *_args: Any) -> None:
        self.ui.loadStructuralButton.enabled = (
            bool(self.ui.structuralPathLineEdit.currentPath)
            and bool(self.ui.structuralNameLineEdit.text.strip())
        )

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

    def onLoadStructuralClicked(self) -> None:
        from kwneuro_slicer_bridge import run_with_progress_dialog

        volume = Path(self.ui.structuralPathLineEdit.currentPath)
        name = self.ui.structuralNameLineEdit.text.strip()

        with slicer.util.tryWithErrorDisplay(
            _("Failed to load structural image."), waitCursor=False,
        ):
            structural = run_with_progress_dialog(
                lambda: self.logic.load_structural_from_disk(volume),
                title=_("KWNeuroImporter"),
                status=_("Reading structural NIfTI from disk..."),
            )
            node_id = self.logic.publish_structural_to_scene(structural, name)
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

    def onLoadEddenClicked(self) -> None:
        # EDDEN goes through urllib download (no tqdm to capture);
        # we feed the bridge's progress queue ourselves via the
        # ``progress_callback`` hook so the dialog log shows
        # MB-counter lines as the download progresses.
        import queue as _queue

        from kwneuro_slicer_bridge import run_with_progress_dialog

        progress_queue: _queue.Queue = _queue.Queue()

        def _worker() -> Any:
            paths = self.logic.fetch_edden_paths(
                progress_callback=progress_queue.put_nowait,
            )
            return self.logic.load_dwi_from_disk(*paths)

        with slicer.util.tryWithErrorDisplay(
            _("Failed to fetch / load EDDEN sample data."), waitCursor=False,
        ):
            dwi = run_with_progress_dialog(
                _worker,
                title=_("KWNeuroImporter"),
                status=_("Fetching EDDEN HCP-protocol DWI..."),
                progress_queue=progress_queue,
            )
            node_id = self.logic.publish_to_scene(dwi, "EDDEN_HCP_protocol")
            self._updateResultLabel(node_id)

    def onLoadDs000221Clicked(self) -> None:
        import queue as _queue

        from kwneuro_slicer_bridge import run_with_progress_dialog

        progress_queue: _queue.Queue = _queue.Queue()

        def _worker() -> tuple[Any, Any]:
            t1, dwi_path, bval, bvec = self.logic.fetch_ds000221_multimodal_paths(
                progress_callback=progress_queue.put_nowait,
            )
            structural = self.logic.load_structural_from_disk(t1)
            dwi = self.logic.load_dwi_from_disk(dwi_path, bval, bvec)
            return structural, dwi

        with slicer.util.tryWithErrorDisplay(
            _("Failed to fetch / load ds000221 multimodal sample data."),
            waitCursor=False,
        ):
            structural, dwi = run_with_progress_dialog(
                _worker,
                title=_("KWNeuroImporter"),
                status=_("Fetching ds000221 T1 + DWI..."),
                progress_queue=progress_queue,
            )
            structural_id = self.logic.publish_structural_to_scene(
                structural, "ds000221_sub-010002_ses-01_T1w",
            )
            dwi_id = self.logic.publish_to_scene(
                dwi, "ds000221_sub-010002_ses-01_DWI",
            )
            structural_node = slicer.mrmlScene.GetNodeByID(structural_id)
            dwi_node = slicer.mrmlScene.GetNodeByID(dwi_id)
            self.ui.resultLabel.text = (
                "Loaded: "
                f"{structural_node.GetName() if structural_node else structural_id}, "
                f"{dwi_node.GetName() if dwi_node else dwi_id}"
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
