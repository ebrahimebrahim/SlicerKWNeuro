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
- "Sample data" section: one-click fetch for the Sherbrooke 3-shell
  dataset (``dipy.data.fetch_sherbrooke_3shell``).

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
            "gradients + b-values), or fetch dipy's Sherbrooke 3-shell "
            "sample dataset."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
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
        from kwneuro_slicer_bridge import run_with_progress_dialog

        with slicer.util.tryWithErrorDisplay(
            _("Failed to fetch / load Sherbrooke sample data."), waitCursor=False,
        ):
            # Worker: network fetch + disk read. Main thread: scene add.
            # capture_tqdm=True so dipy's per-chunk download progress
            # flows into the dialog's Details log — otherwise the user
            # stares at an indeterminate bar for the ~30 MB download.
            paths_and_dwi = run_with_progress_dialog(
                lambda: self._fetch_and_load(),
                title=_("KWNeuroImporter"),
                status=_("Fetching Sherbrooke 3-shell..."),
                capture_tqdm=True,
            )
            _, dwi = paths_and_dwi
            node_id = self.logic.publish_to_scene(dwi, "HARDI193")
            self._updateResultLabel(node_id)

    def _fetch_and_load(self) -> tuple[tuple[Path, Path, Path], Any]:
        """Worker-side: fetch + load into memory, no scene writes."""
        paths = self.logic.fetch_sherbrooke_paths()
        dwi = self.logic.load_dwi_from_disk(*paths)
        return paths, dwi

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
