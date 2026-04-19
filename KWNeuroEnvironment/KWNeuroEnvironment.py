import logging

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)


class KWNeuroEnvironment(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("KWNeuro Environment")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "KWNeuro")]
        self.parent.dependencies = []
        self.parent.contributors = ["Ebrahim Ebrahim (Kitware, Inc.)"]
        self.parent.helpText = _(
            "Phase 0 scaffold for the KWNeuro extension. "
            "In Phase 1 this module will manage the kwneuro Python environment "
            "inside Slicer — install status, optional extras, smoke test."
        )
        self.parent.acknowledgementText = _(
            "Developed at Kitware, Inc. as part of the brain microstructure "
            "exploration tools effort."
        )


class KWNeuroEnvironmentWidget(ScriptedLoadableModuleWidget):
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/KWNeuroEnvironment.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = KWNeuroEnvironmentLogic()


class KWNeuroEnvironmentLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def phase(self) -> str:
        return "phase-0"


class KWNeuroEnvironmentTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_LogicInstantiates()

    def test_LogicInstantiates(self):
        self.delayDisplay("Starting KWNeuroEnvironment Phase 0 scaffold test")
        logic = KWNeuroEnvironmentLogic()
        self.assertEqual(logic.phase(), "phase-0")
        logging.info("KWNeuroEnvironment Phase 0 scaffold test passed")
        self.delayDisplay("Test passed")
