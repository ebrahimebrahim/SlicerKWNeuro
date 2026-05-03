#-----------------------------------------------------------------------------
set(MODULE_NAME kwneuro_slicer_bridge)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  kwneuro_slicer_bridge/__init__.py
  kwneuro_slicer_bridge/async_helpers.py
  kwneuro_slicer_bridge/conversions.py
  kwneuro_slicer_bridge/dti.py
  kwneuro_slicer_bridge/dwi.py
  kwneuro_slicer_bridge/transform.py
  kwneuro_slicer_bridge/volume.py
  )

set(MODULE_PYTHON_RESOURCES
  kwneuro_slicer_bridge/py.typed
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )
