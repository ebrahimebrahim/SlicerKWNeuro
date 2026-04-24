cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(KWNeuro)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/brain-microstructure-exploration-tools/KWNeuro")
set(EXTENSION_CONTRIBUTORS "Ebrahim Ebrahim (Kitware, Inc.)")
set(EXTENSION_DESCRIPTION "Bridge to the kwneuro diffusion MRI processing library — Phase 0 scaffold.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/brain-microstructure-exploration-tools/KWNeuro/main/KWNeuro.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/brain-microstructure-exploration-tools/KWNeuro/main/KWNeuro.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(KWNeuroEnvironment)
add_subdirectory(KWNeuroImporter)
add_subdirectory(KWNeuroBrainExtract)
add_subdirectory(KWNeuroDTI)
add_subdirectory(KWNeuroDenoise)
add_subdirectory(KWNeuroRegister)
add_subdirectory(KWNeuroTemplate)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
