# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Slicer 5.11
#     language: python
#     name: slicer-5.11
# ---

# %% [markdown]
# # KWNeuro pipeline walkthrough
#
# This notebook runs a DWI pipeline end-to-end inside 3D Slicer via
# [SlicerJupyter](https://github.com/Slicer/SlicerJupyter), using the
# scripted modules and `kwneuro_slicer_bridge` that this extension
# provides.
#
# **Kernel requirement**: this notebook expects the `slicer` kernel
# (SlicerJupyter). In a running Slicer session with SlicerJupyter
# installed:
#
# 1. Open the **JupyterKernel** module and start the kernel / server.
# 2. Launch Jupyter from the usual shell — the `slicer` kernel will
#    appear in the New Notebook dropdown.
# 3. Open this file (via jupytext) or its `.ipynb` counterpart.
#
# If you just want to re-run the walkthrough headlessly without
# Jupyter, every Python cell below works verbatim in Slicer's Python
# console (Ctrl+3).

# %% [markdown]
# ## Setup
#
# Confirm the bridge and kwneuro are both importable. If either fails,
# open the **KWNeuro Environment** module in Slicer's UI and click
# *Install / Update*.

# %%
import kwneuro
import kwneuro_slicer_bridge
print(f"kwneuro          {kwneuro.__version__}")
print(f"bridge           {kwneuro_slicer_bridge.__name__} loaded OK")

# %% [markdown]
# ## Load the Sherbrooke 3-shell sample DWI
#
# This is the same 193-direction multi-shell dataset that
# `KWNeuroImporter`'s "Sample data" button fetches (~30 MB, cached to
# `~/.dipy/sherbrooke_3shell/`). Instead of clicking the button, we
# call the logic directly.

# %%
import slicer
slicer.mrmlScene.Clear()

from KWNeuroImporter import KWNeuroImporterLogic
dwi_node_id = KWNeuroImporterLogic().load_sherbrooke(name="HARDI193")
dwi_node = slicer.mrmlScene.GetNodeByID(dwi_node_id)
# GetImageData().GetDimensions() reports only the 3 spatial axes — the
# gradient axis is packed into vtkImageData's scalar components, not
# dimensions — so we print GetNumberOfGradients() alongside to show
# the 4th (DWI-volume) axis.
print(
    f"Loaded {dwi_node.GetName()!r}: "
    f"spatial={dwi_node.GetImageData().GetDimensions()}, "
    f"gradients={dwi_node.GetNumberOfGradients()}, "
    f"class={dwi_node.GetClassName()}"
)

# %% [markdown]
# **Note the node class**: `vtkMRMLDiffusionWeightedVolumeNode`, NOT
# `vtkMRMLScalarVolumeNode`. Slicer's built-in *Add Data* dialog loads
# a 4D NIfTI as a 3D scalar and silently drops the gradients. The
# KWNeuroImporter module + `InSceneDwi.from_nifti_path` avoid that.

# %% [markdown]
# ## Denoise
#
# Runs dipy's Patch2Self via `KWNeuroDenoise`. On the real 193-volume
# dataset this takes a few minutes; in the GUI the progress dialog
# shows per-gradient `tqdm` lines as they stream out.

# %%
from KWNeuroDenoise import KWNeuroDenoiseLogic
denoised_id = KWNeuroDenoiseLogic().process(dwi_node)
print(f"Denoised: {slicer.mrmlScene.GetNodeByID(denoised_id).GetName()}")

# %% [markdown]
# ## Brain extraction (optional, requires `kwneuro[hdbet]`)
#
# `KWNeuroBrainExtract` wraps HD-BET. This cell is intentionally
# wrapped in a try/except so the notebook doesn't hard-fail on a
# machine without the extra — click **KWNeuro Environment** and tick
# the **hdbet** checkbox to install.

# %%
mask_node = None  # guaranteed bound, even if HD-BET fails hard
try:
    from KWNeuroBrainExtract import KWNeuroBrainExtractLogic
    mask_id = KWNeuroBrainExtractLogic().process(dwi_node)
    mask_node = slicer.mrmlScene.GetNodeByID(mask_id)
    print(f"Mask: {mask_node.GetName()} ({mask_node.GetClassName()})")
except Exception as exc:
    # RuntimeError covers the ensure_extras_installed path, but a real
    # HD-BET run can also fail with torch/CUDA errors, nnUNet issues,
    # or network problems downloading the model. Catch broadly so the
    # notebook moves on without a hard NameError in downstream cells.
    print(f"Skipping brain extraction: {type(exc).__name__}: {exc}")
    mask_node = None

# %% [markdown]
# ## Fit DTI + derive FA / MD
#
# Always available — no extras required. If we have a brain mask from
# the previous step, we pass it; otherwise we fit over the whole
# volume.

# %%
from KWNeuroDTI import KWNeuroDTILogic
dti_ids = KWNeuroDTILogic().process(
    dwi_node=dwi_node,
    mask_node=mask_node,  # None is fine — whole-volume fit
    create_fa_md=True,
)
for key, node_id in dti_ids.items():
    if node_id is None:
        continue
    node = slicer.mrmlScene.GetNodeByID(node_id)
    print(f"  {key}: {node.GetName()} ({node.GetClassName()})")

# %% [markdown]
# ## CSD peaks
#
# Fits CSD to produce a 4D peak-vector volume suitable for downstream
# tractography tools. Requires at least 45 diffusion-weighted
# directions (Sherbrooke's 193 easily clears this bar) and a brain
# mask. We use the HD-BET mask if available, falling back to a trivial
# all-ones mask for illustration.

# %%
if mask_node is None:
    # Synthesize a trivial all-in mask if brain extraction was skipped.
    import numpy as np
    from kwneuro.resource import InMemoryVolumeResource
    from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource
    shape = InSceneDwi.from_node(dwi_node).volume.get_array().shape[:3]
    affine = InSceneDwi.from_node(dwi_node).volume.get_affine()
    trivial = InSceneVolumeResource.from_resource(
        InMemoryVolumeResource(
            array=np.ones(shape, dtype="uint8"),
            affine=affine,
            metadata={},
        ),
        name="HARDI193_trivial_mask",
    )
    mask_for_csd = trivial.get_node()
else:
    mask_for_csd = mask_node

from KWNeuroCSD import KWNeuroCSDLogic
csd_id = KWNeuroCSDLogic().process(
    dwi_node=dwi_node,
    mask_node=mask_for_csd,
    n_peaks=5,
    # flip_bvecs_x=True means "flip from FSL convention to MRtrix3
    # convention". Sherbrooke comes from dipy's fetch and is
    # FSL-convention (like most NIfTI data loaded via
    # InSceneDwi.from_nifti_path), so True is correct here. Set False
    # only if your bvecs are already in MRtrix3 convention.
    flip_bvecs_x=True,
)
csd_node = slicer.mrmlScene.GetNodeByID(csd_id)
print(
    f"CSD peaks: {csd_node.GetName()} "
    f"({csd_node.GetClassName()}, "
    f"{csd_node.GetImageData().GetNumberOfScalarComponents()} components)"
)

# %% [markdown]
# ## What's in the scene now
#
# A quick inventory. Each of these nodes is fully-formed MRML, so you
# can slice / render / export it via any Slicer module.

# %%
for cls in (
    "vtkMRMLDiffusionWeightedVolumeNode",
    "vtkMRMLDiffusionTensorVolumeNode",
    "vtkMRMLScalarVolumeNode",
    "vtkMRMLLabelMapVolumeNode",
    "vtkMRMLVectorVolumeNode",
):
    nodes = slicer.mrmlScene.GetNodesByClass(cls)
    names = [nodes.GetItemAsObject(i).GetName() for i in range(nodes.GetNumberOfItems())]
    if names:
        print(f"{cls}:")
        for n in names:
            print(f"  - {n}")

# %% [markdown]
# ## Other modules (GUI-first)
#
# The walkthrough above covers the single-subject pipeline. Two more
# modules are group-level and need multiple subjects:
#
# - **KWNeuroRegister** — register a moving volume to a fixed volume
#   via ANTs (rigid / affine / SyN).
# - **KWNeuroTemplate** — build an unbiased group-wise template via
#   iterative SyN + averaging.
# - **KWNeuroHarmonize** — ComBat cross-site harmonisation across N
#   subjects (requires a covariates CSV).
# - **KWNeuroNODDI** — multi-shell NODDI via AMICO. Requires the
#   `noddi` extra.
# - **KWNeuroTractSeg** — CNN-based white-matter tract segmentation.
#   Requires the `tractseg` extra and really wants a CUDA GPU.
#
# All of them use the same three-phase architecture (materialise
# inputs on the main thread, run the heavy compute on a worker
# thread, publish outputs on the main thread) so they're all
# scriptable from this kernel in exactly the same way as the
# single-subject modules above.
