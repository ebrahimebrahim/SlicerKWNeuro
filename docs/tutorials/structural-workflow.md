# Structural + diffusion workflow

This walkthrough uses the GUI modules for a single-subject structural
and diffusion workflow.

1. Open **KWNeuro Environment**, check the extras you need, and click
   **Apply environment changes**: `hdbet` for brain extraction and
   `antspynet` for Deep Atropos or DKT parcellation.
2. Open **KWNeuro Importer**. Load a structural NIfTI and a DWI
   NIfTI + `.bval` + `.bvec`, or click **Load ds000221 T1 + DWI** to
   fetch the OpenNeuro multimodal sample.
3. Open **KWNeuro Bias Correct**, select the T1 node, and click
   **Apply**. The output is named `{input}_bias_corrected`.
4. Open **KWNeuro Brain Extract** and select either the corrected T1
   or the DWI. DWI inputs are reduced to mean-b0 before HD-BET runs.
5. Open **KWNeuro Tissue Segment** for Atropos or Deep Atropos, or
   **KWNeuro Parcellate** for DKT labels. Outputs are labelmap volume
   nodes, preserving multi-label integer values.
6. Open **KWNeuro DWI To Structural Register**. Select the DWI, the
   corrected T1, optional masks, and optionally a structural labelmap.
   The module publishes transform nodes, a warped mean-b0 QA volume in
   structural space, and inverse-warped labels in DWI space when labels
   are supplied.
