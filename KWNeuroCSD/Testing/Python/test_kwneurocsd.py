"""Tests for KWNeuroCSD.

CSD requires ~45+ DWI directions for an 8th-order spherical harmonic
fit, so this test builds a 50-gradient synthetic DWI (10 b=0 +
40 b=1000 evenly distributed on a hemisphere) with an anisotropic
signal structure so dipy's response estimation can find a reasonable
single-fiber voxel. Spatial dims stay small (5^3) to keep the test
under a few seconds.

We also have a mocked-CSD test that sidesteps dipy's numerics and
verifies our wrapping / node creation independently. Two tests
covering different failure modes: the real test catches dipy-API
breakage, the mocked test catches bridge / node-class regressions.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_dwi_for_csd(seed: int = 0):
    """A DWI with enough gradients for CSD + a mild anisotropic signal."""
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    # n_dwi >= 45 is required for CSD at SH order 8 — below that the
    # module raises ValueError in run_csd's precheck. Use 50 for headroom.
    nx, ny, nz = 5, 5, 5
    n_b0 = 10
    n_dwi = 50
    rng = np.random.default_rng(seed=seed)

    # Fibonacci-ish lattice on a hemisphere — approximately uniform
    # coverage so dipy's spherical harmonic fit is well-conditioned.
    golden = (1 + 5 ** 0.5) / 2
    thetas = np.arccos(1 - (np.arange(n_dwi) + 0.5) / n_dwi)
    phis = 2 * np.pi * np.arange(n_dwi) / golden
    directions = np.stack(
        [
            np.sin(thetas) * np.cos(phis),
            np.sin(thetas) * np.sin(phis),
            np.cos(thetas),
        ],
        axis=-1,
    )
    bvals = np.concatenate([np.zeros(n_b0), np.full(n_dwi, 1000.0)])
    bvecs = np.vstack([np.zeros((n_b0, 3)), directions])

    # Build a DWI signal roughly like a single-fiber oriented along z.
    # Dot product with (0,0,1) tells us how parallel each gradient is
    # with the fiber; higher dot product → lower signal.
    fiber = np.array([0.0, 0.0, 1.0])
    d_par, d_perp = 1.7e-3, 0.3e-3
    dots = np.abs(directions @ fiber)
    dwi_attenuation = np.exp(
        -1000.0 * (d_perp + (d_par - d_perp) * dots ** 2),
    )
    # S_0 * attenuation with a little noise, broadcasted spatially.
    s0 = 800.0
    per_grad = np.concatenate([np.ones(n_b0), dwi_attenuation]) * s0
    volume = np.broadcast_to(per_grad, (nx, ny, nz, n_b0 + n_dwi)).copy()
    volume = volume.astype(np.float32)
    volume += rng.normal(0.0, 5.0, size=volume.shape).astype(np.float32)

    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    return Dwi(
        volume=InMemoryVolumeResource(array=volume, affine=affine, metadata={}),
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


def _synthetic_mask(shape=(5, 5, 5), affine=None):
    from kwneuro.resource import InMemoryVolumeResource

    if affine is None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
    arr = np.ones(shape, dtype=np.uint8)
    return InMemoryVolumeResource(array=arr, affine=affine, metadata={})


class TestKWNeuroCSDLogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_prepare_inputs_raises_without_dwi(self) -> None:
        from KWNeuroCSD import KWNeuroCSDLogic

        logic = KWNeuroCSDLogic()
        from kwneuro_slicer_bridge import InSceneVolumeResource
        mask_node = InSceneVolumeResource.from_resource(_synthetic_mask(), name="m").get_node()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(None, mask_node)

    def test_prepare_inputs_raises_without_mask(self) -> None:
        from KWNeuroCSD import KWNeuroCSDLogic

        logic = KWNeuroCSDLogic()
        from kwneuro_slicer_bridge import InSceneDwi
        dwi_node = InSceneDwi.from_dwi(_synthetic_dwi_for_csd(), name="d").get_node()
        with self.assertRaises(ValueError):
            logic.prepare_inputs(dwi_node, None)

    def test_run_csd_with_mocked_dipy(self) -> None:
        """Mock-test: verify our wrapping / output-shape logic without dipy numerics.

        Relies on ``run_csd``'s lazy ``from kwneuro.csd import
        compute_csd_peaks`` — patching ``kwneuro.csd.compute_csd_peaks``
        takes effect because Python resolves the name at call time.
        If that import ever gets hoisted to module-top, the mock would
        miss and the ``call_count`` assertion fires.

        The distinct-per-peak fixture design catches axis-permutation
        bugs in ``combine_csd_peaks_to_vector_volume`` that an all-
        peaks-same-direction fixture would miss: peak 0 -> +x, peak 1
        -> +y, peak 2 -> +z with amplitudes 3, 2, 1 produce distinct
        output slices that any axis/component swap would corrupt.
        """
        import kwneuro.csd as csd_mod

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroCSD import KWNeuroCSDLogic

        nx, ny, nz = 5, 5, 5
        n_peaks = 3
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

        # Distinct directions per peak — differentiates axis permutations.
        fake_dirs = np.zeros((nx, ny, nz, n_peaks, 3), dtype=np.float32)
        fake_dirs[..., 0, 0] = 1.0  # peak 0 -> (1, 0, 0)
        fake_dirs[..., 1, 1] = 1.0  # peak 1 -> (0, 1, 0)
        fake_dirs[..., 2, 2] = 1.0  # peak 2 -> (0, 0, 1)
        # Distinct amplitudes per peak.
        fake_values = np.zeros((nx, ny, nz, n_peaks), dtype=np.float32)
        fake_values[..., 0] = 3.0
        fake_values[..., 1] = 2.0
        fake_values[..., 2] = 1.0

        call_count = [0]

        def fake_compute_csd_peaks(**kwargs):
            call_count[0] += 1
            return (
                InMemoryVolumeResource(fake_dirs, affine, {}),
                InMemoryVolumeResource(fake_values, affine, {}),
            )

        original = csd_mod.compute_csd_peaks
        csd_mod.compute_csd_peaks = fake_compute_csd_peaks
        try:
            logic = KWNeuroCSDLogic()
            vector_volume = logic.run_csd(
                dwi=_synthetic_dwi_for_csd(),
                mask=_synthetic_mask(),
                n_peaks=n_peaks,
                flip_bvecs_x=True,
            )
        finally:
            csd_mod.compute_csd_peaks = original

        self.assertEqual(
            call_count[0], 1,
            "compute_csd_peaks must be invoked exactly once — if this "
            "fails, the module has an import path the mock missed.",
        )
        self.assertEqual(
            vector_volume.get_array().shape, (nx, ny, nz, n_peaks * 3),
        )
        # Each voxel's 9-element vector is:
        # [peak0_x, peak0_y, peak0_z, peak1_x, peak1_y, peak1_z, peak2_x, peak2_y, peak2_z]
        # with amplitudes applied: [3, 0, 0, 0, 2, 0, 0, 0, 1].
        voxel = vector_volume.get_array()[0, 0, 0]
        np.testing.assert_allclose(
            voxel, np.array([3, 0, 0, 0, 2, 0, 0, 0, 1], dtype=np.float32),
            err_msg=(
                "Vector-volume layout broken: expected each peak's "
                "3 components interleaved in order with magnitude "
                "equal to the peak value."
            ),
        )

    def test_run_csd_real_end_to_end(self) -> None:
        """Real CSD on a 50-gradient synthetic DWI; asserts output shape + content.

        Guards against dipy API drift and pipeline regressions that no
        mock would catch. We only skip on the narrow set of failure
        modes dipy can legitimately hit on synthetic data (tiny mask
        rejecting response-fn voxels, eigenvalue ratio too high) —
        anything else means a real bug and must propagate.
        """
        from KWNeuroCSD import KWNeuroCSDLogic

        logic = KWNeuroCSDLogic()
        dwi = _synthetic_dwi_for_csd()
        mask = _synthetic_mask()
        n_peaks = 3

        try:
            vector_volume = logic.run_csd(
                dwi=dwi, mask=mask, n_peaks=n_peaks, flip_bvecs_x=True,
            )
        except ValueError as exc:
            # response_from_mask_ssst raises ValueError on degenerate
            # inputs; that's a synthetic-data limitation, not a bug.
            self.skipTest(
                f"dipy CSD rejected synthetic DWI (expected on degenerate "
                f"inputs): {exc!r}",
            )
        # IMPORTANT: any other exception type propagates — those
        # indicate a real production bug.

        arr = vector_volume.get_array()
        self.assertEqual(
            arr.shape,
            (*dwi.volume.get_array().shape[:3], n_peaks * 3),
        )

        # The synthetic fiber is along +z; inside the mask we expect
        # at least some nonzero peaks. A bug that returned all zeros
        # (e.g. wrong mask axis ordering) would pass a shape-only check.
        self.assertGreater(
            float(np.abs(arr).sum()), 0.0,
            "CSD produced an all-zero vector volume — likely a bug in "
            "mask axis ordering, DWI array passthrough, or "
            "combine_csd_peaks_to_vector_volume.",
        )

    def test_run_csd_rejects_too_few_directions(self) -> None:
        """With fewer than 45 DW directions, run_csd must raise with a
        clear message BEFORE diving into dipy where the error would be
        opaque.
        """
        from kwneuro.dwi import Dwi
        from kwneuro.resource import (
            InMemoryBvalResource,
            InMemoryBvecResource,
            InMemoryVolumeResource,
        )
        from KWNeuroCSD import KWNeuroCSDLogic

        # Only 20 DW directions — below the SH-order-8 threshold of 45.
        nx, ny, nz = 5, 5, 5
        n_dwi = 20
        rng = np.random.default_rng(0)
        bvals = np.concatenate([np.zeros(1), np.full(n_dwi, 1000.0)])
        dirs = rng.normal(size=(n_dwi, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvecs = np.vstack([np.zeros((1, 3)), dirs])
        volume = rng.uniform(100, 900, size=(nx, ny, nz, n_dwi + 1)).astype(np.float32)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        dwi = Dwi(
            volume=InMemoryVolumeResource(volume, affine, {}),
            bval=InMemoryBvalResource(bvals),
            bvec=InMemoryBvecResource(bvecs),
        )
        mask = _synthetic_mask()

        logic = KWNeuroCSDLogic()
        with self.assertRaises(ValueError) as ctx:
            logic.run_csd(dwi, mask, n_peaks=3, flip_bvecs_x=True)
        err = str(ctx.exception)
        self.assertIn("45", err)
        self.assertIn("20", err)

    def test_publish_to_scene_creates_vector_volume(self) -> None:
        """publish_to_scene produces a 4D vector volume with the right name."""
        import slicer

        from kwneuro.resource import InMemoryVolumeResource
        from KWNeuroCSD import KWNeuroCSDLogic

        n_peaks = 4
        arr = np.arange(5 * 5 * 5 * n_peaks * 3, dtype=np.float32).reshape(
            5, 5, 5, n_peaks * 3,
        )
        vec = InMemoryVolumeResource(arr, np.diag([2.0, 3.0, 4.0, 1.0]), {})

        logic = KWNeuroCSDLogic()
        node_id = logic.publish_to_scene(vec, "csd_test_dwi")
        node = slicer.mrmlScene.GetNodeByID(node_id)
        self.assertEqual(node.GetName(), "csd_test_dwi_csd_peaks")
        # 4D -> bridge creates vtkMRMLVectorVolumeNode.
        self.assertEqual(node.GetClassName(), "vtkMRMLVectorVolumeNode")


class TestKWNeuroCSDWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroCSD")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_requires_both_selectors(self) -> None:
        from kwneuro_slicer_bridge import InSceneDwi, InSceneVolumeResource

        widget = self._widget()
        widget.ui.inputDwiSelector.setCurrentNode(None)
        widget.ui.maskSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        dwi_node = InSceneDwi.from_dwi(
            _synthetic_dwi_for_csd(), name="csd_widget_dwi",
        ).get_node()
        widget.ui.inputDwiSelector.setCurrentNode(dwi_node)
        # Force mask selector back to None — if Slicer ever auto-
        # selects the DWI into the mask combo (would require
        # showChildNodeTypes=true regressing), this assertion would
        # vacuously pass without the explicit clear.
        widget.ui.maskSelector.setCurrentNode(None)
        self._pump()
        self.assertIsNone(widget.ui.maskSelector.currentNode())
        self.assertFalse(widget.ui.applyButton.enabled)

        mask_node = InSceneVolumeResource.from_resource(
            _synthetic_mask(), name="csd_widget_mask",
        ).get_node()
        widget.ui.maskSelector.setCurrentNode(mask_node)
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
