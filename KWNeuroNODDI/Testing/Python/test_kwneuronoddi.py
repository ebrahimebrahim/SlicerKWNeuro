"""Tests for KWNeuroNODDI.

AMICO's real NODDI fit requires kernel generation (downloads /
computes on first call), so tests mock ``Noddi.estimate_noddi`` with
a factory that returns a ``Noddi`` backed by synthetic NDI/ODI/FWF
volumes. This covers:
  * The wrapping: our logic asks for the right outputs and publishes
    them under expected names.
  * The modulated-outputs option: when the checkbox is on, NDI_mod /
    ODI_mod are also published.
  * prepare_inputs gates on the ``noddi`` extra and surfaces a clear
    error if absent.

Bridge / node-class assertions still cover the "output actually lands
as a scalar volume with the right name" contract, independent of the
AMICO numerics.
"""
from __future__ import annotations

import unittest

import numpy as np


def _synthetic_dwi(seed: int = 0):
    from kwneuro.dwi import Dwi
    from kwneuro.resource import (
        InMemoryBvalResource,
        InMemoryBvecResource,
        InMemoryVolumeResource,
    )

    nx, ny, nz, n_grad = 5, 5, 5, 13
    rng = np.random.default_rng(seed=seed)
    bvals = np.concatenate(
        [np.zeros(1), np.full(6, 1000.0), np.full(6, 2000.0)],
    ).astype(np.float64)
    dirs = rng.normal(size=(12, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((1, 3)), dirs]).astype(np.float64)
    volume = rng.uniform(100, 900, size=(nx, ny, nz, n_grad)).astype(np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    return Dwi(
        volume=InMemoryVolumeResource(array=volume, affine=affine, metadata={}),
        bval=InMemoryBvalResource(array=bvals),
        bvec=InMemoryBvecResource(array=bvecs),
    )


def _fake_noddi_maps(shape=(5, 5, 5), affine=None):
    """Construct a fake Noddi whose NDI/ODI/FWF come out as distinct values.

    kwneuro.Noddi stores a single 4D volume (last axis indexes NDI /
    ODI / FWF at positions 0/1/2) plus a directions volume. The fake
    mirrors that layout.
    """
    from kwneuro.noddi import Noddi
    from kwneuro.resource import InMemoryVolumeResource

    if affine is None:
        affine = np.diag([2.0, 3.0, 4.0, 1.0])

    stacked = np.stack(
        [
            np.full(shape, 0.6, dtype=np.float32),  # NDI (index 0)
            np.full(shape, 0.2, dtype=np.float32),  # ODI (index 1)
            np.full(shape, 0.1, dtype=np.float32),  # FWF (index 2)
        ],
        axis=-1,
    )
    directions = np.zeros((*shape, 3), dtype=np.float32)
    directions[..., 2] = 1.0  # placeholder +z
    return Noddi(
        volume=InMemoryVolumeResource(stacked, affine, {}),
        directions=InMemoryVolumeResource(directions, affine, {}),
    )


class TestKWNeuroNODDILogic(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()

    def test_prepare_inputs_requires_dwi(self) -> None:
        from KWNeuroNODDI import KWNeuroNODDILogic

        with self.assertRaises(ValueError):
            KWNeuroNODDILogic().prepare_inputs(None, None)

    def test_prepare_inputs_asks_for_noddi_extra_by_name(self) -> None:
        """Spy on ensure_extras_installed to verify the exact call.

        Watching the error message alone can't distinguish ``["noddi"]``
        from ``["noddi", "tractseg"]`` for future extras; spying on the
        argument is the load-bearing check.
        """
        import kwneuro_slicer_bridge as bridge

        from kwneuro_slicer_bridge import InSceneDwi
        from KWNeuroNODDI import KWNeuroNODDILogic

        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="noddi_spy_check")

        requested: list[list[str]] = []
        original = bridge.ensure_extras_installed

        def spy(names: list[str]) -> None:
            requested.append(list(names))
            # Don't actually check — test pure wiring.

        bridge.ensure_extras_installed = spy
        try:
            logic = KWNeuroNODDILogic()
            logic.prepare_inputs(sdwi.get_node(), None)
        finally:
            bridge.ensure_extras_installed = original

        self.assertEqual(
            requested, [["noddi"]],
            "prepare_inputs must call ensure_extras_installed with "
            "exactly ['noddi'] — catches regressions that add / drop / "
            "rename the extra being checked.",
        )

    def test_prepare_inputs_checks_noddi_extra(self) -> None:
        """When the ``noddi`` extra is missing, prepare_inputs must
        raise with a message pointing at KWNeuroEnvironment — and not
        mistakenly mention a different extra.
        """
        import KWNeuroEnvironment

        from kwneuro_slicer_bridge import InSceneDwi
        from KWNeuroNODDI import KWNeuroNODDILogic

        sdwi = InSceneDwi.from_dwi(_synthetic_dwi(), name="noddi_extras_check")
        logic = KWNeuroNODDILogic()
        original_status = KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status

        def absent_status():
            return {k: False for k in original_status()}

        KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = staticmethod(
            absent_status,
        )
        try:
            with self.assertRaises(RuntimeError) as ctx:
                logic.prepare_inputs(sdwi.get_node(), None)
            err = str(ctx.exception)
            self.assertIn("noddi", err)
            self.assertIn("KWNeuroEnvironment", err)
            # Negative check: don't report the wrong extra.
            for wrong in ("hdbet", "tractseg", "combat"):
                self.assertNotIn(
                    f"[{wrong}]", err,
                    f"Error mentions kwneuro[{wrong}] — the module is "
                    f"asking for the wrong extra.",
                )
        finally:
            KWNeuroEnvironment.KWNeuroEnvironmentLogic.extras_status = (
                staticmethod(original_status)
            )

    def test_run_noddi_with_mock_publishes_three_scalar_volumes(self) -> None:
        """Mock-test: with create_modulated=False, we publish NDI, ODI, FWF.

        Asserts the actual NDI/ODI/FWF values from the fake propagate
        to the published nodes (NDI=0.6, ODI=0.2, FWF=0.1). Without
        value assertions, a bug that quietly bypassed our mock (real
        AMICO got called) would pass the shape-and-name checks.
        """
        import kwneuro.noddi as noddi_mod
        import slicer

        from kwneuro_slicer_bridge import InSceneVolumeResource
        from KWNeuroNODDI import KWNeuroNODDILogic

        call_count = [0]

        def fake_estimate(dwi, mask=None, dpar=1.7e-3, n_kernel_dirs=500):
            call_count[0] += 1
            return _fake_noddi_maps()

        original = noddi_mod.Noddi.estimate_noddi
        noddi_mod.Noddi.estimate_noddi = staticmethod(fake_estimate)
        try:
            logic = KWNeuroNODDILogic()
            volumes = logic.run_noddi(
                dwi=_synthetic_dwi(),
                mask=None,
                dpar=1.7e-3,
                n_kernel_dirs=500,
                create_modulated=False,
            )
        finally:
            noddi_mod.Noddi.estimate_noddi = staticmethod(original)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(set(volumes), {"ndi", "odi", "fwf"})

        # VALUE assertions — the fake produced distinct constants so
        # the real AMICO path being taken instead would fail these.
        np.testing.assert_allclose(volumes["ndi"].get_array(), 0.6)
        np.testing.assert_allclose(volumes["odi"].get_array(), 0.2)
        np.testing.assert_allclose(volumes["fwf"].get_array(), 0.1)

        ids = logic.publish_to_scene(volumes, "noddi_test")
        self.assertEqual(set(ids), {"ndi", "odi", "fwf"})
        for key, nid in ids.items():
            node = slicer.mrmlScene.GetNodeByID(nid)
            self.assertEqual(node.GetClassName(), "vtkMRMLScalarVolumeNode")
            self.assertEqual(node.GetName(), f"noddi_test_{key}")

        # And the published NDI volume should carry the fake's values.
        scene_ndi = InSceneVolumeResource.from_node(
            slicer.mrmlScene.GetNodeByID(ids["ndi"]),
        ).get_array()
        np.testing.assert_allclose(scene_ndi, 0.6)

    def test_run_noddi_modulated_values(self) -> None:
        """create_modulated=True publishes NDI_mod = NDI*(1-FWF), same for ODI.

        With the fake's NDI=0.6, FWF=0.1, ODI=0.2, expect
        NDI_mod = 0.6 * 0.9 = 0.54 and ODI_mod = 0.2 * 0.9 = 0.18.
        A bug that swapped operands in get_modulated_ndi_odi would
        produce wrong values that a keys-only test would not catch.
        """
        import kwneuro.noddi as noddi_mod
        import slicer

        from KWNeuroNODDI import KWNeuroNODDILogic

        noddi_maps = _fake_noddi_maps()
        original = noddi_mod.Noddi.estimate_noddi
        noddi_mod.Noddi.estimate_noddi = staticmethod(
            lambda *a, **kw: noddi_maps,
        )
        try:
            logic = KWNeuroNODDILogic()
            volumes = logic.run_noddi(
                dwi=_synthetic_dwi(),
                mask=None,
                dpar=1.7e-3,
                n_kernel_dirs=500,
                create_modulated=True,
            )
        finally:
            noddi_mod.Noddi.estimate_noddi = staticmethod(original)

        self.assertEqual(
            set(volumes), {"ndi", "odi", "fwf", "ndi_mod", "odi_mod"},
        )
        np.testing.assert_allclose(
            volumes["ndi_mod"].get_array(), 0.54, rtol=1e-5,
            err_msg="NDI_mod should equal NDI * (1 - FWF) = 0.6 * 0.9",
        )
        np.testing.assert_allclose(
            volumes["odi_mod"].get_array(), 0.18, rtol=1e-5,
            err_msg="ODI_mod should equal ODI * (1 - FWF) = 0.2 * 0.9",
        )

        ids = logic.publish_to_scene(volumes, "mod_test")
        self.assertEqual(len(ids), 5)
        for key in ("ndi", "odi", "fwf", "ndi_mod", "odi_mod"):
            node = slicer.mrmlScene.GetNodeByID(ids[key])
            self.assertEqual(node.GetName(), f"mod_test_{key}")


class TestKWNeuroNODDIWidget(unittest.TestCase):
    def setUp(self) -> None:
        import slicer
        slicer.mrmlScene.Clear()
        slicer.app.processEvents()

    def _widget(self):
        import slicer

        module = slicer.util.getModule("KWNeuroNODDI")
        return module.widgetRepresentation().self()

    def _pump(self) -> None:
        import slicer
        slicer.app.processEvents()

    def test_apply_requires_dwi(self) -> None:
        widget = self._widget()
        widget.ui.inputDwiSelector.setCurrentNode(None)
        self._pump()
        self.assertFalse(widget.ui.applyButton.enabled)

        from kwneuro_slicer_bridge import InSceneDwi
        InSceneDwi.from_dwi(_synthetic_dwi(), name="noddi_widget_dwi")
        self._pump()
        self.assertTrue(widget.ui.applyButton.enabled)


if __name__ == "__main__":
    unittest.main()
