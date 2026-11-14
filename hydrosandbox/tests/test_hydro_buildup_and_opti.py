import pytest
import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import Foil, HydroFuselage, HydrofoilAssembly, Strut
from hydrosandbox.hydrodynamics import HydroBuildup
from hydrosandbox.performance import HydroOperatingPoint


def _build_full_assembly() -> HydrofoilAssembly:
    main = Foil(
        name="Main Foil",
        symmetric=True,
        role="main",
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, -0.60], chord=0.16, airfoil=asb.Airfoil("naca2412")),
            asb.WingXSec(xyz_le=[0.02, 0.50, -0.60], chord=0.10, airfoil=asb.Airfoil("naca2412")),
        ],
    )
    stab = Foil(
        name="Stabilizer",
        symmetric=True,
        role="stabilizer",
        xsecs=[
            asb.WingXSec(xyz_le=[0.80, 0.0, -0.60], chord=0.08, twist=-1.5, airfoil=asb.Airfoil("naca0010")),
            asb.WingXSec(xyz_le=[0.81, 0.22, -0.60], chord=0.06, twist=-1.5, airfoil=asb.Airfoil("naca0010")),
        ],
    )
    strut = Strut.from_dimensions(
        name="Mast",
        root_xyz=[0.02, 0.0, -0.60],
        length=0.90,
        root_chord=0.14,
        tip_chord=0.12,
        airfoil=asb.Airfoil("naca0012"),
    )
    pod = HydroFuselage(
        name="Pod",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.15, 0.0, -0.60], radius=0.005),
            asb.FuselageXSec(xyz_c=[0.15, 0.0, -0.60], radius=0.025),
            asb.FuselageXSec(xyz_c=[0.90, 0.0, -0.60], radius=0.005),
        ],
    )
    return HydrofoilAssembly(
        name="T-Foil Test",
        xyz_ref=[0.04, 0.0, -0.50],
        foils=[main, stab],
        struts=[strut],
        fuselages=[pod],
    )


def test_hydro_buildup_drag_identity_and_buoyancy_option():
    """Verify exact drag breakdown identity and include_buoyancy_in_totals behavior."""
    assembly = _build_full_assembly()
    water = Water.seawater()
    op = HydroOperatingPoint(water=water, velocity=12.0, alpha=2.5, surface_z=0.0)

    res_no_b = HydroBuildup(
        assembly=assembly, op_point=op, include_buoyancy_in_totals=False
    ).run_with_stability_derivatives()
    d_sum = (
        res_no_b["D_profile"]
        + res_no_b["D_induced"]
        + res_no_b["D_spray"]
        + res_no_b["D_junction"]
        + res_no_b["D_cavitation"]
    )
    assert np.isclose(res_no_b["D"], d_sum, rtol=1e-9, atol=1e-9)
    assert np.isclose(res_no_b["P_drag"], res_no_b["D"] * 12.0, rtol=1e-9)
    assert res_no_b["D_spray"] > 0.0
    assert res_no_b["D_junction"] > 0.0
    assert res_no_b["CLa"] > 2.0
    assert res_no_b["CLh"] > 0.0
    assert res_no_b["L_h"] > 0.0

    res_with_b = HydroBuildup(
        assembly=assembly, op_point=op, include_buoyancy_in_totals=True
    ).run()
    assert np.isclose(
        res_with_b["F_g"][2] - res_no_b["F_g"][2],
        res_no_b["buoyancy_force"],
        rtol=1e-9,
    )


def test_spray_drag_mesh_refinement_invariance():
    """Subdividing a surface-piercing strut must preserve spray drag within 2%."""
    water = Water.seawater()
    op = HydroOperatingPoint(water=water, velocity=12.0, alpha=0.0, surface_z=0.0)

    strut_coarse = Strut.from_dimensions(
        root_xyz=[0.0, 0.0, -0.70],
        length=1.0,
        root_chord=0.14,
        tip_chord=0.10,
    )
    strut_fine = strut_coarse.subdivide_sections(5)

    r_coarse = HydroBuildup(
        assembly=HydrofoilAssembly(struts=[strut_coarse]), op_point=op
    ).run()
    r_fine = HydroBuildup(
        assembly=HydrofoilAssembly(struts=[strut_fine]), op_point=op
    ).run()

    assert r_coarse["D_spray"] > 1.0
    assert np.isclose(r_coarse["D_spray"], r_fine["D_spray"], rtol=0.02)


def test_hydro_buildup_vectorized_speed_sweep():
    """Vectorized speed sweep should drop cavitation margin and engage cavitation drag."""
    assembly = _build_full_assembly()
    water = Water.seawater()
    op = HydroOperatingPoint(
        water=water,
        velocity=np.linspace(8.0, 28.0, 6),
        alpha=3.0,
        surface_z=0.0,
    )
    res = HydroBuildup(assembly=assembly, op_point=op).run()

    assert res["L"].shape == (6,)
    assert res["cavitation_margin"].shape == (6,)
    assert np.all(np.diff(res["cavitation_margin"]) < 0)
    assert res["cavitation_margin"][0] > 0.0
    assert res["cavitation_margin"][-1] < 0.0
    assert res["D_cavitation"][-1] > res["D_cavitation"][0]


def test_hydro_buildup_opti_mdo_convergence():
    """Solve end-to-end CasADi Opti stern T-foil sizing and optional stabilized trim."""
    from hydrosandbox.examples.optimize_hydrofoil_assembly import run_optimization

    out = run_optimization(verbose=False)
    b = out["buildup"]
    weight_N = out["weight_N"]

    assert len(out["assembly"].foils) == 1
    assert len(out["assembly"].struts) == 1
    assert len(out["assembly"].fuselages) == 1
    assert np.isclose(b["L"] + b["buoyancy_force"], weight_N, rtol=1e-2)
    assert b["cavitation_margin"] >= 0.14
    assert b["ventilation_margin"] >= 0.09
    assert b["L"] / b["D"] > 14.0
    assert 1.0 < out["design"]["wing_incidence"] < 8.0
    assert np.isclose(b["P_drag"], b["D"] * out["velocity"], rtol=1e-9)
    assert np.isclose(out["vlm"]["P_drag"], out["vlm"]["D"] * out["velocity"], rtol=1e-9)
    assert np.isclose(out["lifting_line"]["P_drag"], out["lifting_line"]["D"] * out["velocity"], rtol=1e-9)

    out_stab = run_optimization(include_stabilizer=True, verbose=False)
    bs = out_stab["buildup"]
    assert len(out_stab["assembly"].foils) == 2
    assert np.isclose(bs["L"] + bs["buoyancy_force"], out_stab["weight_N"], rtol=1e-2)
    assert np.isclose(bs["m_b"], 0.0, atol=2.0)
    assert bs["Cma"] < 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_buildup_free_surface_induced_drag_matches_vlm():
    """
    The buildup's free-surface induced-drag multiplier must track the method-of-images
    VLM (which Prandtl's biplane factor reproduces) across submergence depths.
    """
    from hydrosandbox.hydrodynamics import HydroVLM
    from hydrosandbox.library import free_surface_induced_drag_factor

    span, chord = 0.9, 0.15

    def assembly_at(z: float) -> HydrofoilAssembly:
        airfoil = asb.Airfoil("naca2412")
        return HydrofoilAssembly(
            foils=[
                Foil(
                    symmetric=True,
                    xsecs=[
                        asb.WingXSec(xyz_le=[0.0, 0.0, z], chord=chord, airfoil=airfoil),
                        asb.WingXSec(
                            xyz_le=[0.0, span / 2.0, z], chord=chord, airfoil=airfoil
                        ),
                    ],
                )
            ]
        )

    def vlm_induced_drag_parameter(depth: float | None) -> float:
        op = HydroOperatingPoint(
            velocity=10.0, alpha=3.0, surface_z=None if depth is None else 0.0
        )
        res = HydroVLM(
            assembly=assembly_at(-0.6 if depth is None else -depth),
            op_point=op,
            spanwise_resolution=8,
            chordwise_resolution=6,
        ).run()
        return res["CD"] / res["CL"] ** 2

    k_deep = vlm_induced_drag_parameter(None)
    for h_over_b in (0.08, 0.17, 0.33, 0.67):
        k_vlm = vlm_induced_drag_parameter(h_over_b * span) / k_deep
        k_buildup = float(free_surface_induced_drag_factor(h_over_b=h_over_b))
        assert k_buildup == pytest.approx(k_vlm, abs=0.05)

    assert float(free_surface_induced_drag_factor(h_over_b=5.0)) == pytest.approx(
        1.0, abs=1e-3
    )
