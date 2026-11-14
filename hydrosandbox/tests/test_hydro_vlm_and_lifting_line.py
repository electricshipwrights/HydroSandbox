import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import Foil, HydrofoilAssembly, Strut
from hydrosandbox.hydrodynamics import (
    HydroLiftingLine,
    HydroVLM,
    calculate_induced_velocity_horseshoe_hydro,
)
from hydrosandbox.performance import HydroOperatingPoint


def _simple_foil_assembly(z_foil: float = -0.6) -> HydrofoilAssembly:
    main_foil = Foil(
        name="Main Foil",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, z_foil], chord=0.20, airfoil=asb.Airfoil("naca0012")),
            asb.WingXSec(xyz_le=[0.0, 0.60, z_foil], chord=0.20, airfoil=asb.Airfoil("naca0012")),
        ],
    )
    return HydrofoilAssembly(name="Simple Foil", xyz_ref=[0.05, 0.0, z_foil], foils=[main_foil])


def test_method_of_images_boundary_conditions():
    """Verify phi=0 (u=v=0) for high_froude and dphi/dz=0 (w=0) for rigid_lid at z=surface_z."""
    rng = np.random.default_rng(42)
    pts_fs = np.column_stack(
        [
            rng.uniform(-1.0, 3.0, size=25),
            rng.uniform(-2.0, 2.0, size=25),
            np.zeros(25),
        ]
    )

    u_hf, v_hf, w_hf = calculate_induced_velocity_horseshoe_hydro(
        x_field=pts_fs[:, 0],
        y_field=pts_fs[:, 1],
        z_field=pts_fs[:, 2],
        x_left=0.05,
        y_left=-0.5,
        z_left=-0.4,
        x_right=0.05,
        y_right=0.5,
        z_right=-0.4,
        gamma=1.8,
        surface_z=0.0,
        image_weight=1.0,
    )
    assert np.max(np.abs(u_hf)) < 1e-11
    assert np.max(np.abs(v_hf)) < 1e-11
    assert np.max(np.abs(w_hf)) > 1e-3

    u_rl, v_rl, w_rl = calculate_induced_velocity_horseshoe_hydro(
        x_field=pts_fs[:, 0],
        y_field=pts_fs[:, 1],
        z_field=pts_fs[:, 2],
        x_left=0.05,
        y_left=-0.5,
        z_left=-0.4,
        x_right=0.05,
        y_right=0.5,
        z_right=-0.4,
        gamma=1.8,
        surface_z=0.0,
        image_weight=-1.0,
    )
    assert np.max(np.abs(w_rl)) < 1e-11
    assert np.max(np.abs(v_rl)) > 1e-3


def test_vertical_strut_rigid_lid_vs_double_span_mirror():
    """Vertical strut touching rigid-lid surface must match half the load of a double-span strut."""
    water = Water.seawater()
    airfoil = asb.Airfoil("naca0012")

    strut_half = Strut.from_dimensions(
        name="Half Strut",
        root_xyz=[0.0, 0.0, -0.8],
        length=0.8,
        root_chord=0.16,
        tip_chord=0.16,
        airfoil=airfoil,
    )
    assembly_half = HydrofoilAssembly(
        name="Half", xyz_ref=[0.04, 0.0, 0.0], struts=[strut_half]
    )
    op_half = HydroOperatingPoint(
        water=water, velocity=10.0, alpha=0.0, beta=4.0, surface_z=0.0
    )

    strut_double = Strut.from_dimensions(
        name="Double Strut",
        root_xyz=[0.0, 0.0, -0.8],
        length=1.6,
        root_chord=0.16,
        tip_chord=0.16,
        airfoil=airfoil,
    )
    assembly_double = HydrofoilAssembly(
        name="Double", xyz_ref=[0.04, 0.0, 0.0], struts=[strut_double]
    )
    op_double = HydroOperatingPoint(
        water=water, velocity=10.0, alpha=0.0, beta=4.0, surface_z=None
    )

    vlm_half = HydroVLM(
        assembly=assembly_half,
        op_point=op_half,
        spanwise_resolution=8,
        chordwise_resolution=4,
        free_surface_mode="rigid_lid",
        include_wave_drag=False,  # pure image-method equivalence
        clip_to_free_surface=False,
    ).run()
    vlm_double = HydroVLM(
        assembly=assembly_double,
        op_point=op_double,
        spanwise_resolution=16,
        chordwise_resolution=4,
        spanwise_spacing_function=np.linspace,
        free_surface_mode="none",
        clip_to_free_surface=False,
    )
    # Match exact symmetric paneling across z=0 for strict floating-point comparison
    vlm_half_lin = HydroVLM(
        assembly=assembly_half,
        op_point=op_half,
        spanwise_resolution=8,
        chordwise_resolution=4,
        spanwise_spacing_function=np.linspace,
        free_surface_mode="rigid_lid",
        include_wave_drag=False,
        clip_to_free_surface=False,
    ).run()
    vlm_double_res = vlm_double.run()

    assert np.isclose(vlm_half_lin["Y"], 0.5 * vlm_double_res["Y"], rtol=1e-10)
    assert np.isclose(vlm_half_lin["D"], 0.5 * vlm_double_res["D"], rtol=1e-10)
    assert abs(vlm_half["Y"]) > 10.0

    # Same symmetry holds closely in HydroLiftingLine without spray drag
    ll_half = HydroLiftingLine(
        assembly=assembly_half,
        op_point=op_half,
        spanwise_resolution=8,
        spanwise_spacing_function=np.linspace,
        free_surface_mode="rigid_lid",
        include_wave_drag=False,
        clip_to_free_surface=False,
        include_spray_drag=False,
    ).run()
    ll_double = HydroLiftingLine(
        assembly=assembly_double,
        op_point=op_double,
        spanwise_resolution=16,
        spanwise_spacing_function=np.linspace,
        free_surface_mode="none",
        clip_to_free_surface=False,
        include_spray_drag=False,
    ).run()
    assert np.isclose(ll_half["Y"], 0.5 * ll_double["Y"], rtol=1e-4)


def test_hydro_vlm_deep_convergence_and_free_surface_effects():
    """Verify deep-submergence match to VortexLatticeMethod and shallow high-Froude lift drop."""
    water = Water.seawater()
    assembly = _simple_foil_assembly(z_foil=-0.6)

    op_deep = HydroOperatingPoint(water=water, velocity=10.0, alpha=4.0, surface_z=200.0)
    vlm_hydro_deep = HydroVLM(
        assembly=assembly,
        op_point=op_deep,
        spanwise_resolution=8,
        chordwise_resolution=6,
        free_surface_mode="high_froude",
    ).run()
    vlm_asb = asb.VortexLatticeMethod(
        airplane=assembly,
        op_point=op_deep,
        spanwise_resolution=8,
        chordwise_resolution=6,
    ).run()

    assert np.isclose(vlm_hydro_deep["CL"], vlm_asb["CL"], rtol=1e-4)
    assert np.isclose(vlm_hydro_deep["CD"], vlm_asb["CD"], rtol=1e-4)
    assert np.isclose(vlm_hydro_deep["Cm"], vlm_asb["Cm"], rtol=1e-4)

    # Shallow high-Froude free surface reduces lift vs deep water; rigid lid increases lift
    op_shallow = HydroOperatingPoint(
        water=water, velocity=10.0, alpha=4.0, surface_z=-0.50  # h/c = 0.5
    )
    vlm_hf_obj = HydroVLM(
        assembly=assembly,
        op_point=op_shallow,
        spanwise_resolution=8,
        chordwise_resolution=6,
        free_surface_mode="high_froude",
    )
    vlm_hf = vlm_hf_obj.run_with_stability_derivatives()
    vlm_rl = HydroVLM(
        assembly=assembly,
        op_point=op_shallow,
        spanwise_resolution=8,
        chordwise_resolution=6,
        free_surface_mode="rigid_lid",
    ).run()

    assert vlm_hf["CL"] < vlm_hydro_deep["CL"] < vlm_rl["CL"]
    assert vlm_hf["CLh"] > 0.0
    assert vlm_hf["L_h"] > 0.0
    assert vlm_hf["cavitation_margin"] > 0.0
    assert vlm_hf["V_cavitation_inception"] > 10.0

    fig, _ = vlm_hf_obj.draw(backend="matplotlib", show=False)
    plt.close(fig)


def test_hydro_lifting_line_decomposition_cavitation_and_draw():
    """Verify HydroLiftingLine drag decomposition, surface clipping, and matplotlib plot."""
    water = Water.seawater()
    main = Foil(
        name="Main Foil",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, -0.5], chord=0.16, airfoil=asb.Airfoil("naca2412")),
            asb.WingXSec(xyz_le=[0.01, 0.45, -0.5], chord=0.11, airfoil=asb.Airfoil("naca2412")),
        ],
    )
    strut = Strut.from_dimensions(
        name="Mast",
        root_xyz=[0.02, 0.0, -0.5],
        length=0.80,
        root_chord=0.14,
        tip_chord=0.12,
    )
    assembly = HydrofoilAssembly(
        name="T-Foil", xyz_ref=[0.04, 0.0, -0.4], foils=[main], struts=[strut]
    )
    op = HydroOperatingPoint(water=water, velocity=11.0, alpha=2.5, surface_z=0.0)

    ll = HydroLiftingLine(
        assembly=assembly,
        op_point=op,
        spanwise_resolution=5,
        free_surface_mode="high_froude",
        clip_to_free_surface=True,
        include_spray_drag=True,
        include_cavitation_penalty=True,
    )
    res = ll.run_with_stability_derivatives()

    d_sum = (
        res["D_induced"]
        + res["D_profile"]
        + res["D_spray"]
        + res["D_fuselage"]
        + res["D_cavitation"]
        + res["D_wave"]
    )
    assert np.isclose(res["D"], d_sum, rtol=1e-9, atol=1e-9)
    assert res["D_spray"] > 0.0
    assert res["L"] > 100.0
    assert res["CLh"] > 0.0
    assert res["cavitation_margin"] > 0.0
    assert res["buoyancy_force"] > 0.0

    fig, _ = ll.draw(backend="matplotlib", show=False)
    plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _rectangular_foil(z: float, airfoils=("naca2412", "naca2412"), x: float = 0.0):
    return Foil(
        name=f"Foil x={x}",
        symmetric=False,
        xsecs=[
            asb.WingXSec(
                xyz_le=[x, 0.0, z], chord=0.15, airfoil=asb.Airfoil(airfoils[0])
            ),
            asb.WingXSec(
                xyz_le=[x, 0.9, z], chord=0.15, airfoil=asb.Airfoil(airfoils[1])
            ),
        ],
    )


def test_vlm_cavitation_margin_is_mesh_convergent():
    """Suction peak must not track the (singular) leading-edge panel loading."""
    assembly = HydrofoilAssembly(foils=[_rectangular_foil(z=-0.6)])
    op = HydroOperatingPoint(velocity=10.0, alpha=3.0, surface_z=0.0)

    v_inc = [
        HydroVLM(
            assembly=assembly,
            op_point=op,
            spanwise_resolution=6,
            chordwise_resolution=n_c,
        ).run()["V_cavitation_inception"]
        for n_c in (4, 8, 16)
    ]
    assert abs(v_inc[1] - v_inc[0]) / v_inc[0] < 0.03
    assert abs(v_inc[2] - v_inc[1]) / v_inc[1] < 0.015


def test_vlm_panel_thickness_matches_each_foil():
    """Panel t/c on each foil must come from that foil's own (blended) airfoils."""
    pairs = [("naca0006", "naca0008"), ("naca0020", "naca0024"), ("naca0012", "naca0015")]
    assembly = HydrofoilAssembly(
        foils=[
            _rectangular_foil(z=-0.6, airfoils=pair, x=float(i))
            for i, pair in enumerate(pairs)
        ]
    )
    op = HydroOperatingPoint(velocity=10.0, alpha=2.0, surface_z=0.0)
    vlm = HydroVLM(
        assembly=assembly, op_point=op, spanwise_resolution=4, chordwise_resolution=2
    )
    vlm.run()

    tc_by_foil = vlm.panel_t_over_c.reshape(len(pairs), -1)
    for tc, (root, tip) in zip(tc_by_foil, pairs):
        tc_root = int(root[-2:]) / 100.0
        tc_tip = int(tip[-2:]) / 100.0
        assert np.all(np.diff(tc) >= -1e-12)
        assert tc_root - 0.005 <= tc.min() and tc.max() <= tc_tip + 0.005
