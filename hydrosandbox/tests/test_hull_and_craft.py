"""
The hull port (boatmath `hullgeom` / `thinship` -> Python) and the coupled
hull-plus-foil craft model.
"""

import numpy as onp
import pytest
import aerosandbox as asb
import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup, Hull
from hydrosandbox.library import kelvin_wave_resistance

G = 9.80665
# boatmath's fresh water at 15 C (`Fluid::FRESHWATER_15C`).
RHO_FRESH, NU_FRESH = 999.1, 1.1386e-6

# Output of boatmath's `cargo run --release -p thinship --example wigley_curve`
# (commit 2d4b3ac): Wigley hull L = 10 m, B = 1 m, T = 0.625 m, with
# S = 14.877 m^2 and V = 2.778 m^3. Columns: Froude number -> (R_w, R_v) [N].
BOATMATH_WIGLEY = {
    0.20: (25.88, 79.61),
    0.26: (41.75, 128.88),
    0.30: (140.50, 167.68),
    0.40: (318.85, 284.80),
    0.50: (823.17, 429.83),
}


def scalar(value) -> float:
    return float(onp.ravel(value)[0])


def test_wigley_hydrostatics_match_closed_forms_and_boatmath():
    hull = Hull.wigley(10.0, 1.0, 0.625, n_stations=81, n_levels=41)
    statics = hull.hydrostatics()
    assert scalar(statics["volume"]) == pytest.approx(4 / 9 * 10 * 1 * 0.625, rel=2e-3)
    assert scalar(statics["z_cb"]) == pytest.approx(-3 / 8 * 0.625, rel=2e-3)
    assert scalar(statics["x_cb"]) == pytest.approx(0.0, abs=1e-6)
    assert scalar(statics["wetted_surface"]) == pytest.approx(14.877, rel=1e-3)
    assert not hull.has_transom


@pytest.mark.parametrize("froude", sorted(BOATMATH_WIGLEY))
def test_wigley_resistance_matches_boatmath(froude):
    hull = Hull.wigley(10.0, 1.0, 0.625)
    u = froude * onp.sqrt(G * 10.0)
    r_wave = kelvin_wave_resistance(
        velocity=u,
        density=RHO_FRESH,
        length_scales=(1.0, 15.0, 0.02),
        **hull.wave_sources(u),
    )
    r_viscous = hull.viscous_resistance(
        u, RHO_FRESH, NU_FRESH, hull.hydrostatics()["wetted_surface"]
    )
    want_wave, want_viscous = BOATMATH_WIGLEY[froude]
    assert scalar(r_wave) == pytest.approx(want_wave, rel=1e-2)
    assert scalar(r_viscous) == pytest.approx(want_viscous, rel=2e-3)


def _wigley_mesh(l=10.0, b=1.0, t=0.625, n=60):
    """One side of a Wigley hull, wall-sided above the waterline, bow at high x."""
    xi = onp.linspace(-1.0, 1.0, n + 1)
    z = onp.concatenate([onp.linspace(-t, 0.0, n // 2 + 1), [0.3]])
    xx, zz = onp.meshgrid(xi, z, indexing="ij")
    yy = (b / 2) * (1 - xx**2) * (1 - (onp.minimum(zz, 0.0) / t) ** 2)
    vertices = onp.stack([xx * l / 2, yy, zz], axis=-1).reshape(-1, 3)
    idx = onp.arange(vertices.shape[0]).reshape(xx.shape)
    quads = onp.stack(
        [idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1
    ).reshape(-1, 4)
    return vertices, onp.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def test_hull_cut_from_mesh_and_from_boatmath_records():
    exact = 4 / 9 * 10 * 1 * 0.625
    vertices, triangles = _wigley_mesh()

    from_mesh = Hull.from_mesh(vertices, triangles, x_forward=True)
    assert scalar(from_mesh.hydrostatics()["volume"]) == pytest.approx(exact, rel=1e-2)

    mesh_record = {
        "kind": "mesh",
        "hulls": [{"vertices": vertices.tolist(), "triangles": triangles.tolist()}],
    }
    from_record = Hull.from_boatmath_geometry(mesh_record)
    assert onp.allclose(from_record.half_breadths, from_mesh.half_breadths)

    # The underwater Wigley surface is exactly one biquadratic Bezier patch.
    l, b, t = 10.0, 1.0, 0.625
    ctrl = [
        [x, (b / 2) * yu * yv, z]
        for x, yu in zip([-l / 2, 0.0, l / 2], [0.0, 2.0, 0.0])
        for z, yv in zip([0.0, -t / 2, -t], [1.0, 1.0, 0.0])
    ]
    nurbs_record = {
        "kind": "nurbs",
        "hulls": [
            {
                "patches": [
                    {
                        "degree": [2, 2],
                        "n_ctrl": [3, 3],
                        "knots_u": [0, 0, 0, 1, 1, 1],
                        "knots_v": [0, 0, 0, 1, 1, 1],
                        "ctrl": ctrl,
                        "trim_uv": None,
                    }
                ]
            }
        ],
    }
    from_nurbs = Hull.from_boatmath_geometry(nurbs_record)
    # This patch stops at the waterline (no topsides), so the smoothed waterline
    # cut gives up a few millimetres of draft: allow 2%.
    assert scalar(from_nurbs.hydrostatics()["volume"]) == pytest.approx(exact, rel=2e-2)


def test_transom_closure_keeps_wave_resistance_finite_and_ordered():
    """A hull cut off at midships: closing its transom over a longer hollow radiates less."""
    full = Hull.wigley(10.0, 1.0, 0.625, n_stations=41, n_levels=13)
    half = len(full.stations) // 2 + 1

    def resistance(closure):
        hull = Hull(
            full.stations[:half],
            full.levels,
            full.half_breadths[:half],
            transom_closure=closure,
        )
        assert hull.has_transom
        u = 4.0
        return scalar(
            kelvin_wave_resistance(
                velocity=u, density=1000.0, length_scales=(1.0, 20.0, 0.02),
                **hull.wave_sources(u),
            )
        )

    short, auto, long = resistance(0.5), resistance("auto"), resistance(30.0)
    assert short > auto > 0.0
    assert short > long > 0.0


def _small_craft():
    water = hsb.Water.seawater()
    hull = Hull.wigley(5.0, 0.8, 0.25, freeboard=0.3, n_stations=21, n_levels=13)
    mass = scalar(hull.hydrostatics()["volume"]) * water.density()
    return water, hull, mass


def _t_foil(x_foil: float) -> hsb.HydrofoilAssembly:
    airfoil = asb.Airfoil("naca2412")
    foil = hsb.Foil(
        name="Main",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[x_foil, 0, -0.5], chord=0.16, twist=2.0, airfoil=airfoil),
            asb.WingXSec(xyz_le=[x_foil + 0.02, 0.6, -0.5], chord=0.10, twist=2.0, airfoil=airfoil),
        ],
    )
    strut = hsb.Strut.from_dimensions(
        root_xyz=[x_foil + 0.02, 0, -0.5], length=0.48, root_chord=0.12, tip_chord=0.12
    )
    return hsb.HydrofoilAssembly(name="T-foil", foils=[foil], struts=[strut])


def test_bare_hull_floats_at_its_design_waterline():
    water, hull, mass = _small_craft()
    common = dict(hull=hull, velocity=0.3, mass=mass, xyz_cg=[0.0, 0.0, 0.1], water=water)
    res = CraftBuildup.solve_equilibrium(include_hull_dynamics=False, **common)
    assert res["sinkage"] == pytest.approx(0.0, abs=1e-4)
    assert res["trim"] == pytest.approx(0.0, abs=1e-3)
    assert scalar(res["buoyancy_hull"]) == pytest.approx(mass * G, rel=1e-6)

    # With its own pressure field the hull squats, by about 0.02 Fn^2 L at low speed
    # (Havelock; boatmath's Wigley coefficient is 0.021).
    squat = CraftBuildup.solve_equilibrium(**common)
    froude = 0.3 / onp.sqrt(G * hull.length)
    assert squat["sinkage"] / hull.length / froude**2 == pytest.approx(0.02, rel=0.5)
    assert scalar(squat["F_z_hull_dynamic"]) < 0.0


def test_foil_lifts_the_hull_and_its_position_sets_trim_and_interference():
    water, hull, mass = _small_craft()
    common = dict(
        hull=hull, velocity=4.5, mass=mass, xyz_cg=[0.2, 0.0, 0.1], water=water,
        xyz_thrust=[2.0, 0.0, -0.3],
    )
    bare = CraftBuildup.solve_equilibrium(**common)
    aft = CraftBuildup.solve_equilibrium(assembly=_t_foil(1.5), **common)
    mid = CraftBuildup.solve_equilibrium(assembly=_t_foil(0.0), **common)

    for res in (bare, aft, mid):
        assert abs(scalar(res["F_z_net"])) < 1e-3 * mass * G
        assert abs(scalar(res["M_y_net"])) < 1e-3 * mass * G * hull.length

    # The foil carries part of the weight: the hull rides higher and wets less.
    for res in (aft, mid):
        assert scalar(res["L_foil"]) > 0.1 * mass * G
        assert res["sinkage"] < bare["sinkage"] - 0.01
        assert scalar(res["wetted_surface_hull"]) < scalar(bare["wetted_surface_hull"])
    # Lift aft of the centre of gravity pushes the bow down relative to lift amidships.
    assert aft["trim"] < mid["trim"] - 0.5

    # The wave breakdown is an identity, and interference depends on foil position.
    for res in (aft, mid):
        total = (
            scalar(res["D_wave_hull_alone"])
            + scalar(res["D_wave_foil_alone"])
            + scalar(res["D_wave_interference"])
        )
        assert total == pytest.approx(scalar(res["D_wave"]), rel=1e-9)
    assert abs(scalar(mid["D_wave_interference"]) - scalar(aft["D_wave_interference"])) > 5.0


# Output of a boatmath example calling `thinship::sectional::dynamic_force` on the
# same Wigley hull in sea water (rho = 1025.9), about midships (commit 2d4b3ac):
# Froude number -> (force_up [N], moment_bow_up [N m]).
BOATMATH_WIGLEY_DYNAMIC = {
    0.20: (-610.7129, 22.3468),
    0.30: (-1508.8475, 188.6157),
    0.40: (-3451.4538, 4270.7062),
    0.50: (-4132.9551, 10672.1180),
}


@pytest.mark.parametrize("froude", sorted(BOATMATH_WIGLEY_DYNAMIC))
def test_wigley_dynamic_sinkage_force_and_trim_moment_match_boatmath(froude):
    hull = Hull.wigley(10.0, 1.0, 0.625)
    u = froude * onp.sqrt(G * 10.0)
    dynamic = hull.dynamic_force(u, 1025.9)
    want_force, want_moment = BOATMATH_WIGLEY_DYNAMIC[froude]
    assert scalar(dynamic["force_up"]) == pytest.approx(want_force, rel=1e-2)
    assert scalar(dynamic["moment_bow_up"]) == pytest.approx(want_moment, rel=1.5e-2)
    # Fore-aft symmetric hull: the local part of the moment vanishes; trim is all wave.
    assert abs(scalar(dynamic["moment_local"])) < 1e-6 * abs(want_moment)
    # Moving the reference 1 m forward adds the (downward) force's moment, bow-up.
    forward = hull.dynamic_force(u, 1025.9, x_ref=-1.0)
    assert scalar(forward["moment_bow_up"]) == pytest.approx(
        scalar(dynamic["moment_bow_up"]) - scalar(dynamic["force_up"]), rel=1e-6
    )


def _transom_hull() -> Hull:
    """A Wigley forebody cut off at 70% of its length: an immersed transom."""
    full = Hull.wigley(6.0, 0.8, 0.3, freeboard=0.3, n_stations=41, n_levels=17)
    keep = int(0.7 * len(full.stations))
    return Hull(full.stations[:keep], full.levels, full.half_breadths[:keep])


def test_station_contracted_amplitude_equals_per_cell_sources():
    hull = _transom_hull()
    attitude = dict(sinkage=0.02, trim=1.5, pivot_xz=(0.5, 0.1))
    u, scales = 3.5, (1.0, 9.0, 0.02)
    per_cell = kelvin_wave_resistance(
        velocity=u, density=1000.0, length_scales=scales,
        **hull.wave_sources(u, **attitude),
    )
    contracted = kelvin_wave_resistance(
        xyz_a=([], [], []), xyz_b=([], [], []), velocity=u, density=1000.0,
        length_scales=scales, extra_source_amplitude=hull.wave_amplitude(u, **attitude),
    )
    # Identical but for the trimmed cells' fore-aft lean, which the contraction drops.
    assert scalar(contracted) == pytest.approx(scalar(per_cell), rel=2e-2)


def test_transom_hollow_stays_level_when_the_hull_trims():
    """
    The closing hollow carries the transom's immersed section straight aft: its
    sources must not dive with a bow-up trim.
    """
    hull = _transom_hull()
    strips = hull._strips(4.0, G, sinkage=0.0, trim=3.0, pivot_xz=(0.0, 0.0))
    transom = strips["transom"]
    assert scalar(transom["hollow_length"]) > 0.5
    z_lo = onp.asarray(strips["z_lo_hollow"]).reshape(strips["n_hollow"], -1)
    assert onp.allclose(z_lo, onp.asarray(transom["z_lo"]).reshape(1, -1))
    assert onp.all(onp.diff(onp.ravel(strips["x_hollow"])) > 0)


def test_symbolic_attitude_reproduces_numeric_and_solves_in_opti():
    import casadi as cas

    hull = _transom_hull()
    u, rho, pivot = 3.5, 1025.0, (0.5, 0.1)
    s, t = cas.MX.sym("sinkage"), cas.MX.sym("trim")
    at = (0.015, 1.2)

    def evaluate(expr):
        return float(cas.Function("f", [s, t], [expr])(*at))

    numeric = hull.dynamic_force(u, rho, G, at[0], at[1], pivot)
    symbolic = hull.dynamic_force(u, rho, G, s, t, pivot)
    for key in ("force_up", "moment_local", "moment_wave"):
        assert evaluate(symbolic[key]) == pytest.approx(scalar(numeric[key]), rel=1e-9)

    # Equilibrium as two Opti constraints agrees with the Newton solve.
    water = hsb.Water.seawater()
    statics = hull.hydrostatics()
    mass = 0.9 * scalar(statics["volume"]) * water.density()
    common = dict(
        hull=hull, velocity=2.5, mass=mass, water=water,
        xyz_cg=[scalar(statics["x_cb"]), 0.0, 0.1],  # over the centre of buoyancy
        include_interference_breakdown=False,
    )
    newton = CraftBuildup.solve_equilibrium(**common)

    opti = asb.Opti()
    # (started near the Newton attitude, as a design loop would be)
    sinkage = opti.variable(
        init_guess=newton["sinkage"] + 0.003, lower_bound=-0.2, upper_bound=0.2
    )
    trim = opti.variable(
        init_guess=newton["trim"] + 0.3, lower_bound=-15.0, upper_bound=15.0
    )
    res = CraftBuildup(sinkage=sinkage, trim=trim, **common).run()
    weight = mass * G
    opti.subject_to(
        [res["F_z_net"] / weight == 0, res["M_y_net"] / (weight * hull.length) == 0]
    )
    opti.minimize(0)
    sol = opti.solve(verbose=False, options={"ipopt.hessian_approximation": "limited-memory"})
    assert sol(sinkage) == pytest.approx(newton["sinkage"], abs=2e-4)
    assert sol(trim) == pytest.approx(newton["trim"], abs=2e-2)


def test_static_stability_at_rest_is_the_hydrostatic_metacentric_height():
    """
    At rest, heave stiffness is rho g A_wp and the heave-free pitch stiffness is
    m g GM_L with GM_L = I_L / V + z_cb - z_cg (closed forms for the Wigley hull).
    """
    water = hsb.Water.seawater()
    l, b, t, z_cg = 5.0, 0.8, 0.25, 0.1
    hull = Hull.wigley(l, b, t, freeboard=0.3, n_stations=41, n_levels=33)
    mass = scalar(hull.hydrostatics()["volume"]) * water.density()
    res = CraftBuildup(
        hull=hull, velocity=0.05, mass=mass, xyz_cg=[0.0, 0.0, z_cg], water=water,
        include_hull_dynamics=False,
    ).run_with_stability()

    waterplane = 2.0 / 3.0 * l * b
    second_moment = b * 4.0 * (l / 2.0) ** 3 / 15.0
    gm_l = second_moment / (4.0 / 9.0 * l * b * t) - 3.0 / 8.0 * t - z_cg
    assert scalar(res["k_heave"]) == pytest.approx(
        water.density() * G * waterplane, rel=2e-2
    )
    assert scalar(res["GM_L"]) == pytest.approx(gm_l, rel=3e-2)
    assert scalar(res["k_pitch"]) == pytest.approx(mass * G * gm_l, rel=3e-2)


def test_lifting_the_hull_costs_pitch_stiffness_and_an_aft_foil_gives_some_back():
    water, hull, mass = _small_craft()
    common = dict(
        hull=hull, velocity=4.5, mass=mass, xyz_cg=[0.2, 0.0, 0.1], water=water,
        include_interference_breakdown=False,
    )

    def stiffness(assembly):
        attitude = CraftBuildup.solve_equilibrium(assembly=assembly, **common)
        return scalar(
            CraftBuildup(
                assembly=assembly, sinkage=attitude["sinkage"], trim=attitude["trim"], **common
            ).run_with_stability()["k_pitch"]
        )

    bare, amidships, aft = stiffness(None), stiffness(_t_foil(0.0)), stiffness(_t_foil(1.5))
    assert bare > 0.0
    # A lifting foil under the centre of gravity unloads the waterplane and adds no
    # pitch stiffness of its own; the same foil well aft adds its own back.
    assert amidships < bare
    assert aft > amidships
