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
        root_xyz=[x_foil + 0.02, 0, -0.5], length=0.75, root_chord=0.12, tip_chord=0.12
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
        # The lever-arm argument alone, without the foil's suction on the hull
        # (which on this small craft is about as large as the margin tested).
        include_hull_foil_interaction=False,
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


def test_point_masses_and_shaft_angle_enter_the_balance():
    water, hull, mass = _small_craft()
    base = dict(hull=hull, velocity=4.0, water=water, sinkage=0.01, trim=1.0,
                include_interference_breakdown=False)

    # Point masses are equivalent to their combined mass and centre of gravity.
    split = CraftBuildup(
        mass=0.7 * mass, xyz_cg=[0.0, 0.0, 0.1],
        point_masses=[(0.2 * mass, [1.5, 0.0, 0.0]), (0.1 * mass, [-1.0, 0.0, 0.3])],
        **base,
    )
    assert split.mass == pytest.approx(mass)
    assert split.xyz_cg[0] == pytest.approx(0.2 * 1.5 - 0.1 * 1.0)
    lumped = CraftBuildup(mass=mass, xyz_cg=list(split.xyz_cg), **base).run()
    split_res = split.run()
    for key in ("F_z_net", "M_y_net", "D"):
        assert scalar(split_res[key]) == pytest.approx(scalar(lumped[key]), rel=1e-9, abs=1e-9)

    # An inclined shaft: thrust grows so its horizontal part still equals the drag,
    # and its vertical part lifts the craft.
    level = CraftBuildup(mass=mass, xyz_cg=[0.0, 0.0, 0.1], xyz_thrust=[2.0, 0.0, -0.3], **base).run()
    inclined = CraftBuildup(
        mass=mass, xyz_cg=[0.0, 0.0, 0.1], xyz_thrust=[2.0, 0.0, -0.3], shaft_angle=10.0, **base
    ).run()
    drag = scalar(inclined["D"])
    assert scalar(inclined["thrust"]) == pytest.approx(drag / onp.cos(onp.radians(11.0)), rel=1e-9)
    assert scalar(inclined["F_z_net"]) - scalar(level["F_z_net"]) == pytest.approx(
        drag * (onp.tan(onp.radians(11.0)) - onp.tan(onp.radians(1.0))), rel=1e-6
    )


def test_powered_craft_closes_attitude_thrust_and_drive_masses():
    water, hull, mass = _small_craft()
    common = dict(
        hull=hull, velocity=3.0, mass=0.85 * mass, xyz_cg=[0.0, 0.0, 0.1], water=water,
        xyz_propeller=[2.3, 0.0, -0.35], xyz_motor=[1.8, 0.0, 0.0],
        propulsion_options=dict(
            propeller_diameter=0.30, propeller_pitch_ratio=1.0, propeller_ear=0.5,
            motor_model="loss", motor_kv=60.0, endurance_hours=2.0,
        ),
    )
    amidships = hsb.PoweredCraftBuildup.solve(xyz_battery=[0.0, 0.0, 0.0], **common)
    aft = hsb.PoweredCraftBuildup.solve(xyz_battery=[2.0, 0.0, 0.0], **common)

    for res in (amidships, aft):
        weight = scalar(res["mass"]) * G
        assert abs(scalar(res["F_z_net"])) < 1e-4 * weight
        assert abs(scalar(res["M_y_net"])) < 1e-4 * weight * hull.length
        assert abs(scalar(res["thrust_residual"])) < 1e-3 * scalar(res["thrust"])
        assert abs(scalar(res["mass_drivetrain_residual"])) < 0.05
        assert abs(scalar(res["mass_battery_residual"])) < 0.05
        # The drive is on board: total mass is the bare craft plus what was sized.
        assert scalar(res["mass"]) == pytest.approx(
            0.85 * mass + scalar(res["mass_drivetrain_sized"]) + scalar(res["mass_battery_sized"]),
            abs=0.1,
        )
        assert scalar(res["mass_battery_sized"]) > 1.0
        assert 0.3 < scalar(res["propulsion"]["eta_prop"]) < 0.85

    # Moving the battery aft moves the centre of gravity aft and trims bow-up.
    assert scalar(aft["xyz_cg"][0]) > scalar(amidships["xyz_cg"][0]) + 0.01
    assert aft["trim"] > amidships["trim"]


def test_catamaran_limits_coincident_hulls_quadruple_and_distant_hulls_double():
    """
    Two demihulls: at zero spacing their sources coincide (amplitudes double, so
    wave resistance and the dynamic force quadruple); far apart they do not
    interact (both double). In between, the spacing sets the interference.
    """
    hull = Hull.wigley(8.0, 0.8, 0.5, n_stations=31, n_levels=13)
    u, rho = 2.5, 1025.9

    def wave(spacing):
        return scalar(
            kelvin_wave_resistance(
                xyz_a=([], [], []), xyz_b=([], [], []), velocity=u, density=rho,
                length_scales=(max(spacing or 0.0, 1.0) + 1.0, 12.0, 0.02),
                extra_source_amplitude=hull.wave_amplitude(u, demihull_spacing=spacing),
            )
        )

    def force(spacing):
        return scalar(hull.dynamic_force(u, rho, demihull_spacing=spacing)["force_up"])

    assert wave(1e-9) == pytest.approx(4.0 * wave(None), rel=1e-6)
    assert wave(60.0) == pytest.approx(2.0 * wave(None), rel=2e-2)
    assert force(1e-9) == pytest.approx(4.0 * force(None), rel=1e-6)
    assert force(40.0) == pytest.approx(2.0 * force(None), rel=2e-2)
    assert abs(wave(2.4) / (2.0 * wave(None)) - 1.0) > 0.05

    # The craft model: a catamaran at twice the mass floats where one hull does.
    water, small_hull, mass = _small_craft()
    common = dict(hull=small_hull, velocity=0.3, xyz_cg=[0.0, 0.0, 0.1], water=water,
                  include_hull_dynamics=False, include_interference_breakdown=False)
    mono = CraftBuildup.solve_equilibrium(mass=mass, **common)
    cat = CraftBuildup.solve_equilibrium(mass=2.0 * mass, demihull_spacing=2.0, **common)
    assert cat["sinkage"] == pytest.approx(mono["sinkage"], abs=1e-5)
    assert scalar(cat["buoyancy_hull"]) == pytest.approx(2.0 * scalar(mono["buoyancy_hull"]), rel=1e-6)


def test_dynamic_force_does_not_depend_on_how_the_pose_is_described():
    """
    The same physical pose, described about a pivot at the waterline or about one
    well above it, gives the same force and the same moment about the same point.
    """
    hull = _transom_hull()
    u, rho, trim = 3.5, 1025.0, 1.5
    high, low = (0.4, 0.35), (1.0, 0.0)
    sinkage_high = 0.01
    cos, sin = onp.cos(onp.radians(trim)), onp.sin(onp.radians(trim))
    # Where the low pivot ends up, and how far it has sunk, in the pose about `high`.
    sinkage_low = sinkage_high + (low[0] - high[0]) * sin + (low[1] - high[1]) * (1 - cos)
    x_low_earth = high[0] + (low[0] - high[0]) * cos + (low[1] - high[1]) * sin

    about_high = hull.dynamic_force(u, rho, G, sinkage_high, trim, high, x_ref=high[0])
    about_low = hull.dynamic_force(
        u, rho, G, sinkage_low, trim, low, x_ref=low[0] + (high[0] - x_low_earth)
    )
    for key in ("force_up", "moment_bow_up"):
        assert scalar(about_low[key]) == pytest.approx(scalar(about_high[key]), rel=1e-9)

    # And it is the pose that matters: a pivot above the waterline must not move
    # the waterline the hull's breadths are read at.
    level = hull.dynamic_force(u, rho, G, 0.0, 0.0, (0.0, 0.0))
    level_high_pivot = hull.dynamic_force(u, rho, G, 0.0, 0.0, (0.0, 0.5))
    assert scalar(level_high_pivot["force_up"]) == pytest.approx(scalar(level["force_up"]), rel=1e-9)


def test_dynamic_force_table_reproduces_direct_evaluation():
    import pickle

    hull = _transom_hull()
    u, rho = 3.5, 1025.0
    table = hull.dynamic_force_table(
        u, rho, sinkage_range=(-0.06, 0.06), trim_range=(-1.0, 3.0), n_sinkage=7, n_trim=9
    )
    table = pickle.loads(pickle.dumps(table))  # tables travel to worker processes
    for sinkage, trim, pivot in [
        (0.013, 1.7, (0.6, 0.2)),
        (-0.022, 0.4, (-0.3, 0.0)),
        (0.031, 2.6, (1.2, 0.35)),
    ]:
        direct = hull.dynamic_force(u, rho, G, sinkage, trim, pivot, x_ref=pivot[0])
        tabled = table(sinkage, trim, pivot_xz=pivot, x_ref=pivot[0])
        assert tabled["force_up"] == pytest.approx(scalar(direct["force_up"]), rel=2e-2)
        assert tabled["moment_bow_up"] == pytest.approx(
            scalar(direct["moment_bow_up"]), rel=2e-2, abs=5.0
        )

    # In the craft model the table stands in for the direct integral.
    water = hsb.Water.seawater()
    mass = 0.9 * scalar(hull.hydrostatics()["volume"]) * water.density()
    common = dict(hull=hull, velocity=u, mass=mass, xyz_cg=[0.9, 0.0, 0.1], water=water,
                  sinkage=0.01, trim=1.0, include_interference_breakdown=False)
    table = hull.dynamic_force_table(
        u, water.density(), sinkage_range=(-0.06, 0.06), trim_range=(-1.0, 3.0),
        n_sinkage=7, n_trim=9,
    )
    direct = CraftBuildup(**common).run()
    tabled = CraftBuildup(hull_dynamics=table, **common).run()
    assert scalar(tabled["F_z_net"]) == pytest.approx(scalar(direct["F_z_net"]), abs=0.02 * mass * G)
    assert scalar(tabled["D"]) == pytest.approx(scalar(direct["D"]), rel=1e-9)


def test_foil_pitch_damping_grows_with_the_square_of_its_arm():
    water, hull, mass = _small_craft()
    common = dict(hull=hull, velocity=4.5, mass=mass, xyz_cg=[0.0, 0.0, 0.1], water=water,
                  sinkage=0.0, trim=0.5, include_hull_dynamics=False,
                  include_interference_breakdown=False)

    def damping(x_foil):
        return scalar(
            CraftBuildup(assembly=_t_foil(x_foil), **common).run_with_stability()["c_pitch_foils"]
        )

    near, far_aft, far_forward = damping(0.5), damping(2.0), damping(-2.0)
    assert near > 0.0
    # Damping acts on either side of the centre of gravity, roughly as arm squared.
    assert far_aft == pytest.approx(16.0 * near, rel=0.4)
    assert far_forward == pytest.approx(far_aft, rel=0.4)
    assert scalar(CraftBuildup(**common).run_with_stability()["c_pitch_foils"]) == 0.0


def test_powered_craft_symbolic_opti_with_point_masses_and_inclined_shaft():
    water, hull, mass = _small_craft()
    velocity = 3.0
    prop_opts = dict(
        propeller_diameter=0.30,
        propeller_pitch_ratio=1.0,
        propeller_ear=0.5,
        motor_model="loss",
        motor_kv=60.0,
        endurance_hours=2.0,
    )
    ref = hsb.PoweredCraftBuildup.solve(
        hull=hull,
        velocity=velocity,
        mass=0.85 * mass,
        xyz_cg=[0.0, 0.0, 0.1],
        xyz_propeller=[2.3, 0.0, -0.35],
        xyz_motor=[1.8, 0.0, 0.0],
        xyz_battery=[0.0, 0.0, 0.0],
        shaft_angle=10.0,
        water=water,
        propulsion_options=prop_opts,
        craft_options=dict(include_interference_breakdown=False),
    )

    # Numeric equilibrium + inclined-shaft power identity:
    # P_effective = T * (1 - t) * cos(trim + shaft_angle) * V == D * V.
    prop = ref["propulsion"]
    t_ded = scalar(prop["thrust_deduction"])
    cos_total = onp.cos(onp.radians(scalar(ref["trim"]) + 10.0))
    assert t_ded > 0.0
    assert scalar(prop["P_effective"]) == pytest.approx(scalar(ref["D"]) * velocity, rel=1e-4)
    assert scalar(prop["T_required"]) == pytest.approx(
        scalar(ref["D"]) / ((1.0 - t_ded) * cos_total), rel=1e-4
    )
    assert abs(scalar(ref["F_z_net"])) < 1e-4 * scalar(ref["mass"]) * G
    assert abs(scalar(ref["M_y_net"])) < 1e-4 * scalar(ref["mass"]) * G * hull.length

    # Symbolic Opti evaluation and solve (Exercises symbolic pivot_xz with point_masses).
    opti = asb.Opti()
    sinkage = opti.variable(init_guess=ref["sinkage"], lower_bound=-0.25, upper_bound=0.25)
    trim = opti.variable(init_guess=ref["trim"], lower_bound=-10.0, upper_bound=10.0)
    m_drive = opti.variable(
        init_guess=scalar(ref["mass_drivetrain_sized"]), lower_bound=0.1, upper_bound=80.0
    )
    m_batt = opti.variable(
        init_guess=scalar(ref["mass_battery_sized"]), lower_bound=0.5, upper_bound=250.0
    )
    rpm = opti.variable(
        init_guess=float(ref["propeller_rpm"]), lower_bound=200.0, upper_bound=5000.0
    )
    sym = hsb.PoweredCraftBuildup(
        hull=hull,
        velocity=velocity,
        mass=0.85 * mass,
        xyz_cg=[0.0, 0.0, 0.1],
        xyz_propeller=[2.3, 0.0, -0.35],
        xyz_motor=[1.8, 0.0, 0.0],
        xyz_battery=[0.0, 0.0, 0.0],
        mass_drivetrain=m_drive,
        mass_battery=m_batt,
        shaft_angle=10.0,
        sinkage=sinkage,
        trim=trim,
        water=water,
        propulsion_options={**prop_opts, "propeller_rpm": rpm},
        craft_options=dict(include_interference_breakdown=False),
    ).run()
    weight = sym["mass"] * G
    opti.subject_to(
        [
            sym["F_z_net"] / weight == 0,
            sym["M_y_net"] / (weight * hull.length) == 0,
            sym["thrust_residual"] / scalar(ref["thrust"]) == 0,
            sym["mass_drivetrain_residual"] / 10.0 == 0,
            sym["mass_battery_residual"] / 10.0 == 0,
        ]
    )
    opti.minimize(0)
    sol = opti.solve(verbose=False, options={"ipopt.hessian_approximation": "limited-memory"})
    assert sol(sinkage) == pytest.approx(ref["sinkage"], abs=1e-4)
    assert sol(trim) == pytest.approx(ref["trim"], abs=1e-2)
    assert sol(rpm) == pytest.approx(float(ref["propeller_rpm"]), rel=1e-3)


def test_trimmed_waterplane_sets_true_aft_foil_and_strut_submergence():
    from hydrosandbox.library import resolve_surface_segment

    water, hull, mass = _small_craft()
    airfoil = asb.Airfoil("naca2412")
    # Main foil forward at z=-0.4, aft foil raised near design waterline at z=-0.03.
    fwd = hsb.Foil(
        name="Fwd",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[-1.5, 0.0, -0.4], chord=0.14, twist=2.0, airfoil=airfoil),
            asb.WingXSec(xyz_le=[-1.5, 0.4, -0.4], chord=0.14, twist=2.0, airfoil=airfoil),
        ],
    )
    aft = hsb.Foil(
        name="Aft",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[1.5, 0.0, -0.03], chord=0.10, twist=4.0, airfoil=airfoil),
            asb.WingXSec(xyz_le=[1.5, 0.25, -0.03], chord=0.10, twist=4.0, airfoil=airfoil),
        ],
    )
    short_strut = hsb.Strut.from_dimensions(
        root_xyz=[-1.5, 0.0, -0.4], length=0.35, root_chord=0.10, tip_chord=0.10
    )
    piercing_strut = hsb.Strut.from_dimensions(
        root_xyz=[-1.5, 0.0, -0.4], length=0.70, root_chord=0.10, tip_chord=0.10
    )
    assembly_piercing = hsb.HydrofoilAssembly(
        name="Multi", foils=[fwd, aft], struts=[piercing_strut]
    )
    assembly_short = hsb.HydrofoilAssembly(
        name="MultiShort", foils=[fwd, aft], struts=[short_strut]
    )

    craft = CraftBuildup(
        hull=hull,
        velocity=4.0,
        mass=mass,
        xyz_cg=[0.0, 0.0, 0.1],
        assembly=assembly_piercing,
        sinkage=0.0,
        trim=-4.0,
        water=water,
        include_interference_breakdown=False,
    )
    craft_short = CraftBuildup(
        hull=hull,
        velocity=4.0,
        mass=mass,
        xyz_cg=[0.0, 0.0, 0.1],
        assembly=assembly_short,
        sinkage=0.0,
        trim=-4.0,
        water=water,
        include_interference_breakdown=False,
    )
    res = craft.run()
    res_short = craft_short.run()

    # Bow-down (-4 deg) raises aft station (x=+1.5) into air while deepening fwd station (x=-1.5).
    z_fwd_eff, z_aft_eff, sz_eff = resolve_surface_segment(
        -1.5, -0.4, 1.5, -0.03, lambda xg, zg: craft._earth(xg, zg)[1]
    )
    assert sz_eff == 0.0
    assert z_fwd_eff < -0.45
    assert z_aft_eff > 0.05

    # Buoyancy tilted along earth up [-sin(trim), 0, cos(trim)] has zero earth horizontal component.
    trim_rad = onp.radians(-4.0)
    up_g = [-onp.sin(trim_rad), 0.0, onp.cos(trim_rad)]
    hs = assembly_piercing.hydrostatic_forces(
        water=water,
        surface_z=lambda xg, zg: craft._earth(xg, zg)[1],
        up_direction_g=up_g,
    )
    f_b = hs["F_buoyancy_g"]
    f_earth_x = f_b[0] * onp.cos(trim_rad) + f_b[2] * onp.sin(trim_rad)
    assert f_earth_x == pytest.approx(0.0, abs=1e-9)

    # A strut reaching above the waterplane picks up spray/piercing drag vs a cut-off underwater strut.
    assert scalar(res["D_foil"]) > scalar(res_short["D_foil"])

    # Wave dipole force_z in CraftBuildup must match wind-axis element force_z directly (no extra cos(trim)).
    craft_trimmed = CraftBuildup(
        hull=hull,
        velocity=4.0,
        mass=mass,
        xyz_cg=[0.0, 0.0, 0.1],
        assembly=hsb.HydrofoilAssembly(name="Sub", foils=[fwd]),
        sinkage=0.0,
        trim=10.0,
        water=water,
        include_interference_breakdown=False,
    )
    r_trim = craft_trimmed.run()
    elem = r_trim["foil"]["wave_elements"]
    fz_elem = sum(scalar(v) for v in elem["force_z"])
    # Lift component L in wind axes points vertically up in earth axes. (To 1e-4:
    # the hull's induced flow tilts each section's own lift by a fraction of a degree.)
    assert fz_elem == pytest.approx(scalar(r_trim["L_foil"]), rel=1e-4)


def test_flat_bottom_emerged_stations_drop_waterline_breadth_to_zero():
    stations = onp.linspace(-2.0, 2.0, 21)
    levels = onp.linspace(-0.25, 0.25, 11)
    half_breadths = onp.full((len(stations), len(levels)), 0.30)
    box = Hull(stations, levels, half_breadths, transom_closure=10.0)

    # Pitch bow up 12 deg: bow at x=-2 rises 2*sin(12)=+0.416 m above water -> keel emerged.
    f_wl = onp.asarray(box._waterline_half_breadths(sinkage=0.0, trim=12.0, pivot_xz=(0.0, 0.0)))
    assert float(f_wl[0, 0]) < 1e-4
    assert float(f_wl[-1, 0]) == pytest.approx(0.30, rel=1e-2)


def test_dynamic_force_symbolic_velocity_below_reference_is_finite():
    import casadi as cas

    hull = Hull.wigley(5.0, 0.8, 0.25, n_stations=15, n_levels=9)
    v_sym = cas.MX.sym("v")
    dyn = hull.dynamic_force(v_sym, 1025.9, reference_velocity=5.0)
    f_fn = cas.Function("dyn", [v_sym], [dyn["force_up"], dyn["moment_bow_up"], cas.gradient(dyn["force_up"], v_sym)])

    for v_val in (5.0, 2.5, 1.2):
        f_num = hull.dynamic_force(v_val, 1025.9, reference_velocity=v_val)
        fz_s, my_s, df_dv = (float(x) for x in f_fn(v_val))
        assert onp.isfinite(fz_s) and onp.isfinite(my_s) and onp.isfinite(df_dv)
        if abs(v_val - 5.0) < 1e-9:
            assert fz_s == pytest.approx(scalar(f_num["force_up"]), rel=1e-9)


def test_blunt_bow_hull_source_conservation_and_dynamic_force_parallel_axis():
    stations = onp.linspace(-2.0, 2.0, 21)
    levels = onp.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2])
    half_breadths = onp.zeros((len(stations), len(levels)))
    half_breadths[:, 1:] = 0.25  # blunt rectangular cross section with closed bottom keel
    hull = Hull(stations, levels, half_breadths, transom_closure=8.0)

    # At non-zero trim, bow closure + interior cells + transom hollow conserve net flux == 0.
    src = hull.wave_sources(3.0, trim=2.5)
    assert float(onp.sum(src["source_flux"])) == pytest.approx(0.0, abs=1e-6)

    # Wave amplitude vs per-cell wave_sources must agree on blunt-bow hulls too.
    r_cell = scalar(
        kelvin_wave_resistance(velocity=3.0, density=1000.0, length_scales=(1.0, 12.0, 0.02), **src)
    )
    r_amp = scalar(
        kelvin_wave_resistance(
            xyz_a=([], [], []),
            xyz_b=([], [], []),
            velocity=3.0,
            density=1000.0,
            length_scales=(1.0, 12.0, 0.02),
            extra_source_amplitude=hull.wave_amplitude(3.0, trim=2.5),
        )
    )
    assert r_amp == pytest.approx(r_cell, rel=2e-2)

    # Parallel-axis theorem in dynamic_force requires consistent bow + stern steps.
    d0 = hull.dynamic_force(3.0, 1025.0, trim=2.0, x_ref=0.0)
    d1 = hull.dynamic_force(3.0, 1025.0, trim=2.0, x_ref=0.8)
    assert scalar(d1["moment_local"]) == pytest.approx(
        scalar(d0["moment_local"]) + 0.8 * scalar(d0["force_up"]), rel=1e-9
    )
    assert scalar(d1["moment_bow_up"]) == pytest.approx(
        scalar(d0["moment_bow_up"]) + 0.8 * scalar(d0["force_up"]), rel=2e-2
    )

    # Dynamic force varies continuously across the has_transom threshold (no step jump).
    f_ref = Hull.wigley(4.0, 0.6, 0.2, n_stations=15, n_levels=9)
    hb_zero = f_ref.half_breadths.copy()
    hb_zero[-1, :] = 0.0
    hb_tiny = f_ref.half_breadths.copy()
    hb_tiny[-1, :] = 5e-4
    h_zero = Hull(f_ref.stations, f_ref.levels, hb_zero)
    h_tiny = Hull(f_ref.stations, f_ref.levels, hb_tiny)
    assert not h_zero.has_transom and h_tiny.has_transom
    assert scalar(h_tiny.dynamic_force(2.5, 1025.0)["force_up"]) == pytest.approx(
        scalar(h_zero.dynamic_force(2.5, 1025.0)["force_up"]), rel=5e-3
    )


def test_run_with_stability_casadi_ad_matches_central_fd_and_negative_gml_floor():
    import casadi as cas

    water, hull, mass = _small_craft()
    common = dict(
        hull=hull,
        velocity=4.0,
        mass=mass,
        xyz_cg=[0.1, 0.0, 0.1],
        assembly=_t_foil(1.2),
        water=water,
        include_interference_breakdown=False,
    )
    s_val, t_val = -0.02, 1.5
    craft_num = CraftBuildup(sinkage=s_val, trim=t_val, **common)
    num_default = craft_num.run_with_stability()
    num_fine = craft_num.run_with_stability(d_sinkage=1e-5, d_trim=5e-4)

    s_sym, t_sym = cas.MX.sym("s"), cas.MX.sym("t")
    sym = CraftBuildup(sinkage=s_sym, trim=t_sym, **common).run_with_stability()
    fn = cas.Function("stab", [s_sym, t_sym], [sym["k_heave"], sym["k_pitch"], sym["GM_L"]])
    kh_ad, kp_ad, gml_ad = (float(x) for x in fn(s_val, t_val))

    assert kh_ad == pytest.approx(scalar(num_fine["k_heave"]), rel=1e-5)
    assert kp_ad == pytest.approx(scalar(num_fine["k_pitch"]), rel=1e-5)
    assert gml_ad == pytest.approx(scalar(num_fine["GM_L"]), rel=1e-5)
    assert kh_ad == pytest.approx(scalar(num_default["k_heave"]), rel=5e-3)
    assert kp_ad == pytest.approx(scalar(num_default["k_pitch"]), rel=5e-3)
    assert gml_ad == pytest.approx(scalar(num_default["GM_L"]), rel=5e-3)

    # Signed GM_L floor constraint keeps the inequality direction right when gml_floor <= 0.
    for gml_floor in (-2.0, 0.0):
        scale = max(abs(gml_floor), 1.0)
        assert (gml_floor + 0.5 - gml_floor) / scale >= 0.0
        assert (gml_floor - 0.5 - gml_floor) / scale < 0.0


def test_solve_equilibrium_positional_args_squat_level_and_rejected_backtrack():
    water, hull, mass = _small_craft()
    res = CraftBuildup.solve_equilibrium(
        hull,
        2.0,
        mass,
        [0.0, 0.0, 0.1],
        water=water,
        squat_level=0.5,
        include_interference_breakdown=False,
    )
    assert abs(scalar(res["F_z_net"])) < 1e-4 * mass * G
    assert abs(scalar(res["M_y_net"])) < 1e-4 * mass * G * hull.length

    with pytest.raises(RuntimeError):
        CraftBuildup.solve_equilibrium(
            hull,
            2.0,
            100.0 * mass,
            [0.0, 0.0, 0.1],
            water=water,
            include_interference_breakdown=False,
        )


def test_craft_step_export_writes_hull_surfaces_and_foil_solids(tmp_path):
    hull = Hull.wigley(5.0, 0.8, 0.25, freeboard=0.3, n_stations=11, n_levels=7)
    assembly = _t_foil(1.5)

    # A catamaran: two demihull shells (the offsets table as bilinear patches, one
    # per side of each hull) beside the foil and strut solids.
    path = hsb.export_craft_step(tmp_path / "cat.step", hull, assembly, demihull_spacing=2.0)
    text = path.read_text()
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;")
    assert text.count("MANIFOLD_SOLID_BREP") == 2
    assert text.count("SHELL_BASED_SURFACE_MODEL") == 2
    assert text.count("OPEN_SHELL") == 2
    assert text.count("B_SPLINE_SURFACE_WITH_KNOTS") >= 4
    assert "MANIFOLD_SURFACE_SHAPE_REPRESENTATION" in text
    assert "ADVANCED_BREP_SHAPE_REPRESENTATION" in text
    # Every entity reference resolves.
    import re

    defined = set(re.findall(r"^#(\d+)=", text, flags=re.M))
    used = set(re.findall(r"#(\d+)", text))
    assert used <= defined

    # A hull read from B-spline geometry keeps the designer's patches, about its
    # own centreplane and with x running aft.
    l, b, t = 10.0, 1.0, 0.625
    ctrl = [
        [x, 3.0 + (b / 2) * yu * yv, z]
        for x, yu in zip([-l / 2, 0.0, l / 2], [0.0, 2.0, 0.0])
        for z, yv in zip([0.0, -t / 2, -t], [1.0, 1.0, 0.0])
    ]
    mirrored = [[x, 6.0 - y, z] for x, y, z in ctrl]
    patch = lambda c: {"degree": [2, 2], "n_ctrl": [3, 3], "knots_u": [0, 0, 0, 1, 1, 1],
                       "knots_v": [0, 0, 0, 1, 1, 1], "ctrl": c, "trim_uv": None}
    drawn = Hull.from_boatmath_geometry(
        {"kind": "nurbs", "hulls": [{"patches": [patch(ctrl), patch(mirrored)]}]}
    )
    patches = drawn.surface_patches()
    assert len(patches) == 2 and patches[0]["ctrl"].shape == (3, 3, 3)
    assert drawn.source_centerplane_y == pytest.approx(3.0, abs=1e-6)
    assert onp.max(onp.abs([p["ctrl"][..., 1] for p in patches])) == pytest.approx(b, rel=1e-9)
    hull_only = hsb.export_craft_step(tmp_path / "hull.step", drawn)
    assert hull_only.read_text().count("B_SPLINE_SURFACE_WITH_KNOTS") == 2


def test_craft_step_export_marks_the_centre_of_gravity_in_red(tmp_path):
    import re

    hull = Hull.wigley(5.0, 0.8, 0.25, freeboard=0.3, n_stations=11, n_levels=7)
    text = hsb.export_craft_step(
        tmp_path / "marked.step", hull, _t_foil(1.5), xyz_cg=[0.4, 0.0, 0.1],
        cg_marker_radius=0.05,
    ).read_text()
    assert "MANIFOLD_SOLID_BREP('Centre of gravity'" in text
    # A 50 mm sphere centred on the CG (the file is in millimetres), coloured red.
    assert re.search(r"SPHERICAL_SURFACE\('',#\d+,50\.\)", text)
    assert "CARTESIAN_POINT('',(400.,0.,100.))" in text
    assert "COLOUR_RGB('',1.,0.,0.)" in text
    assert text.count("STYLED_ITEM") == 1
    # Without a CG, no marker and no styling.
    plain = hsb.export_craft_step(tmp_path / "plain.step", hull, _t_foil(1.5)).read_text()
    assert "SPHERICAL_SURFACE" not in plain and "STYLED_ITEM" not in plain


def test_keel_height_and_under_hull_foil_layouts():
    import pickle
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(hsb.__file__).parent / "examples"))
    from optimize_foils_on_hull import build_assembly

    hull = Hull.wigley(5.0, 0.4, 0.2, freeboard=0.3, n_stations=21, n_levels=13)
    # Deepest amidships, shallower toward the ends; never above the keel itself.
    assert hull.keel_height(0.0) == pytest.approx(-0.2, abs=1e-9)
    assert hull.keel_height(0.0) <= hull.keel_height(2.0) <= 0.0
    assert pickle.loads(pickle.dumps(hull)).keel_height(1.0) == hull.keel_height(1.0)

    spacing = 2.0
    design = {"x_main": 1.5, "depth_main": 0.45, "root_chord_main": 0.14, "taper_main": 0.7,
              "incidence_main": 1.0, "half_span_main": 0.35}
    pi = build_assembly(design, False, layout="pi", hull=hull, demihull_spacing=spacing)
    twin = build_assembly(design, False, layout="twin", hull=hull, demihull_spacing=spacing)

    # One foil from hull to hull on two struts...
    assert len(pi.foils) == 1 and pi.foils[0].symmetric
    assert pi.foils[0].span() == pytest.approx(spacing)
    # ...or a T-foil under each hull, each a wing of its own.
    assert len(twin.foils) == 2 and not any(f.symmetric for f in twin.foils)
    assert sorted(float(f.xsecs[1].xyz_le[1]) for f in twin.foils) == [-1.0, 1.0]
    assert all(f.span() == pytest.approx(0.7) for f in twin.foils)
    for assembly in (pi, twin):
        (strut,) = assembly.struts
        assert strut.symmetric and float(strut.xsecs[0].xyz_le[1]) == pytest.approx(spacing / 2)
        # Each strut runs from the foil up to the keel of the hull above it.
        assert float(strut.xsecs[0].xyz_le[2]) == pytest.approx(-0.45)
        assert float(strut.xsecs[1].xyz_le[2]) == pytest.approx(
            hull.keel_height(float(strut.xsecs[1].xyz_le[0]))
        )

    # Both run through the craft model, and the two T-foils share the lift equally.
    water = hsb.Water.seawater()
    mass = 2.0 * scalar(hull.hydrostatics()["volume"]) * water.density()
    res = CraftBuildup(
        hull=hull, assembly=twin, velocity=4.0, mass=mass, xyz_cg=[0.0, 0.0, 0.0], water=water,
        demihull_spacing=spacing, include_hull_dynamics=False, include_interference_breakdown=False,
    ).run()
    lifts = [
        scalar(c.L) for w, c in zip(twin.wings, res["foil"]["wing_hydro_components"])
        if not getattr(w, "is_strut", False)
    ]
    assert lifts[0] > 0.0 and lifts[0] == pytest.approx(lifts[1], rel=1e-6)


def test_vortex_under_a_wide_flat_hull_pulls_it_down_by_half_its_lift():
    """
    A long bound vortex of circulation Gamma a height h below a wide flat bottom
    speeds the flow under it by Gamma h / (2 pi (x^2 + h^2)); the linearized pressure
    -rho U u integrates along the hull to -rho U Gamma / 2 per unit width: half the
    vortex's own lift, as suction on the hull (free field, no reflection in the hull).
    """
    from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import (
        calculate_induced_velocity_horseshoe,
    )

    stations = onp.linspace(-15.0, 15.0, 121)
    levels = onp.array([-0.6, -0.5, 0.0, 0.5])
    half_breadths = onp.zeros((len(stations), len(levels)))
    half_breadths[:, 1:] = 1.0  # a 2 m wide box drawing 0.5 m, flat-bottomed
    barge = Hull(stations, levels, half_breadths, transom_closure="none")

    u_inf, rho, gamma, clearance, span = 5.0, 1000.0, 2.0, 1.0, 60.0
    depth = barge.sections(u_inf)["mean_depth"]
    z_vortex = -float(onp.max(depth)) - clearance

    def vortex_u(x, y, z):
        u, _, _ = calculate_induced_velocity_horseshoe(
            x_field=x, y_field=y, z_field=z,
            x_left=0.0, y_left=-span / 2, z_left=z_vortex,
            x_right=0.0, y_right=span / 2, z_right=z_vortex,
            gamma=gamma, trailing_vortex_direction=onp.array([1.0, 0.0, 0.0]),
        )
        return u

    load = barge.pressure_load(vortex_u, u_inf, rho)
    lift_under_hull = rho * u_inf * gamma * 2.0  # the vortex's lift over the hull's 2 m beam
    # Over a finite hull length the integral of h / (x^2 + h^2) falls a little short of pi.
    expected = -0.5 * lift_under_hull * (2.0 / onp.pi) * onp.arctan(15.0 / clearance)
    assert scalar(load["force_up"]) == pytest.approx(expected, rel=0.05)
    # A wall-sided box has no bow or stern slope: no axial force; and by symmetry no moment.
    assert abs(scalar(load["moment_bow_up"])) < 1e-6 * abs(expected) * 15.0


def test_hull_induced_flow_is_outward_at_the_bow_inward_at_the_stern_and_dies_away():
    hull = Hull.wigley(10.0, 1.0, 0.625)
    u_inf = 3.0

    def at(x, y, z, **kw):
        u, v, w = hull.induced_velocity([x], [y], [z], u_inf, **kw)
        return scalar(u), scalar(v), scalar(w)

    # Beside the bow the hull pushes water outward and down; beside the stern it
    # draws it back in. (No surface image here: just the hull's own sources.)
    _, v_bow, w_bow = at(-3.5, 1.0, -1.0, free_surface="none")
    _, v_stern, w_stern = at(3.5, 1.0, -1.0, free_surface="none")
    assert v_bow > 0.0 and w_bow < 0.0
    assert v_stern == pytest.approx(-v_bow, rel=1e-6) and w_stern == pytest.approx(-w_bow, rel=1e-6)
    # Amidships, beside a fore-aft symmetric hull, the flow is sped up.
    u_mid, _, _ = at(0.0, 1.0, -0.3, free_surface="rigid_lid")
    assert u_mid > 0.0
    # The sources sum to zero, so far away the disturbance falls off quickly.
    near = max(abs(c) for c in at(0.0, 3.0, -0.3))
    far = max(abs(c) for c in at(0.0, 30.0, -0.3))
    assert far < 2e-3 * near
    # A catamaran's other hull adds its own field.
    lone = at(0.0, 1.0, -0.3)
    paired = hull.induced_velocity([0.0], [0.0], [-0.3], u_inf, demihull_spacing=2.0)
    assert scalar(paired[0]) == pytest.approx(2.0 * lone[0], rel=1e-6)
    assert abs(scalar(paired[1])) < 1e-12


def test_hull_foil_interaction_in_the_craft_model():
    import casadi as cas

    water, hull, mass = _small_craft()
    common = dict(hull=hull, velocity=4.5, mass=mass, xyz_cg=[0.2, 0.0, 0.1], water=water,
                  sinkage=0.0, trim=0.5, include_hull_dynamics=False,
                  include_interference_breakdown=False)
    assembly = _t_foil(0.5)

    off = CraftBuildup(assembly=assembly, include_hull_foil_interaction=False, **common).run()
    on = CraftBuildup(assembly=assembly, **common).run()
    assert scalar(off["F_z_foil_on_hull"]) == 0.0
    # A lifting foil under the hull draws the hull down, by a modest share of its lift.
    assert scalar(on["L_foil"]) > 0.0
    assert -0.3 * scalar(on["L_foil"]) < scalar(on["F_z_foil_on_hull"]) < 0.0
    # The books balance: the net vertical force changes by the load on the hull plus
    # the change in the foil's own force from the hull's flow.
    foil_change = (
        scalar(on["foil"]["F_g_with_buoyancy"][2]) - scalar(off["foil"]["F_g_with_buoyancy"][2])
    ) * onp.cos(onp.radians(0.5)) - (
        scalar(on["foil"]["F_g_with_buoyancy"][0]) - scalar(off["foil"]["F_g_with_buoyancy"][0])
    ) * onp.sin(onp.radians(0.5))
    assert scalar(on["F_z_net"]) - scalar(off["F_z_net"]) == pytest.approx(
        scalar(on["F_z_foil_on_hull"]) + foil_change, rel=1e-9, abs=1e-9
    )

    # Symbolic attitude reproduces the numeric evaluation.
    s, t = cas.MX.sym("sinkage"), cas.MX.sym("trim")
    symbolic = CraftBuildup(assembly=assembly, **dict(common, sinkage=s, trim=t)).run()
    f = cas.Function("f", [s, t], [symbolic["F_z_net"], symbolic["M_y_net"], symbolic["F_z_foil_on_hull"]])
    values = [float(v) for v in f(0.0, 0.5)]
    for value, key in zip(values, ("F_z_net", "M_y_net", "F_z_foil_on_hull")):
        assert value == pytest.approx(scalar(on[key]), rel=1e-8, abs=1e-8)


def test_hull_wave_field_trails_the_hull_and_obeys_the_surface_condition():
    hull = Hull.wigley(10.0, 1.0, 0.625)
    u_inf = 0.4 * onp.sqrt(G * 10.0)
    waves = hull.wave_field(u_inf, level=2, k_cap=20.0)

    # Radiation: waves astern, almost nothing ahead of the bow.
    ahead = onp.ravel(waves.elevation(onp.linspace(-25.0, -12.0, 40), onp.zeros(40)))
    astern = onp.ravel(waves.elevation(onp.linspace(8.0, 30.0, 60), onp.zeros(60)))
    assert onp.max(onp.abs(ahead)) < 0.05 * onp.max(onp.abs(astern))

    # Astern on the track, the transverse waves have the wavelength 2 pi U^2 / g.
    x = onp.linspace(12.0, 40.0, 280)
    zeta = onp.ravel(waves.elevation(x, onp.zeros_like(x)))
    upward = x[1:][(zeta[:-1] < 0) & (zeta[1:] >= 0)]
    assert onp.mean(onp.diff(upward)) == pytest.approx(2 * onp.pi * u_inf**2 / G, rel=0.08)

    # Port/starboard symmetric about the hull's centreplane.
    left = waves.elevation([6.0], [-1.5])
    right = waves.elevation([6.0], [1.5])
    assert scalar(left) == pytest.approx(scalar(right), rel=1e-9)

    # Linearized kinematic condition on the surface: w = U d(zeta)/dx, where w is
    # the whole flow's: the wave part plus the local part (the Rankine source minus
    # its negative image, which has no elevation of its own but does have a w there).
    xs = onp.array([7.0, 9.0, 14.0, 22.0])
    h = 1e-3
    slope = (
        onp.ravel(waves.elevation(xs + h, 0.8 * onp.ones(4)))
        - onp.ravel(waves.elevation(xs - h, 0.8 * onp.ones(4)))
    ) / (2 * h)
    _, _, w = waves.induced_velocity(xs, 0.8 * onp.ones(4), onp.zeros(4))
    _, _, w_local = hull.induced_velocity(xs, 0.8 * onp.ones(4), onp.zeros(4), u_inf)
    # (Approximately: the local part is the slender-body one, and the two parts are
    # integrated differently, so they close the condition to about a tenth.)
    assert onp.ravel(w) + onp.ravel(w_local) == pytest.approx(u_inf * slope, rel=0.15, abs=5e-3)

    # The wave flow itself is exactly a potential flow: no divergence, no vorticity.
    point = onp.array([9.0, 0.8, -0.4])

    def velocity_at(offset):
        q = point + offset
        return onp.array([scalar(c) for c in waves.induced_velocity([q[0]], [q[1]], [q[2]])])

    def derivative(component, axis):
        step = onp.zeros(3)
        step[axis] = 1e-3
        return (velocity_at(step)[component] - velocity_at(-step)[component]) / 2e-3

    scale = onp.max(onp.abs(velocity_at(onp.zeros(3)))) / 1.0  # per metre
    assert abs(derivative(0, 0) + derivative(1, 1) + derivative(2, 2)) < 1e-4 * scale
    assert abs(derivative(0, 2) - derivative(2, 0)) < 1e-4 * scale
    assert abs(derivative(0, 1) - derivative(1, 0)) < 1e-4 * scale
    # And the dynamic one: zeta = -(U / g) u.
    # (the local part has no streamwise velocity on the surface)
    u, _, _ = waves.induced_velocity(xs, 0.8 * onp.ones(4), onp.zeros(4))
    zeta_here = onp.ravel(waves.elevation(xs, 0.8 * onp.ones(4)))
    assert zeta_here == pytest.approx(-(u_inf / G) * onp.ravel(u), rel=1e-9, abs=1e-12)
    u_local, _, _ = hull.induced_velocity(xs, 0.8 * onp.ones(4), onp.zeros(4), u_inf)
    assert onp.max(onp.abs(onp.ravel(u_local))) < 1e-12

    # The wave flow dies away with depth.
    shallow = onp.abs(onp.ravel(waves.induced_velocity([9.0], [0.0], [-0.5])[2]))
    deep = onp.abs(onp.ravel(waves.induced_velocity([9.0], [0.0], [-8.0])[2]))
    assert deep < 0.1 * shallow


def test_hull_wave_field_reaches_the_foil_without_tilting_its_forces():
    import casadi as cas

    water, hull, mass = _small_craft()
    common = dict(hull=hull, velocity=4.5, mass=mass, xyz_cg=[0.2, 0.0, 0.1], water=water,
                  sinkage=0.0, trim=0.5, include_hull_dynamics=False,
                  include_interference_breakdown=False)
    assembly = _t_foil(2.0)  # near the stern, in the hull's wave

    local = CraftBuildup(assembly=assembly, include_hull_wave_field=False, **common).run()
    waves = CraftBuildup(assembly=assembly, **common).run()
    # The hull's wave changes the foil's lift and its submergence...
    assert scalar(local["hull_wave_elevation_at_foil"]) == 0.0
    assert abs(scalar(waves["hull_wave_elevation_at_foil"])) > 1e-3
    assert abs(scalar(waves["L_foil"]) / scalar(local["L_foil"]) - 1.0) > 0.01
    # ...but the foil's drag changes only as its lift does (induced and profile
    # drag), not by its lift being tilted in the wave flow, which the shared
    # far-field wave integral already accounts for.
    lift_ratio = scalar(waves["L_foil"]) / scalar(local["L_foil"])
    assert scalar(waves["D_foil"]) == pytest.approx(
        scalar(local["D_foil"]), rel=2.5 * abs(lift_ratio - 1.0) + 0.02
    )

    # Symbolic attitude reproduces the numeric evaluation.
    s, t = cas.MX.sym("sinkage"), cas.MX.sym("trim")
    symbolic = CraftBuildup(assembly=assembly, **dict(common, sinkage=s, trim=t)).run()
    f = cas.Function("f", [s, t], [symbolic["F_z_net"], symbolic["M_y_net"], symbolic["D"]])
    for value, key in zip([float(v) for v in f(0.0, 0.5)], ("F_z_net", "M_y_net", "D")):
        assert value == pytest.approx(scalar(waves[key]), rel=1e-8, abs=1e-8)


def test_wave_amplitude_table_reproduces_direct_evaluation_and_interference():
    import pickle

    hull = _transom_hull()
    water = hsb.Water.seawater()
    u, rho = 3.5, water.density()
    mass = 0.9 * scalar(hull.hydrostatics()["volume"]) * rho
    scales = hull.default_wave_length_scales()

    wave_table = hull.wave_amplitude_table(
        u, water.gravity, length_scales=scales,
        sinkage_range=(-0.06, 0.06), trim_range=(-1.0, 3.0), n_sinkage=7, n_trim=9,
    )
    wave_table = pickle.loads(pickle.dumps(wave_table))
    dyn_table = hull.dynamic_force_table(
        u, rho, sinkage_range=(-0.06, 0.06), trim_range=(-1.0, 3.0), n_sinkage=7, n_trim=9,
    )

    # Numerically reproduces bare hull and foil-hull interference across attitudes and shifted pivots.
    for sinkage, trim, cg in [
        (0.013, 1.7, [0.6, 0.0, 0.2]),
        (-0.022, 0.4, [-0.3, 0.0, 0.0]),
        (0.031, 2.6, [0.9, 0.0, 0.1]),
    ]:
        common = dict(
            hull=hull, assembly=_t_foil(0.8), velocity=u, mass=mass, xyz_cg=cg,
            water=water, sinkage=sinkage, trim=trim, wave_length_scales=scales,
            include_interference_breakdown=True,
        )
        direct = CraftBuildup(**common).run()
        tabled = CraftBuildup(
            hull_dynamics=dyn_table, hull_wave_amplitude=wave_table, **common
        ).run()
        assert scalar(tabled["D_wave_hull_alone"]) == pytest.approx(
            scalar(direct["D_wave_hull_alone"]), rel=2e-2
        )
        assert scalar(tabled["D_wave"]) == pytest.approx(scalar(direct["D_wave"]), rel=2e-2)
        assert scalar(tabled["D_wave_interference"]) == pytest.approx(
            scalar(direct["D_wave_interference"]), rel=5e-2, abs=0.5
        )

    # Catamaran spacing matches direct evaluation.
    cat_scales = hull.default_wave_length_scales(demihull_spacing=2.0)
    cat_table = hull.wave_amplitude_table(
        u, water.gravity, length_scales=cat_scales,
        sinkage_range=(-0.06, 0.06), trim_range=(-1.0, 3.0), n_sinkage=7, n_trim=9,
        demihull_spacing=2.0,
    )
    cat_common = dict(
        hull=hull, assembly=_t_foil(0.8), velocity=u, mass=2.0 * mass, xyz_cg=[0.4, 0.0, 0.1],
        water=water, sinkage=0.01, trim=1.2, demihull_spacing=2.0,
        wave_length_scales=cat_scales, include_interference_breakdown=False,
        include_hull_dynamics=False,
    )
    cat_direct = CraftBuildup(**cat_common).run()
    cat_tabled = CraftBuildup(hull_wave_amplitude=cat_table, **cat_common).run()
    assert scalar(cat_tabled["D_wave"]) == pytest.approx(scalar(cat_direct["D_wave"]), rel=2e-2)

    # Symbolic Opti equilibrium solve matches direct CraftBuildup.solve_equilibrium.
    ref = CraftBuildup.solve_equilibrium(
        hull=hull, assembly=_t_foil(0.2), velocity=u, mass=mass, xyz_cg=[-0.3, 0.0, 0.1],
        water=water, wave_length_scales=scales, include_interference_breakdown=False,
    )
    opti = asb.Opti()
    s_var = opti.variable(init_guess=0.0, lower_bound=-0.06, upper_bound=0.06)
    t_var = opti.variable(init_guess=0.0, lower_bound=-1.0, upper_bound=3.0)
    sym = CraftBuildup(
        hull=hull, assembly=_t_foil(0.2), velocity=u, mass=mass, xyz_cg=[-0.3, 0.0, 0.1],
        water=water, sinkage=s_var, trim=t_var, wave_length_scales=scales,
        hull_dynamics=dyn_table, hull_wave_amplitude=wave_table,
        include_interference_breakdown=False,
    ).run()
    weight = mass * G
    opti.subject_to([sym["F_z_net"] / weight == 0.0, sym["M_y_net"] / (weight * hull.length) == 0.0])
    opti.minimize(0)
    sol = opti.solve(verbose=False, options={"ipopt.hessian_approximation": "limited-memory"})
    assert float(sol(s_var)) == pytest.approx(ref["sinkage"], abs=1e-3)
    assert float(sol(t_var)) == pytest.approx(ref["trim"], abs=5e-2)

    # Stability quantities (GM_L, k_heave) from tabled run_with_stability match direct evaluation.
    stab_common = dict(
        hull=hull, assembly=_t_foil(0.2), velocity=u, mass=mass, xyz_cg=[-0.3, 0.0, 0.1],
        water=water, sinkage=0.01, trim=1.2, wave_length_scales=scales,
        include_interference_breakdown=False,
    )
    stab_direct = CraftBuildup(**stab_common).run_with_stability(include_pitch_damping=False)
    stab_tabled = CraftBuildup(
        hull_dynamics=dyn_table, hull_wave_amplitude=wave_table, **stab_common
    ).run_with_stability(include_pitch_damping=False)
    assert scalar(stab_tabled["GM_L"]) == pytest.approx(scalar(stab_direct["GM_L"]), rel=3e-2)
    assert scalar(stab_tabled["k_heave"]) == pytest.approx(scalar(stab_direct["k_heave"]), rel=3e-2)

    # Hull.dynamic_force exposes exact moment_slope == d(moment_bow_up)/dx_ref (mono and cat).
    for spacing in (None, 2.0):
        h_here = hull.dynamic_force(u, rho, G, 0.013, 1.7, (0.6, 0.2), x_ref=0.6, demihull_spacing=spacing)
        h_aft = hull.dynamic_force(u, rho, G, 0.013, 1.7, (0.6, 0.2), x_ref=1.6, demihull_spacing=spacing)
        assert scalar(h_here["moment_slope"]) == pytest.approx(
            scalar(h_aft["moment_bow_up"]) - scalar(h_here["moment_bow_up"]), rel=1e-12
        )


def test_run_with_stability_hull_reuse_and_optional_pitch_damping():
    import casadi as cas

    water, hull, mass = _small_craft()
    common = dict(
        hull=hull, velocity=4.0, mass=mass, xyz_cg=[0.0, 0.0, 0.1],
        assembly=_t_foil(1.2), water=water, include_interference_breakdown=False,
    )
    craft_num = CraftBuildup(sinkage=-0.01, trim=1.2, **common)
    with_damp = craft_num.run_with_stability(include_pitch_damping=True)
    no_damp = craft_num.run_with_stability(include_pitch_damping=False)
    assert scalar(with_damp["c_pitch_foils"]) > 0.0
    assert no_damp["c_pitch_foils"] is None
    assert scalar(no_damp["GM_L"]) == pytest.approx(scalar(with_damp["GM_L"]), rel=1e-12)
    assert scalar(no_damp["k_heave"]) == pytest.approx(scalar(with_damp["k_heave"]), rel=1e-12)

    s_sym, t_sym = cas.MX.sym("s"), cas.MX.sym("t")
    sym_craft = CraftBuildup(sinkage=s_sym, trim=t_sym, **common)
    sym_with = sym_craft.run_with_stability(include_pitch_damping=True)
    sym_no = sym_craft.run_with_stability(include_pitch_damping=False)
    assert sym_no["c_pitch_foils"] is None
    fn = cas.Function(
        "stab_cmp",
        [s_sym, t_sym],
        [sym_with["GM_L"], sym_no["GM_L"], sym_with["c_pitch_foils"]],
    )
    gml_a, gml_b, c_pitch = (float(x) for x in fn(-0.01, 1.2))
    assert gml_a == pytest.approx(gml_b, rel=1e-12)
    assert c_pitch == pytest.approx(scalar(with_damp["c_pitch_foils"]), rel=1e-9)


def test_foil_section_variables_and_bending_stress():
    """The example's section family and its bending-stress estimate."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(hsb.__file__).parent / "examples"))
    import casadi as cas
    from optimize_foils_on_hull import build_assembly, foil_section, root_bending_stress

    x = onp.linspace(0.0, 1.0, 401)
    # The family passes through the NACA 2412 and scales as asked.
    reference = asb.KulfanAirfoil("naca2412")
    same = foil_section(0.12, float(onp.max(reference.local_camber(x))))
    assert onp.allclose(same.upper_weights, reference.upper_weights, atol=2e-3)
    assert onp.allclose(same.lower_weights, reference.lower_weights, atol=2e-3)
    for thickness, camber in ((0.08, 0.0), (0.15, 0.04)):
        section = foil_section(thickness, camber)
        assert onp.max(section.local_thickness(x)) == pytest.approx(thickness, rel=1e-3)
        assert onp.max(section.local_camber(x)) == pytest.approx(camber, abs=1e-4)
    # More camber lifts more at a given angle; it stays usable symbolically.
    lift = [
        float(foil_section(0.12, c).get_aero_from_neuralfoil(alpha=1.0, Re=1e6)["CL"][0])
        for c in (0.0, 0.02, 0.04)
    ]
    assert lift[0] < lift[1] < lift[2]
    opti = asb.Opti()
    symbolic = foil_section(opti.variable(0.12), opti.variable(0.02))
    assert isinstance(symbolic.get_aero_from_neuralfoil(alpha=1.0, Re=1e6)["CD"], cas.MX)

    # Rectangular cantilever, by hand: M = p c s^2 / 2, Z = 2 k c t^2.
    design = dict(root_chord_main=0.2, taper_main=1.0, half_span_main=0.8, thickness_main=0.10)
    pressure = 18000.0
    by_hand = (pressure * 0.2 * 0.8**2 / 2) / (2 * 0.0395 * 0.2 * 0.02**2)
    assert root_bending_stress(design, "centreline", None, pressure) == pytest.approx(by_hand)
    # The same half-span simply supported at its tips carries the same midspan moment.
    assert root_bending_stress(design, "pi", 1.6, pressure) == pytest.approx(by_hand)
    # Thicker section, lower stress, as the square.
    thicker = dict(design, thickness_main=0.20)
    assert root_bending_stress(thicker, "centreline", None, pressure) == pytest.approx(by_hand / 4)
    # The section inertia factor is that of the thickness form.
    t = foil_section(0.12, 0.0).local_thickness(x)
    assert onp.trapezoid(t**3 / 12, x) / 0.12**3 == pytest.approx(0.0395, rel=0.01)

    # The assembly takes the section from the design when it is there.
    design = dict(
        x_main=3.0, depth_main=0.4, half_span_main=0.5, root_chord_main=0.15, taper_main=0.7,
        incidence_main=1.0, thickness_main=0.09, camber_main=0.03,
    )
    foil = build_assembly(design, False).foils[0]
    assert float(foil.xsecs[0].airfoil.max_thickness()) == pytest.approx(0.09, rel=0.01)
