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
    res = CraftBuildup.solve_equilibrium(
        hull=hull, velocity=0.3, mass=mass, xyz_cg=[0.0, 0.0, 0.1], water=water
    )
    assert res["sinkage"] == pytest.approx(0.0, abs=1e-4)
    assert res["trim"] == pytest.approx(0.0, abs=1e-3)
    assert scalar(res["buoyancy_hull"]) == pytest.approx(mass * G, rel=1e-6)


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
