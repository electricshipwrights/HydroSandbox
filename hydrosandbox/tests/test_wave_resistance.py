"""
Wave-making resistance: the far-field kernel against closed forms (the same ones
boatmath's `thinship` crate validates against), and its consistency with the
method-of-images solvers.
"""

import numpy as onp
import pytest
import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import Foil, HydrofoilAssembly, Strut
from hydrosandbox.hydrodynamics import HydroBuildup, HydroLiftingLine, HydroVLM
from hydrosandbox.library import kelvin_wave_resistance, wing_thickness_wave_sources
from hydrosandbox.performance import HydroOperatingPoint

G = 9.80665
RHO = 1000.0


def _wigley_michell_reference(l: float, b: float, t: float, u: float) -> float:
    """
    Michell's integral for the Wigley hull f = (B/2)(1-(2x/L)^2)(1-(z/T)^2), with the
    closed-form inner integral J(lambda) (boatmath `thinship::validation::wigley_j`).
    """
    nu = G / u**2
    theta = onp.linspace(0.0, onp.arccos(1.0 / 200.0), 400_001)
    lam = 1.0 / onp.cos(theta)
    a, k, kappa = l / 2.0, nu * lam, nu * lam**2
    x_int = 2.0 * (onp.sin(k * a) / k**2 - a * onp.cos(k * a) / k)
    e = onp.exp(-kappa * t)
    z_int = (1.0 - e) / kappa - (
        2.0 - e * (kappa**2 * t**2 + 2.0 * kappa * t + 2.0)
    ) / (kappa**3 * t**2)
    j = -(4.0 * b / l**2) * x_int * z_int
    return 4.0 * RHO * G**2 / (onp.pi * u**2) * onp.trapezoid(j**2 * lam**3, theta)


def _wigley_source_elements(l, b, t, u, nx=60, nz=24) -> dict:
    """Centreplane source panels (vertical line elements) for the Wigley hull."""
    x_edges = -l / 2.0 + l * 0.5 * (1.0 - onp.cos(onp.pi * onp.linspace(0, 1, nx + 1)))
    z_edges = t * onp.linspace(0.0, 1.0, nz + 1)

    def half_beam(x, z):
        return (b / 2.0) * (1.0 - (2.0 * x / l) ** 2) * (1.0 - (z / t) ** 2)

    x_mid, z_top, z_bot, flux, smear = [], [], [], [], []
    for i in range(nx):
        for j in range(nz):
            z0, z1 = z_edges[j], z_edges[j + 1]
            zm = 0.5 * (z0 + z1)
            # Depth-averaged (Simpson) jump in half-beam across the strip.
            df = sum(
                w * (half_beam(x_edges[i + 1], z) - half_beam(x_edges[i], z))
                for w, z in ((1 / 6, z0), (4 / 6, zm), (1 / 6, z1))
            )
            x_mid.append(0.5 * (x_edges[i] + x_edges[i + 1]))
            z_top.append(-z0)
            z_bot.append(-z1)
            flux.append(2.0 * u * df * (z1 - z0))
            smear.append(x_edges[i + 1] - x_edges[i])
    zeros = onp.zeros(len(x_mid))
    return dict(
        xyz_a=(onp.array(x_mid), zeros, onp.array(z_top)),
        xyz_b=(onp.array(x_mid), zeros, onp.array(z_bot)),
        source_flux=onp.array(flux),
        smear_length=onp.array(smear),
    )


@pytest.mark.parametrize("froude", [0.25, 0.35, 0.5])
def test_thin_ship_sources_match_michell_for_wigley_hull(froude):
    l, b, t = 10.0, 1.0, 0.625
    u = froude * onp.sqrt(G * l)
    got = kelvin_wave_resistance(
        velocity=u, density=RHO, **_wigley_source_elements(l, b, t, u)
    )
    assert got == pytest.approx(_wigley_michell_reference(l, b, t, u), rel=5e-3)


@pytest.mark.parametrize("u, depth", [(3.0, 0.5), (6.0, 0.35), (10.0, 1.0)])
def test_wide_lifting_line_matches_two_dimensional_vortex(u, depth):
    """R / B = rho * g * Gamma^2 / U^2 * exp(-2 g h / U^2) for a submerged vortex."""
    span = 400.0 * depth
    lift_per_span = 1000.0
    got = kelvin_wave_resistance(
        xyz_a=([0.0], [-span / 2], [-depth]),
        xyz_b=([0.0], [span / 2], [-depth]),
        force_z=[lift_per_span * span],
        velocity=u,
        density=RHO,
        length_scales=(span, 1.0, depth),
    )
    gamma = lift_per_span / (RHO * u)
    want = RHO * G * gamma**2 / u**2 * onp.exp(-2.0 * G * depth / u**2) * span
    assert got == pytest.approx(want, rel=1e-2)


def test_infinite_froude_limit_is_twice_the_biplane_image_mutual_drag():
    """
    Uniformly loaded line of span b at depth d: as Fr -> inf the wave resistance tends
    to twice the drag induced by its phi = 0 image,
    2 * D_m = rho * Gamma^2 / (2 pi) * ln(1 + b^2 / (4 d^2)).
    """
    u, depth, span, lift_per_span = 200.0, 0.3, 0.9, 1000.0
    got = kelvin_wave_resistance(
        xyz_a=([0.0], [-span / 2], [-depth]),
        xyz_b=([0.0], [span / 2], [-depth]),
        force_z=[lift_per_span * span],
        velocity=u,
        density=RHO,
    )
    gamma = lift_per_span / (RHO * u)
    want = RHO * gamma**2 / (2 * onp.pi) * onp.log(1 + span**2 / (4 * depth**2))
    assert got == pytest.approx(want, rel=2e-3)


def _foil_assembly(z: float = -0.3, twist: float = 3.0) -> HydrofoilAssembly:
    airfoil = asb.Airfoil("naca2412")
    return HydrofoilAssembly(
        foils=[
            Foil(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, z], chord=0.15, twist=twist, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[0, 0.45, z], chord=0.15, twist=twist, airfoil=airfoil),
                ],
            )
        ]
    )


@pytest.mark.parametrize(
    "solver, kwargs",
    [(HydroVLM, dict(chordwise_resolution=4)), (HydroLiftingLine, {})],
)
def test_lattice_wave_drag_recovers_image_method_at_high_froude(solver, kwargs):
    """
    Two independent computations must agree as Fr -> inf: the far-field wave integral
    of the solved loading, and the near-field drag induced by the phi = 0 image.
    """
    op = HydroOperatingPoint(velocity=200.0, alpha=0.0, surface_z=0.0)
    with_waves = solver(_foil_assembly(), op, spanwise_resolution=8, **kwargs).run()
    image_only = solver(
        _foil_assembly(), op, spanwise_resolution=8, include_wave_drag=False, **kwargs
    ).run()

    assert with_waves["L"] == pytest.approx(image_only["L"], rel=1e-12)
    assert with_waves["D"] == pytest.approx(image_only["D"], rel=3e-3)
    assert with_waves["D_wave"] > 0.2 * with_waves["D_induced"]


def test_wave_drag_grows_as_froude_number_falls_and_vanishes_when_deep():
    def drag_due_to_lift(u, z):
        op = HydroOperatingPoint(velocity=u, alpha=0.0, surface_z=0.0)
        res = HydroVLM(
            _foil_assembly(z=z), op, spanwise_resolution=8, chordwise_resolution=4
        ).run()
        image = HydroVLM(
            _foil_assembly(z=z),
            op,
            spanwise_resolution=8,
            chordwise_resolution=4,
            include_wave_drag=False,
        ).run()
        return res, image

    # Depth Froude number ~3.5 -> ~1.7: waves add progressively more than the
    # high-Froude image model accounts for.
    excess = []
    for u in (6.0, 4.0, 3.0):
        res, image = drag_due_to_lift(u, z=-0.3)
        assert res["D_wave"] > 0.0
        excess.append(res["D"] / image["D"] - 1.0)
    assert 0.02 < excess[0] < excess[1] < excess[2] < 0.5

    deep, deep_image = drag_due_to_lift(6.0, z=-20.0)
    assert deep["D_wave"] < 1e-3 * deep["D_induced"]
    assert deep["D"] == pytest.approx(deep_image["D"], rel=1e-3)


def test_surface_piercing_strut_thickness_makes_waves_without_lift():
    """A symmetric strut at zero leeway is a (very) thin ship: pure Michell resistance."""
    strut = Strut.from_dimensions(
        root_xyz=[0.0, 0.0, -0.6], length=0.9, root_chord=0.3, tip_chord=0.3
    )
    assembly = HydrofoilAssembly(struts=[strut], s_ref=0.18, c_ref=0.3, b_ref=0.6)
    op = HydroOperatingPoint(velocity=2.0, alpha=0.0, surface_z=0.0)

    buildup = HydroBuildup(assembly, op).run()
    lifting_line = HydroLiftingLine(assembly, op).run()
    d_buildup = float(onp.ravel(buildup["D_wave"])[0])
    assert d_buildup > 0.0
    # Smoothly-trimmed (buildup) and hard-clipped (lifting line) strut sources agree.
    assert lifting_line["D_wave"] == pytest.approx(d_buildup, rel=0.05)

    # Converged with respect to the chordwise source discretization.
    def michell(n_chordwise):
        sources = wing_thickness_wave_sources(
            strut, velocity=2.0, surface_z=0.0, n_chordwise=n_chordwise
        )
        return kelvin_wave_resistance(
            velocity=2.0, density=1025.0, **{k: v for k, v in sources.items()}
        )

    assert michell(8) == pytest.approx(michell(32), rel=0.05)


def test_buildup_wave_drag_is_differentiable_in_opti():
    """Depth is free; deeper is always better for wave drag alone, up to the bound."""
    opti = asb.Opti()
    depth = opti.variable(init_guess=0.3, lower_bound=0.15, upper_bound=0.6)
    airfoil = asb.Airfoil("naca2412")
    assembly = HydrofoilAssembly(
        foils=[
            Foil(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, -depth], chord=0.15, twist=3.0, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[0, 0.45, -depth], chord=0.15, twist=3.0, airfoil=airfoil),
                ],
            )
        ],
        s_ref=0.135,
        c_ref=0.15,
        b_ref=0.9,
    )
    op = HydroOperatingPoint(velocity=4.0, alpha=0.0, surface_z=0.0)
    res = HydroBuildup(
        assembly, op, wave_length_scales=(0.9, 0.15, 0.15), include_spray_drag=False
    ).run()
    opti.minimize(res["D_wave"])
    sol = opti.solve(verbose=False)
    assert sol(depth) == pytest.approx(0.6, abs=1e-3)

    # The symbolic expression reproduces the numeric evaluation at the optimum.
    numeric_assembly = HydrofoilAssembly(
        foils=[
            Foil(
                symmetric=True,
                xsecs=[
                    asb.WingXSec(xyz_le=[0, 0, -0.6], chord=0.15, twist=3.0, airfoil=airfoil),
                    asb.WingXSec(xyz_le=[0, 0.45, -0.6], chord=0.15, twist=3.0, airfoil=airfoil),
                ],
            )
        ],
        s_ref=0.135,
        c_ref=0.15,
        b_ref=0.9,
    )
    numeric = HydroBuildup(
        numeric_assembly, op, wave_length_scales=(0.9, 0.15, 0.15), include_spray_drag=False
    ).run()
    assert float(sol(res["D_wave"])) == pytest.approx(
        float(onp.ravel(numeric["D_wave"])[0]), rel=1e-6
    )
