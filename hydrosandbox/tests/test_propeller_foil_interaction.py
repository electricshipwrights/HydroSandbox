"""Propeller-foil interaction models and the wave field of line elements."""

import numpy as onp
import pytest

import aerosandbox as asb
from hydrosandbox.hull.hull import wave_field_from_source_amplitude
from hydrosandbox.library.wave_resistance import element_source_amplitude, kelvin_wave_resistance
from hydrosandbox.propulsion.interaction import (
    actuator_disc_velocity,
    foils_at_propeller,
    horseshoe_velocity,
)


def test_actuator_disc_follows_momentum_theory_on_its_axis():
    U, rho, R, T = 6.0, 1025.0, 0.13, 210.0
    c_t = T / (0.5 * rho * U**2 * onp.pi * R**2)
    u_jet = U * (onp.sqrt(1 + c_t) - 1)
    for x in (-4, -2, -1, -0.5, 0.0, 0.5, 1, 2, 4):
        u, v, w = actuator_disc_velocity(x * R, 0.0, 0.0, [0, 0, 0], R, T, U, rho)
        assert u == pytest.approx(0.5 * u_jet * (1 + x / onp.sqrt(x**2 + 1)), abs=0.01 * u_jet)
        assert abs(v) < 1e-9 and abs(w) < 1e-9
    # Beside the disc the flow runs inward; outside the jet there is no speed-up.
    u, v, _ = actuator_disc_velocity(0.0, 1.5 * R, 0.0, [0, 0, 0], R, T, U, rho)
    assert v < 0 and abs(u) < 0.01 * u_jet
    u, _, _ = actuator_disc_velocity(3 * R, 1.6 * R, 0.0, [0, 0, 0], R, T, U, rho)
    assert abs(u) < 0.05 * u_jet
    # The swirl carries the torque: sum of rho u_axial r u_theta over the jet.
    Q = 8.0
    r = onp.linspace(0.0, 0.9 * R, 400)
    u_axial = U + onp.array([actuator_disc_velocity(0.0, ri, 0.0, [0, 0, 0], R, T, U, rho, torque=Q)[0] for ri in r])
    swirl = onp.array([actuator_disc_velocity(10 * R, ri, 0.0, [0, 0, 0], R, T, U, rho, torque=Q)[2] for ri in r])
    k = swirl[200] / (r[200] / R)
    assert rho * (U + 0.5 * u_jet) * onp.pi * k * R**3 / 2 == pytest.approx(Q, rel=0.02)
    assert onp.all(u_axial > U)
    # Opposite hand, opposite swirl.
    assert actuator_disc_velocity(10 * R, 0.5 * R, 0.0, [0, 0, 0], R, T, U, rho, torque=Q, rotation=-1.0)[2] == pytest.approx(-swirl[onp.argmin(abs(r - 0.5 * R))], rel=0.02)


def test_horseshoe_vortex_limits():
    b, gamma = 2.0, 1.0
    a_, b_ = [0.0, -b / 2, 0.0], [0.0, b / 2, 0.0]
    # Far downstream, between the legs: two infinite lines.
    assert horseshoe_velocity([80.0, 0.0, 0.0], a_, b_, gamma, 1e-3)[2] == pytest.approx(-2 * gamma / (onp.pi * b), rel=1e-3)
    # Faster above a lifting vortex, slower below; upwash ahead and outboard.
    assert horseshoe_velocity([0.0, 0.0, 0.3], a_, b_, gamma, 1e-3)[0] > 0
    assert horseshoe_velocity([0.0, 0.0, -0.3], a_, b_, gamma, 1e-3)[0] < 0
    assert horseshoe_velocity([-1.0, 0.0, 0.0], a_, b_, gamma, 1e-3)[2] > 0
    assert horseshoe_velocity([5.0, 2.0, 0.0], a_, b_, gamma, 1e-3)[2] > 0


def test_foils_at_propeller_wake_and_downwash():
    class Component:
        def __init__(self, lift, drag):
            self.L, self.D_profile = lift, drag

    U, rho, R = 6.0, 1025.0, 0.13
    airfoil = asb.Airfoil("naca0012")
    wing = asb.Wing(
        name="Foil", symmetric=True,
        xsecs=[asb.WingXSec(xyz_le=[0, 0, -0.5], chord=0.2, airfoil=airfoil),
               asb.WingXSec(xyz_le=[0, 1.0, -0.5], chord=0.2, airfoil=airfoil)],
    )
    drag = 20.0
    behind = foils_at_propeller([wing], [Component(1500.0, drag)], [1.0, 0.3, -0.5], R, U, rho)
    # The trailing edge crosses the disc on a diameter: 2R of 2 m of span.
    expected = drag / 2.0 * (2 * R) / (rho * U**2 * onp.pi * R**2)
    assert behind["wake_fraction_viscous"] == pytest.approx(expected, rel=0.05)
    assert behind["inflow_angle"] > 0  # downwash behind a lifting foil
    # A propeller ahead of the foil, or clear above it, is out of its wake.
    ahead = foils_at_propeller([wing], [Component(1500.0, drag)], [-1.0, 0.3, -0.5], R, U, rho)
    above = foils_at_propeller([wing], [Component(1500.0, drag)], [1.0, 0.3, -0.5 + 3 * R], R, U, rho)
    assert ahead["wake_fraction_viscous"] < 0.01 * expected
    assert above["wake_fraction_viscous"] < 0.01 * expected
    # Above a lifting foil the flow is faster: a negative potential wake fraction.
    over = foils_at_propeller([wing], [Component(1500.0, 0.0)], [0.05, 0.0, -0.5 + 0.2], R, U, rho)
    assert over["wake_fraction"] < 0


def test_element_amplitude_gives_the_elements_wave_resistance_and_2d_wave_height():
    rho, g = 1000.0, 9.80665
    U, h, n = 3.0, 0.5, 40
    ys = onp.linspace(-1.0, 1.0, n + 1)
    load = onp.sqrt(1 - (0.5 * (ys[1:] + ys[:-1])) ** 2) * 100.0 * onp.diff(ys)
    el = dict(xyz_a=(onp.zeros(n), ys[:-1], onp.full(n, -h)), xyz_b=(onp.zeros(n), ys[1:], onp.full(n, -h)))
    for strengths in (dict(force_z=load), dict(source_flux=load / 5000), dict(force_z=load, source_flux=load / 5000)):
        direct = kelvin_wave_resistance(velocity=U, density=rho, length_scales=(2.0, 0.5, h), **el, **strengths)
        via = kelvin_wave_resistance(
            ([], [], []), ([], [], []), velocity=U, density=rho, length_scales=(2.0, 0.5, h),
            extra_source_amplitude=element_source_amplitude(velocity=U, density=rho, **el, **strengths),
        )
        assert float(via) == pytest.approx(float(direct), rel=1e-6)

    # A long lifting line: the waves behind it reach the two-dimensional height
    # 2 Gamma / U exp(-nu h), and there is next to nothing ahead.
    U, h, span, gamma = 3.0, 1.0, 80.0, 1.0
    nu = g / U**2
    wavelength = 2 * onp.pi / nu
    n = 160
    ys = onp.linspace(-span / 2, span / 2, n + 1)
    el = dict(xyz_a=(onp.zeros(n), ys[:-1], onp.full(n, -h)), xyz_b=(onp.zeros(n), ys[1:], onp.full(n, -h)))
    field = wave_field_from_source_amplitude(
        element_source_amplitude(velocity=U, density=rho, force_z=rho * U * gamma * onp.diff(ys), **el),
        U, length=span, k_cap=30.0, level=2,
    )
    field.chunk = 256
    x = onp.linspace(2 * wavelength, 5 * wavelength, 120)
    elevation = onp.ravel(field.elevation(x, 0 * x))
    assert 0.5 * (elevation.max() - elevation.min()) == pytest.approx(2 * gamma / U * onp.exp(-nu * h), rel=0.03)
    assert abs(onp.ravel(field.elevation(onp.array([-3.0 * wavelength]), onp.array([0.0])))[0]) < 0.01 * 2 * gamma / U
    # Lift pushes the surface down just behind the foil.
    assert onp.ravel(field.elevation(onp.array([1.0]), onp.array([0.0])))[0] < 0
