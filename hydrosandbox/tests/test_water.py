import casadi as cas
import pytest
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water


def test_ittc_reference_properties():
    """Compare freshwater & seawater density and viscosity against ITTC 2011 values."""
    fresh_15 = Water.freshwater(temperature_C=15.0)
    assert np.isclose(fresh_15.density(), 999.10, atol=0.15)
    assert np.isclose(fresh_15.dynamic_viscosity(), 1.138e-3, rtol=0.02)
    assert np.isclose(fresh_15.kinematic_viscosity(), 1.139e-6, rtol=0.02)

    fresh_20 = Water.freshwater(temperature_C=20.0)
    assert np.isclose(fresh_20.density(), 998.21, atol=0.15)
    assert np.isclose(fresh_20.kinematic_viscosity(), 1.003e-6, rtol=0.02)

    fresh_4 = Water.freshwater(temperature_C=4.0)
    assert fresh_4.density() > fresh_15.density()

    sea_15 = Water.seawater(temperature_C=15.0)
    assert np.isclose(sea_15.density(), 1025.97, atol=0.25)
    assert np.isclose(sea_15.kinematic_viscosity(), 1.188e-6, rtol=0.03)
    assert sea_15.density() > fresh_15.density()

    sea_20 = Water.seawater(temperature_C=20.0)
    assert np.isclose(sea_20.density(), 1024.76, atol=0.25)
    assert np.isclose(sea_20.kinematic_viscosity(), 1.050e-6, rtol=0.03)


def test_vapor_pressure_and_hydrostatic_depth():
    """Verify vapor pressure trends and smooth hydrostatic depth pressure."""
    fresh_20 = Water.freshwater(temperature_C=20.0)
    fresh_50 = Water.freshwater(temperature_C=50.0)
    fresh_100 = Water.freshwater(temperature_C=99.97)
    sea_20 = Water.seawater(temperature_C=20.0)

    assert np.isclose(fresh_20.vapor_pressure(), 2339.0, rtol=0.02)
    assert fresh_50.vapor_pressure() > fresh_20.vapor_pressure()
    assert np.isclose(fresh_100.vapor_pressure(), 101325.0, rtol=0.02)
    # Salinity lowers vapor pressure slightly via Raoult's law
    assert sea_20.vapor_pressure() < fresh_20.vapor_pressure()

    w0 = Water.seawater(depth=0.0)
    w10 = Water.seawater(depth=10.0)
    rho = float(w0.density())
    assert np.isclose(w10.pressure() - w0.pressure(), rho * w0.gravity * 10.0, rtol=1e-4)

    # Above the surface (depth < 0), pressure smoothly floors near atmospheric pressure
    w_air = Water.seawater(depth=-2.0)
    assert np.isclose(w_air.pressure(), w_air.atmospheric_pressure, atol=200.0)

    assert np.isclose(w0.temperature(), 273.15 + 15.0, atol=1e-9)
    assert np.isclose(w0.temperature_celsius(), 15.0, atol=1e-9)
    assert 1480.0 < float(w0.speed_of_sound()) < 1530.0
    assert 0.06 < float(w0.surface_tension()) < 0.08


def test_water_vectorization_and_slicing():
    """Verify array inputs and __getitem__ slicing on Water."""
    depths = np.linspace(0.0, 20.0, 5)
    w = Water.seawater(depth=depths, temperature_C=15.0)
    assert len(w) == 5
    p = w.pressure()
    assert p.shape == (5,)
    assert np.all(np.diff(p) > 0)

    w_slice = w[1:3]
    assert len(w_slice) == 2
    assert np.allclose(w_slice.depth, depths[1:3])


def test_water_casadi_first_and_second_derivatives():
    """Verify 1st and 2nd CasADi derivatives are finite everywhere including salinity=0."""
    depth_sym = cas.MX.sym("depth")
    temp_sym = cas.MX.sym("temp")
    sal_sym = cas.MX.sym("salinity")
    x = cas.vertcat(depth_sym, temp_sym, sal_sym)

    w_sym = Water(depth=depth_sym, temperature_C=temp_sym, salinity=sal_sym)
    outputs = cas.vertcat(
        w_sym.density(),
        w_sym.dynamic_viscosity(),
        w_sym.kinematic_viscosity(),
        w_sym.vapor_pressure(),
        w_sym.pressure(),
        w_sym.speed_of_sound(),
        w_sym.surface_tension(),
    )

    jac_fn = cas.Function("water_jac", [x], [cas.jacobian(outputs, x)])
    for test_point in ([0.0, 15.0, 0.0], [10.0, 20.0, 35.0], [0.5, 4.0, 0.0]):
        j_val = np.array(jac_fn(test_point))
        assert np.all(np.isfinite(j_val)), f"Non-finite Jacobian at {test_point}"

    # Check scalar Hessian of every output at S = 0.0 (where S**1.5 would fail without smoothing)
    for idx in range(outputs.size1()):
        hess_expr, _ = cas.hessian(outputs[idx], x)
        hess_fn = cas.Function(f"hess_{idx}", [x], [hess_expr])
        for test_point in ([0.5, 15.0, 0.0], [5.0, 18.0, 35.0]):
            h_val = np.array(hess_fn(test_point))
            assert np.all(np.isfinite(h_val)), f"Non-finite Hessian idx={idx} at {test_point}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
