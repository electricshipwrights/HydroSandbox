import pytest
import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.hydrodynamics import cavitation_bucket, evaluate_hydrofoil_2D
from hydrosandbox.performance import HydroOperatingPoint


def test_hydro_operating_point_physics_and_state():
    """Verify Bernoulli pressure, cavitation number, Froude/Weber numbers, and state round-trip."""
    water = Water.seawater(temperature_C=18.0)
    op = HydroOperatingPoint(water=water, velocity=12.0, alpha=2.5, beta=1.0, surface_z=0.0)

    assert op.atmosphere is water
    assert np.isclose(op.local_depth(-0.65), 0.65, atol=1e-12)

    p_loc = op.local_pressure(-0.65)
    q = op.dynamic_pressure()
    p_tot = op.total_pressure(-0.65)
    assert np.isclose(p_tot, p_loc + q, rtol=1e-12)

    # Higher depth raises sigma; higher velocity lowers sigma
    op_fast = HydroOperatingPoint(water=water, velocity=20.0, alpha=2.5, surface_z=0.0)
    assert op.cavitation_number(-1.0) > op.cavitation_number(-0.2)
    assert op.cavitation_number(-0.5) > op_fast.cavitation_number(-0.5)

    # Froude and Weber numbers
    fr_c = op.froude_number(reference_length=0.2)
    fr_h = op.depth_froude_number(-0.65)
    we = op.weber_number(reference_length=0.2)
    assert fr_c > 1.0
    assert fr_h > 1.0
    assert we > 100.0

    # State packing/unpacking round-trip
    state = op.pack_state()
    op2 = op.get_new_instance_with_state(state)
    assert np.isclose(op2.velocity, op.velocity)
    assert np.isclose(op2.surface_z, op.surface_z)
    assert np.isclose(op2.water.salinity, op.water.salinity)


def test_hydro_operating_point_vectorization_and_indexing():
    """Verify vectorized HydroOperatingPoint and slice indexing."""
    vel = np.linspace(6.0, 16.0, 5)
    op = HydroOperatingPoint(
        water=Water.freshwater(temperature_C=20.0),
        velocity=vel,
        alpha=2.0,
        surface_z=0.0,
    )
    assert len(op) == 5
    sig = op.cavitation_number(-0.5)
    assert sig.shape == (5,)
    assert np.all(np.diff(sig) < 0)

    op_sub = op[1:4]
    assert len(op_sub) == 3
    assert np.allclose(op_sub.velocity, vel[1:4])


def test_evaluate_hydrofoil_2d_and_cavitation_bucket():
    """Verify 2D hydrofoil polar evaluation, cavitation drag rise, and bucket peak."""
    af_sym = asb.Airfoil("naca0012")
    water = Water.seawater(temperature_C=15.0)

    # Scalar evaluation below cavitation inception
    res_low = evaluate_hydrofoil_2D(
        airfoil=af_sym,
        velocity=10.0,
        chord=0.18,
        alpha=2.0,
        depth=0.5,
        water=water,
        include_cavitation_penalty=True,
    )
    assert res_low["CL"] > 0.15
    assert res_low["L_over_D"] > 10.0
    assert res_low["Cpmin"] < 0.0
    assert res_low["cavitation_margin"] > 0.0
    assert np.isclose(res_low["CD_cavitation"], 0.0, atol=1e-4)

    # Vectorized sweep well past inception velocity
    res_high = evaluate_hydrofoil_2D(
        airfoil=af_sym,
        velocity=35.0,
        chord=0.18,
        alpha=np.array([0.0, 3.0, 6.0]),
        depth=0.3,
        water=water,
        include_cavitation_penalty=True,
    )
    assert res_high["cavitation_margin"].shape == (3,)
    assert res_high["cavitation_margin"][-1] < 0.0
    assert res_high["CD_cavitation"][-1] > 0.01

    # Symmetric NACA0012 cavitation bucket peak should occur near CL ~ 0
    bucket = cavitation_bucket(
        airfoil=af_sym,
        alphas=np.linspace(-6.0, 8.0, 29),
        velocity=12.0,
        chord=0.15,
        depth=0.5,
        water=water,
    )
    assert bucket["V_bucket_max"] > 12.0
    assert abs(bucket["CL_bucket_peak"]) < 0.15
    assert np.allclose(bucket["sigma_required"], -bucket["Cpmin"])
    assert bucket["CL_max_bucket"] > bucket["CL_min_bucket"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
