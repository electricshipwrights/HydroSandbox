"""
Unit, golden-parity, hull-interaction, and Opti co-design tests for
hydrosandbox.propulsion.
"""

from __future__ import annotations

import json
import os
import math
from pathlib import Path

import casadi as ca
import numpy as _onp
import pytest

import aerosandbox as asb
import hydrosandbox as hsb
from hydrosandbox.propulsion import (
    BSeriesInputs,
    ElectricPropulsionBuildup,
    MotorDatabase,
    TopOperatingPoint,
    evaluate_bseries,
    hull_propeller_interaction,
    mass_battery_marine,
    mass_controller_marine,
    mass_gearbox_marine,
    mass_motor_marine,
    match_motors_to_curve,
    motor_performance_drela,
    motor_performance_loss,
    optimize_bseries_propeller,
    propeller_actuator_disk_power,
    sweep_bseries_rpm,
    through_gearbox,
)

# boatmath's propeller goldens: `$BOATMATH_DIR`, else a `boatmath` checkout beside
# this repository (or beside the repository a worktree belongs to).
_GOLDEN_SUBPATH = Path("crates") / "propeller" / "tests" / "golden"
_here = Path(__file__).resolve()
_candidates = [Path(os.environ["BOATMATH_DIR"])] if os.environ.get("BOATMATH_DIR") else []
_candidates += [_here.parents[3] / "boatmath", _here.parents[5] / "boatmath"]
GOLDEN_DIR = next(
    (c / _GOLDEN_SUBPATH for c in _candidates if (c / _GOLDEN_SUBPATH).exists()),
    _candidates[0] / _GOLDEN_SUBPATH,
)


@pytest.mark.parametrize("case_name", ["small_twin", "top_speed", "web_default"])
def test_boatmath_golden_parity(case_name: str) -> None:
    """B-series RPM sweep and discrete motor matching match ../boatmath goldens."""
    golden_path = GOLDEN_DIR / f"{case_name}.json"
    if not golden_path.exists():
        pytest.skip(f"External boatmath golden fixture missing: {golden_path}")

    g = json.loads(golden_path.read_text(encoding="utf-8"))
    o = g["opts"]
    top = TopOperatingPoint(speed=o["V_top"], thrust=o["T_top"]) if "V_top" in o else None
    inp = BSeriesInputs(
        speed=o["V_s"],
        thrust=o["T"],
        shafts=o["shafts"],
        wake=o["wake"],
        thrust_deduction=o["thrustDeduction"],
        d_min=o["D_min"],
        d_max=o["D_max"],
        blades=tuple(o["Z"]),
        depth=o["depth"],
        keller_k=o["kellerK"],
        cavitation=o["cavitation"],
        re_correct=o["reCorrect"],
        strict_ear=o["strictEAR"],
        top=top,
    )

    sweep = sweep_bseries_rpm(inp, power_cap=g.get("powerCap", 1.6))
    gb = g["sweep"]["best"]
    b = sweep["best"]
    assert b is not None
    assert b["Z"] == gb["Z"]
    for key in ["rpm", "D", "PD", "EAR", "P_shaft", "eta0", "Q"]:
        rel = abs(b[key] - gb[key]) / max(abs(gb[key]), 1e-9)
        assert rel < 1e-5, f"{case_name} sweep.best[{key}] mismatch: {b[key]} vs {gb[key]}"

    db = MotorDatabase.vendored(include_foil_assist_pods=False)
    scatter = match_motors_to_curve(db, sweep["points"], check_top=(top is not None))
    g_all = {d["motor"]: d for d in g["motors"]["all"]}
    p_all = {d["motor"].id: d for d in scatter["all"]}
    assert set(p_all.keys()) == set(g_all.keys())
    assert [d["motor"].id for d in scatter["dots"]] == [d["motor"] for d in g["motors"]["dots"]]

    for mid, gd in g_all.items():
        pd = p_all[mid]
        d = pd["point"]["design"]
        run = pd["run"]
        checks = [
            (d["rpm"], gd["rpm"]),
            (d["P_shaft"], gd["P_shaft"]),
            (run["P_elec"], gd["P_elec"]),
            (run["eta"], gd["eta"]),
            (run["etaDrive"], gd["etaDrive"]),
            (run["ratio"], gd["ratio"]),
            (run["rpm"], gd["motorRpm"]),
            (run["Q"], gd["motorQ"]),
            (run["V_bus"], gd["V_bus"]),
            (run["I_dc"], gd["I_dc"]),
            (pd["load"], gd["load"]),
        ]
        for pv, gv in checks:
            if gv is None:
                continue
            rel = abs(pv - gv) / max(abs(gv), 1e-9)
            assert rel < 1e-5, f"{case_name} motor={mid}: {pv} vs {gv}"


def test_bseries_physics_and_opti() -> None:
    """Open-water B-series evaluation, oblique flow, cavitation, and Opti sizing."""
    res0 = evaluate_bseries(
        rpm=1400.0,
        diameter=0.22,
        pitch_ratio=1.05,
        ear=0.55,
        n_blades=3,
        v_ship=4.5,
        wake_fraction=0.05,
        depth=0.35,
    )
    assert res0["thrust"] > 100.0
    assert 0.5 < res0["eta_0"] < 0.85
    assert res0["keller_margin"] > 0.0
    assert res0["burrill_margin"] > 0.0
    assert res0["Re_075"] > 1e5

    res_inc = evaluate_bseries(
        rpm=1400.0,
        diameter=0.22,
        pitch_ratio=1.05,
        ear=0.55,
        n_blades=3,
        v_ship=4.5,
        wake_fraction=0.05,
        depth=0.35,
        shaft_angle_deg=12.0,
    )
    assert res_inc["thrust"] != pytest.approx(res0["thrust"])
    assert math.isfinite(res_inc["eta_0"])

    act = propeller_actuator_disk_power(
        thrust=250.0, diameter=0.22, v_advance=4.5, rho=1025.9
    )
    assert act["P_shaft"] > 250.0 * 4.5
    assert 0.5 < act["eta_0"] < 0.9

    opt = optimize_bseries_propeller(
        thrust=250.0,
        v_ship=4.5,
        d_min=0.12,
        d_max=0.32,
        blades=(3,),
        depth=0.35,
    )
    best = opt["best"]
    assert best["thrust"] == pytest.approx(250.0, rel=1e-4)
    assert 0.55 < best["eta_0"] < 0.85
    assert best["keller_margin"] >= -1e-6
    assert best["burrill_margin"] >= -1e-6


def test_gearbox_direction_and_motor_models_ad() -> None:
    """Verify reduction direction (motor_rpm = prop_rpm * ratio) and AD gradients."""
    db = MotorDatabase.vendored(include_foil_assist_pods=True)
    mot = db.motor("emrax_208_hv")
    assert mot is not None
    r_direct = through_gearbox(db, mot, prop_rpm=1000.0, prop_torque=20.0, ratio=1.0)
    r_geared = through_gearbox(db, mot, prop_rpm=1000.0, prop_torque=20.0, ratio=3.0)
    assert r_direct is not None and r_direct["ok"]
    assert r_geared is not None and r_geared["ok"]
    assert r_geared["rpm"] == pytest.approx(3000.0)
    assert r_geared["Q"] == pytest.approx(20.0 / (3.0 * 0.97))

    # CasADi AD vs finite differences on Drela & 4-term polynomial loss models
    rpm_sym = ca.MX.sym("rpm")
    tq_sym = ca.MX.sym("torque")
    drela = motor_performance_drela(rpm=rpm_sym, torque=tq_sym, kv=140.0, resistance=0.04, no_load_current=1.2)
    f_d = ca.Function("f_d", [rpm_sym, tq_sym], [drela["electrical_power"], ca.gradient(drela["electrical_power"], rpm_sym), ca.gradient(drela["electrical_power"], tq_sym)])
    p0, dp_drpm, dp_dtq = [float(x) for x in f_d(3500.0, 4.0)]
    eps = 1e-4
    p_rpm_hi = float(f_d(3500.0 + eps, 4.0)[0])
    p_rpm_lo = float(f_d(3500.0 - eps, 4.0)[0])
    p_tq_hi = float(f_d(3500.0, 4.0 + eps)[0])
    p_tq_lo = float(f_d(3500.0, 4.0 - eps)[0])
    assert dp_drpm == pytest.approx((p_rpm_hi - p_rpm_lo) / (2.0 * eps), rel=1e-5)
    assert dp_dtq == pytest.approx((p_tq_hi - p_tq_lo) / (2.0 * eps), rel=1e-5)
    assert p0 > 3500.0 * 2.0 * math.pi / 60.0 * 4.0

    loss_sym = motor_performance_loss(
        rpm=rpm_sym,
        torque=tq_sym,
        c_cu=0.25,
        k_h=0.02,
        k_e=1e-5,
        k_0=5.0,
        kt=0.068,
        kv=140.0,
        r_ll=0.04,
        v_bus=48.0,
    )
    f_l = ca.Function(
        "f_l",
        [rpm_sym, tq_sym],
        [
            loss_sym["P_elec"],
            loss_sym["I_dc_pack"],
            loss_sym["I_dc_min_bus"],
            ca.gradient(loss_sym["P_elec"], tq_sym),
        ],
    )
    pl, idc_pack, idc_min, dpl_dtq = [float(x) for x in f_l(2500.0, 3.0)]
    pl_hi = float(f_l(2500.0, 3.0 + eps)[0])
    pl_lo = float(f_l(2500.0, 3.0 - eps)[0])
    assert dpl_dtq == pytest.approx((pl_hi - pl_lo) / (2.0 * eps), rel=1e-5)
    assert pl > 2500.0 * 2.0 * math.pi / 60.0 * 3.0
    assert idc_pack < idc_min  # 48V bus > back-EMF at 2500 RPM


def test_mass_regressions_and_foil_assist_pods() -> None:
    """Marine torque-based motor mass laws and optional ~100W pod catalog."""
    q_ref = 20.0
    m_ax = mass_motor_marine(q_ref, topology="axial_pmsm")
    m_rad = mass_motor_marine(q_ref, topology="radial_pmsm")
    m_dd = mass_motor_marine(q_ref, topology="direct_drive_torque")
    assert 0.5 < m_ax < m_rad < m_dd

    assert mass_controller_marine(5000.0, waterproof=True) > mass_controller_marine(5000.0, waterproof=False)
    assert mass_gearbox_marine(5000.0, rpm_in=1500.0, rpm_out=1500.0) == 0.0
    assert mass_gearbox_marine(5000.0, rpm_in=4500.0, rpm_out=1500.0) > 0.1
    assert mass_battery_marine(1000.0, chemistry="lfp") > mass_battery_marine(1000.0, chemistry="nmc")

    db_base = MotorDatabase.vendored(include_foil_assist_pods=False)
    db_pods = MotorDatabase.vendored(include_foil_assist_pods=True)
    assert len(db_base.motors) == 159
    assert len(db_pods.motors) == 163

    # Small ~100W stern foil-assist duty point (4.5 m/s, 18 N -> ~81 W effective)
    inp_small = BSeriesInputs(
        speed=4.5,
        thrust=22.0,
        d_min=0.06,
        d_max=0.14,
        blades=(3,),
        depth=0.25,
        cavitation=True,
        strict_ear=True,
    )
    sweep_small = sweep_bseries_rpm(inp_small, samples=60)
    assert sweep_small["feasible"]
    pod_scatter = match_motors_to_curve(
        db_pods,
        sweep_small["points"],
        all_windings=True,
        include=lambda m: m.id.startswith(("bluerobotics_", "flipsky_")),
    )
    matched_ids = {d["motor"].id for d in pod_scatter["all"]}
    assert "bluerobotics_m200_24v" in matched_ids or any("flipsky_" in mid for mid in matched_ids)


def test_hull_propeller_interaction_wigley() -> None:
    """Thin-ship wake & Lagally thrust deduction symmetry, aft decay, reciprocity."""
    hull = hsb.Hull.wigley(length=10.0, beam=1.0, draft=0.625)
    near = hull_propeller_interaction(
        hull, [5.2, 0.0, -0.35], 0.15, velocity=5.0, thrust_per_shaft=400.0
    )
    far = hull_propeller_interaction(
        hull, [6.5, 0.0, -0.35], 0.15, velocity=5.0, thrust_per_shaft=400.0
    )
    assert near["w"] > 0.0
    assert near["w_potential"] > far["w_potential"] > 0.0
    assert near["w_friction"] > far["w_friction"] > 0.0
    assert near["t"] > far["t"] > 0.0
    assert near["eta_H"] > 1.0
    assert near["delta_r"] == pytest.approx(near["delta_r_reciprocal"], rel=1e-6)
    assert len(near["radial"]) == 6

    port = hull_propeller_interaction(
        hull, [5.2, 0.25, -0.35], 0.12, velocity=5.0, thrust_per_shaft=200.0, n_shafts=2
    )
    stbd = hull_propeller_interaction(
        hull, [5.2, -0.25, -0.35], 0.12, velocity=5.0, thrust_per_shaft=200.0, n_shafts=2
    )
    assert port["w"] == pytest.approx(stbd["w"], rel=1e-12)
    assert port["t"] == pytest.approx(stbd["t"], rel=1e-12)


def test_coupled_electric_propulsion_opti_codesign() -> None:
    """End-to-end Opti co-design over D, PD, EAR, rpm, gearbox_ratio with envelope."""
    hull = hsb.Hull.wigley(length=10.0, beam=1.0, draft=0.625)
    db = MotorDatabase.vendored(include_foil_assist_pods=True)
    mot = db.motor("flipsky_6384_120kv_36v")
    assert mot is not None

    opti = asb.Opti()
    d = opti.variable(init_guess=0.20, lower_bound=0.10, upper_bound=0.35)
    pd = opti.variable(init_guess=1.1, lower_bound=0.5, upper_bound=1.4)
    ear = opti.variable(init_guess=0.55, lower_bound=0.35, upper_bound=1.05)
    rpm = opti.variable(init_guess=1800.0, lower_bound=500.0, upper_bound=5000.0)
    gr = opti.variable(init_guess=2.0, lower_bound=1.0, upper_bound=10.0)

    buildup = ElectricPropulsionBuildup(
        op_point=hsb.HydroOperatingPoint(velocity=6.0),
        drag_bare_N=80.0,
        hull=hull,
        disc_xyz=[5.2, 0.0, -0.35],
        propeller_diameter=d,
        propeller_pitch_ratio=pd,
        propeller_ear=ear,
        propeller_rpm=rpm,
        gearbox_ratio=gr,
        motor=mot,
        endurance_hours=1.5,
    )
    res = buildup.run()

    opti.subject_to(
        [
            res["thrust_residual_N"] == 0.0,
            res["propeller"]["J"] >= 0.1,
            res["propeller"]["J"] <= 1.35,
            res["propeller"]["keller_margin"] >= 0.0,
            res["propeller"]["burrill_margin"] >= 0.0,
            res["P_prop_shaft"] >= 10.0,
        ]
    )
    for margin in res["envelope_margins"].values():
        opti.subject_to(margin >= 0.0)
    opti.minimize(res["P_battery"])
    sol = opti.solve(verbose=False)

    p_batt = float(_onp.ravel(sol(res["P_battery"]))[0])
    p_eff = float(_onp.ravel(sol(res["P_effective"]))[0])
    eta_tot = float(_onp.ravel(sol(res["eta_overall"]))[0])
    gr_sol = float(_onp.ravel(sol(gr))[0])
    rpm_sol = float(_onp.ravel(sol(rpm))[0])
    m_rpm_sol = float(_onp.ravel(sol(res["motor_rpm"]))[0])
    m_tot = float(_onp.ravel(sol(res["mass_propulsion_total_kg"]))[0])

    assert p_eff == pytest.approx(480.0, rel=1e-4)
    assert 550.0 < p_batt < 850.0
    assert 0.55 < eta_tot < 0.85
    assert m_rpm_sol == pytest.approx(rpm_sol * gr_sol, rel=1e-6)
    assert m_tot > mot.mass_kg

    # Also verify actuator_disk + drela branch of ElectricPropulsionBuildup
    ad_buildup = ElectricPropulsionBuildup(
        op_point=hsb.HydroOperatingPoint(velocity=6.0),
        drag_bare_N=80.0,
        hull=hull,
        disc_xyz=[5.2, 0.0, -0.35],
        propeller_model="actuator_disk",
        propeller_diameter=0.18,
        propeller_rpm=1500.0,
        gearbox_ratio=2.5,
        motor_model="drela",
        motor_kv=120.0,
        motor_r_ll=0.05,
        motor_i0=1.2,
        endurance_hours=1.0,
    ).run()
    assert 500.0 < ad_buildup["P_battery"] < 900.0
    assert 0.5 < ad_buildup["eta_overall"] < 0.85


def test_hull_propeller_interaction_takes_trim_in_degrees() -> None:
    """Trim is in degrees, as on `Hull`: 2 deg bow-up sinks the stern toward the disc."""
    hull = hsb.Hull.wigley(length=10.0, beam=1.0, draft=0.625)
    common = dict(velocity=3.0, pivot_xz=(0.0, 0.0))
    level = hull_propeller_interaction(hull, [5.2, 0.0, -0.35], 0.15, trim=0.0, **common)
    trimmed = hull_propeller_interaction(hull, [5.2, 0.0, -0.35], 0.15, trim=2.0, **common)
    # The stern sits ~0.17 m deeper beside the (fixed) disc, so the potential wake
    # grows moderately. Read as radians, 2 would be 115 degrees and flip the sign of
    # the hull's sources, making it negative.
    assert 0.0 < float(level["w_potential"]) < float(trimmed["w_potential"])
    assert float(trimmed["w_potential"]) < 2.0 * float(level["w_potential"])
