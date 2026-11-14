"""
Hull-propeller interaction and coupled electric marine propulsion analysis.

All equations are written in ``aerosandbox.numpy`` with smooth transitions so
propulsion sizes, operating RPM, gearbox reduction ratio, cavitation margins,
and motor ratings can be optimized directly inside ``asb.Opti``.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import numpy as _onp

import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.hull.hull import Hull
from hydrosandbox.performance import HydroOperatingPoint
from hydrosandbox.propulsion.bseries import (
    evaluate_bseries,
    propeller_actuator_disk_power,
)
from hydrosandbox.propulsion.motor import (
    CONTROLLER_ETA_DEFAULT,
    GEAR_ETA_DEFAULT,
    CatalogMotor,
    mass_battery_marine,
    mass_controller_marine,
    mass_gearbox_marine,
    mass_motor_marine,
    motor_performance_drela,
    motor_performance_loss,
)

Vectorizable = float | _onp.ndarray | Any


def _is_symbolic(x: Any) -> bool:
    return bool(np.is_casadi_type(x, recursive=True))


def _smooth_clamp(x: Vectorizable, lo: float, hi: float) -> Vectorizable:
    """Differentiable clamp to ``[lo, hi]`` using ``np.fmin``/``np.fmax``."""
    return np.fmin(np.fmax(x, lo), hi)


def _piecewise_hat_weights(grid: _onp.ndarray, x: Vectorizable) -> Vectorizable:
    """
    Exact continuous piecewise-linear interpolation weights along a strictly
    increasing 1D grid ``grid`` of length ``n >= 2``, supporting both scalar/vector
    NumPy inputs and CasADi symbolics.

    Always returns shape ``(m, n)``.
    """
    g = _onp.asarray(grid, dtype=float)
    n = g.size
    if n < 2:
        raise ValueError("Grid must contain at least two points.")
    xc = np.fmin(np.fmax(x, float(g[0])), float(g[-1]))
    xv = np.reshape(xc, (-1, 1))  # (m, 1)

    cols = []
    # Left endpoint hat: 1 at g[0], 0 at g[1]
    d0 = float(g[1] - g[0])
    cols.append(np.fmax(0.0, 1.0 - (xv - float(g[0])) / max(d0, 1e-12)))
    for i in range(1, n - 1):
        dl = float(g[i] - g[i - 1])
        dr = float(g[i + 1] - g[i])
        left = (xv - float(g[i - 1])) / max(dl, 1e-12)
        right = (float(g[i + 1]) - xv) / max(dr, 1e-12)
        cols.append(np.fmax(0.0, np.fmin(left, right)))
    dn = float(g[-1] - g[-2])
    cols.append(np.fmax(0.0, 1.0 - (float(g[-1]) - xv) / max(dn, 1e-12)))
    return np.concatenate(cols, axis=1)


def _interp_half_breadth(
    hull: Hull, x_body: Vectorizable, z_body: Vectorizable
) -> Vectorizable:
    """
    Smoothly bilinearly interpolate ``hull.half_breadths`` at body coordinates
    ``(x_body, z_body)`` where ``+x`` is aft and ``+z`` is up.
    """
    wx = _piecewise_hat_weights(hull.stations, x_body)  # (m, nx)
    wz = _piecewise_hat_weights(hull.levels, z_body)    # (m, nz)
    hb = np.asarray(hull.half_breadths, dtype=float)     # (nx, nz)
    return np.reshape(np.sum((wx @ hb) * wz, axis=1), (-1, 1))


def hull_propeller_interaction(
    hull: Hull,
    disc_xyz: Sequence[float] | Vectorizable,
    radius: Vectorizable,
    *,
    hub_ratio: Vectorizable = 0.2,
    velocity: Vectorizable = 6.0,
    sinkage: Vectorizable = 0.0,
    trim: Vectorizable = 0.0,
    pivot_xz: tuple[float, float] | None = None,
    thrust_per_shaft: Vectorizable | None = None,
    n_shafts: int = 1,
    water: Water | None = None,
    n_radial: int = 6,
    n_azimuth: int = 24,
    free_surface_bc: Literal["zero_potential", "rigid_lid"] = "zero_potential",
) -> dict[str, Any]:
    """
    Evaluate coupled hull–propeller hydrodynamic interaction using thin-ship
    Rankine sources, a Schlichting 1/7th-law frictional boundary-layer / wake
    defect model, and Dickmann / Lagally thrust deduction.

    Coordinate convention (matching ``hydrosandbox.Hull``): ``+x`` points **aft**,
    ``+y`` athwartships (centerplane ``y = 0``), and ``+z`` points **up** with
    still waterline at ``z = 0``.

    Parameters
    ----------
    hull : Hull
        Bare-hull offset-table geometry.
    disc_xyz : array-like of length 3
        Propeller disc center ``(x, y, z)`` in earth axes [m]. For twin
        symmetric shafts (``n_shafts == 2``), pass one shaft's coordinates
        ``(x, y_off, z)``.
    radius : float or Opti variable
        Propeller tip radius ``D / 2`` [m].
    hub_ratio : float or Opti variable, optional
        Hub-to-tip radius ratio ``r_hub / R`` (default ``0.2``).
    velocity : float or Opti variable, optional
        Ship forward speed ``V_s`` [m/s].
    sinkage : float or Opti variable, optional
        Positive-down hull sinkage [m].
    trim : float or Opti variable, optional
        Bow-up trim angle [deg] (the convention of ``Hull`` and ``CraftBuildup``).
    pivot_xz : tuple[float, float], optional
        Trim/sinkage pivot in body axes; defaults to ``(0.0, 0.0)``.
    thrust_per_shaft : float or Opti variable, optional
        Thrust produced by each propeller [N]. Required to evaluate actuator-disc
        suction velocity ``u_a``, added Lagally hull resistance ``delta_r``, and
        thrust deduction fraction ``t``.
    n_shafts : int, optional
        Number of symmetric propeller shafts (default ``1``).
    water : Water, optional
        Water properties (default: ``Water()``).
    n_radial, n_azimuth : int, optional
        Annular Gauss–Legendre radial rings × midpoint azimuthal quadrature
        points (defaults ``6 x 24`` matching ``boatmath::thinship::propulsion``).
    free_surface_bc : {"zero_potential", "rigid_lid"}, optional
        Free-surface Rankine image sign: ``"zero_potential"`` (``phi = 0`` at
        ``z = 0``, default matching ``boatmath``) or ``"rigid_lid"`` (double-body).

    Returns
    -------
    dict
        Keys include:
        ``w`` (Taylor wake fraction), ``w_potential``, ``w_friction``,
        ``v_advance``, ``area_disc``, ``r_nodes``, ``radial`` (per-ring
        wake breakdown when numeric), and — when ``thrust_per_shaft`` is given —
        ``u_a``, ``delta_r``, ``delta_r_reciprocal``, ``t`` (thrust deduction),
        and ``eta_H = (1 - t) / (1 - w)``.
    """
    if water is None:
        water = Water()
    rho = float(water.density())
    nu = float(water.kinematic_viscosity())

    if pivot_xz is None:
        pivot_xz_eff = (0.0, 0.0)
    else:
        pivot_xz_eff = (pivot_xz[0], pivot_xz[1])

    r_tip = np.fmax(radius, 1e-6)
    r_hub = np.fmax(hub_ratio, 0.0) * r_tip
    area_disc = np.pi * np.fmax(r_tip**2 - r_hub**2, 1e-12)

    # Gauss-Legendre radial quadrature in r^2 and midpoint azimuthal quadrature
    xi_r, wi_r = _onp.polynomial.legendre.leggauss(int(n_radial))
    s_nodes = 0.5 * (1.0 + xi_r)  # shape (nr,)
    r_sq = r_hub**2 + (r_tip**2 - r_hub**2) * s_nodes
    r_nodes = np.sqrt(np.fmax(r_sq, 1e-12))  # (nr,)
    ring_area = 0.5 * np.pi * (r_tip**2 - r_hub**2) * wi_r  # (nr,)

    theta = 2.0 * math.pi * (_onp.arange(n_azimuth) + 0.5) / float(n_azimuth)
    sin_th = _onp.sin(theta)  # (na,)
    cos_th = _onp.cos(theta)  # (na,)

    xc = disc_xyz[0]
    yc = disc_xyz[1]
    zc = disc_xyz[2]

    # Flattened disc coordinates (N = nr * na) as explicit (N, 1) column vectors
    # so NumPy and CasADi MX broadcasting stay identical without [None, :].
    n_pts = int(n_radial) * int(n_azimuth)
    idx_r = _onp.repeat(_onp.arange(int(n_radial)), int(n_azimuth))
    idx_a = _onp.tile(_onp.arange(int(n_azimuth)), int(n_radial))
    r_flat = np.reshape(r_nodes, (-1, 1))[idx_r, 0]
    yq = np.reshape(yc + r_flat * sin_th[idx_a], (n_pts, 1))
    zq = np.reshape(zc + r_flat * cos_th[idx_a], (n_pts, 1))
    xq = np.reshape(_onp.ones((n_pts, 1)) * xc, (n_pts, 1))
    da_q = np.reshape((ring_area / float(n_azimuth))[idx_r], (n_pts, 1))

    # Hull thin-ship source panels
    cells = hull._cells(sinkage=sinkage, trim=trim, pivot_xz=pivot_xz_eff)
    xi_c = np.reshape(0.5 * (cells["x_lo"] + cells["x_hi"]), (-1, 1))
    zeta_c = np.reshape(0.5 * (cells["z_lo"] + cells["z_hi"]), (-1, 1))
    # Trim is in degrees, as everywhere on `Hull` and `CraftBuildup`.
    cos_t = np.cosd(trim)
    sin_t = np.sind(trim)
    flux_c = np.reshape(
        2.0
        * velocity
        * cells["f_sub"]
        * (cos_t * hull._cell_dfx * hull._cell_dz + sin_t * hull._cell_dfz * hull._cell_dx),
        (-1, 1),
    )

    # Filter permanently dry hull cells when hull/sinkage/trim are numeric
    m_cells = int(hull._cell_dx.size)
    if (
        not _is_symbolic(sinkage)
        and not _is_symbolic(trim)
        and not _is_symbolic(cells["f_sub"])
    ):
        active = _onp.asarray(cells["f_sub"], dtype=float) > 0.0
        if _onp.any(active):
            xi_c = xi_c[active, :]
            zeta_c = zeta_c[active, :]
            flux_c = flux_c[active, :]
            m_cells = int(_onp.sum(active))

    image_sign = -1.0 if free_surface_bc == "zero_potential" else 1.0
    eps_r2 = 1e-10

    # Pairwise kernel matrix G_x (M hull cells, N disc points):
    # u_x(q <- c) = (flux_c / 4pi) * (xq - xi_c) * (r^{-3} + image_sign * r_img^{-3})
    ones_m = _onp.ones((m_cells, 1))
    ones_n = _onp.ones((1, n_pts))
    dx_qc = (ones_m @ np.reshape(xq, (1, n_pts))) - (xi_c @ ones_n)
    dy_qc = ones_m @ np.reshape(yq, (1, n_pts))
    dz_qc = (ones_m @ np.reshape(zq, (1, n_pts))) - (zeta_c @ ones_n)
    dz_img = (ones_m @ np.reshape(zq, (1, n_pts))) + (zeta_c @ ones_n)
    r2 = dx_qc**2 + dy_qc**2 + dz_qc**2 + eps_r2
    ri2 = dx_qc**2 + dy_qc**2 + dz_img**2 + eps_r2
    gx_qc = (1.0 / (4.0 * math.pi)) * dx_qc * (r2**(-1.5) + image_sign * (ri2**(-1.5)))

    u_hull_x = np.reshape(np.reshape(flux_c, (1, m_cells)) @ gx_qc, (n_pts, 1))
    w_p_pts = -u_hull_x / np.fmax(velocity, 1e-6)

    # Smooth Schlichting 1/7th-law frictional boundary-layer & stern wake defect
    # matching boatmath::thinship::propulsion::frictional_wake_at (+x aft, +z up).
    ux_b = xq - pivot_xz_eff[0]
    uz_b = zq - (pivot_xz_eff[1] - sinkage)
    x_body = pivot_xz_eff[0] + ux_b * cos_t - uz_b * sin_t
    z_body = pivot_xz_eff[1] + ux_b * sin_t + uz_b * cos_t

    x_bow = float(hull.stations[0])
    x_stern = float(hull.stations[-1])
    l_hull = max(x_stern - x_bow, 1e-4)
    z_keel = float(hull.levels[0])

    x_station = np.fmin(np.fmax(x_body, x_bow), x_stern)
    l_run = np.fmax(x_station - x_bow, 1e-4)
    re_x = np.fmax(np.abs(velocity) * l_run / max(nu, 1e-12), 1e3)
    delta0 = 0.37 * l_run * (re_x ** (-0.2))

    d_x = np.softmax(x_body - x_stern, 0.0, softness=0.002 * l_hull)
    delta = delta0 * np.sqrt(1.0 + 0.5 * d_x / np.fmax(delta0, 1e-4))
    decay = 1.0 / np.sqrt(1.0 + 0.8 * d_x / np.fmax(delta0, 1e-4))

    f_hull = np.reshape(_interp_half_breadth(hull, x_station, z_body), (n_pts, 1))
    abs_y = np.sqrt(yq**2 + 1e-12)
    dy_pos = np.softmax(abs_y - f_hull, 0.0, softness=0.002 * l_hull)
    dz_keel = np.softmax(z_keel - z_body, 0.0, softness=0.002 * l_hull)
    dist = np.sqrt(dy_pos**2 + dz_keel**2 + 1e-12)
    xi_bl = np.fmin(dist / np.fmax(delta, 1e-4), 1.0)
    raw_wf = np.softmax(1.0 - (xi_bl + 1e-7) ** (1.0 / 7.0), 0.0, softness=0.005)

    subm_mask = 0.5 * (1.0 + np.tanh(-zq / 0.01))
    bow_mask = 0.5 * (1.0 + np.tanh((x_body - x_bow) / (0.01 * l_hull)))
    w_f_pts = bow_mask * subm_mask * raw_wf * decay

    w_potential = np.sum(da_q * w_p_pts) / area_disc
    w_friction = np.sum(da_q * w_f_pts) / area_disc
    w_total = w_potential + w_friction
    v_advance = velocity * (1.0 - w_total)

    out: dict[str, Any] = {
        "w": w_total,
        "w_potential": w_potential,
        "w_friction": w_friction,
        "v_advance": v_advance,
        "area_disc": area_disc,
        "r_nodes": r_nodes,
    }

    if not any(
        _is_symbolic(v)
        for v in (disc_xyz, radius, hub_ratio, velocity, sinkage, trim, pivot_xz_eff)
    ):
        w_p_mat = _onp.asarray(w_p_pts, dtype=float).reshape((n_radial, n_azimuth))
        w_f_mat = _onp.asarray(w_f_pts, dtype=float).reshape((n_radial, n_azimuth))
        r_arr = _onp.asarray(r_nodes, dtype=float)
        r_tip_f = float(r_tip)
        out["radial"] = [
            {
                "r": float(r_arr[i]),
                "r_over_R": float(r_arr[i] / r_tip_f),
                "w_potential": float(_onp.mean(w_p_mat[i])),
                "w_friction": float(_onp.mean(w_f_mat[i])),
                "w": float(_onp.mean(w_p_mat[i] + w_f_mat[i])),
            }
            for i in range(n_radial)
        ]

    if thrust_per_shaft is not None:
        v_a_pos = np.fmax(v_advance, 1e-4)
        u_a = 0.5 * (
            -v_a_pos
            + np.sqrt(
                v_a_pos**2 + 2.0 * np.fmax(thrust_per_shaft, 0.0) / np.fmax(rho * area_disc, 1e-12)
            )
        )
        m_q = -2.0 * u_a * da_q  # sink strength at each Gauss point (N,)

        # Force on hull sources by propeller sinks:
        # u_sink_at_c = sum_q m_q * (1/4pi) * (xi_c - xq) * (...) = - sum_q m_q * gx_qc[c, q]
        u_sink_x = -gx_qc @ np.reshape(m_q, (-1, 1))
        delta_r_single = np.sum(-rho * np.reshape(flux_c, (-1, 1)) * u_sink_x)
        delta_r = float(n_shafts) * delta_r_single

        # Reciprocal check: +rho * sum_q m_q * u_hull_x(q)
        delta_r_rec_single = rho * np.sum(m_q * u_hull_x)
        delta_r_reciprocal = float(n_shafts) * delta_r_rec_single

        t_denom = np.fmax(float(n_shafts) * thrust_per_shaft, 1e-9)
        t_ded = delta_r / t_denom
        eta_h = (1.0 - t_ded) / np.fmax(1.0 - w_total, 1e-6)

        out.update(
            {
                "u_a": u_a,
                "delta_r": delta_r,
                "delta_r_reciprocal": delta_r_reciprocal,
                "t": t_ded,
                "eta_H": eta_h,
            }
        )

    return out


class ElectricPropulsionBuildup(asb.ExplicitAnalysis):
    """
    Coupled marine electric propulsion system analysis (wake & thrust deduction
    -> propeller -> reduction gearbox -> motor -> controller -> wiring/battery).

    Compatible with both plain float evaluation and ``aerosandbox.Opti``
    gradient-based design optimization.
    """

    def __init__(
        self,
        op_point: HydroOperatingPoint,
        water: Water | None = None,
        *,
        thrust_total: Vectorizable | None = None,
        drag_bare_N: Vectorizable | None = None,
        hull: Hull | None = None,
        n_shafts: int = 1,
        propeller_model: Literal["bseries", "actuator_disk"] = "bseries",
        propeller_diameter: Vectorizable = 0.25,
        propeller_pitch_ratio: Vectorizable = 0.9,
        propeller_ear: Vectorizable = 0.55,
        propeller_n_blades: int = 3,
        propeller_rpm: Vectorizable = 900.0,
        propeller_hub_ratio: float = 0.2,
        propeller_cop: float = 0.85,
        shaft_depth: Vectorizable = 0.35,
        shaft_angle_deg: Vectorizable = 0.0,
        keller_k: float = 0.1,
        re_correct: bool = True,
        wake_fraction: Vectorizable | None = None,
        thrust_deduction: Vectorizable | None = None,
        disc_xyz: Sequence[float] | Vectorizable | None = None,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: tuple[Vectorizable, Vectorizable] | None = None,
        gearbox_ratio: Vectorizable = 1.0,
        gearbox_efficiency: float = GEAR_ETA_DEFAULT,
        motor: CatalogMotor | None = None,
        motor_model: Literal["loss", "drela"] = "loss",
        motor_kv: Vectorizable = 150.0,
        motor_kt: Vectorizable | None = None,
        motor_r_ll: Vectorizable = 0.05,
        motor_i0: Vectorizable = 1.0,
        motor_c_cu: Vectorizable = 0.05,
        motor_k_h: Vectorizable = 0.0,
        motor_k_e: Vectorizable = 0.0,
        motor_k_0: Vectorizable = 0.0,
        motor_topology: Literal["pmsm", "axial_pmsm", "radial_pmsm", "bldc", "outrunner", "inrunner"] = "axial_pmsm",
        motor_max_cont_torque: Vectorizable | None = None,
        controller_efficiency: float = CONTROLLER_ETA_DEFAULT,
        controller_waterproof: bool = True,
        bus_voltage: Vectorizable | None = None,
        wire_resistance: float = 0.0,
        battery_voltage: Vectorizable | None = None,
        battery_internal_resistance: float = 0.0,
        endurance_hours: Vectorizable | None = None,
        battery_chemistry: Literal["nmc", "lfp", "lifepo4", "li_ion", "lipo"] = "lfp",
        battery_cell_specific_energy_wh_kg: float | None = None,
        battery_pack_cell_fraction: float | None = None,
        hotel_power_w: Vectorizable = 0.0,
    ):
        self.op_point = op_point
        self.water = Water() if water is None else water
        self.thrust_total = thrust_total
        self.drag_bare_N = drag_bare_N
        self.hull = hull
        self.n_shafts = int(n_shafts)
        self.propeller_model = propeller_model
        self.propeller_diameter = propeller_diameter
        self.propeller_pitch_ratio = propeller_pitch_ratio
        self.propeller_ear = propeller_ear
        self.propeller_n_blades = int(propeller_n_blades)
        self.propeller_rpm = propeller_rpm
        self.propeller_hub_ratio = propeller_hub_ratio
        self.propeller_cop = propeller_cop
        self.shaft_depth = shaft_depth
        self.shaft_angle_deg = shaft_angle_deg
        self.keller_k = keller_k
        self.re_correct = re_correct
        self.wake_fraction = wake_fraction
        self.thrust_deduction = thrust_deduction
        self.disc_xyz = disc_xyz
        self.sinkage = sinkage
        self.trim = trim
        self.pivot_xz = pivot_xz
        self.gearbox_ratio = gearbox_ratio
        self.gearbox_efficiency = gearbox_efficiency
        self.motor = motor
        self.motor_model = motor_model
        self.motor_kv = motor_kv
        self.motor_kt = motor_kt
        self.motor_r_ll = motor_r_ll
        self.motor_i0 = motor_i0
        self.motor_c_cu = motor_c_cu
        self.motor_k_h = motor_k_h
        self.motor_k_e = motor_k_e
        self.motor_k_0 = motor_k_0
        self.motor_topology = motor_topology
        self.motor_max_cont_torque = motor_max_cont_torque
        self.controller_efficiency = controller_efficiency
        self.controller_waterproof = controller_waterproof
        self.bus_voltage = bus_voltage
        self.wire_resistance = wire_resistance
        self.battery_voltage = battery_voltage
        self.battery_internal_resistance = battery_internal_resistance
        self.endurance_hours = endurance_hours
        self.battery_chemistry = battery_chemistry
        self.battery_cell_specific_energy_wh_kg = battery_cell_specific_energy_wh_kg
        self.battery_pack_cell_fraction = battery_pack_cell_fraction
        self.hotel_power_w = hotel_power_w

    def run(self) -> dict[str, Any]:
        """Run the coupled propulsion buildup and return performance, margins, and masses."""
        v_ship = self.op_point.velocity
        rho = float(self.water.density())

        cos_shaft = np.fmax(np.cosd(self.shaft_angle_deg), 1e-6)

        # 1. Hull interaction (wake fraction & thrust deduction)
        interaction: dict[str, Any] | None = None
        if self.hull is not None and self.disc_xyz is not None:
            t_seed = (
                self.thrust_total / float(self.n_shafts)
                if self.thrust_total is not None
                else (
                    self.drag_bare_N / (float(self.n_shafts) * cos_shaft)
                    if self.drag_bare_N is not None
                    else None
                )
            )
            interaction = hull_propeller_interaction(
                self.hull,
                self.disc_xyz,
                0.5 * self.propeller_diameter,
                hub_ratio=self.propeller_hub_ratio,
                velocity=v_ship,
                sinkage=self.sinkage,
                trim=self.trim,
                pivot_xz=self.pivot_xz,
                thrust_per_shaft=t_seed,
                n_shafts=self.n_shafts,
                water=self.water,
            )
            w = interaction["w"] if self.wake_fraction is None else self.wake_fraction
            t_ded = (
                interaction.get("t", 0.0)
                if self.thrust_deduction is None
                else self.thrust_deduction
            )
        else:
            w = 0.0 if self.wake_fraction is None else self.wake_fraction
            t_ded = 0.0 if self.thrust_deduction is None else self.thrust_deduction

        # Resolve required bare thrust vs total shaft thrust
        if self.thrust_total is not None:
            t_total_req = self.thrust_total
        elif self.drag_bare_N is not None:
            t_total_req = self.drag_bare_N / np.fmax((1.0 - t_ded) * cos_shaft, 1e-6)
        elif self.propeller_model == "bseries":
            t_total_req = None
        else:
            raise ValueError("Provide either `thrust_total` or `drag_bare_N`.")

        v_advance = v_ship * (1.0 - w)

        # 2. Propeller evaluation
        if self.propeller_model == "bseries":
            prop = evaluate_bseries(
                rpm=self.propeller_rpm,
                diameter=self.propeller_diameter,
                pitch_ratio=self.propeller_pitch_ratio,
                ear=self.propeller_ear,
                n_blades=self.propeller_n_blades,
                v_ship=v_ship,
                wake_fraction=w,
                thrust_deduction=t_ded,
                depth=self.shaft_depth,
                water=self.water,
                shaft_angle_deg=self.shaft_angle_deg,
                keller_k=self.keller_k,
                re_correct=self.re_correct,
            )
            t_per_shaft = prop["thrust"]
            q_prop = prop["torque"]
            p_prop_shaft = prop["P_shaft"]
            eta_prop = prop["eta_0"]
            t_total_prop = float(self.n_shafts) * t_per_shaft
            thrust_residual = (
                t_total_prop - t_total_req if t_total_req is not None else 0.0
            )
            if t_total_req is None:
                t_total_req = t_total_prop
        elif self.propeller_model == "actuator_disk":
            t_per_shaft = t_total_req / float(self.n_shafts)
            act = propeller_actuator_disk_power(
                thrust=t_per_shaft,
                diameter=self.propeller_diameter,
                v_advance=v_advance,
                rho=rho,
                hub_ratio=self.propeller_hub_ratio,
                coefficient_of_performance=self.propeller_cop,
            )
            p_prop_shaft = act["P_shaft"]
            omega_prop = 2.0 * math.pi * (self.propeller_rpm / 60.0)
            q_prop = p_prop_shaft / np.fmax(omega_prop, 1e-9)
            eta_prop = act["eta_0"]
            t_total_prop = t_total_req
            thrust_residual = 0.0
            prop = dict(act)
            prop["T"] = t_per_shaft
            prop["Q"] = q_prop
            prop["v_advance"] = v_advance
        else:
            raise ValueError(f"Unknown propeller_model {self.propeller_model!r}.")

        # 3. Gearbox reduction (always motor_rpm = prop_rpm * gearbox_ratio)
        g_is_unity = not _is_symbolic(self.gearbox_ratio) and abs(float(self.gearbox_ratio) - 1.0) < 1e-12
        eff_g = 1.0 if g_is_unity else float(self.gearbox_efficiency)
        motor_rpm = self.propeller_rpm * self.gearbox_ratio
        motor_torque = q_prop / (np.fmax(self.gearbox_ratio, 1e-6) * eff_g)
        p_motor_shaft = p_prop_shaft / eff_g

        # 4. Motor + Controller
        envelope_margins: dict[str, Vectorizable] = {}
        if self.motor is not None:
            m_perf = self.motor.performance_smooth(
                rpm=motor_rpm,
                torque=motor_torque,
                controller_efficiency=self.controller_efficiency,
                v_bus_override=self.bus_voltage,
            )
            envelope_margins = self.motor.envelope_margins(
                rpm=motor_rpm,
                torque=motor_torque,
                controller_efficiency=self.controller_efficiency,
                v_bus_override=self.bus_voltage,
            )
            eta_motor = m_perf["eta_motor"]
            p_elec_one = m_perf["P_elec"]
            v_bus_eff = (
                self.bus_voltage
                if self.bus_voltage is not None
                else (
                    m_perf["V_req"]
                    if m_perf.get("V_req") is not None
                    else (float(self.motor.v_dc) if self.motor.v_dc is not None else 48.0)
                )
            )
            i_dc_one = (
                m_perf["I_dc_pack"]
                if m_perf.get("I_dc_pack") is not None
                else p_elec_one / np.fmax(v_bus_eff, 1e-6)
            )
        elif self.motor_model == "drela":
            d_perf = motor_performance_drela(
                rpm=motor_rpm,
                torque=motor_torque,
                kv=self.motor_kv,
                resistance=self.motor_r_ll,
                no_load_current=self.motor_i0,
            )
            eta_motor = d_perf["efficiency"]
            p_elec_motor = d_perf["electrical_power"]
            p_elec_one = p_elec_motor / max(float(self.controller_efficiency), 1e-6)
            v_bus_eff = d_perf["voltage"] if self.bus_voltage is None else self.bus_voltage
            i_dc_one = p_elec_one / np.fmax(v_bus_eff, 1e-6)
            m_perf = dict(d_perf)
            m_perf["eta_motor"] = eta_motor
            m_perf["P_elec"] = p_elec_one
            m_perf["V_req"] = d_perf["voltage"]
            m_perf["I_dc_pack"] = i_dc_one
        elif self.motor_model == "loss":
            m_perf = motor_performance_loss(
                rpm=motor_rpm,
                torque=motor_torque,
                c_cu=self.motor_c_cu,
                k_h=self.motor_k_h,
                k_e=self.motor_k_e,
                k_0=self.motor_k_0,
                kt=self.motor_kt,
                kv=self.motor_kv,
                r_ll=self.motor_r_ll,
                i0=self.motor_i0,
                v_bus=self.bus_voltage,
                controller_efficiency=self.controller_efficiency,
                topology="bldc" if self.motor_topology == "bldc" else "pmsm",
            )
            eta_motor = m_perf["eta_motor"]
            p_elec_one = m_perf["P_elec"]
            v_bus_eff = (
                self.bus_voltage
                if self.bus_voltage is not None
                else (m_perf["V_req"] if m_perf.get("V_req") is not None else 48.0)
            )
            i_dc_one = (
                p_elec_one / np.fmax(v_bus_eff, 1e-6)
                if m_perf.get("I_dc_pack") is None
                else m_perf["I_dc_pack"]
            )
        else:
            raise ValueError(f"Unknown motor_model {self.motor_model!r}.")

        # 5. System electrical & battery buildup
        p_prop_total = float(self.n_shafts) * p_prop_shaft
        p_elec_total = float(self.n_shafts) * p_elec_one
        i_bus_total = float(self.n_shafts) * i_dc_one
        v_pack = v_bus_eff if self.battery_voltage is None else self.battery_voltage
        i_pack = (p_elec_total + self.hotel_power_w) / np.fmax(v_pack, 1e-6)
        p_wire_loss = (i_pack**2) * float(self.wire_resistance)
        p_batt_loss = (i_pack**2) * float(self.battery_internal_resistance)
        p_battery_prop = p_elec_total + p_wire_loss + p_batt_loss
        p_battery_total = p_battery_prop + self.hotel_power_w

        eta_h = (1.0 - t_ded) / np.fmax(1.0 - w, 1e-6)
        eta_drive = p_prop_total / np.fmax(p_elec_total, 1e-12)
        p_effective = t_total_prop * (1.0 - t_ded) * cos_shaft * v_ship
        eta_overall = p_effective / np.fmax(p_battery_prop, 1e-12)

        # 6. Component masses
        if self.motor is not None and self.motor.mass_kg is not None:
            m_motor_one = float(self.motor.mass_kg)
        else:
            q_rating = (
                self.motor_max_cont_torque
                if self.motor_max_cont_torque is not None
                else (
                    self.motor.tq_cont_nm
                    if (self.motor is not None and self.motor.tq_cont_nm is not None)
                    else motor_torque
                )
            )
            topo = (
                "outrunner_pod"
                if self.motor_topology in ("bldc", "outrunner", "outrunner_pod")
                else (
                    "direct_drive_torque"
                    if self.motor_topology == "direct_drive_torque"
                    else ("radial_pmsm" if self.motor_topology == "radial_pmsm" else "axial_pmsm")
                )
            )
            m_motor_one = mass_motor_marine(q_rating, topology=topo)
        mass_motor = float(self.n_shafts) * m_motor_one
        mass_controller = float(self.n_shafts) * mass_controller_marine(
            p_elec_one, waterproof=self.controller_waterproof
        )
        mass_gearbox = (
            0.0
            if g_is_unity
            else float(self.n_shafts)
            * mass_gearbox_marine(p_prop_shaft, rpm_in=motor_rpm, rpm_out=self.propeller_rpm)
        )
        if self.endurance_hours is not None:
            battery_energy_wh = p_battery_total * self.endurance_hours
            chem = self.battery_chemistry
            if chem == "lifepo4":
                chem_norm: Literal["lfp", "nmc"] = "lfp"
            elif chem in ("nmc", "li_ion", "lipo"):
                chem_norm = "nmc"
            else:
                chem_norm = "lfp"
            mass_battery = mass_battery_marine(
                battery_energy_wh,
                chemistry=chem_norm,
                cell_specific_energy_wh_kg=self.battery_cell_specific_energy_wh_kg,
                pack_cell_fraction=self.battery_pack_cell_fraction,
            )
        else:
            battery_energy_wh = 0.0
            mass_battery = 0.0

        mass_total = mass_motor + mass_controller + mass_gearbox + mass_battery

        return {
            "V_ship": v_ship,
            "v_advance": v_advance,
            "wake_fraction": w,
            "thrust_deduction": t_ded,
            "eta_H": eta_h,
            "T_per_shaft": t_per_shaft,
            "T_total": t_total_prop,
            "T_required": t_total_req,
            "thrust_residual_N": thrust_residual,
            "Q_prop": q_prop,
            "P_prop_shaft": p_prop_total,
            "P_motor_shaft": float(self.n_shafts) * p_motor_shaft,
            "P_elec": p_elec_total,
            "P_wire_loss": p_wire_loss,
            "P_battery_loss": p_batt_loss,
            "P_battery": p_battery_total,
            "P_effective": p_effective,
            "motor_rpm": motor_rpm,
            "motor_torque": motor_torque,
            "gearbox_ratio": self.gearbox_ratio,
            "gearbox_efficiency": eff_g,
            "eta_prop": eta_prop,
            "eta_motor": eta_motor,
            "eta_drive": eta_drive,
            "eta_overall": eta_overall,
            "V_bus": v_bus_eff,
            "I_dc_per_shaft": i_dc_one,
            "I_bus_total": i_bus_total,
            "battery_energy_Wh": battery_energy_wh,
            "mass_motor_kg": mass_motor,
            "mass_controller_kg": mass_controller,
            "mass_gearbox_kg": mass_gearbox,
            "mass_battery_kg": mass_battery,
            "mass_propulsion_total_kg": mass_total,
            "propeller": prop,
            "motor": m_perf,
            "envelope_margins": envelope_margins,
            "interaction": interaction,
        }


__all__ = [
    "hull_propeller_interaction",
    "ElectricPropulsionBuildup",
]
