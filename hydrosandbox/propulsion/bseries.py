"""
Wageningen B-series marine propeller model and actuator-disk baseline.

Provides both:
1. Smooth, CasADi-differentiable evaluations (``evaluate_bseries``,
   ``bseries_cavitation``, ``propeller_actuator_disk_power``) for gradient-based
   design optimization with ``aerosandbox.Opti``.
2. Numerical pitch-matching, pattern-search sizing, and RPM curve sweeping
   (``BSeriesInputs``, ``evaluate_bseries_at_thrust``, ``sweep_bseries_rpm``)
   following Oosterveld & van Oossanen (1975) and the ``boatmath`` / ``propopt``
   formulation.

All equations use SI units (m, m/s, N, N*m, W, Pa) with rotational speed inputs
in rpm where explicitly named ``rpm``.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Sequence
import numpy as _onp
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.fluid import Water

# Wageningen B-series polynomial terms (Re = 2e6 fit):
# [i_J, j_PD, k_EAR, l_Z, coeff] representing coeff * J^i * (P/D)^j * EAR^k * Z^l.
KT_TERMS: tuple[tuple[int, int, int, int, float], ...] = (
    (0, 0, 0, 0, 0.00880496),
    (0, 0, 0, 1, 0.0144043),
    (0, 0, 0, 2, -0.000606848),
    (0, 0, 1, 1, -0.0125894),
    (0, 0, 1, 2, 0.000690904),
    (0, 0, 2, 0, -0.0507214),
    (0, 1, 0, 0, 0.166351),
    (0, 1, 0, 1, 0.0143481),
    (0, 2, 0, 0, 0.158114),
    (0, 2, 1, 0, 0.415437),
    (0, 2, 2, 1, -0.00410798),
    (0, 3, 0, 0, -0.133698),
    (0, 3, 0, 1, -0.00841728),
    (0, 3, 1, 1, -0.0317791),
    (0, 3, 1, 2, 0.00421749),
    (0, 3, 2, 2, -0.00146564),
    (0, 6, 0, 0, 0.00638407),
    (1, 0, 0, 0, -0.204554),
    (1, 0, 0, 2, -0.0049819),
    (1, 0, 1, 1, 0.0109689),
    (1, 0, 2, 1, 0.018604),
    (1, 1, 0, 1, 0.0606826),
    (1, 1, 1, 0, -0.481497),
    (1, 2, 0, 2, -0.00163652),
    (1, 3, 0, 1, 0.0168424),
    (1, 6, 0, 2, -0.000328787),
    (1, 6, 2, 0, 0.010465),
    (2, 0, 0, 1, -0.0530054),
    (2, 0, 0, 2, 0.0025983),
    (2, 0, 1, 0, -0.147581),
    (2, 0, 2, 0, 0.0854559),
    (2, 6, 0, 0, -0.00132718),
    (2, 6, 0, 2, 0.000116502),
    (2, 6, 2, 0, -0.00648272),
    (3, 0, 0, 2, -0.000560528),
    (3, 0, 1, 0, 0.168496),
    (3, 0, 2, 0, -0.0504475),
    (3, 3, 0, 1, -0.00102296),
    (3, 6, 1, 2, 5.65229e-05),
)

KQ_TERMS: tuple[tuple[int, int, int, int, float], ...] = (
    (0, 0, 0, 0, 0.00379368),
    (0, 0, 2, 0, 0.015896),
    (0, 0, 2, 2, -0.0001843),
    (0, 1, 0, 1, 0.00513696),
    (0, 1, 1, 0, -0.0408811),
    (0, 1, 2, 0, -0.0502782),
    (0, 2, 0, 0, 0.00344778),
    (0, 2, 1, 0, 0.188561),
    (0, 2, 1, 1, -0.0269403),
    (0, 2, 1, 2, 0.00155334),
    (0, 2, 2, 1, 0.0126803),
    (0, 3, 1, 0, 0.0161886),
    (0, 3, 2, 0, -0.0397722),
    (0, 3, 2, 2, -0.000425399),
    (0, 6, 0, 1, -0.000313912),
    (0, 6, 1, 1, -0.00142121),
    (0, 6, 1, 2, 0.000302683),
    (0, 6, 2, 0, -0.00350024),
    (0, 6, 2, 1, 0.00334268),
    (0, 6, 2, 2, -0.0004659),
    (1, 0, 0, 1, -0.00370871),
    (1, 0, 1, 2, 0.000269551),
    (1, 0, 2, 0, 0.0471729),
    (1, 0, 2, 1, -0.00383637),
    (1, 1, 0, 0, -0.032241),
    (1, 1, 0, 1, 0.0209449),
    (1, 1, 0, 2, -0.00183491),
    (1, 1, 1, 0, -0.108009),
    (1, 1, 1, 1, 0.00438388),
    (1, 3, 1, 0, 0.00318086),
    (1, 6, 2, 2, 5.54194e-05),
    (2, 0, 0, 0, 0.00886523),
    (2, 0, 1, 1, -0.00723408),
    (2, 0, 1, 2, 0.00083265),
    (2, 1, 0, 1, 0.00474319),
    (2, 1, 1, 0, -0.0885381),
    (2, 2, 2, 0, 0.0417122),
    (2, 3, 2, 1, -0.00318278),
    (3, 0, 0, 1, -0.0106854),
    (3, 0, 1, 0, 0.0558082),
    (3, 0, 1, 1, 0.0035985),
    (3, 0, 2, 0, 0.0196283),
    (3, 1, 2, 0, -0.030055),
    (3, 2, 0, 2, 0.000112451),
    (3, 3, 0, 1, 0.00110903),
    (3, 3, 2, 2, 8.69243e-05),
    (3, 6, 0, 2, -2.97228e-05),
)

J_BOUNDS: tuple[float, float] = (0.001, 2.0)
PD_BOUNDS: tuple[float, float] = (0.6, 1.4)
EAR_BOUNDS: tuple[float, float] = (0.30, 1.05)

RE_FIT: float = 2.0e6
RE_CD_FLOOR: float = 5.0e4
BURRILL_A: float = 0.26
BURRILL_B: float = 0.57
BURRILL_SIGMA_RANGE: tuple[float, float] = (0.1, 2.0)
HYSTERESIS: float = 0.002

_EAR_RANGE_BY_Z: dict[int, tuple[float, float]] = {
    2: (0.30, 0.38),
    3: (0.35, 0.80),
    4: (0.40, 1.00),
    5: (0.45, 1.05),
    6: (0.50, 0.80),
    7: (0.55, 0.85),
}


def ear_series_bounds(n_blades: int) -> tuple[float, float] | None:
    """Return the Wageningen B-series tested blade-area-ratio bounds for ``n_blades``."""
    return _EAR_RANGE_BY_Z.get(int(n_blades))


def _is_symbolic(val: Any) -> bool:
    return bool(np.is_casadi_type(val, recursive=True))


def _polyval6(c: Sequence[Any], x: Any) -> Any:
    return (((((c[6] * x + c[5]) * x + c[4]) * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0]


def bseries_pitch_polynomials(
    J: Vectorizable,
    ear: Vectorizable,
    n_blades: int,
) -> tuple[list[Vectorizable], list[Vectorizable]]:
    """
    Compute degree-6 polynomial coefficients in ``P/D`` for ``K_T`` and ``K_Q``
    at a given advance ratio ``J``, expanded area ratio ``ear``, and blade count.
    """
    z = float(n_blades)
    jp = (1.0, J, J * J, J * J * J)
    bp = (1.0, ear, ear * ear)
    zp = (1.0, z, z * z)
    a = [0.0] * 7
    b = [0.0] * 7
    for i, j, k, l, coeff in KT_TERMS:
        a[j] = a[j] + coeff * jp[i] * bp[k] * zp[l]
    for i, j, k, l, coeff in KQ_TERMS:
        b[j] = b[j] + coeff * jp[i] * bp[k] * zp[l]
    return a, b


def _bseries_kt_poly_num(J: float, ear: float, z: float) -> list[float]:
    jp = (1.0, J, J * J, J * J * J)
    bp = (1.0, ear, ear * ear)
    zp = (1.0, z, z * z)
    a = [0.0] * 7
    for i, j, k, l, coeff in KT_TERMS:
        a[j] += coeff * jp[i] * bp[k] * zp[l]
    return a


def _bseries_kq_poly_num(J: float, ear: float, z: float) -> list[float]:
    jp = (1.0, J, J * J, J * J * J)
    bp = (1.0, ear, ear * ear)
    zp = (1.0, z, z * z)
    b = [0.0] * 7
    for i, j, k, l, coeff in KQ_TERMS:
        b[j] += coeff * jp[i] * bp[k] * zp[l]
    return b


def _bseries_pitch_polynomials_num(
    J: float,
    ear: float,
    z: float,
) -> tuple[list[float], list[float]]:
    """Fast scalar NumPy/Python path for numeric B-series sweep loops."""
    jp = (1.0, J, J * J, J * J * J)
    bp = (1.0, ear, ear * ear)
    zp = (1.0, z, z * z)
    a = [0.0] * 7
    b = [0.0] * 7
    for i, j, k, l, coeff in KT_TERMS:
        a[j] += coeff * jp[i] * bp[k] * zp[l]
    for i, j, k, l, coeff in KQ_TERMS:
        b[j] += coeff * jp[i] * bp[k] * zp[l]
    return a, b


def _effective_pitch_polynomials(
    v_advance: Vectorizable,
    shaft_angle_deg: Vectorizable,
    rev_per_sec: Vectorizable,
    diameter: Vectorizable,
    ear: Vectorizable,
    n_blades: int,
    numeric: bool = False,
) -> tuple[list[Any], list[Any], Any]:
    """
    Compute effective degree-6 ``P/D`` polynomials for ``K_T`` and ``K_Q`` with
    Gutsche's 16-azimuthal-point oblique-flow correction when inclined.
    """
    if not _is_symbolic(shaft_angle_deg) and float(shaft_angle_deg) == 0.0:
        j = v_advance / (rev_per_sec * diameter)
        if numeric:
            a, b = _bseries_pitch_polynomials_num(float(j), float(ear), float(n_blades))
        else:
            a, b = bseries_pitch_polynomials(j, ear, n_blades)
        return a, b, j

    if numeric:
        eps = math.radians(float(shaft_angle_deg))
        sin_e, cos_e = math.sin(eps), math.cos(eps)
        v_ax = float(v_advance) * cos_e
        j_a = v_ax / (float(rev_per_sec) * float(diameter))
        lam = (j_a * abs(sin_e) / max(abs(cos_e), 1e-6)) / (0.7 * math.pi)
        denom0 = j_a * j_a + (0.7 * math.pi) ** 2
        p_a = [0.0] * 7
        p_b = [0.0] * 7
        n_az = 16
        z = float(n_blades)
        for k in range(n_az):
            theta = 2.0 * math.pi * k / n_az
            denom_th = max(1.0 - lam * math.sin(theta), 0.05)
            j_th = min(max(j_a / denom_th, J_BOUNDS[0]), J_BOUNDS[1])
            w_th = (j_a * j_a + (0.7 * math.pi * denom_th) ** 2) / denom0 / n_az
            pa, pb = _bseries_pitch_polynomials_num(j_th, float(ear), z)
            for idx in range(7):
                p_a[idx] += pa[idx] * w_th
                p_b[idx] += pb[idx] * w_th
        return p_a, p_b, j_a

    sin_e = np.sind(shaft_angle_deg)
    cos_e = np.cosd(shaft_angle_deg)
    abs_sin_e = (
        np.sqrt(sin_e**2 + 1e-12) - 1e-6
        if _is_symbolic(shaft_angle_deg)
        else _onp.abs(sin_e)
    )
    abs_cos_e = np.fmax(np.sqrt(cos_e**2 + 1e-12), 1e-6)
    v_ax = v_advance * cos_e
    j_a = v_ax / (rev_per_sec * diameter)
    lam = (j_a * abs_sin_e / abs_cos_e) / (0.7 * math.pi)
    denom0 = j_a * j_a + (0.7 * math.pi) ** 2
    p_a: list[Any] = [0.0] * 7
    p_b: list[Any] = [0.0] * 7
    n_az = 16
    for k in range(n_az):
        theta = 2.0 * math.pi * k / n_az
        denom_th = np.fmax(1.0 - lam * math.sin(theta), 0.05)
        j_th = np.clip(j_a / denom_th, J_BOUNDS[0], J_BOUNDS[1])
        w_th = (j_a * j_a + (0.7 * math.pi * denom_th) ** 2) / denom0 / n_az
        pa, pb = bseries_pitch_polynomials(j_th, ear, n_blades)
        for idx in range(7):
            p_a[idx] = p_a[idx] + pa[idx] * w_th
            p_b[idx] = p_b[idx] + pb[idx] * w_th
    return p_a, p_b, j_a


def bseries_kt(
    J: Vectorizable,
    pitch_ratio: Vectorizable,
    ear: Vectorizable,
    n_blades: int = 3,
) -> Vectorizable:
    """Evaluate raw Wageningen B-series thrust coefficient ``K_T`` at ``Re = 2e6``."""
    a, _ = bseries_pitch_polynomials(J, ear, n_blades)
    return _polyval6(a, pitch_ratio)


def bseries_kq(
    J: Vectorizable,
    pitch_ratio: Vectorizable,
    ear: Vectorizable,
    n_blades: int = 3,
) -> Vectorizable:
    """Evaluate raw Wageningen B-series torque coefficient ``K_Q`` at ``Re = 2e6``."""
    _, b = bseries_pitch_polynomials(J, ear, n_blades)
    return _polyval6(b, pitch_ratio)


def _section_drag_coefficient(re: Vectorizable, t_over_c: Vectorizable, numeric: bool = False):
    if numeric:
        r = max(float(re), 1.0)
        return 2.0 * (1.0 + 2.0 * float(t_over_c)) * (0.044 / (r ** (1.0 / 6.0)) - 5.0 / (r ** (2.0 / 3.0)))
    r = np.fmax(re, 1.0)
    return 2.0 * (1.0 + 2.0 * t_over_c) * (0.044 / r ** (1.0 / 6.0) - 5.0 / r ** (2.0 / 3.0))


def bseries_reynolds_correction(
    diameter: Vectorizable,
    ear: Vectorizable,
    n_blades: int,
    v_advance: Vectorizable,
    rev_per_sec: Vectorizable,
    kinematic_viscosity: Vectorizable = 1.0e-6,
    shaft_angle_deg: Vectorizable = 0.0,
) -> dict[str, Vectorizable]:
    """
    Compute the ITTC / Oosterveld-van Oossanen Reynolds correction from the
    series ``Re = 2e6`` polynomial fit to actual operating Reynolds number at
    ``0.75 R``.
    """
    numeric = not (
        _is_symbolic(diameter)
        or _is_symbolic(ear)
        or _is_symbolic(v_advance)
        or _is_symbolic(rev_per_sec)
        or _is_symbolic(kinematic_viscosity)
        or _is_symbolic(shaft_angle_deg)
    )
    if numeric and not any(
        isinstance(v, (_onp.ndarray,)) for v in (diameter, ear, v_advance, rev_per_sec, kinematic_viscosity, shaft_angle_deg)
    ):
        eps = math.radians(float(shaft_angle_deg))
        sin_e, cos_e = math.sin(eps), math.cos(eps)
        z = float(n_blades)
        c075 = 2.073 * (float(ear) / z) * float(diameter)
        tc075 = (0.0185 - 0.00125 * z) / (2.073 * float(ear) / z)
        vr2 = (
            (float(v_advance) * cos_e) ** 2
            + (0.75 * math.pi * float(rev_per_sec) * float(diameter)) ** 2
            + 0.5 * (float(v_advance) * sin_e) ** 2
        )
        re = c075 * math.sqrt(max(vr2, 1e-18)) / max(float(kinematic_viscosity), 1e-12)
        d_cd = _section_drag_coefficient(re, tc075, numeric=True) - _section_drag_coefficient(RE_FIT, tc075, numeric=True)
        cz_d = 2.073 * float(ear)
        return {
            "Re_075": re,
            "t_over_c_075": tc075,
            "dC_D": d_cd,
            "dkt_per_pd": -0.30 * d_cd * cz_d,
            "dkq": 0.25 * d_cd * cz_d,
            "cd_formula_ok": re >= RE_CD_FLOOR,
        }

    cos_e = np.cosd(shaft_angle_deg)
    sin_e = np.sind(shaft_angle_deg)
    z = float(n_blades)
    c075 = 2.073 * (ear / z) * diameter
    tc075 = (0.0185 - 0.00125 * z) / (2.073 * ear / z)
    vr2 = (
        (v_advance * cos_e) ** 2
        + (0.75 * math.pi * rev_per_sec * diameter) ** 2
        + 0.5 * (v_advance * sin_e) ** 2
    )
    re = c075 * np.sqrt(np.fmax(vr2, 1e-18)) / np.fmax(kinematic_viscosity, 1e-12)
    d_cd = _section_drag_coefficient(re, tc075) - _section_drag_coefficient(RE_FIT, tc075)
    cz_d = 2.073 * ear
    return {
        "Re_075": re,
        "t_over_c_075": tc075,
        "dC_D": d_cd,
        "dkt_per_pd": -0.30 * d_cd * cz_d,
        "dkq": 0.25 * d_cd * cz_d,
    }


def _resolve_fluid_scalars(
    water: Water | None,
    depth: Vectorizable,
    rho: Vectorizable | None,
    kinematic_viscosity: Vectorizable | None,
    atmospheric_pressure: Vectorizable | None,
    vapor_pressure: Vectorizable | None,
    gravity: Vectorizable | None,
):
    if water is None and all(
        x is None for x in (rho, kinematic_viscosity, atmospheric_pressure, vapor_pressure, gravity)
    ):
        water = Water.seawater()
    if water is not None:
        rho_v = water.density() if rho is None else rho
        nu_v = water.kinematic_viscosity() if kinematic_viscosity is None else kinematic_viscosity
        p_atm_v = water.atmospheric_pressure if atmospheric_pressure is None else atmospheric_pressure
        p_vap_v = water.vapor_pressure() if vapor_pressure is None else vapor_pressure
        g_v = water.gravity if gravity is None else gravity
        p_stat = (
            water.pressure(depth=depth)
            if (rho is None and atmospheric_pressure is None and gravity is None)
            else p_atm_v + rho_v * g_v * np.fmax(depth, 0.0)
        )
    else:
        rho_v = 1000.0 if rho is None else rho
        nu_v = 1.0e-6 if kinematic_viscosity is None else kinematic_viscosity
        p_atm_v = 101325.0 if atmospheric_pressure is None else atmospheric_pressure
        p_vap_v = 2339.0 if vapor_pressure is None else vapor_pressure
        g_v = 9.81 if gravity is None else gravity
        p_stat = p_atm_v + rho_v * g_v * depth
    return rho_v, nu_v, p_stat, p_vap_v


def bseries_cavitation(
    thrust: Vectorizable,
    diameter: Vectorizable,
    pitch_ratio: Vectorizable,
    ear: Vectorizable,
    n_blades: int,
    v_advance: Vectorizable,
    rev_per_sec: Vectorizable,
    depth: Vectorizable = 0.35,
    water: Water | None = None,
    keller_k: float = 0.1,
    shaft_angle_deg: Vectorizable = 0.0,
    rho: Vectorizable | None = None,
    atmospheric_pressure: Vectorizable | None = None,
    vapor_pressure: Vectorizable | None = None,
    gravity: Vectorizable | None = None,
) -> dict[str, Vectorizable]:
    """
    Evaluate Keller minimum expanded-area ratio and Burrill 5%-back-cavitation
    limits at ``0.7 R`` for a Wageningen B-series propeller.

    Positive ``keller_margin`` and ``burrill_margin`` indicate non-cavitating
    operation within each criterion.
    """
    rho_v, _, p_static, p_vap = _resolve_fluid_scalars(
        water, depth, rho, None, atmospheric_pressure, vapor_pressure, gravity
    )
    cos_e = np.cosd(shaft_angle_deg)
    sin_e = np.sind(shaft_angle_deg)
    abs_sin_e = (
        np.sqrt(sin_e**2 + 1e-12) - 1e-6
        if _is_symbolic(shaft_angle_deg)
        else np.fabs(sin_e)
    )
    vr2_07 = (v_advance * cos_e) ** 2 + (
        0.7 * math.pi * rev_per_sec * diameter + v_advance * abs_sin_e
    ) ** 2
    dp = np.fmax(p_static - p_vap, 1.0)
    sigma_07 = dp / np.fmax(0.5 * rho_v * vr2_07, 1e-12)
    ear_min = (1.3 + 0.3 * n_blades) * thrust / np.fmax(dp * diameter**2, 1e-12) + keller_k
    a_proj = ear * (1.067 - 0.229 * pitch_ratio) * (math.pi * diameter**2 / 4.0)
    tau_c = thrust / np.fmax(0.5 * rho_v * np.fmax(a_proj, 1e-12) * vr2_07, 1e-12)
    tau_allow = BURRILL_A * np.fmax(sigma_07, 1e-12) ** BURRILL_B
    return {
        "sigma_07": sigma_07,
        "EAR_min": ear_min,
        "keller_margin": ear - ear_min,
        "tau_c": tau_c,
        "tau_allow": tau_allow,
        "burrill_margin": tau_allow - tau_c,
    }


def evaluate_bseries(
    rpm: Vectorizable,
    diameter: Vectorizable,
    pitch_ratio: Vectorizable,
    ear: Vectorizable,
    n_blades: int = 3,
    v_ship: Vectorizable = 6.0,
    wake_fraction: Vectorizable = 0.0,
    thrust_deduction: Vectorizable = 0.0,
    depth: Vectorizable = 0.35,
    water: Water | None = None,
    shaft_angle_deg: Vectorizable = 0.0,
    keller_k: float = 0.1,
    re_correct: bool = True,
    rho: Vectorizable | None = None,
    kinematic_viscosity: Vectorizable | None = None,
    atmospheric_pressure: Vectorizable | None = None,
    vapor_pressure: Vectorizable | None = None,
    gravity: Vectorizable | None = None,
) -> dict[str, Vectorizable]:
    """
    Evaluate a Wageningen B-series propeller (single shaft) using smooth,
    CasADi-differentiable ``aerosandbox.numpy`` operations.
    """
    rho_v, nu_v, _, _ = _resolve_fluid_scalars(
        water, depth, rho, kinematic_viscosity, atmospheric_pressure, vapor_pressure, gravity
    )
    n = rpm / 60.0
    v_a = v_ship * (1.0 - wake_fraction)
    p_a, p_b, j = _effective_pitch_polynomials(
        v_a, shaft_angle_deg, n, diameter, ear, n_blades, numeric=False
    )
    kt = _polyval6(p_a, pitch_ratio)
    kq = _polyval6(p_b, pitch_ratio)

    re_info = bseries_reynolds_correction(
        diameter=diameter,
        ear=ear,
        n_blades=n_blades,
        v_advance=v_a,
        rev_per_sec=n,
        kinematic_viscosity=nu_v,
        shaft_angle_deg=shaft_angle_deg,
    )
    if re_correct:
        kt = kt + re_info["dkt_per_pd"] * pitch_ratio
        kq = kq + re_info["dkq"]

    thrust = rho_v * n**2 * diameter**4 * kt
    torque = rho_v * n**2 * diameter**5 * kq
    p_shaft = 2.0 * math.pi * n * torque

    cos_e = np.cosd(shaft_angle_deg)
    v_ax = v_a * cos_e
    eta_0 = thrust * v_ax / np.fmax(p_shaft, 1e-12)
    thrust_net = thrust * (1.0 - thrust_deduction) * cos_e
    eta_h = (1.0 - thrust_deduction) / np.fmax(1.0 - wake_fraction, 1e-6)
    eta_propulsive = thrust_net * v_ship / np.fmax(p_shaft, 1e-12)

    a0 = math.pi * diameter**2 / 4.0
    c_th = thrust / np.fmax(0.5 * rho_v * v_ax**2 * a0, 1e-12)
    eta_ideal = 2.0 / (1.0 + np.sqrt(1.0 + np.fmax(c_th, 0.0)))

    cav = bseries_cavitation(
        thrust=thrust,
        diameter=diameter,
        pitch_ratio=pitch_ratio,
        ear=ear,
        n_blades=n_blades,
        v_advance=v_a,
        rev_per_sec=n,
        depth=depth,
        water=water,
        keller_k=keller_k,
        shaft_angle_deg=shaft_angle_deg,
        rho=rho,
        atmospheric_pressure=atmospheric_pressure,
        vapor_pressure=vapor_pressure,
        gravity=gravity,
    )
    return {
        "n": n,
        "rpm": rpm,
        "D": diameter,
        "PD": pitch_ratio,
        "EAR": ear,
        "Z": n_blades,
        "J": j,
        "v_ship": v_ship,
        "v_advance": v_a,
        "K_T": kt,
        "K_Q": kq,
        "thrust": thrust,
        "thrust_net": thrust_net,
        "torque": torque,
        "P_shaft": p_shaft,
        "eta_0": eta_0,
        "eta_hull": eta_h,
        "eta_propulsive": eta_propulsive,
        "C_Th": c_th,
        "eta_ideal": eta_ideal,
        "eta_of_ideal": eta_0 / np.fmax(eta_ideal, 1e-12),
        "Re_075": re_info["Re_075"],
        "dC_D": re_info["dC_D"],
        **cav,
    }


def propeller_actuator_disk_power(
    thrust: Vectorizable,
    diameter: Vectorizable,
    v_advance: Vectorizable,
    rho: Vectorizable = 1025.9,
    hub_ratio: float = 0.2,
    coefficient_of_performance: float = 0.85,
) -> dict[str, Vectorizable]:
    """
    Smooth actuator-disk shaft power model for conceptual sizing when blade
    geometry is not yet specified.
    """
    area = 0.25 * math.pi * diameter**2 * (1.0 - hub_ratio**2)
    q_va = np.fmax(v_advance, 1e-6)
    u_a = 0.5 * (-q_va + np.sqrt(q_va**2 + 2.0 * thrust / np.fmax(rho * area, 1e-12)))
    p_ideal = thrust * (v_advance + u_a)
    p_shaft = p_ideal / np.fmax(coefficient_of_performance, 1e-6)
    eta_0 = thrust * v_advance / np.fmax(p_shaft, 1e-12)
    return {
        "area_disc": area,
        "u_a": u_a,
        "P_ideal": p_ideal,
        "P_shaft": p_shaft,
        "eta_0": eta_0,
    }


# ---------------------------------------------------------------------------
# Numeric pitch-matched evaluation and RPM curve sweep (boatmath / propopt)
# ---------------------------------------------------------------------------


@dataclass
class TopOperatingPoint:
    """Optional second operating condition (e.g. sprint / takeoff hump)."""

    speed: float
    thrust: float


@dataclass
class BSeriesInputs:
    """Inputs matching ``boatmath::propeller::bseries::Inputs``."""

    speed: float
    thrust: float
    shafts: int = 1
    wake: float = 0.0
    thrust_deduction: float = 0.0
    d_min: float = 0.08
    d_max: float = 0.35
    blades: tuple[int, ...] = (3, 4)
    depth: float = 0.35
    keller_k: float = 0.1
    cavitation: bool = True
    re_correct: bool = True
    strict_ear: bool = True
    top: TopOperatingPoint | None = None
    shaft_angle_deg: float = 0.0
    # Fluid properties: defaults match boatmath::propeller::bseries (freshwater 20 C).
    rho: float = 1000.0
    nu: float = 1.0e-6
    p_atm: float = 101325.0
    p_vapour: float = 2339.0
    gravity: float = 9.81
    water: Water | None = None

    @classmethod
    def with_water(
        cls,
        water: Water,
        speed: float,
        thrust: float,
        depth: float = 0.35,
        **kwargs,
    ) -> "BSeriesInputs":
        """Construct inputs populated from a ``hydrosandbox.Water`` instance."""
        return cls(
            speed=speed,
            thrust=thrust,
            depth=depth,
            water=water,
            rho=float(water.density()),
            nu=float(water.kinematic_viscosity()),
            p_atm=float(water.atmospheric_pressure),
            p_vapour=float(water.vapor_pressure()),
            gravity=float(water.gravity),
            **kwargs,
        )


@dataclass
class _BSeriesConfig:
    v_s: float
    v_a: float
    r_t: float
    shafts: int
    t_total: float
    t_target: float
    p_e: float
    eta_h: float
    d_min: float
    d_max: float
    z: tuple[float, ...]
    depth: float
    keller_k: float
    cavitation: bool
    strict_ear: bool
    re_correct: bool
    grid_d: int
    grid_ear: int
    top: dict[str, float] | None
    shaft_angle_deg: float
    rho: float
    nu: float
    p_atm: float
    p_vapour: float
    gravity: float

    @classmethod
    def from_inputs(cls, i: BSeriesInputs) -> "_BSeriesConfig":
        rho = float(i.water.density()) if i.water is not None else float(i.rho)
        nu = float(i.water.kinematic_viscosity()) if i.water is not None else float(i.nu)
        p_atm = float(i.water.atmospheric_pressure) if i.water is not None else float(i.p_atm)
        p_vapour = float(i.water.vapor_pressure()) if i.water is not None else float(i.p_vapour)
        gravity = float(i.water.gravity) if i.water is not None else float(i.gravity)

        w, t = float(i.wake), float(i.thrust_deduction)
        shafts = max(int(i.shafts), 1)
        r_t = float(i.thrust) * (1.0 - t)
        top_cfg = None
        if i.top is not None and i.top.speed > 0.0 and i.top.thrust > 0.0:
            top_cfg = {
                "v_s": float(i.top.speed),
                "v_a": float(i.top.speed) * (1.0 - w),
                "t_total": float(i.top.thrust),
                "t_target": float(i.top.thrust) / shafts,
            }
        return cls(
            v_s=float(i.speed),
            v_a=float(i.speed) * (1.0 - w),
            r_t=r_t,
            shafts=shafts,
            t_total=float(i.thrust),
            t_target=float(i.thrust) / shafts,
            p_e=r_t * float(i.speed),
            eta_h=(1.0 - t) / (1.0 - w),
            d_min=min(float(i.d_min), float(i.d_max)),
            d_max=max(float(i.d_min), float(i.d_max)),
            z=tuple(float(b) for b in i.blades),
            depth=float(i.depth),
            keller_k=float(i.keller_k),
            cavitation=bool(i.cavitation),
            strict_ear=bool(i.strict_ear),
            re_correct=bool(i.re_correct),
            grid_d=16,
            grid_ear=12,
            top=top_cfg,
            shaft_angle_deg=float(i.shaft_angle_deg),
            rho=rho,
            nu=nu,
            p_atm=p_atm,
            p_vapour=p_vapour,
            gravity=gravity,
        )


def solve_bseries_pitch(
    a_coeffs: Sequence[float],
    kt_req: float,
    dkt_per_pd: float = 0.0,
    pd_bounds: tuple[float, float] = PD_BOUNDS,
    iters: int = 40,
) -> float | None:
    """Bisect ``P/D`` inside ``pd_bounds`` to match ``K_T == kt_req``."""
    a = list(a_coeffs)
    a[1] += dkt_per_pd
    lo, hi = pd_bounds
    f_lo = _polyval6(a, lo) - kt_req
    f_hi = _polyval6(a, hi) - kt_req
    if f_lo > 0.0 or f_hi < 0.0:
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _polyval6(a, mid) - kt_req < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _num_reynolds_correction(
    c: _BSeriesConfig, d: float, ear: float, z: float, v_a: float, n: float
) -> dict[str, Any]:
    eps = math.radians(c.shaft_angle_deg)
    sin_e, cos_e = math.sin(eps), math.cos(eps)
    chord075 = 2.073 * (ear / z) * d
    vr2 = (v_a * cos_e) ** 2 + (0.75 * math.pi * n * d) ** 2 + 0.5 * (v_a * sin_e) ** 2
    re = chord075 * math.sqrt(vr2) / c.nu
    tc = (0.0185 - 0.00125 * z) / (2.073 * ear / z)
    r_clamped = max(re, 1.0)
    cd_re = 2.0 * (1.0 + 2.0 * tc) * (0.044 / (r_clamped ** (1.0 / 6.0)) - 5.0 / (r_clamped ** (2.0 / 3.0)))
    cd_fit = 2.0 * (1.0 + 2.0 * tc) * (0.044 / (RE_FIT ** (1.0 / 6.0)) - 5.0 / (RE_FIT ** (2.0 / 3.0)))
    d_cd = cd_re - cd_fit
    cz_d = 2.073 * ear
    return {
        "re": re,
        "d_cd": d_cd,
        "dkt_per_pd": -0.30 * d_cd * cz_d,
        "dkq": 0.25 * d_cd * cz_d,
        "cd_formula_ok": re >= RE_CD_FLOOR,
    }


def _num_cavitation_metrics(
    c: _BSeriesConfig, t: float, d: float, pd: float, ear: float, z: float, v_a: float, n: float
) -> dict[str, Any]:
    dp = max(c.p_atm + c.rho * c.gravity * c.depth - c.p_vapour, 1e-6)
    ear_min = (1.3 + 0.3 * z) * t / (dp * d * d) + c.keller_k
    eps = math.radians(c.shaft_angle_deg)
    sin_e, cos_e = math.sin(eps), math.cos(eps)
    vr2 = (v_a * cos_e) ** 2 + (0.7 * math.pi * n * d + v_a * abs(sin_e)) ** 2
    sigma = dp / (0.5 * c.rho * max(vr2, 1e-12))
    a_p = ear * (1.067 - 0.229 * pd) * (math.pi * d * d / 4.0)
    tau_c = t / (0.5 * c.rho * max(a_p, 1e-12) * max(vr2, 1e-12))
    tau_allow = BURRILL_A * max(sigma, 1e-12) ** BURRILL_B
    return {
        "EAR_min": ear_min,
        "kellerOk": ear >= ear_min,
        "sigma": sigma,
        "tau_c": tau_c,
        "tau_allow": tau_allow,
        "burrillOk": tau_c <= tau_allow,
    }


def _at_top_condition(
    c: _BSeriesConfig, top: dict[str, float], d: float, pd: float, ear: float, z: float
) -> dict[str, Any] | None:
    eps = math.radians(c.shaft_angle_deg)
    sin_e, cos_e = math.sin(eps), math.cos(eps)
    v_top = top["v_a"]
    v_ax = v_top * cos_e
    want = top["t_target"]

    # Hoist n-invariant Reynolds terms outside the bisection loop.
    if c.re_correct:
        chord075 = 2.073 * (ear / z) * d
        tc = (0.0185 - 0.00125 * z) / (2.073 * ear / z)
        cd_scale = 2.0 * (1.0 + 2.0 * tc)
        cd_fit = cd_scale * (0.044 / (RE_FIT ** (1.0 / 6.0)) - 5.0 / (RE_FIT ** (2.0 / 3.0)))
        cz_d = 2.073 * ear
        vr_va2 = (v_top * cos_e) ** 2 + 0.5 * (v_top * sin_e) ** 2
        tip075 = 0.75 * math.pi * d

        def _dkt_fast(n_val: float) -> float:
            vr2 = vr_va2 + (tip075 * n_val) ** 2
            re = chord075 * math.sqrt(vr2) / c.nu
            r_c = max(re, 1.0)
            cd_re = cd_scale * (0.044 / (r_c ** (1.0 / 6.0)) - 5.0 / (r_c ** (2.0 / 3.0)))
            return -0.30 * (cd_re - cd_fit) * cz_d * pd
    else:
        def _dkt_fast(n_val: float) -> float:
            return 0.0

    if c.shaft_angle_deg == 0.0:
        pd_p = (1.0, pd, pd * pd, pd**3, pd**4, pd**5, pd**6)
        ep_p = (1.0, ear, ear * ear)
        zp_p = (1.0, z, z * z)
        c0 = c1 = c2 = c3 = 0.0
        for i, j_pow, k, l, coeff in KT_TERMS:
            term = coeff * pd_p[j_pow] * ep_p[k] * zp_p[l]
            if i == 0:
                c0 += term
            elif i == 1:
                c1 += term
            elif i == 2:
                c2 += term
            else:
                c3 += term

        def thrust_at(n: float) -> float | None:
            j = v_top / (n * d)
            if not (J_BOUNDS[0] <= j <= J_BOUNDS[1]):
                return None
            kt = c0 + j * (c1 + j * (c2 + j * c3)) + _dkt_fast(n)
            return c.rho * n * n * (d**4) * kt if kt > 0.0 else None
    else:
        def thrust_at(n: float) -> float | None:
            p_a, _, j = _effective_pitch_polynomials(
                v_top, c.shaft_angle_deg, n, d, ear, int(z), numeric=True
            )
            if not (J_BOUNDS[0] <= j <= J_BOUNDS[1]):
                return None
            kt = _polyval6(p_a, pd) + _dkt_fast(n)
            return c.rho * n * n * (d**4) * kt if kt > 0.0 else None

    lo = v_ax / (J_BOUNDS[1] * d)
    hi = v_ax / (J_BOUNDS[0] * d)
    t_hi = thrust_at(hi * 0.999)
    if t_hi is None or t_hi < want:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        tm = thrust_at(mid)
        if tm is not None and tm >= want:
            hi = mid
        else:
            lo = mid
    n = 0.5 * (lo + hi)
    p_a, p_b, j = _effective_pitch_polynomials(
        top["v_a"], c.shaft_angle_deg, n, d, ear, int(z), numeric=True
    )
    if c.re_correct:
        rc = _num_reynolds_correction(c, d, ear, z, top["v_a"], n)
        dkt, dkq = rc["dkt_per_pd"] * pd, rc["dkq"]
    else:
        dkt, dkq = 0.0, 0.0
    kt = _polyval6(p_a, pd) + dkt
    kq = _polyval6(p_b, pd) + dkq
    if not (kq > 0.0 and kt > 0.0):
        return None
    q = c.rho * n * n * (d**5) * kq
    p_shaft = 2.0 * math.pi * n * q
    if not (p_shaft > 0.0):
        return None
    cav = _num_cavitation_metrics(c, want, d, pd, ear, z, top["v_a"], n)
    return {
        "n": n,
        "rpm": 60.0 * n,
        "J": j,
        "K_T": kt,
        "K_Q": kq,
        "T": want,
        "Q": q,
        "P_shaft": p_shaft,
        "eta0": want * v_ax / p_shaft,
        "sigma": cav["sigma"],
        "tau_c": cav["tau_c"],
        "tau_allow": cav["tau_allow"],
        "burrillOk": cav["burrillOk"],
        "EAR_min": cav["EAR_min"],
        "kellerOk": cav["kellerOk"],
    }


def _evaluate_cfg(
    c: _BSeriesConfig, n: float, d: float, ear: float, z: float
) -> dict[str, Any] | None:
    if c.shaft_angle_deg == 0.0:
        j = c.v_a / (n * d)
        if not (J_BOUNDS[0] <= j <= J_BOUNDS[1]):
            return None
        p_a = _bseries_kt_poly_num(j, ear, z)
        p_b = None
    else:
        p_a, p_b, j = _effective_pitch_polynomials(
            c.v_a, c.shaft_angle_deg, n, d, ear, int(z), numeric=True
        )
        if not (J_BOUNDS[0] <= j <= J_BOUNDS[1]):
            return None
    kt_req = c.t_target / (c.rho * n * n * (d**4))
    if not (kt_req > 0.0):
        return None
    rc = None
    dkt_per_pd, dkq = 0.0, 0.0
    if c.re_correct:
        rc = _num_reynolds_correction(c, d, ear, z, c.v_a, n)
        dkt_per_pd = rc["dkt_per_pd"]
        dkq = rc["dkq"]
    pd = solve_bseries_pitch(p_a, kt_req, dkt_per_pd)
    if pd is None or not (pd > 0.0):
        return None
    if p_b is None:
        p_b = _bseries_kq_poly_num(j, ear, z)
    kq = _polyval6(p_b, pd) + dkq
    if not (kq > 0.0):
        return None
    q = c.rho * n * n * (d**5) * kq
    p_shaft = 2.0 * math.pi * n * q
    if not (p_shaft > 0.0):
        return None
    cos_e = math.cos(math.radians(c.shaft_angle_deg))
    eta0 = c.t_target * (c.v_a * cos_e) / p_shaft
    if not (eta0 > 0.0 and math.isfinite(eta0)):
        return None

    cav = _num_cavitation_metrics(c, c.t_target, d, pd, ear, z, c.v_a, n)
    if c.cavitation and not (cav["kellerOk"] and cav["burrillOk"]):
        return None
    er = ear_series_bounds(int(z))
    ear_in = er is not None and (er[0] <= ear <= er[1])
    if c.strict_ear and not ear_in:
        return None

    a0 = math.pi * d * d / 4.0
    cth = c.t_target / (0.5 * c.rho * (c.v_a * cos_e) ** 2 * a0)
    eta_ideal = 2.0 / (1.0 + math.sqrt(1.0 + cth)) if (cth > 0.0 and math.isfinite(cth)) else None
    top = _at_top_condition(c, c.top, d, pd, ear, z) if c.top is not None else None
    top_ok = True if c.top is None else (
        top is not None and (not c.cavitation or (top["kellerOk"] and top["burrillOk"]))
    )
    if rc is None:
        rc = _num_reynolds_correction(c, d, ear, z, c.v_a, n)

    return {
        "n": n,
        "rpm": 60.0 * n,
        "D": d,
        "PD": pd,
        "EAR": ear,
        "Z": z,
        "J": j,
        "K_T": kt_req,
        "K_Q": kq,
        "T": c.t_target,
        "Q": q,
        "P_shaft": p_shaft,
        "eta0": eta0,
        "Cth": cth,
        "etaIdeal": eta_ideal,
        "etaOfIdeal": (eta0 / eta_ideal) if (eta_ideal is not None and eta_ideal > 0.0) else None,
        "top": top,
        "topOk": top_ok,
        "Re": rc["re"],
        "dC_D": rc["d_cd"],
        "cdFormulaOk": rc["cd_formula_ok"],
        "EAR_min": cav["EAR_min"],
        "kellerOk": cav["kellerOk"],
        "sigma": cav["sigma"],
        "tau_c": cav["tau_c"],
        "tau_allow": cav["tau_allow"],
        "burrillOk": cav["burrillOk"],
        "sigmaCharted": BURRILL_SIGMA_RANGE[0] <= cav["sigma"] <= BURRILL_SIGMA_RANGE[1],
        "earInSeries": ear_in,
        "perZ": [],
    }


def evaluate_bseries_at_thrust(
    inputs: BSeriesInputs,
    rpm: float,
    diameter: float,
    ear: float,
    n_blades: int = 3,
) -> dict[str, Any] | None:
    """Evaluate ``(D, EAR, Z)`` at ``rpm`` with pitch bisected to deliver ``inputs.thrust``."""
    cfg = _BSeriesConfig.from_inputs(inputs)
    return _evaluate_cfg(cfg, rpm / 60.0, diameter, ear, float(n_blades))


def _pattern_search(
    c: _BSeriesConfig,
    n: float,
    z: float,
    d0: float,
    ear0: float,
    d_bounds: tuple[float, float],
    e_bounds: tuple[float, float],
    best: dict[str, Any] | None,
) -> dict[str, Any] | None:
    d, ear = d0, ear0
    seed = _evaluate_cfg(c, n, d, ear, z)
    if seed is None:
        return best
    if best is not None and seed["eta0"] <= best["eta0"]:
        d, ear = best["D"], best["EAR"]
    else:
        best = seed

    d_lo, d_hi = d_bounds
    e_lo, e_hi = e_bounds
    sd, se = (d_hi - d_lo) * 0.08, (e_hi - e_lo) * 0.08
    min_d, min_e = (d_hi - d_lo) * 1e-5, (e_hi - e_lo) * 1e-5
    axes = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
    diags = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    it = 0
    while it < 400 and (sd > min_d or se > min_e):
        moved = False
        for step_set in (axes, diags):
            for ud, ue in step_set:
                d2 = min(d_hi, max(d_lo, d + ud * sd))
                e2 = min(e_hi, max(e_lo, ear + ue * se))
                if d2 == d and e2 == ear:
                    continue
                r = _evaluate_cfg(c, n, d2, e2, z)
                if r is not None and r["eta0"] > best["eta0"]:
                    best = r
                    d, ear = d2, e2
                    moved = True
            if moved:
                break
        if not moved:
            sd *= 0.5
            se *= 0.5
        it += 1
    return best


def _at_rpm(
    c: _BSeriesConfig, rpm: float, seeds: list[tuple[float, float, float]] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    n = rpm / 60.0
    per_z: list[dict[str, Any]] = []
    for z in c.z:
        e_lo, e_hi = EAR_BOUNDS
        if c.strict_ear:
            er = ear_series_bounds(int(z))
            if er is None:
                continue
            e_lo = max(e_lo, er[0])
            e_hi = min(e_hi, er[1])
        d_lo, d_hi = c.d_min, c.d_max
        warm = next((w for w in seeds if w[0] == z), None) if seeds else None
        bz = None
        if warm is not None:
            bz = _pattern_search(
                c, n, z, min(d_hi, max(d_lo, warm[1])), min(e_hi, max(e_lo, warm[2])),
                (d_lo, d_hi), (e_lo, e_hi), None
            )
        n_d = (c.grid_d + 1) // 2 if bz is not None else c.grid_d
        n_e = (c.grid_ear + 1) // 2 if bz is not None else c.grid_ear
        seed = None
        for i in range(n_d):
            d = d_lo if n_d == 1 else d_lo + (d_hi - d_lo) * i / (n_d - 1)
            for k in range(n_e):
                ear = e_lo if n_e == 1 else e_lo + (e_hi - e_lo) * k / (n_e - 1)
                r = _evaluate_cfg(c, n, d, ear, z)
                if r is not None and (seed is None or r["eta0"] > seed["eta0"]):
                    seed = r
        if seed is not None:
            bz = _pattern_search(c, n, z, seed["D"], seed["EAR"], (d_lo, d_hi), (e_lo, e_hi), bz)
        if bz is not None:
            per_z.append(bz)
    if not per_z:
        return None
    per_z.sort(key=lambda r: r["eta0"], reverse=True)
    best = dict(per_z[0])
    best["perZ"] = [
        {"Z": x["Z"], "D": x["D"], "PD": x["PD"], "EAR": x["EAR"], "eta0": x["eta0"], "P_shaft": x["P_shaft"]}
        for x in per_z
    ]
    return best, per_z


def _refine_edge(c: _BSeriesConfig, bad: float, good: float) -> float:
    for _ in range(22):
        mid = 0.5 * (bad + good)
        if _at_rpm(c, mid) is not None:
            good = mid
        else:
            bad = mid
    return good


def _refine_optimum(c: _BSeriesConfig, best: dict[str, Any], span: float) -> dict[str, Any]:
    phi = 0.6180339887
    lo, hi = best["rpm"] - span, best["rpm"] + span

    def eval_rpm(r: float):
        res = _at_rpm(c, r)
        return res[0] if res is not None else None

    x1 = hi - phi * (hi - lo)
    x2 = lo + phi * (hi - lo)
    f1 = eval_rpm(x1)
    f2 = eval_rpm(x2)
    i = 0
    while i < 24 and (hi - lo) > 1e-4:
        p1 = f1["P_shaft"] if f1 is not None else math.inf
        p2 = f2["P_shaft"] if f2 is not None else math.inf
        if p1 < p2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - phi * (hi - lo)
            f1 = eval_rpm(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + phi * (hi - lo)
            f2 = eval_rpm(x2)
        i += 1
    win = best
    for cand in (f1, f2):
        if cand is not None and cand["P_shaft"] < win["P_shaft"]:
            win = cand
    return win


def sweep_bseries_rpm(
    inputs: BSeriesInputs,
    probe_min: float = 20.0,
    probe_max: float = 40000.0,
    probes: int = 90,
    samples: int = 180,
    power_cap: float = 1.6,
) -> dict[str, Any]:
    """
    Sweep propeller shaft speed and find the highest-efficiency Wageningen
    B-series geometry at each RPM (matching ``boatmath::propeller::bseries::sweep``).
    """
    c = _BSeriesConfig.from_inputs(inputs)
    probe: list[tuple[float, dict[str, Any] | None]] = []
    feas_lo = None
    feas_hi = None
    best_probe = None
    for i in range(probes):
        rpm = probe_min * (probe_max / probe_min) ** (i / (probes - 1))
        at = _at_rpm(c, rpm)
        r = at[0] if at is not None else None
        if r is not None:
            if feas_lo is None:
                feas_lo = rpm
            feas_hi = rpm
            if best_probe is None or r["P_shaft"] < best_probe["P_shaft"]:
                best_probe = r
        probe.append((rpm, r))

    if best_probe is None:
        return {
            "feasible": False,
            "rpmLo": None,
            "rpmHi": None,
            "feasLo": None,
            "feasHi": None,
            "best": None,
            "bestUnconstrained": None,
            "topOkCount": 0,
            "topFeasible": False,
            "points": [],
        }

    ratio = (probe_max / probe_min) ** (1.0 / (probes - 1))
    feas_lo = _refine_edge(c, feas_lo / ratio, feas_lo)
    feas_hi = _refine_edge(c, feas_hi * ratio, feas_hi)

    rpm_lo, rpm_hi = feas_lo, feas_hi
    cap = best_probe["P_shaft"] * power_cap
    for rpm, r in reversed(probe):
        if rpm < best_probe["rpm"] and r is not None and r["P_shaft"] > cap:
            rpm_lo = rpm
            break
    for rpm, r in probe:
        if rpm > best_probe["rpm"] and r is not None and r["P_shaft"] > cap:
            rpm_hi = rpm
            break
    rpm_lo = max(rpm_lo, feas_lo)
    rpm_hi = min(rpm_hi, feas_hi)

    rpms = [rpm_lo + (rpm_hi - rpm_lo) * i / (samples - 1) for i in range(samples)]
    mid = min(range(samples), key=lambda idx: abs(rpms[idx] - best_probe["rpm"]))

    raw: list[tuple[dict[str, Any], list[dict[str, Any]]] | None] = [None] * samples
    for direction in (1, -1):
        seeds = None
        i = mid if direction == 1 else mid - 1
        while 0 <= i < samples:
            at = _at_rpm(c, rpms[i], seeds)
            seeds = [(pz["Z"], pz["D"], pz["EAR"]) for pz in at[0]["perZ"]] if at is not None else None
            raw[i] = at
            i += direction

    points: list[dict[str, Any]] = [{} for _ in range(samples)]
    best = None
    best_any = None
    for direction in (1, -1):
        held_z = None
        i = mid if direction == 1 else mid - 1
        while 0 <= i < samples:
            at = raw[i]
            if at is None:
                held_z = None
                points[i] = {"rpm": rpms[i], "ok": False, "design": None}
                i += direction
                continue
            r = dict(at[0])
            per_z = at[1]
            if held_z is not None and r["Z"] != held_z:
                held = next((x for x in per_z if x["Z"] == held_z), None)
                if held is not None and held["eta0"] >= r["eta0"] * (1.0 - HYSTERESIS):
                    r = dict(held)
                    r["perZ"] = at[0]["perZ"]
            held_z = r["Z"]
            if r["topOk"] and (best is None or r["P_shaft"] < best["P_shaft"]):
                best = r
            if best_any is None or r["P_shaft"] < best_any["P_shaft"]:
                best_any = r
            points[i] = {"rpm": rpms[i], "ok": True, "design": r}
            i += direction

    span = (rpm_hi - rpm_lo) / (samples - 1)
    if best is not None:
        rf = _refine_optimum(c, best, span)
        if rf["topOk"] or c.top is None:
            best = rf
    if best_any is not None and not (
        best is not None
        and best["rpm"] == best_any["rpm"]
        and best["Z"] == best_any["Z"]
        and best["P_shaft"] == best_any["P_shaft"]
    ):
        best_any = _refine_optimum(c, best_any, span)

    top_ok_count = sum(1 for p in points if p["ok"] and p["design"]["topOk"])
    return {
        "feasible": True,
        "rpmLo": rpm_lo,
        "rpmHi": rpm_hi,
        "feasLo": feas_lo,
        "feasHi": feas_hi,
        "best": best,
        "bestUnconstrained": best_any if c.top is not None else None,
        "topOkCount": top_ok_count,
        "topFeasible": (c.top is None) or (top_ok_count > 0),
        "points": points,
    }


def optimize_bseries_propeller(
    thrust: float,
    v_ship: float,
    n_shafts: int = 1,
    d_min: float = 0.08,
    d_max: float = 0.40,
    blades: Sequence[int] = (3, 4),
    depth: float = 0.35,
    water: Water | None = None,
    wake_fraction: float = 0.0,
    thrust_deduction: float = 0.0,
    shaft_angle_deg: float = 0.0,
    keller_k: float = 0.1,
    cavitation: bool = True,
    re_correct: bool = True,
    strict_ear: bool = True,
    rpm_bounds: tuple[float, float] = (60.0, 12000.0),
) -> dict[str, Any]:
    """
    Smooth ``asb.Opti`` sizing of the minimum-shaft-power Wageningen B-series
    propeller across candidate blade counts.
    """
    import aerosandbox as asb

    if water is None:
        water = Water.seawater()
    thrust_per_shaft = float(thrust) / max(int(n_shafts), 1)
    v_a = float(v_ship) * (1.0 - float(wake_fraction))

    best_sol: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for z in blades:
        e_lo, e_hi = EAR_BOUNDS
        if strict_ear:
            er = ear_series_bounds(int(z))
            if er is None:
                continue
            e_lo, e_hi = max(e_lo, er[0]), min(e_hi, er[1])

        d0 = 0.8 * d_max if d_max > d_min else d_min
        ear0 = max(e_lo, min(e_hi, 0.5 * (e_lo + e_hi)))
        rpm0 = min(rpm_bounds[1], max(rpm_bounds[0], 60.0 * v_a / (0.8 * max(d0, 1e-3))))

        opti = asb.Opti()
        diameter = opti.variable(init_guess=d0, lower_bound=d_min, upper_bound=d_max)
        pitch_ratio = opti.variable(init_guess=1.1, lower_bound=PD_BOUNDS[0], upper_bound=PD_BOUNDS[1])
        ear = opti.variable(init_guess=ear0, lower_bound=e_lo, upper_bound=e_hi)
        rpm = opti.variable(
            init_guess=rpm0,
            lower_bound=rpm_bounds[0],
            upper_bound=rpm_bounds[1],
            log_transform=True,
        )

        res = evaluate_bseries(
            rpm=rpm,
            diameter=diameter,
            pitch_ratio=pitch_ratio,
            ear=ear,
            n_blades=int(z),
            v_ship=v_ship,
            wake_fraction=wake_fraction,
            thrust_deduction=thrust_deduction,
            depth=depth,
            water=water,
            shaft_angle_deg=shaft_angle_deg,
            keller_k=keller_k,
            re_correct=re_correct,
        )
        opti.subject_to(
            [
                res["thrust"] / thrust_per_shaft == 1.0,
                res["J"] >= J_BOUNDS[0],
                res["J"] <= J_BOUNDS[1],
            ]
        )
        if cavitation:
            opti.subject_to(
                [
                    res["keller_margin"] >= 0.0,
                    res["burrill_margin"] >= 0.0,
                ]
            )
        p_scale = max(thrust_per_shaft * v_ship, 1.0)
        opti.minimize(res["P_shaft"] / p_scale)

        try:
            sol = opti.solve(verbose=False)
        except Exception:
            continue

        solved = {
            k: (float(_onp.ravel(sol(v))[0]) if _is_symbolic(v) else v)
            for k, v in res.items()
        }
        candidates.append(solved)
        if best_sol is None or solved["P_shaft"] < best_sol["P_shaft"]:
            best_sol = solved

    if best_sol is None:
        raise RuntimeError("No feasible B-series propeller found for the given constraints.")
    return {"best": best_sol, "candidates": candidates}
