import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def cavitation_number(
    local_pressure: Vectorizable,
    vapor_pressure: Vectorizable,
    dynamic_pressure: Vectorizable,
) -> Vectorizable:
    """
    Compute the cavitation number ``sigma = (p_local - p_vapor) / q`` [-].
    """
    return (local_pressure - vapor_pressure) / np.maximum(dynamic_pressure, 1e-12)


def cavitation_margin(
    Cpmin: Vectorizable,
    sigma: Vectorizable,
) -> Vectorizable:
    """
    Compute the cavitation margin ``sigma + Cpmin`` [-].

    - ``> 0``: Non-cavitating (safe margin above vapor pressure).
    - ``= 0``: Incipient cavitation (minimum surface pressure equals vapor pressure).
    - ``< 0``: Cavitating.
    """
    return sigma + Cpmin


def cavitation_inception_speed(
    Cpmin: Vectorizable,
    local_pressure: Vectorizable,
    vapor_pressure: Vectorizable,
    density: Vectorizable,
) -> Vectorizable:
    """
    Compute the cavitation inception speed [m/s] given ``Cpmin`` and ambient pressure.
    """
    dp = np.softmax(local_pressure - vapor_pressure, 1.0, softness=10.0)
    suction = np.softmax(-Cpmin, 1e-3, softness=1e-2)
    return np.sqrt(2.0 * dp / (density * suction))


def estimate_Cpmin_inviscid(
    Cl_local: Vectorizable,
    thickness_to_chord: Vectorizable,
    sweep_rad: Vectorizable = 0.0,
) -> Vectorizable:
    """
    Analytical potential-flow suction-peak ``Cpmin`` estimate for 3D panel solvers
    where full boundary-layer Cp distributions are not solved directly.
    """
    tc = np.maximum(thickness_to_chord, 0.0)
    u_t_over_v = 1.0 + 1.20 * tc + 0.8 * tc**2
    u_l_over_v = 0.55 * np.sqrt(Cl_local**2 + 1e-6) / (1.0 + 2.5 * tc)
    cpmin_2d = 1.0 - (u_t_over_v + u_l_over_v) ** 2
    return cpmin_2d * np.cos(sweep_rad) ** 2


def cavitation_drag_increment(
    Cpmin: Vectorizable,
    sigma: Vectorizable,
    k_cav: float = 0.5,
    softness: float = 0.02,
) -> Vectorizable:
    """
    Compute a smooth, differentiable section drag coefficient increment ``Delta_CD_cav``
    due to cavitation once ``-Cpmin`` exceeds ``sigma``.

    Returns ~0 well below cavitation inception and increases smoothly and
    quadratically once cavitation margin becomes negative.
    """
    overload = np.softmax(-Cpmin - sigma, 0.0, softness=softness)
    return k_cav * overload**2


def ventilation_margin_lifting_foil(
    h_over_c: Vectorizable,
    CL_local: Vectorizable,
) -> Vectorizable:
    """
    Compute an empirical ventilation safety margin [-] for a submerged lifting foil
    near the free surface (positive = safe against vortex air draw-down).
    """
    return h_over_c - 0.35 * np.sqrt(CL_local**2 + 1e-4) - 0.10


def ventilation_margin_strut(
    CL_strut_abs: Vectorizable,
    froude_depth: Vectorizable,
    CL_crit: float = 0.55,
) -> Vectorizable:
    """
    Compute an empirical ventilation safety margin [-] for a surface-piercing strut
    operating under yaw/leeway (positive = safe against full air-path ventilation).
    """
    f_fr = froude_depth**2 / (froude_depth**2 + 2.25)
    return CL_crit - f_fr * CL_strut_abs
