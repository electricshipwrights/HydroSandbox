from typing import Any, Literal
from aerosandbox.geometry import Airfoil, ControlSurface
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.fluid import Water
from hydrosandbox.library.cavitation import (
    cavitation_drag_increment,
    cavitation_inception_speed,
    cavitation_margin,
    cavitation_number,
)


def _is_scalar_numeric(x: Any) -> bool:
    """Return True if ``x`` is a non-CasADi 0D numeric scalar."""
    if hasattr(x, "is_symbolic") or "casadi" in type(x).__module__:
        return False
    return np.ndim(x) == 0


def evaluate_hydrofoil_2D(
    airfoil: Airfoil,
    velocity: Vectorizable = 10.0,
    chord: Vectorizable = 0.2,
    alpha: Vectorizable = 0.0,
    depth: Vectorizable = 0.5,
    water: Water | None = None,
    control_surfaces: list[ControlSurface] | None = None,
    model_size: Literal[
        "xxsmall", "xsmall", "small", "medium", "large", "xlarge"
    ] = "medium",
    include_cavitation_penalty: bool = False,
) -> dict[str, Any]:
    """
    Evaluate 2D incompressible hydrofoil section performance and cavitation limits.

    Uses NeuralFoil at ``mach = 0.0`` to obtain viscous ``CL``, ``CD``, ``CM``, and
    incompressible suction-peak ``Cpmin_0``, paired with local hydrostatic cavitation
    number ``sigma(depth, velocity)``.

    Args:
        airfoil: Section ``Airfoil`` (or ``KulfanAirfoil``).
        velocity: Flow speed [m/s].
        chord: Section chord [m].
        alpha: Angle of attack [deg].
        depth: Submergence depth below free surface [m].
        water: Fluid properties (defaults to ``Water.seawater()``).
        control_surfaces: Optional list of section control surfaces / flaps.
        model_size: NeuralFoil model size.
        include_cavitation_penalty: If True, adds smooth drag rise when
            ``sigma < -Cpmin_0``.

    Returns:
        Dictionary containing ``CL``, ``CD``, ``CD_profile``, ``CD_cavitation``,
        ``CM``, ``L_over_D``, ``Cpmin``, ``sigma``, ``cavitation_margin``,
        ``V_inception``, ``Re``, ``local_pressure``, and ``dynamic_pressure``.
    """
    if water is None:
        water = Water.seawater()

    rho = water.density()
    nu = water.kinematic_viscosity()
    p_vapor = water.vapor_pressure()
    p_local = water.pressure(depth=depth)

    Re = velocity * chord / nu
    q = 0.5 * rho * velocity**2

    aero = airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=0.0,
        control_surfaces=control_surfaces,
        model_size=model_size,
    )

    CL = aero["CL"]
    CD_profile = aero["CD"]
    CM = aero["CM"]
    Cpmin = aero["Cpmin_0"]

    sigma = cavitation_number(
        local_pressure=p_local,
        vapor_pressure=p_vapor,
        dynamic_pressure=q,
    )
    cav_margin = cavitation_margin(Cpmin=Cpmin, sigma=sigma)
    v_inc = cavitation_inception_speed(
        Cpmin=Cpmin,
        local_pressure=p_local,
        vapor_pressure=p_vapor,
        density=rho,
    )

    if include_cavitation_penalty:
        CD_cav = cavitation_drag_increment(Cpmin=Cpmin, sigma=sigma)
    else:
        CD_cav = 0.0 * CD_profile

    CD = CD_profile + CD_cav
    L_over_D = CL / np.maximum(CD, 1e-12)

    result: dict[str, Any] = {
        "CL": CL,
        "CD": CD,
        "CD_profile": CD_profile,
        "CD_cavitation": CD_cav,
        "CM": CM,
        "L_over_D": L_over_D,
        "Cpmin": Cpmin,
        "sigma": sigma,
        "sigma_required": -Cpmin,
        "cavitation_margin": cav_margin,
        "V_inception": v_inc,
        "Re": Re,
        "local_pressure": p_local,
        "dynamic_pressure": q,
    }
    if "Top_Xtr" in aero:
        result["Top_Xtr"] = aero["Top_Xtr"]
    if "Bot_Xtr" in aero:
        result["Bot_Xtr"] = aero["Bot_Xtr"]

    all_scalar = all(
        _is_scalar_numeric(v) for v in (velocity, chord, alpha, depth, rho)
    )
    if all_scalar:
        for k, val in result.items():
            if np.ndim(val) == 1 and len(val) == 1:
                result[k] = float(val[0])
            elif np.ndim(val) == 0:
                result[k] = float(val)

    return result


def cavitation_bucket(
    airfoil: Airfoil,
    alphas: np.ndarray | None = None,
    velocity: float = 10.0,
    chord: float = 0.2,
    depth: float = 0.5,
    water: Water | None = None,
    control_surfaces: list[ControlSurface] | None = None,
    model_size: Literal[
        "xxsmall", "xsmall", "small", "medium", "large", "xlarge"
    ] = "medium",
) -> dict[str, Any]:
    """
    Compute the 2D cavitation bucket (cavitation-free operating envelope) across
    angles of attack.

    Args:
        airfoil: Section ``Airfoil``.
        alphas: Angle-of-attack sweep [deg]. Defaults to ``np.linspace(-6, 12, 73)``.
        velocity: Reference speed [m/s] (sets Reynolds number and ``cavitation_margin``).
        chord: Reference chord [m].
        depth: Submergence depth [m].
        water: Fluid properties (defaults to ``Water.seawater()``).
        control_surfaces: Optional control surfaces.
        model_size: NeuralFoil model size.

    Returns:
        Dictionary with arrays ``alpha``, ``CL``, ``CD``, ``Cpmin``,
        ``sigma_required`` (``-Cpmin``), ``V_inception``, ``cavitation_margin``,
        plus summary scalars ``V_bucket_max``, ``alpha_bucket_peak``,
        ``CL_bucket_peak``, ``CL_min_bucket``, and ``CL_max_bucket``.
    """
    if alphas is None:
        alphas = np.linspace(-6.0, 12.0, 73)
    else:
        alphas = np.asarray(alphas, dtype=float).reshape(-1)

    res = evaluate_hydrofoil_2D(
        airfoil=airfoil,
        velocity=velocity,
        chord=chord,
        alpha=alphas,
        depth=depth,
        water=water,
        control_surfaces=control_surfaces,
        model_size=model_size,
        include_cavitation_penalty=False,
    )

    v_inc = np.asarray(res["V_inception"], dtype=float).reshape(-1)
    CL = np.asarray(res["CL"], dtype=float).reshape(-1)
    CD = np.asarray(res["CD"], dtype=float).reshape(-1)
    CM = np.asarray(res["CM"], dtype=float).reshape(-1)
    Cpmin = np.asarray(res["Cpmin"], dtype=float).reshape(-1)
    cav_margin = np.asarray(res["cavitation_margin"], dtype=float).reshape(-1)

    peak_idx = int(np.argmax(v_inc))
    in_bucket = cav_margin >= 0.0
    if np.any(in_bucket):
        cl_min_bucket = float(np.min(CL[in_bucket]))
        cl_max_bucket = float(np.max(CL[in_bucket]))
    else:
        cl_min_bucket = float("nan")
        cl_max_bucket = float("nan")

    return {
        "alpha": alphas,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Cpmin": Cpmin,
        "sigma": float(res["sigma"]) if np.ndim(res["sigma"]) == 0 else res["sigma"],
        "sigma_required": -Cpmin,
        "V_inception": v_inc,
        "cavitation_margin": cav_margin,
        "V_bucket_max": float(v_inc[peak_idx]),
        "alpha_bucket_peak": float(alphas[peak_idx]),
        "CL_bucket_peak": float(CL[peak_idx]),
        "CL_min_bucket": cl_min_bucket,
        "CL_max_bucket": cl_max_bucket,
    }
