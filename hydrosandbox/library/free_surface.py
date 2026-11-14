import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from typing import Literal


def smooth_submergence_fraction(
    z_a: Vectorizable,
    z_b: Vectorizable,
    surface_z: Vectorizable,
    transition_thickness: float = 0.005,
) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
    """
    Compute the smooth, CasADi-differentiable submerged fraction of a linear segment
    between geometry coordinates ``z_a`` and ``z_b`` below a free water surface ``surface_z``.

    Positive submergence depth is ``h = surface_z - z``. Uses the exact analytical
    integral of a logistic step over the segment.

    Parameters
    ----------
    z_a : Vectorizable
        Start geometry z-coordinate (+z up) [m].
    z_b : Vectorizable
        End geometry z-coordinate (+z up) [m].
    surface_z : Vectorizable
        Free-surface elevation in geometry z-coordinates [m].
    transition_thickness : float
        Smoothing scale delta [m]. Default is 0.005 m.

    Returns
    -------
    tuple[Vectorizable, Vectorizable, Vectorizable]
        - ``f_sub``: Submerged fraction in ``[0, 1]``.
        - ``s_sub``: Interpolation parameter in ``[0, 1]`` along ``a -> b`` of the
          submerged segment midpoint.
        - ``h_sub_mid``: Mean depth [m] of the submerged portion below ``surface_z``.
    """
    delta = transition_thickness
    h_a = surface_z - z_a
    h_b = surface_z - z_b

    phi_a = np.softmax(h_a, 0.0, softness=delta)
    phi_b = np.softmax(h_b, 0.0, softness=delta)

    dh = h_b - h_a
    # Safe non-zero denominator so CasADi 2nd-order AD never evaluates 0/0 on inactive branches.
    dh_safe = np.where(dh >= 0, np.maximum(dh, 1e-7), np.minimum(dh, -1e-7))

    h_mid = 0.5 * (h_a + h_b)
    sigmoid_mid = 0.5 + 0.5 * np.tanh(0.5 * np.clip(h_mid / delta, -200.0, 200.0))

    f_sub = np.where(
        np.fabs(dh) > 1e-4,
        np.clip((phi_b - phi_a) / dh_safe, 0.0, 1.0),
        sigmoid_mid,
    )

    h_sub_mid = 0.5 * (phi_a + phi_b)
    z_sub_mid = surface_z - h_sub_mid

    z_diff = z_b - z_a
    z_diff_safe = np.where(
        z_diff >= 0, np.maximum(z_diff, 1e-7), np.minimum(z_diff, -1e-7)
    )
    s_sub = np.where(
        np.fabs(z_diff) > 1e-4,
        np.clip((z_sub_mid - z_a) / z_diff_safe, 0.0, 1.0),
        0.5,
    )

    return f_sub, s_sub, h_sub_mid


def smooth_surface_piercing_weight(
    z_a: Vectorizable,
    z_b: Vectorizable,
    surface_z: Vectorizable,
    transition_thickness: float = 0.005,
) -> tuple[Vectorizable, Vectorizable]:
    """
    Compute a smooth, mesh-refinement-invariant weight in ``[0, 1]`` indicating
    whether the segment ``[z_a, z_b]`` pierces the free surface ``surface_z``,
    together with the waterline crossing parameter ``t_pierce`` along ``a -> b``.

    Returns
    -------
    tuple[Vectorizable, Vectorizable]
        - ``pierce_weight``: Smooth indicator (~1 when one endpoint is submerged and
          the other is emerged, ~0 otherwise; sums to 1 across contiguous sections).
        - ``t_pierce``: Interpolation parameter in ``[0, 1]`` from ``a`` to ``b``
          where ``z = surface_z``.
    """
    h_a = surface_z - z_a
    h_b = surface_z - z_b

    span_z = np.sqrt((h_b - h_a) ** 2 + 1e-10)
    delta_eff = np.minimum(transition_thickness, 0.2 * span_z + 1e-5)

    h_max = np.softmax(h_a, h_b, softness=delta_eff)
    h_min = np.softmin(h_a, h_b, softness=delta_eff)

    s_max = 0.5 + 0.5 * np.tanh(0.5 * np.clip(h_max / delta_eff, -200.0, 200.0))
    s_min = 0.5 + 0.5 * np.tanh(0.5 * np.clip(-h_min / delta_eff, -200.0, 200.0))

    pierce_weight = s_max * s_min

    dh_ab = h_a - h_b
    dh_ab_safe = np.where(
        dh_ab >= 0, np.maximum(dh_ab, 1e-7), np.minimum(dh_ab, -1e-7)
    )
    t_pierce = np.where(
        np.fabs(dh_ab) > 1e-6,
        np.clip(h_a / dh_ab_safe, 0.0, 1.0),
        0.5,
    )

    return pierce_weight, t_pierce


def prandtl_biplane_interference_factor(
    h_over_b: Vectorizable,
) -> Vectorizable:
    """
    Compute Prandtl's biplane mutual-interference factor ``sigma`` between a foil of
    span ``b`` at depth ``h`` and its free-surface image (an equal-span "biplane"
    with gap ``G = 2 * h``).

    Uses Prandtl's rational fit ``sigma = (1 - 0.66 * G/b) / (1.05 + 3.7 * G/b)``
    (Prandtl 1924, NACA TR-116; nominally valid for ``0.05 <= G/b <= 0.5``), with
    the numerator smoothly clamped at zero so that ``sigma`` stays positive and
    decays monotonically in deep water. Agrees with ``HydroVLM`` method-of-images
    results to within ~0.02 for ``0.04 <= h/b <= 0.7``.
    """
    gap_over_span = 2.0 * np.softmax(h_over_b, 0.0, softness=0.01)
    numerator = np.softmax(1.0 - 0.66 * gap_over_span, 0.0, softness=0.05)
    return numerator / (1.05 + 3.7 * gap_over_span)


def froude_image_weight(
    froude_depth: Vectorizable = 10.0,
    mode: Literal["high_froude", "froude_blended", "rigid_lid", "none"] = "high_froude",
) -> Vectorizable:
    """
    Compute the method-of-images reflection weight ``kappa`` across the free surface.

    With ``gamma_image = kappa * gamma`` and mirrored geometry across ``z = surface_z``:
    - ``kappa = +1.0`` (``"high_froude"``): ``phi = 0`` free-surface condition; reduces
      lift slope and increases induced drag near the surface.
    - ``kappa = -1.0`` (``"rigid_lid"``): ``dphi/dz = 0`` wall/ground-effect condition.
    - ``"froude_blended"``: ``(Fr_h^2 - 1) / (Fr_h^2 + 1)`` smoothly blending from
      rigid lid at ``Fr_h -> 0`` to high-Froude free surface as ``Fr_h -> inf``.
    - ``"none"``: ``0.0`` (unbounded infinite fluid).
    """
    if mode == "high_froude":
        return 1.0
    elif mode == "rigid_lid":
        return -1.0
    elif mode == "none":
        return 0.0
    elif mode == "froude_blended":
        fr2 = froude_depth**2
        return (fr2 - 1.0) / (fr2 + 1.0)
    else:
        raise ValueError(
            f"Bad free-surface mode {mode!r}. Valid options: "
            "'high_froude', 'froude_blended', 'rigid_lid', 'none'."
        )


def free_surface_lift_slope_factor(
    h_over_c: Vectorizable,
    aspect_ratio: Vectorizable,
    froude_depth: Vectorizable = 10.0,
    mode: Literal["high_froude", "froude_blended", "rigid_lid", "none"] = "high_froude",
) -> Vectorizable:
    """
    Compute the ratio ``CL_alpha(h) / CL_alpha(inf)`` for a submerged lifting hydrofoil
    at nondimensional depth ``h / c`` below the free surface.

    Combines 2D biplane-image lift reduction (Wadlin/Hoerner) with Prandtl 3D
    mutual-interference biplane correction for finite-aspect-ratio foils.
    """
    if mode == "none":
        return 1.0

    hc = np.softmax(h_over_c, 0.0, softness=0.02)
    ar = np.maximum(aspect_ratio, 0.1)
    kappa = froude_image_weight(froude_depth=froude_depth, mode=mode)

    k2d_high = (1.0 + 16.0 * hc**2) / (2.0 + 16.0 * hc**2)
    k2d_lid = 1.0 + 1.0 / (4.0 + 16.0 * hc**2)
    k2d = 0.5 * (1.0 + kappa) * k2d_high + 0.5 * (1.0 - kappa) * k2d_lid

    h_over_b = hc / ar
    sigma_fs = prandtl_biplane_interference_factor(h_over_b)

    f_inf = ar / (ar + 2.0)
    f_fs = (k2d * ar) / (ar + 2.0 * k2d * (1.0 + kappa * sigma_fs))

    return f_fs / f_inf


def free_surface_induced_drag_factor(
    h_over_b: Vectorizable,
    froude_depth: Vectorizable = 10.0,
    mode: Literal["high_froude", "froude_blended", "rigid_lid", "none"] = "high_froude",
) -> Vectorizable:
    """
    Compute the induced drag multiplier ``CD_i(h) / CD_i(inf)`` at fixed lift due
    to the free surface, using Prandtl's biplane mutual-interference factor ``sigma``.

    Returns ``1 + kappa * sigma``, where ``sigma`` is evaluated for the biplane formed
    by the foil and its image (gap ``2 * h``). Approaches ``~1.9`` at very shallow
    high-Froude submergence, ``1.0`` in deep water, and decreases toward ``0.05`` in
    rigid-lid ground effect.
    """
    if mode == "none":
        return 1.0

    hb = np.softmax(h_over_b, 0.02, softness=0.02)
    kappa = froude_image_weight(froude_depth=froude_depth, mode=mode)
    sigma_fs = prandtl_biplane_interference_factor(hb)

    return np.softmax(1.0 + kappa * sigma_fs, 0.05, softness=0.02)


def spray_drag_strut(
    waterline_chord: Vectorizable,
    thickness_to_chord: Vectorizable,
    dynamic_pressure: Vectorizable,
    froude_chord: Vectorizable = 10.0,
) -> Vectorizable:
    """
    Compute surface-piercing strut spray drag [N] using Hoerner's correlation:
    ``D_spray = 0.24 * f(Fr_c) * q * t_wl^2`` (Hoerner, Fluid-Dynamic Drag, Ch. 10).
    """
    t_wl = waterline_chord * thickness_to_chord
    f_fr = froude_chord**2 / (froude_chord**2 + 1.0)
    return 0.24 * f_fr * dynamic_pressure * t_wl**2


def junction_interference_drag(
    chord: Vectorizable,
    thickness_to_chord: Vectorizable,
    dynamic_pressure: Vectorizable,
    n_junctions: int = 1,
) -> Vectorizable:
    """
    Compute T-junction / strut-foil interference drag [N] using Hoerner's
    corner-interference correlation (Hoerner, Fluid-Dynamic Drag, Ch. 8, Eq. 2:
    ``C_D_t2 = 0.75 * (t/c) - 0.0003 / (t/c)^2``).
    """
    t = chord * thickness_to_chord
    tc = np.softmax(thickness_to_chord, 0.04, softness=0.01)
    c_d_t2 = np.softmax(
        0.75 * tc - 0.0003 / tc**2,
        0.005,
        softness=0.005,
    )
    return n_junctions * dynamic_pressure * c_d_t2 * t**2
