"""
Steady wave-making resistance of submerged and surface-piercing hydrofoil assemblies,
by linearized (Michell / Havelock / Kochin) far-field free-wave theory.

Theory
------
With ``nu = g / U^2`` and the Kelvin wave fan parameterized by ``lambda = sec(theta)``
(``kx = nu * lambda``, ``ky = +/- nu * lambda * sqrt(lambda^2 - 1)``,
``kappa = nu * lambda^2``), the wave resistance of a set of singularities is

    R_w = (4 rho g^2) / (pi U^2) * int_0^{pi/2} 1/2 (|A+|^2 + |A-|^2) sec^3(theta) dtheta

where ``A+/-`` is the free-wave amplitude of the two (+/- theta) Kelvin half-systems.
This is Michell's integral in the form used by the ``thinship`` crate of the
``boatmath`` project (Tuck 1989), whose conventions this module follows. Three kinds
of singularity contribute, and because the theory is linear they simply superpose
(so all mutual interference is captured by the one integral):

- **Thickness** radiates as sources. A thin body of full thickness ``t`` sheds a
  source flux ``U * dt/dx`` per unit area of its centre-surface; for a vertical
  surface-piercing strut this is exactly Michell's thin ship
  (``A = iint (df/dx) exp(-kappa d) exp(i kx x) dx dd``, ``f = t / 2``).
- **Lift** radiates as a vertical pressure dipole. A force ``F_z`` on the body at
  depth ``d`` is equivalent to a source of flux ``i * lambda * F_z / (rho U)``
  (the bottom-pressure amplitude of ``thinship::pressure``).
- **Side force** radiates as a horizontal dipole, equivalent to a source of flux
  ``+/- sqrt(lambda^2 - 1) * F_y / (rho U)`` in the two half-systems (the
  centreplane dipole system of boatmath's ``michell-calculation-part2`` notes).

Relation to the method of images
--------------------------------
For a given circulation distribution, linear theory gives the drag due to lift as

    D = D_vortex(rigid-lid image) + R_w

i.e. the trailing-vortex (Trefftz-plane) drag evaluated with a *rigid-lid* image
system, plus the wave resistance above: far downstream the steady trailing-vortex
flow is x-independent, for which the linearized free-surface condition reduces to
a rigid wall. As ``Fr -> infinity``, ``R_w`` tends to twice the biplane-image mutual
induced drag, recovering the classical ``phi = 0`` ("high-Froude") image result
``D_i * (1 + sigma)``; as ``Fr -> 0`` it vanishes, leaving ground effect
``D_i * (1 - sigma)``. ``R_w`` must therefore never be added on top of a
high-Froude-image induced drag: that double-counts the image term.

All functions here are written with ``aerosandbox.numpy`` and real arithmetic, so
they are compatible with CasADi automatic differentiation inside ``asb.Opti``.
"""

import numpy as _onp
import aerosandbox.numpy as np
from aerosandbox.geometry import Fuselage, Wing
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.library.free_surface import (
    resolve_surface_segment,
    smooth_submergence_fraction,
)


def _column(values) -> Vectorizable:
    """Stack a sequence (or array) of N scalars into an (N, 1) column."""
    if isinstance(values, (list, tuple)):
        values = np.array(values) if len(values) > 0 else _onp.zeros(0)
    return np.reshape(values, (-1, 1))


def _numeric_or_none(values) -> _onp.ndarray | None:
    """Return ``values`` as a flat float array, or None if it is symbolic (CasADi)."""
    try:
        if isinstance(values, (list, tuple)):
            if any(np.is_casadi_type(v, recursive=False) for v in values):
                return None
        elif np.is_casadi_type(values, recursive=False):
            return None
        return _onp.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return None


def _numeric_entries(values) -> _onp.ndarray:
    """Extract any plain numeric scalar entries from a possibly mixed symbolic/numeric sequence."""
    if isinstance(values, (list, tuple)):
        nums = []
        for v in values:
            arr = _numeric_or_none(v)
            if arr is not None:
                nums.append(arr)
        return _onp.concatenate(nums) if nums else _onp.zeros(0, dtype=float)
    arr = _numeric_or_none(values)
    return arr if arr is not None else _onp.zeros(0, dtype=float)


def wave_quadrature_nodes(
    nu: float,
    y_extent: float,
    x_extent: float,
    min_depth: float,
    phase_per_panel: float = 2.0,
    nodes_per_panel: int = 6,
    kappa_tail_factor: float = 300.0,
    max_panels: int = 600,
) -> tuple[_onp.ndarray, _onp.ndarray]:
    """
    Build composite Gauss-Legendre nodes and weights for the outer wave integral.

    The integration variable is ``sigma = sqrt(nu) * tan(theta)``, so that
    ``kappa = nu + sigma^2`` [1/m]. Panels are sized, as in
    ``thinship::michell::integrate_outer``, so that the transverse
    (``ky * y_extent``) and longitudinal (``kx * x_extent``) phases advance by at
    most ``phase_per_panel`` radians across each; they run until submerged
    singularities have decayed (``kappa * min_depth = 12``), and are then continued
    geometrically to ``kappa_tail_factor / min_depth`` to capture the slowly
    (algebraically) decaying spectrum of surface-piercing elements.

    Returns
    -------
    (sigma, weights) : numeric arrays for ``int_0^inf f(sigma) dsigma``.
    """
    nu = float(nu)
    y_extent = max(float(y_extent), 1e-6)
    x_extent = max(float(x_extent), 1e-6)
    min_depth = max(float(min_depth), 1e-4)

    def phase_rate(sigma: float) -> float:
        root = _onp.sqrt(nu + sigma**2)
        return (
            y_extent * (nu + 2.0 * sigma**2) / root
            + x_extent * _onp.sqrt(nu) * sigma / root
        )

    sigma_decayed = _onp.sqrt(12.0 / min_depth)
    sigma_max = _onp.sqrt(kappa_tail_factor / min_depth)

    edges = [0.0]
    while edges[-1] < sigma_decayed and len(edges) <= max_panels:
        sigma = edges[-1]
        # Evaluate the rate at the far edge of a trial panel so the step shrinks
        # where the phase accelerates.
        step = phase_per_panel / max(phase_rate(sigma), 1e-12)
        step = phase_per_panel / max(phase_rate(sigma + step), 1e-12)
        edges.append(sigma + step)
    while edges[-1] < sigma_max:
        edges.append(edges[-1] * 1.25)

    edges = _onp.array(edges)
    gx, gw = _onp.polynomial.legendre.leggauss(nodes_per_panel)
    half = 0.5 * (edges[1:] - edges[:-1])
    mid = 0.5 * (edges[1:] + edges[:-1])
    sigma_nodes = (mid[:, None] + half[:, None] * gx[None, :]).reshape(-1)
    weights = (half[:, None] * gw[None, :]).reshape(-1)
    return sigma_nodes, weights


def kelvin_wave_resistance(
    xyz_a: tuple[Vectorizable, Vectorizable, Vectorizable],
    xyz_b: tuple[Vectorizable, Vectorizable, Vectorizable],
    velocity: Vectorizable,
    density: Vectorizable,
    source_flux: Vectorizable | None = None,
    force_y: Vectorizable | None = None,
    force_z: Vectorizable | None = None,
    smear_length: Vectorizable | None = None,
    surface_z: Vectorizable = 0.0,
    gravity: float = 9.80665,
    length_scales: tuple[float, float, float] | None = None,
    reference_velocity: float | None = None,
    phase_per_panel: float = 2.0,
    nodes_per_panel: int = 6,
    extra_source_amplitude=None,
) -> Vectorizable:
    """
    Compute the steady deep-water wave resistance [N] of a set of straight line
    singularities moving at ``velocity`` beneath a free surface at ``z = surface_z``.

    Each of the ``N`` elements is a straight segment from ``xyz_a`` to ``xyz_b`` in
    geometry axes (``+x`` downstream, ``+y`` starboard, ``+z`` up), carrying,
    uniformly distributed along its length, any combination of:

    - ``source_flux``: total volume flux [m^3/s] (thickness: ``U * delta_t * span``),
    - ``force_z``: total upward force on the body [N] (lift),
    - ``force_y``: total starboard force on the body [N] (side force).

    Each element may additionally be smeared uniformly over a streamwise length
    ``smear_length`` centred on the segment (e.g. half the chord, for a lifting line
    at the quarter-chord). Points above the free surface are clamped onto it.

    Parameters
    ----------
    xyz_a, xyz_b : tuples ``(x, y, z)`` of length-N vectors (or lists of scalars).
    velocity : Speed through the water [m/s].
    density : Water density [kg/m^3].
    length_scales : ``(y_extent, x_extent, min_depth)`` [m] used only to size the
        quadrature panels. Inferred from numeric coordinates when available (even in
        partially symbolic ``Opti`` graphs), falling back to ``(1.0, 0.5, 0.15)`` for
        any missing dimension.
    reference_velocity : Numeric speed used to size the quadrature when ``velocity``
        is symbolic. Required then: the node layout depends on speed, so there
        is no safe default.
    extra_source_amplitude : Optional callable ``(kx, kappa, ky) -> (re, im)``
        giving, at the quadrature's wavenumbers (1 x M rows), the source-flux
        transform ``sum Q exp(kappa z - i kx x)`` of a further wave-maker that is
        symmetric about the centreplane ``y = 0``: a hull on it, or a pair of
        demihulls either side of it (e.g. ``Hull.wave_amplitude``). It is added to
        both half-systems, so it interferes with the elements. With it,
        ``length_scales`` is required.

    Returns
    -------
    Wave resistance ``R_w`` [N] (scalar; a CasADi expression if inputs are symbolic).
    """
    raw_xa, raw_ya, raw_za = xyz_a
    raw_xb, raw_yb, raw_zb = xyz_b
    xa, ya, za = (_column(v) for v in xyz_a)
    xb, yb, zb = (_column(v) for v in xyz_b)
    n_elements = xa.shape[0]
    if n_elements == 0 and extra_source_amplitude is None:
        return 0.0
    if extra_source_amplitude is not None and length_scales is None:
        raise ValueError("`length_scales` is required with `extra_source_amplitude`.")

    def strengths(values):
        if values is None:
            return None
        return np.reshape(_column(values), (1, -1))

    q_row = strengths(source_flux)
    fy_row = strengths(force_y)
    fz_row = strengths(force_z)
    if n_elements == 0:
        q_row = fy_row = fz_row = None
    if (
        q_row is None
        and fy_row is None
        and fz_row is None
        and extra_source_amplitude is None
    ):
        return 0.0

    # Depth-like coordinate: z relative to the free surface, clamped to z <= 0.
    if callable(surface_z):
        za = np.minimum(surface_z(xa, za), 0.0)
        zb = np.minimum(surface_z(xb, zb), 0.0)
    else:
        za = np.minimum(za - surface_z, 0.0)
        zb = np.minimum(zb - surface_z, 0.0)

    ### Quadrature (numeric nodes; sized from numeric estimates of the problem scales)
    v_num = _numeric_or_none(velocity)
    if v_num is None:
        if reference_velocity is None:
            raise ValueError(
                "`velocity` is symbolic; pass a numeric `reference_velocity` "
                "(`wave_reference_velocity` on the solvers) so the wave-resistance "
                "quadrature can be sized."
            )
        v_ref = float(reference_velocity)
    else:
        v_ref = float(v_num[0])
    if v_ref <= 0:
        return 0.0

    if length_scales is None:
        numeric = [_numeric_or_none(v) for v in (xa, ya, za, xb, yb, zb)]
        if any(v is None for v in numeric):
            x_nums = _onp.concatenate([_numeric_entries(raw_xa), _numeric_entries(raw_xb)])
            y_nums = _onp.concatenate([_numeric_entries(raw_ya), _numeric_entries(raw_yb)])
            s_num = None if callable(surface_z) else _numeric_or_none(surface_z)
            if s_num is not None and s_num.size == 1:
                z_nums = (
                    _onp.concatenate([_numeric_entries(raw_za), _numeric_entries(raw_zb)])
                    - float(s_num[0])
                )
            else:
                z_nums = _onp.zeros(0, dtype=float)
            y_ext = (
                float(_onp.ptp(y_nums))
                if y_nums.size >= 2 and _onp.ptp(y_nums) > 1e-3
                else 1.0
            )
            x_ext = (
                float(_onp.ptp(x_nums))
                if x_nums.size >= 2 and _onp.ptp(x_nums) > 1e-3
                else 0.5
            )
            if z_nums.size >= 1:
                d_pos = -_onp.minimum(z_nums, 0.0)
                d_nonzero = d_pos[d_pos > 1e-3]
                min_d = float(_onp.min(d_nonzero)) if d_nonzero.size > 0 else 0.15
            else:
                min_d = 0.15
            length_scales = (y_ext, x_ext, min_d)
        else:
            xa_n, ya_n, za_n, xb_n, yb_n, zb_n = numeric
            x_all = _onp.concatenate([xa_n, xb_n])
            y_all = _onp.concatenate([ya_n, yb_n])
            mid_depth = -0.5 * (za_n + zb_n)
            seg_len = _onp.sqrt(
                (xb_n - xa_n) ** 2 + (yb_n - ya_n) ** 2 + (zb_n - za_n) ** 2
            )
            length_scales = (
                float(_onp.ptp(y_all)),
                float(_onp.ptp(x_all)),
                float(_onp.min(_onp.maximum(mid_depth, 0.25 * seg_len))),
            )
    y_extent, x_extent, min_depth = length_scales
    min_depth = max(float(min_depth), 1e-3)
    y_extent = max(float(y_extent), min_depth)
    x_extent = max(float(x_extent), min_depth)

    sigma, weights = wave_quadrature_nodes(
        nu=gravity / v_ref**2,
        y_extent=y_extent,
        x_extent=x_extent,
        min_depth=min_depth,
        phase_per_panel=phase_per_panel,
        nodes_per_panel=nodes_per_panel,
    )
    sigma = sigma.reshape(1, -1)
    weights = weights.reshape(1, -1)

    ### Wavenumbers along the dispersion curve (rows, 1 x M)
    nu = gravity / velocity**2
    kappa = nu + sigma**2  # = nu * lambda^2
    lam = np.sqrt(kappa / nu)
    s = sigma / np.sqrt(nu)  # = tan(theta) = sqrt(lambda^2 - 1)
    kx = np.sqrt(nu * kappa)  # = nu * lambda
    ky = sigma * np.sqrt(kappa)  # = nu * lambda * sqrt(lambda^2 - 1)

    def outer(col, row):
        return col @ row

    if extra_source_amplitude is not None:
        extra_re, extra_im = extra_source_amplitude(kx, kappa, ky)
        extra_re = np.reshape(extra_re, (1, -1))
        extra_im = np.reshape(extra_im, (1, -1))
    else:
        extra_re = extra_im = None
    has_elements = q_row is not None or fy_row is not None or fz_row is not None
    smear_col = (
        0.5 * _column(smear_length)
        if smear_length is not None and has_elements
        else None
    )

    def _eval_wave_chunk(kx_c, kappa_c, ky_c, lam_c, s_c, w_c, ex_re_c, ex_im_c):
        ### Streamwise smearing: uniform over `smear_length` -> sinc(kx * c / 2)
        if smear_col is not None:
            arg = outer(smear_col, kx_c)
            arg_safe = np.where(np.fabs(arg) < 1e-6, 1.0, arg)
            smear = np.where(np.fabs(arg) < 1e-6, 1.0, np.sin(arg_safe) / arg_safe)
        else:
            smear = 1.0

        if has_elements:
            re_a = outer(za, kappa_c)
            re_b = outer(zb, kappa_c)
            d_re = re_b - re_a
            px_a = outer(xa, kx_c)
            px_b = outer(xb, kx_c)
            py_a = outer(ya, ky_c)
            py_b = outer(yb, ky_c)
            exp_a = np.exp(re_a)
            exp_b = np.exp(re_b)

        def segment_transform(sign: float):
            """
            E = int_0^1 exp(w(t)) dt along each segment, w = kappa*z - i(kx*x +/- ky*y),
            returned as (real, imag) (N x M) matrices: (e^wb - e^wa) / (wb - wa).
            """
            im_a = -(px_a + sign * py_a)
            im_b = -(px_b + sign * py_b)
            d_im = im_b - im_a

            num_re = exp_b * np.cos(im_b) - exp_a * np.cos(im_a)
            num_im = exp_b * np.sin(im_b) - exp_a * np.sin(im_a)
            mag2 = d_re**2 + d_im**2
            small = mag2 < 1e-8
            mag2_safe = np.where(small, 1.0, mag2)

            e_re = (num_re * d_re + num_im * d_im) / mag2_safe
            e_im = (num_im * d_re - num_re * d_im) / mag2_safe

            # Degenerate (constant-phase) limit: exp(w_mid).
            exp_mid = np.exp(0.5 * (re_a + re_b))
            im_mid = 0.5 * (im_a + im_b)
            e_re = np.where(small, exp_mid * np.cos(im_mid), e_re) * smear
            e_im = np.where(small, exp_mid * np.sin(im_mid), e_im) * smear
            return e_re, e_im

        integrand = 0.0
        for sign in (1.0, -1.0):
            b_re = 0.0 if ex_re_c is None else ex_re_c
            b_im = 0.0 if ex_im_c is None else ex_im_c
            if has_elements:
                e_re, e_im = segment_transform(sign)
            if q_row is not None:
                b_re = b_re + q_row @ e_re
                b_im = b_im + q_row @ e_im
            if fz_row is not None:
                # i * lambda * F_z / (rho U) * E
                coeff = lam_c / (density * velocity)
                b_re = b_re - coeff * (fz_row @ e_im)
                b_im = b_im + coeff * (fz_row @ e_re)
            if fy_row is not None:
                # +/- sqrt(lambda^2 - 1) * F_y / (rho U) * E
                coeff = sign * s_c / (density * velocity)
                b_re = b_re + coeff * (fy_row @ e_re)
                b_im = b_im + coeff * (fy_row @ e_im)
            integrand = integrand + 0.5 * (b_re**2 + b_im**2)

        return np.sum(integrand * lam_c * w_c)

    # R = (4 rho g^2 / (pi U^2)) * int |A|^2 sec^3 dtheta, with A = B / (2U) and
    # sec^3(theta) dtheta = lambda * d(tan theta) = lambda * dsigma / sqrt(nu).
    m = sigma.shape[1]
    is_sym_elements = (
        has_elements
        and m > 256
        and any(
            v is not None and np.is_casadi_type(v, recursive=False)
            for v in (xa, ya, za, xb, yb, zb, q_row, fy_row, fz_row, smear_col, velocity, density)
        )
    )
    if is_sym_elements:
        integral = 0.0
        for s0 in range(0, m, 256):
            s1 = min(s0 + 256, m)
            integral = integral + _eval_wave_chunk(
                kx[:, s0:s1],
                kappa[:, s0:s1],
                ky[:, s0:s1],
                lam[:, s0:s1],
                s[:, s0:s1],
                weights[:, s0:s1],
                None if extra_re is None else extra_re[:, s0:s1],
                None if extra_im is None else extra_im[:, s0:s1],
            )
    else:
        integral = _eval_wave_chunk(kx, kappa, ky, lam, s, weights, extra_re, extra_im)
    return density * gravity**2 / (np.pi * velocity**4) * (integral / np.sqrt(nu))


def wing_thickness_wave_sources(
    wing: Wing,
    velocity: Vectorizable,
    surface_z: Vectorizable | None = 0.0,
    n_chordwise: int = 8,
) -> dict[str, list]:
    """
    Represent the thickness of a ``Wing`` / ``Foil`` / ``Strut`` as line sources for
    ``kelvin_wave_resistance``.

    Each wing section is cut into ``n_chordwise`` cosine-spaced chordwise strips.
    Every strip sheds the thin-body source flux ``U * (t_aft - t_fore)`` per unit
    span (``t`` = local full thickness), carried by a spanwise line at the strip's
    mid-chord and smeared over the strip's width. Trailing edges are treated as
    closed, so each section's net flux is zero. For a vertical surface-piercing
    strut this is a discretization of Michell's thin-ship integral.

    Sections crossing ``surface_z`` are smoothly trimmed to their submerged part.

    Returns
    -------
    Dict with lists ``xyz_a`` (3 lists), ``xyz_b`` (3 lists), ``source_flux`` and
    ``smear_length``, ready to concatenate with other elements.
    """
    out = {
        "xyz_a": ([], [], []),
        "xyz_b": ([], [], []),
        "source_flux": [],
        "smear_length": [],
    }
    n_xsecs = len(wing.xsecs)
    if n_xsecs < 2:
        return out

    edges = 0.5 * (1.0 - _onp.cos(_onp.pi * _onp.linspace(0.0, 1.0, n_chordwise + 1)))
    mids = 0.5 * (edges[1:] + edges[:-1])
    widths = edges[1:] - edges[:-1]

    # Thickness-to-chord jump across each strip, per cross-section (TE closed).
    jumps = []
    for xsec in wing.xsecs:
        tau = xsec.airfoil.local_thickness(x_over_c=edges)
        tau = [tau[j] for j in range(n_chordwise + 1)]
        tau[0] = 0.0
        tau[-1] = 0.0
        jumps.append([tau[j + 1] - tau[j] for j in range(n_chordwise)])

    surface_is_callable = callable(surface_z)
    for i in range(n_xsecs - 1):
        xsec_a = wing.xsecs[i]
        xsec_b = wing.xsecs[i + 1]

        if surface_z is None:
            t_lo, t_hi, f_sub = 0.0, 1.0, 1.0
        elif not surface_is_callable:
            f_sub, s_sub, _ = smooth_submergence_fraction(
                xsec_a.xyz_le[2], xsec_b.xyz_le[2], surface_z
            )
            t_lo = s_sub - 0.5 * f_sub
            t_hi = s_sub + 0.5 * f_sub

        for j in range(n_chordwise):
            p_a = wing._compute_xyz_of_WingXSec(i, x_nondim=mids[j], z_nondim=0.0)
            p_b = wing._compute_xyz_of_WingXSec(i + 1, x_nondim=mids[j], z_nondim=0.0)
            if surface_is_callable:
                za, zb, sz = resolve_surface_segment(
                    p_a[0], p_a[2], p_b[0], p_b[2], surface_z
                )
                f_sub, s_sub, _ = smooth_submergence_fraction(za, zb, sz)
                t_lo = s_sub - 0.5 * f_sub
                t_hi = s_sub + 0.5 * f_sub
            span_yz = np.sqrt((p_b[1] - p_a[1]) ** 2 + (p_b[2] - p_a[2]) ** 2 + 1e-16)

            c_lo = (1.0 - t_lo) * xsec_a.chord + t_lo * xsec_b.chord
            c_hi = (1.0 - t_hi) * xsec_a.chord + t_hi * xsec_b.chord
            j_lo = (1.0 - t_lo) * jumps[i][j] + t_lo * jumps[i + 1][j]
            j_hi = (1.0 - t_hi) * jumps[i][j] + t_hi * jumps[i + 1][j]
            flux = velocity * span_yz * f_sub * 0.5 * (c_lo * j_lo + c_hi * j_hi)
            smear = widths[j] * 0.5 * (c_lo + c_hi)

            start = [(1.0 - t_lo) * p_a[k] + t_lo * p_b[k] for k in range(3)]
            end = [(1.0 - t_hi) * p_a[k] + t_hi * p_b[k] for k in range(3)]

            for mirror in (False, True) if wing.symmetric else (False,):
                y_sign = -1.0 if mirror else 1.0
                out["xyz_a"][0].append(start[0])
                out["xyz_a"][1].append(y_sign * start[1])
                out["xyz_a"][2].append(start[2])
                out["xyz_b"][0].append(end[0])
                out["xyz_b"][1].append(y_sign * end[1])
                out["xyz_b"][2].append(end[2])
                out["source_flux"].append(flux)
                out["smear_length"].append(smear)

    return out


def fuselage_wave_sources(
    fuselage: Fuselage,
    velocity: Vectorizable,
    surface_z: Vectorizable | None = 0.0,
) -> dict[str, list]:
    """
    Represent a slender ``Fuselage`` / pod as axial line sources for
    ``kelvin_wave_resistance``: each section between cross-sections sheds the
    slender-body flux ``U * (A_aft - A_fore)``, spread along its axis.

    Sections above ``surface_z`` are smoothly faded out by their submerged fraction.
    """
    out = {
        "xyz_a": ([], [], []),
        "xyz_b": ([], [], []),
        "source_flux": [],
        "smear_length": [],
    }
    areas = [xsec.xsec_area() for xsec in fuselage.xsecs]
    for i in range(len(fuselage.xsecs) - 1):
        c_a = fuselage.xsecs[i].xyz_c
        c_b = fuselage.xsecs[i + 1].xyz_c
        if surface_z is None:
            f_sub = 1.0
        else:
            za, zb, sz = resolve_surface_segment(
                c_a[0], c_a[2], c_b[0], c_b[2], surface_z
            )
            f_sub, _, _ = smooth_submergence_fraction(za, zb, sz)
        for k in range(3):
            out["xyz_a"][k].append(c_a[k])
            out["xyz_b"][k].append(c_b[k])
        out["source_flux"].append(velocity * f_sub * (areas[i + 1] - areas[i]))
        out["smear_length"].append(0.0)
    return out


def merge_wave_elements(*element_sets: dict[str, list]) -> dict[str, list]:
    """
    Concatenate element dictionaries (as returned by ``wing_thickness_wave_sources``,
    ``fuselage_wave_sources``, or built by hand) into keyword arguments for
    ``kelvin_wave_resistance``. Missing strengths are filled with zeros.
    """
    merged: dict[str, list] = {
        "xyz_a": ([], [], []),
        "xyz_b": ([], [], []),
        "source_flux": [],
        "force_y": [],
        "force_z": [],
        "smear_length": [],
    }
    for elements in element_sets:
        n = len(elements["xyz_a"][0])
        for k in range(3):
            merged["xyz_a"][k].extend(elements["xyz_a"][k])
            merged["xyz_b"][k].extend(elements["xyz_b"][k])
        for key in ("source_flux", "force_y", "force_z", "smear_length"):
            merged[key].extend(elements.get(key, [0.0] * n))
    return merged


def element_source_amplitude(
    xyz_a,
    xyz_b,
    velocity: float,
    density: float,
    source_flux=None,
    force_y=None,
    force_z=None,
    smear_length=None,
    surface_z: float = 0.0,
):
    """
    The source-flux transform of a set of line elements (as taken by
    ``kelvin_wave_resistance``), as a callable ``(kx, kappa, ky) -> (re, im)`` of
    ``sum Q exp(kappa z - i kx x)`` [m^3/s], for wavenumbers on or off the
    dispersion curve. Numeric only.

    The set must be symmetric about ``y = 0`` (every element with its mirror
    image); the transform is then the same for waves heading to port and to
    starboard, and is what ``kelvin_wave_resistance(extra_source_amplitude=...)``
    and ``hydrosandbox.hull.hull.wave_field_from_source_amplitude`` take.

    A lifting element is the vortex sheet it trails: vertical dipoles of strength
    ``F_z / (rho U)`` per unit length over a strip running aft from it, whose
    transform is that of a source ``i (kappa / kx) F_z / (rho U)`` at the element
    (on the dispersion curve ``kappa / kx = sec(theta)``). A side force likewise,
    with ``ky / kx``.
    """
    xa, ya, za = (_onp.reshape(_onp.asarray(v, dtype=float), (-1, 1)) for v in xyz_a)
    xb, yb, zb = (_onp.reshape(_onp.asarray(v, dtype=float), (-1, 1)) for v in xyz_b)
    za, zb = _onp.minimum(za - surface_z, 0.0), _onp.minimum(zb - surface_z, 0.0)
    n = xa.shape[0]

    def strength(values):
        if values is None:
            return _onp.zeros((1, n))
        return _onp.reshape(_onp.asarray(values, dtype=float), (1, -1))

    q, fy, fz = strength(source_flux), strength(force_y), strength(force_z)
    smear = None if smear_length is None else 0.5 * _onp.reshape(
        _onp.asarray(smear_length, dtype=float), (-1, 1)
    )

    def amplitude(kx, kappa, ky, chunk: int = 4096):
        kx, kappa, ky = (_onp.reshape(_onp.asarray(v, dtype=float), (1, -1)) for v in (kx, kappa, ky))
        m = kx.shape[1]
        total = _onp.zeros(m, dtype=complex)
        for start in range(0, m, chunk):
            cut = slice(start, min(start + chunk, m))
            kx_c, kappa_c, ky_c = kx[:, cut], kappa[:, cut], ky[:, cut]
            both = 0.0
            for sign in (1.0, -1.0):
                w_a = za * kappa_c - 1j * (xa * kx_c + sign * ya * ky_c)
                w_b = zb * kappa_c - 1j * (xb * kx_c + sign * yb * ky_c)
                d = w_b - w_a
                small = _onp.abs(d) < 1e-4
                e = _onp.where(
                    small,
                    _onp.exp(0.5 * (w_a + w_b)),
                    (_onp.exp(w_b) - _onp.exp(w_a)) / _onp.where(small, 1.0, d),
                )
                if smear is not None:
                    e = e * _onp.sinc(smear * kx_c / _onp.pi)
                per_flow = 1.0 / (density * velocity)
                both = both + _onp.sum(
                    (q.T + 1j * (kappa_c / kx_c) * per_flow * fz.T
                     + sign * (ky_c / kx_c) * per_flow * fy.T) * e,
                    axis=0,
                )
            total[cut] = 0.5 * both
        return total.real.reshape(1, -1), total.imag.reshape(1, -1)

    return amplitude
