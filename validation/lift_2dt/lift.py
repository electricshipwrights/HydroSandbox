"""
2D+t (slender-body) dynamic lift of a hull at an attitude, after boatmath's
`lift.rs`, in closed form so it can be tested against tank data.

Each station's section, cut at the running attitude, has an infinite-frequency
heave added mass `m_a` (here from the Lewis form with the section's beam,
draught and area, not a panel solve). A hull at angle of attack `alpha` carries

    f(x) = C_T(x) [ U^2 sin(alpha) cos(alpha) dm_a/dxi + 1/2 rho C_D 2c U^2 sin(alpha)|sin(alpha)| ]

per unit length (xi aft, 2c the wetted beam), plus the step in `m_a` where the
wetted length begins. `C_T = tanh(k (x_T - x) / (B C_V))` is Garme's transom
reduction. `alpha` is the trim alone (boatmath's default), or the trim less the
slope of the keel over the aft 40% of the wetted length ("run"). With
`transom_reduction=False` the sections closing toward the transom keep their
full (downward) load.
"""

import math

import numpy as np

CROSSFLOW_CD, TRANSOM_DECAY, RUN_FRACTION = 1.0, 2.5, 0.4


def lewis_added_mass(half_beam, draught, area, density):
    """Infinite-frequency heave added mass per unit length of a Lewis-form section."""
    if half_beam < 1e-9 or draught < 1e-9:
        return 0.0
    h0 = half_beam / draught
    sigma = min(max(area / (2.0 * half_beam * draught), 0.3), 0.99)
    c1 = 3.0 + 4.0 * sigma / math.pi + (1.0 - 4.0 * sigma / math.pi) * ((h0 - 1.0) / (h0 + 1.0)) ** 2
    a3 = (-c1 + 3.0 + math.sqrt(max(9.0 - 2.0 * c1, 0.0))) / c1
    a1 = (h0 - 1.0) / (h0 + 1.0) * (a3 + 1.0)
    return 0.5 * density * math.pi * half_beam**2 * ((1.0 + a1) ** 2 + 3.0 * a3**2) / (1.0 + a1 + a3) ** 2


def sections(hull, sinkage, trim, pivot_xz):
    """Wetted half-beam, draught, area and keel height (hull axes) of each station at the attitude."""
    out = []
    sin, cos = math.sin(math.radians(trim)), math.cos(math.radians(trim))
    for x, hb in zip(hull.stations, hull.half_breadths):
        z_e = pivot_xz[1] - sinkage - (x - pivot_xz[0]) * sin + (hull.levels - pivot_xz[1]) * cos
        wet = hb > 1e-6 * hull.half_breadths.max()
        keel_hull = hull.levels[np.argmax(wet)] if wet.any() else hull.levels[-1]
        keel_e = z_e[np.argmax(wet)] if wet.any() else 1.0
        if keel_e >= 0.0:
            out.append((0.0, 0.0, 0.0, keel_hull))
            continue
        zz = np.linspace(keel_e, 0.0, 60)
        y = np.interp(zz, z_e, hb)
        out.append((float(y[-1]), float(-keel_e), float(2.0 * np.trapezoid(y, zz)), keel_hull))
    return np.array(out)


def lift_2dt(hull, velocity, density, sinkage, trim, pivot_xz, x_cg, alpha_model="trim", gravity=9.80665,
             transom_reduction=True, crossflow_cd=CROSSFLOW_CD):
    """Upward force [N] and bow-up moment about `x_cg` [N m] of one hull (x runs aft)."""
    sec = sections(hull, sinkage, trim, pivot_xz)
    x = hull.stations
    wet = sec[:, 1] > 0
    if wet.sum() < 3:
        return 0.0, 0.0
    m_a = np.array([lewis_added_mass(c, d, a, density) for c, d, a, _ in sec])
    i_fwd, i_aft = np.where(wet)[0][0], np.where(wet)[0][-1]
    alpha = math.radians(trim)
    if alpha_model == "run":
        run = wet & (x >= x[i_aft] - RUN_FRACTION * (x[i_aft] - x[i_fwd]))
        alpha -= math.atan(np.polyfit(x[run], sec[run, 3], 1)[0])  # a keel rising aft unloads the run
    beam = 2.0 * sec[:, 0].max()
    c_v = velocity / math.sqrt(gravity * beam)
    transom = sec[i_aft, 0] > 0.05 * sec[:, 0].max() and i_aft == len(x) - 1
    c_t = np.tanh(TRANSOM_DECAY * (x[i_aft] - x) / (beam * c_v)) if (transom and transom_reduction) else np.ones_like(x)
    sa, ca = math.sin(alpha), math.cos(alpha)
    dm = np.gradient(m_a, x)  # d m_a / d xi, xi aft
    f = c_t * (velocity**2 * sa * ca * dm + 0.5 * density * crossflow_cd * 2.0 * sec[:, 0] * velocity**2 * sa * abs(sa))
    f = np.where(wet, f, 0.0)
    force = np.trapezoid(f, x) + c_t[i_fwd] * velocity**2 * sa * ca * m_a[i_fwd]
    moment = np.trapezoid(f * (x_cg - x), x) + c_t[i_fwd] * velocity**2 * sa * ca * m_a[i_fwd] * (x_cg - x[i_fwd])
    return float(force), float(moment)
