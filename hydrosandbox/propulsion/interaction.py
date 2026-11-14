"""
Steady interaction between propellers and foils.

- ``actuator_disc_velocity``: the flow a loaded propeller induces around itself (the
  inflow ahead of it, the jet and its swirl behind), for the foils and struts near it.
- ``foils_at_propeller``: what the foils and struts do to a propeller's inflow: the
  momentum their viscous wakes take out of its disc, and the velocity their lift
  induces there.

Both work in any right-handed axes with ``x`` along the flow (downstream), and take
numbers or CasADi expressions. Neither knows about the free surface.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _onp

import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def actuator_disc_velocity(
    x: Vectorizable,
    y: Vectorizable,
    z: Vectorizable,
    centre: Sequence[Vectorizable],
    radius: float,
    thrust: Vectorizable,
    velocity: Vectorizable,
    density: Vectorizable,
    torque: Vectorizable = 0.0,
    rotation: float = 1.0,
    n_rings: int = 3,
    n_sectors: int = 8,
) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
    """
    Velocity ``(u, v, w)`` [m/s] induced at a point by a uniformly loaded actuator
    disc of ``radius`` at ``centre``, its axis along ``+x`` (downstream), giving
    ``thrust`` [N] in a stream of ``velocity``.

    Momentum theory sets the jet's excess speed far downstream,
    ``u_jet = U (sqrt(1 + C_T) - 1)``, half of which is reached at the disc. The
    field is that of the disc's vortex cylinder (contraction neglected), which is
    exactly a uniform jet of ``u_jet`` inside the cylinder behind the disc plus a
    disc of sinks of density ``u_jet``: the sinks give the inflow ahead and around.
    They are lumped at ``n_rings * n_sectors`` points and softened over their own
    spacing, and the jet's edges over the same lengths, so the field is smooth.

    ``torque`` [N m] adds the jet's swirl as a solid-body rotation carrying the
    shaft's angular momentum; ``rotation`` is +1 for a propeller turning
    right-handed about ``+x``, -1 for the other hand.
    """
    area = _onp.pi * radius**2
    c_t = thrust / (0.5 * density * velocity**2 * area)
    u_jet = velocity * (np.sqrt(1.0 + np.fmax(c_t, -0.99)) - 1.0)

    dx, dy, dz = x - centre[0], y - centre[1], z - centre[2]
    n = n_rings * n_sectors
    d_area = area / n
    core2 = d_area / _onp.pi  # each lump is spread over its own patch
    u = v = w = 0.0
    for ring in range(n_rings):
        # Equal-area rings, each lump at its ring's area-mean radius.
        r_in, r_out = radius * _onp.sqrt(ring / n_rings), radius * _onp.sqrt((ring + 1) / n_rings)
        r = 2.0 / 3.0 * (r_out**3 - r_in**3) / (r_out**2 - r_in**2)
        for sector in range(n_sectors):
            phi = 2.0 * _onp.pi * (sector + 0.5 * (ring % 2)) / n_sectors
            ry, rz = dy - r * _onp.cos(phi), dz - r * _onp.sin(phi)
            inv = (dx**2 + ry**2 + rz**2 + core2) ** -1.5
            strength = -u_jet * d_area / (4.0 * _onp.pi)
            u, v, w = u + strength * dx * inv, v + strength * ry * inv, w + strength * rz * inv

    # The jet: inside the cylinder, behind the disc.
    edge = _onp.sqrt(core2)
    behind = 0.5 * (1.0 + dx / np.sqrt(dx**2 + core2))
    rho_r = np.sqrt(dy**2 + dz**2 + 1e-12)
    inside = 0.5 * (1.0 - np.tanh((rho_r - radius) / (0.5 * edge)))
    jet = behind * inside
    u = u + u_jet * jet
    # Swirl: u_theta = k r / R, with torque = rho (U + u_jet / 2) pi k R^3 / 2.
    k = 2.0 * torque / (_onp.pi * density * (velocity + 0.5 * u_jet) * radius**3)
    v = v - rotation * k * dz / radius * jet
    w = w + rotation * k * dy / radius * jet
    return u, v, w


def horseshoe_velocity(
    point: Sequence[Vectorizable],
    a: Sequence[Vectorizable],
    b: Sequence[Vectorizable],
    circulation: Vectorizable,
    core: float,
) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
    """
    Velocity at ``point`` from a horseshoe vortex: bound from ``a`` to ``b``, its
    legs trailing to ``x = +infinity``. With the flow along ``+x``, ``a`` to port
    and ``b`` to starboard (``+y``), positive ``circulation`` is upward lift
    ``rho U circulation`` per unit span, and gives downwash between the legs. The
    vortex is softened over ``core``.
    """
    px, py, pz = point
    core2 = core**2

    def bound(p0, p1):
        r0 = [px - p0[0], py - p0[1], pz - p0[2]]
        r1 = [px - p1[0], py - p1[1], pz - p1[2]]
        cross = [
            r0[1] * r1[2] - r0[2] * r1[1],
            r0[2] * r1[0] - r0[0] * r1[2],
            r0[0] * r1[1] - r0[1] * r1[0],
        ]
        n0 = np.sqrt(r0[0] ** 2 + r0[1] ** 2 + r0[2] ** 2 + core2)
        n1 = np.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2 + core2)
        dot = r0[0] * r1[0] + r0[1] * r1[1] + r0[2] * r1[2]
        length2 = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
        factor = (n0 + n1) / (n0 * n1 * (n0 * n1 + dot) + core2 * length2)
        return [circulation / (4.0 * _onp.pi) * factor * c for c in cross]

    def leg(p0, sign):
        """A leg from ``p0`` to ``+x`` infinity, circulation running that way times ``sign``."""
        ry, rz = py - p0[1], pz - p0[2]
        rx = px - p0[0]
        d2 = ry**2 + rz**2 + core2
        factor = sign * circulation / (4.0 * _onp.pi) / d2 * (1.0 + rx / np.sqrt(rx**2 + d2))
        # direction of circulation: +x; (ex cross r) = (0, -rz, ry)
        return [0.0, -factor * rz, factor * ry]

    parts = [bound(a, b), leg(b, 1.0), leg(a, -1.0)]
    return tuple(sum(part[k] for part in parts) for k in range(3))


def foils_at_propeller(
    wings: Sequence[Any],
    components: Sequence[Any],
    centre: Sequence[Vectorizable],
    radius: float,
    velocity: Vectorizable,
    density: Vectorizable,
    n_samples: int = 12,
) -> dict[str, Vectorizable]:
    """
    What a set of wings (foils and struts, with their ``HydroBuildup`` component
    results) does to the inflow of a propeller disc at ``centre``, axis along ``+x``.

    Returns:

    - ``wake_fraction_viscous``: the mean speed deficit over the disc, as a fraction
      of ``velocity``, from the wings' boundary-layer wakes. Each wing sheds its
      profile drag as a momentum deficit along its trailing edge, carried straight
      downstream; the part of each trailing edge that projects inside the disc, and
      lies ahead of it, counts.
    - ``u``, ``v``, ``w``: the velocity the lifting wings' circulation induces at the
      disc's centre (softened over the disc), each as one horseshoe vortex of the
      wing's lift, of span ``pi / 4`` of the wing's.
    - ``wake_fraction``: the total, ``wake_fraction_viscous - u / velocity``.
    - ``inflow_angle``: the downward tilt of the inflow [deg], ``atan(-w / velocity)``.
    """
    area = _onp.pi * radius**2
    deficit = 0.0
    u = v = w = 0.0
    for wing, component in zip(wings, components):
        sides = (1.0, -1.0) if wing.symmetric else (1.0,)
        trailing = [wing._compute_xyz_of_WingXSec(i, x_nondim=1.0, z_nondim=0.0) for i in range(len(wing.xsecs))]
        quarter = [wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0.0) for i in range(len(wing.xsecs))]
        lengths = [
            np.sqrt(sum((trailing[i + 1][k] - trailing[i][k]) ** 2 for k in (1, 2)) + 1e-12)
            for i in range(len(trailing) - 1)
        ]
        span = len(sides) * sum(lengths)
        drag_per_span = component.D_profile / span
        for side in sides:
            for i, length in enumerate(lengths):
                for s in (_onp.arange(n_samples) + 0.5) / n_samples:
                    px, py, pz = ((1.0 - s) * trailing[i][k] + s * trailing[i + 1][k] for k in range(3))
                    r = np.sqrt((side * py - centre[1]) ** 2 + (pz - centre[2]) ** 2 + 1e-12)
                    inside = 0.5 * (1.0 - np.tanh((r - radius) / (0.2 * radius)))
                    ahead = 0.5 * (1.0 + np.tanh((centre[0] - px) / (0.5 * radius)))
                    deficit = deficit + drag_per_span * (length / n_samples) * inside * ahead
        if not getattr(wing, "is_strut", False):
            a, b = quarter[-1] if wing.symmetric else quarter[0], quarter[-1]
            a = [a[0], -a[1], a[2]] if wing.symmetric else a
            # (a to port, b to starboard)
            mid = [0.5 * (a[k] + b[k]) for k in range(3)]
            a_eff = [mid[k] + _onp.pi / 4 * (a[k] - mid[k]) for k in range(3)]
            b_eff = [mid[k] + _onp.pi / 4 * (b[k] - mid[k]) for k in range(3)]
            width = np.sqrt(sum((b_eff[k] - a_eff[k]) ** 2 for k in (1, 2)) + 1e-12)
            circulation = component.L / (density * velocity * width)
            du, dv, dw = horseshoe_velocity(centre, a_eff, b_eff, circulation, core=radius)
            u, v, w = u + du, v + dv, w + dw
    viscous = deficit / (density * velocity**2 * area)
    return {
        "wake_fraction_viscous": viscous,
        "u": u,
        "v": v,
        "w": w,
        "wake_fraction": viscous - u / velocity,
        "inflow_angle": np.arctan2d(-w, velocity),
    }
