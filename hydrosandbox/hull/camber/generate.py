"""
Camber documents generated from a few physical numbers, for hulls no one drew.

camber's model is a section swept along the sheer plan and trimmed by the sheer,
the centreline and the transom. The keel, the rocker and the stem are not drawn:
they are where the section crosses the centreline, so they follow from the plan's
half-breadth ``b(x)`` and the section's depth ``z(n)`` at ``n = b``. This
generator draws the two things camber does draw:

- the sheer plan: a clamped B-spline from the transom, leaving it square (so the
  aft section is the transom) at half-breadth ``transom_ratio * half_beam``,
  through its widest point (``half_beam``, about ``x_max_beam`` of the length
  from the transom) to the stem, which it reaches at the half-angle
  ``entry_angle``;
- one section per end, lofted between them: topsides from the sheer down
  ``side`` [mm], ``flare`` [mm] inboard at their foot, then a superellipse
  quadrant (exponent ``shape``: 1 a straight V, 2 an ellipse, more a box) that
  leaves them tangent and flattens out ``bottom`` [mm] lower and
  ``width_ratio * half_beam`` further inboard, flat beyond.

Narrow stations, where the centreline cuts the section near the sheer, so get
fine, deep-forefooted ends (a plumb stem when there is no flare), and wide ones
are cut where the bottom has flattened: a constant-camber hull. The topsides'
depth sets where the bottom starts, so ``side_aft`` against ``side_fwd`` is the
rocker and lifts the transom.

The deck is flat (the sheer at the deck datum) and the transom vertical and flat.
The design waterline is set afterwards for a displacement (``with_displacement``).
Units are millimetres; ``length`` is the hull from transom to stem.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as _onp

from hydrosandbox.hull.camber.curves import PlanCurve
from hydrosandbox.hull.camber.document import CamberHull, Station, StationPoint

#: Points on each section's superellipse quadrant (after its first).
BOTTOM_POINTS = 6


@dataclass(frozen=True)
class DemihullParameters:
    """A slender hull's numbers [mm, deg, -]. See the module docstring."""

    length: float = 5500.0
    half_beam: float = 190.0  # widest half-breadth of the sheer plan
    x_max_beam: float = 0.50  # its position, fraction of length from the transom
    transom_ratio: float = 0.70  # transom half-breadth / half_beam
    entry_angle: float = 9.0  # plan half-angle at the stem [deg]
    flare: float = 15.0  # topsides' inboard offset at their foot
    side_aft: float = 380.0  # topsides' depth below the sheer, aft station
    side_fwd: float = 450.0  # ... forward station
    bottom_aft: float = 200.0  # bottom's depth below the topsides, aft station
    bottom_fwd: float = 260.0  # ... forward station
    width_ratio: float = 0.95  # bottom's flat-out offset / half_beam
    shape_aft: float = 2.2  # bottom superellipse exponent, aft
    shape_fwd: float = 2.2  # ... forward

    def replace(self, **changes) -> "DemihullParameters":
        return DemihullParameters(**{**asdict(self), **changes})

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _plan_polygon(p: DemihullParameters) -> list[tuple[float, float]]:
    L, b = p.length, p.half_beam
    xm = p.x_max_beam * L
    b_t = p.transom_ratio * b
    x3 = xm + 0.12 * L
    x4 = x3 + 0.5 * (L - x3)
    y4 = min((L - x4) * math.tan(math.radians(p.entry_angle)), b)
    return [
        (0.0, b_t),
        (0.06 * L, b_t),  # square to the transom
        (0.5 * xm, b_t + 0.8 * (b - b_t)),
        (xm - 0.10 * L, b),
        (x3, b),
        (x4, y4),
        (L, 0.0),
    ]


def plan_polygon(p: DemihullParameters) -> list[tuple[float, float]]:
    """The sheer plan's control polygon, scaled in breadth so its widest point is ``half_beam``."""
    poly = _plan_polygon(p)
    curve = PlanCurve(poly)
    widest = max(curve.at(float(u))[1] for u in _onp.linspace(0.0, 1.0, 401))
    s = p.half_beam / widest
    return [(x, y * s) for x, y in poly]


def section_points(
    side: float, flare: float, bottom: float, width: float, shape: float, reach: float
) -> list[StationPoint]:
    """Topsides to ``(flare, -side)``, a superellipse bottom to ``(flare + width, -side - bottom)``, flat to ``reach``."""
    pts = [StationPoint(0.0, 0.1 * side, 0.0)]  # above the sheer, which trims it
    pts += [StationPoint(f * flare, -f * side, 0.0) for f in (0.0, 0.5)]
    theta = _onp.linspace(0.0, 0.5 * math.pi, BOTTOM_POINTS + 1)
    e = 2.0 / shape
    n = flare + width * (1.0 - _onp.cos(theta) ** e)
    z = -side - bottom * _onp.sin(theta) ** e
    pts += [StationPoint(float(a), float(c), 0.0) for a, c in zip(n, z)]
    pts.append(StationPoint(max(reach, flare + 1.05 * width), -side - bottom, 0.0))
    return pts


def demihull(p: DemihullParameters, name: str = "generated demihull") -> CamberHull:
    """The camber document of a slender hull (waterline at the topsides' foot; see ``with_displacement``)."""
    plan = plan_polygon(p)
    width = p.width_ratio * p.half_beam
    reach = 1.5 * max(width, p.half_beam) + p.flare + 0.05 * p.length
    stations = [
        Station(0.0, 0.0, section_points(p.side_aft, p.flare, p.bottom_aft, width, p.shape_aft, reach)),
        Station(1.0, 0.0, section_points(p.side_fwd, p.flare, p.bottom_fwd, width, p.shape_fwd, reach)),
    ]
    L = p.length
    deepest = max(p.side_aft + p.bottom_aft, p.side_fwd + p.bottom_fwd)
    return CamberHull(
        name=name,
        unit="mm",
        sheer_plan=plan,
        sheer_trim=[(0.0, 0.0, 0.0), (0.5 * L, 0.0, 0.0), (L, 0.0, 0.0)],
        # Just aft of the plan's start, so it trims nothing: the hull ends square at x = 0.
        transom=[(-1.0, 0.1 * deepest), (-1.0, -2.0 * deepest)],
        stations=stations,
        waterline=min(p.side_aft, p.side_fwd),
        deck_trim=0.0,
        extra={"generator": "hydrosandbox.hull.camber.generate.demihull", "parameters": asdict(p)},
    )


def with_displacement(doc: CamberHull, volume: float, iterations: int = 3, **hull_kwargs) -> CamberHull:
    """
    The document with its design waterline moved so the hull, level, displaces
    ``volume`` [m^3]. Each pass builds the hull and finds its level sinkage for
    the volume; a few passes settle the band-averaged table.
    """
    from scipy.optimize import brentq

    from hydrosandbox.hull import Hull

    doc = doc.copy()
    for _ in range(iterations):
        hull = Hull.from_camber(doc, **hull_kwargs)
        levels = _onp.asarray(hull.levels)
        f = lambda s: float(hull.hydrostatics(s, 0.0, (0.0, 0.0))["volume"]) - volume
        lo, hi = float(levels[0]) + 1e-4, float(levels[-1]) - 1e-4
        if f(lo) * f(hi) > 0:
            raise ValueError(f"no waterline displaces {volume} m^3 (range {f(lo) + volume:.4g}..{f(hi) + volume:.4g})")
        s = brentq(f, lo, hi, xtol=1e-6)
        doc.waterline -= s / doc.unit_m  # sinking deeper raises the waterline on the hull
        if abs(s) < 1e-4:
            break
    return doc
