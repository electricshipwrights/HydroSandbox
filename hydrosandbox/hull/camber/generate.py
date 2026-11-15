"""
Camber documents generated from a few physical numbers, for hulls no one drew.

camber's model is a section swept along the sheer plan and trimmed by the sheer,
the centreline and the transom. The keel, the rocker and the stem are not drawn:
they are where the section crosses the centreline, so they follow from the plan's
half-breadth ``b(x)`` and the section's depth ``z(n)`` at ``n = b``. This
generator draws what camber draws, with as few points as a designer would:

- the sheer plan, five control points: the transom (half-breadth
  ``transom_half_beam``), a second point level with it (so the plan leaves the
  transom square and the aft section is the transom), ``mid_half_beam`` at
  ``x_mid`` of the length from the transom, a point that sets the stem's
  half-angle ``entry_angle``, and the stem;
- the sheer, flat, three points;
- three or four stations (``station_u``) of four points each, as in hand-drawn
  documents: a knuckle at the sheer; the topsides' foot ``side`` down and
  ``flare`` inboard; a bilge point; and a knuckle at the keel, ``bottom`` below
  the topsides' foot and ``keel_reach`` inboard of the sheer (past the
  centreline, which trims it). ``fullness`` places the bilge point between the
  straight line foot-to-keel (0, a V) and the corner under the foot (1, a box).
  The foot gets just enough knuckle that the topsides never swing outboard of
  the sheer, so the bow closes (``inboard``).

Narrow stations, where the centreline cuts the section near the sheer, get fine,
deep-forefooted ends (a plumb stem when there is no flare); wide ones are cut
low on the bottom: a constant-camber hull. Stations' ``side`` against each other
is the rocker, and lifts the transom.

The transom is vertical and flat. The design waterline is set afterwards for a
displacement (``with_displacement``). Units are millimetres; ``length`` is the
hull from transom to stem.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as _onp

from hydrosandbox.hull.camber.document import CamberHull, Station, StationPoint

#: The deck datum's height above the (flat) sheer [mm]: the sections start at the
#: datum, so the sheer trims them.
DATUM = 20.0


@dataclass(frozen=True)
class DemihullParameters:
    """A slender hull's numbers [mm, deg, -]. See the module docstring. Per-station values are tuples."""

    length: float = 5500.0
    transom_half_beam: float = 160.0
    mid_half_beam: float = 185.0
    x_mid: float = 0.45  # fraction of length from the transom
    entry_angle: float = 10.0  # plan half-angle at the stem [deg]
    flare: float = 10.0
    keel_reach: float = 260.0  # keel knuckle's offset inboard of the sheer (past the centreline)
    station_u: tuple = (0.0, 0.4, 1.0)
    side: tuple = (330.0, 420.0, 480.0)  # topsides' depth below the sheer
    bottom: tuple = (180.0, 230.0, 220.0)  # keel knuckle's depth below the topsides' foot
    fullness: tuple = (0.6, 0.5, 0.4)

    def __post_init__(self):
        k = len(self.station_u)
        if not (len(self.side) == len(self.bottom) == len(self.fullness) == k >= 2):
            raise ValueError("station_u, side, bottom and fullness need one value per station (at least two)")
        if list(self.station_u) != sorted(self.station_u) or self.station_u[0] != 0.0 or self.station_u[-1] != 1.0:
            raise ValueError("station_u must rise from 0 to 1")

    def replace(self, **changes) -> "DemihullParameters":
        return DemihullParameters(**{**asdict(self), **changes})

    @property
    def n_stations(self) -> int:
        return len(self.station_u)

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def plan_polygon(p: DemihullParameters) -> list[tuple[float, float]]:
    """The sheer plan's five control points."""
    L = p.length
    xm = p.x_mid * L
    x3 = xm + 0.55 * (L - xm)
    y3 = min((L - x3) * math.tan(math.radians(p.entry_angle)), max(p.mid_half_beam, p.transom_half_beam))
    return [
        (0.0, p.transom_half_beam),
        (0.4 * xm, p.transom_half_beam),  # square to the transom
        (xm, p.mid_half_beam),
        (x3, y3),
        (L, 0.0),
    ]


def section_points(side: float, bottom: float, fullness: float, flare: float, reach: float) -> list[StationPoint]:
    """Sheer knuckle, topsides' foot, bilge point, keel knuckle."""
    foot = _onp.array([flare, -side])
    keel = _onp.array([reach, -side - bottom])
    straight = 0.5 * (foot + keel)
    corner = _onp.array([flare + 0.15 * (reach - flare), -side - 0.9 * bottom])
    bilge = straight + fullness * (corner - straight)
    return [
        StationPoint(0.0, DATUM, 1.0),
        StationPoint(float(foot[0]), float(foot[1]), 0.0),
        StationPoint(float(bilge[0]), float(bilge[1]), 0.0),
        StationPoint(float(keel[0]), float(keel[1]), 1.0),
    ]


def outboard_bulge(points: list[StationPoint]) -> float:
    """How far [mm] a station's curve swings outboard of the sheer (``-min n``, 0 if never)."""
    from hydrosandbox.hull.camber.sweep import Section

    pts = _onp.array([(q.n, q.z) for q in points])
    sec = Section(pts, [q.k for q in points], 1.0)
    return max(0.0, -float(sec.at(_onp.linspace(0.0, sec.vmax, 400))[:, 0].min()))


def inboard(points: list[StationPoint], tolerance: float = 0.05) -> list[StationPoint]:
    """
    The station with the least knuckle at the topsides' foot that keeps its curve
    inboard of the sheer. A smooth foot lets the topsides swing outboard of the
    sheer, and at the stem, where the plan meets the centreline, that swing
    survives the centreline trim and leaves the bow open.
    """
    def with_k(k):
        return [q if i != 1 else StationPoint(q.n, q.z, k) for i, q in enumerate(points)]

    if outboard_bulge(points) <= tolerance:
        return points
    lo, hi = 0.0, 1.0
    if outboard_bulge(with_k(hi)) > tolerance:
        return with_k(hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if outboard_bulge(with_k(mid)) > tolerance:
            lo = mid
        else:
            hi = mid
    return with_k(hi)


def demihull(p: DemihullParameters, name: str = "generated demihull") -> CamberHull:
    """The camber document of a slender hull (waterline at the shallowest topsides' foot; see ``with_displacement``)."""
    plan = plan_polygon(p)
    widest = max(y for _, y in plan)
    reach = max(p.keel_reach, 1.1 * widest + p.flare)  # the keel must be past the centreline everywhere
    stations = [
        Station(float(u), 1.0, inboard(section_points(s, b, f, p.flare, reach)))
        for u, s, b, f in zip(p.station_u, p.side, p.bottom, p.fullness)
    ]
    L = p.length
    deepest = max(s + b for s, b in zip(p.side, p.bottom))
    return CamberHull(
        name=name,
        unit="mm",
        sheer_plan=plan,
        sheer_trim=[(0.0, 0.0, 0.0), (0.5 * L, 0.0, 0.0), (L, 0.0, 0.0)],
        # Just aft of the plan's start, so it trims nothing: the hull ends square at x = 0.
        transom=[(-1.0, DATUM), (-1.0, -2.0 * deepest)],
        stations=stations,
        waterline=min(p.side),
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
