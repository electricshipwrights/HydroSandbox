"""
camber hull documents (``src/core/{document,json,hull}.ts`` and
``src/legacy/v1/convert.ts``): reading format versions 1 and 2, units, and
writing version 2 back out.

A document is one hull, in camber's own frame: ``x`` forward from the sheer
plan's start, ``y`` the half-breadth to starboard, ``z`` up from the flat deck
datum (the hull hangs below it), in the document's ``unit``.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

from hydrosandbox.hull.camber.curves import Pchip, PlanCurve, _clamp

#: The format version written, and the newest read.
VERSION = 2
#: Millimetres per unit.
UNIT_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}


@dataclass
class StationPoint:
    n: float  # inboard offset from the sheer, in the station plane
    z: float  # height above the deck datum (<= 0)
    k: float  # knuckle strength in [0, 1]


@dataclass
class Station:
    u: float  # position along the sheer plan's parameter, in [0, 1]
    keel_k: float
    points: list[StationPoint]


@dataclass
class CamberHull:
    """A camber hull's authored state (camber's ``HullState``)."""

    name: str
    unit: str
    sheer_plan: list[tuple[float, float]]  # (x, y) control polygon
    sheer_trim: list[tuple[float, float, float]]  # (x, z, k)
    transom: list[tuple[float, float]]  # [(x, z) top, (x, z) bottom]
    stations: list[Station]
    waterline: float = 0.0  # depth of the design waterline below the deck datum
    deck_trim: float = 0.0  # [rad], bow up
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def unit_m(self) -> float:
        """Metres per document unit."""
        return UNIT_MM[self.unit] / 1000.0

    @property
    def loa(self) -> float:
        return self.sheer_plan[-1][0] - self.sheer_plan[0][0]

    @property
    def topology(self) -> tuple[int, int, int, int]:
        """(plan points, trim points, points per station, stations)."""
        return (
            len(self.sheer_plan),
            len(self.sheer_trim),
            len(self.stations[0].points),
            len(self.stations),
        )

    def copy(self) -> "CamberHull":
        return copy.deepcopy(self)

    def scaled(self, factor: float) -> "CamberHull":
        """The same shape scaled uniformly by ``factor`` (lengths and waterline depth)."""
        out = self.converted(self.unit)
        out.sheer_plan = [(x * factor, y * factor) for x, y in out.sheer_plan]
        out.sheer_trim = [(x * factor, z * factor, k) for x, z, k in out.sheer_trim]
        out.transom = [(x * factor, z * factor) for x, z in out.transom]
        for st in out.stations:
            st.points = [StationPoint(p.n * factor, p.z * factor, p.k) for p in st.points]
        out.waterline *= factor
        return out

    def converted(self, unit: str) -> "CamberHull":
        """The same hull with its lengths expressed in ``unit``."""
        out = self.copy()
        if unit == self.unit:
            return out
        s = UNIT_MM[self.unit] / UNIT_MM[unit]
        out.unit = unit
        out.sheer_plan = [(x * s, y * s) for x, y in out.sheer_plan]
        out.sheer_trim = [(x * s, z * s, k) for x, z, k in out.sheer_trim]
        out.transom = [(x * s, z * s) for x, z in out.transom]
        for st in out.stations:
            st.points = [StationPoint(p.n * s, p.z * s, p.k) for p in st.points]
        out.waterline = out.waterline * s
        return out

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The version-2 document."""
        return {
            "version": VERSION,
            "name": self.name,
            "unit": self.unit,
            "waterline": self.waterline,
            "deckTrimDeg": math.degrees(self.deck_trim),
            "sheerPlan": [{"x": x, "y": y} for x, y in self.sheer_plan],
            "sheerTrim": [{"x": x, "z": z, "k": k} for x, z, k in self.sheer_trim],
            "transom": [{"x": x, "z": z} for x, z in self.transom],
            "stations": [
                {
                    "u": st.u,
                    "keelK": st.keel_k,
                    "points": [{"z": p.z, "n": p.n, "k": p.k} for p in st.points],
                }
                for st in self.stations
            ],
        }

    def to_json(self, path: str | None = None) -> str:
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w") as f:
                f.write(text)
        return text

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "CamberHull":
        return parse_document(doc)

    @classmethod
    def from_json(cls, path_or_text) -> "CamberHull":
        """Read a camber document from a path or from its JSON text."""
        text = path_or_text = str(path_or_text) if hasattr(path_or_text, "__fspath__") else path_or_text
        if not path_or_text.lstrip().startswith("{"):
            with open(path_or_text) as f:
                text = f.read()
        return parse_document(json.loads(text))


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def _num(v: Any, ctx: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{ctx} must be a finite number")
    return float(v)


def _knuckle(v: Any) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        return _clamp(float(v), 0.0, 1.0)
    return 0.0


def station_knuckle(count: int, idx: int, k: float) -> float:
    """A section is cut at its ends, so its first and last points are corners (k = 1)."""
    return 1.0 if idx == 0 or idx == count - 1 else k


def document_version(doc: dict[str, Any]) -> float:
    v = doc.get("version")
    if v is None:
        return 1.0
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else math.nan


def parse_document(doc: dict[str, Any]) -> CamberHull:
    """Decode a camber document (version 1 or 2) as camber's ``parseDocument`` does."""
    if not isinstance(doc, dict):
        raise ValueError("document must be an object")
    version = document_version(doc)
    if not version <= VERSION:
        raise ValueError(f"camber document version {doc.get('version')!r}: this reads version {VERSION} and earlier")
    if "sheerPlan" not in doc:
        raise ValueError("not a camber hull document (no sheerPlan)")
    if version < 2:
        doc = convert_v1_to_v2(doc)

    def arr(v, ctx, n_min):
        if not isinstance(v, list):
            raise ValueError(f"{ctx} must be an array")
        if len(v) < n_min:
            raise ValueError(f"{ctx} must have at least {n_min} entries")
        return v

    plan = [
        (_num(p["x"], f"sheerPlan[{i}].x"), _num(p["y"], f"sheerPlan[{i}].y"))
        for i, p in enumerate(arr(doc.get("sheerPlan"), "sheerPlan", 2))
    ]
    for i in range(1, len(plan)):
        if plan[i][0] <= plan[i - 1][0]:
            raise ValueError(f"sheerPlan[{i}].x must be > the previous point's x")
    trim = [
        (_num(p["x"], f"sheerTrim[{i}].x"), _num(p["z"], f"sheerTrim[{i}].z"), _knuckle(p.get("k")))
        for i, p in enumerate(arr(doc.get("sheerTrim"), "sheerTrim", 2))
    ]
    for i in range(1, len(trim)):
        if trim[i][0] <= trim[i - 1][0]:
            raise ValueError(f"sheerTrim[{i}].x must be > the previous point's x")
    raw_tr = arr(doc.get("transom"), "transom", 2)
    if len(raw_tr) != 2:
        raise ValueError("transom must have exactly 2 points (top and bottom)")
    transom = [(_num(p["x"], f"transom[{i}].x"), _num(p["z"], f"transom[{i}].z")) for i, p in enumerate(raw_tr)]
    if transom[1][1] >= transom[0][1]:
        raise ValueError("transom[1] (the bottom) must be below transom[0]")
    raw_sts = arr(doc.get("stations"), "stations", 1)
    n_sec = len(arr(raw_sts[0].get("points"), "stations[0].points", 2))
    stations = []
    for j, s in enumerate(raw_sts):
        pts = arr(s.get("points"), f"stations[{j}].points", 2)
        if len(pts) != n_sec:
            raise ValueError(f"stations[{j}].points must have {n_sec} points (every station shares one count)")
        stations.append(
            Station(
                u=_clamp(_num(s["u"], f"stations[{j}].u"), 0.0, 1.0),
                keel_k=_knuckle(s.get("keelK")),
                points=[
                    StationPoint(
                        n=_num(q["n"], f"stations[{j}].points[{i}].n"),
                        z=_num(q["z"], f"stations[{j}].points[{i}].z"),
                        k=station_knuckle(len(pts), i, _knuckle(q.get("k"))),
                    )
                    for i, q in enumerate(pts)
                ],
            )
        )
    stations.sort(key=lambda st: st.u)
    for j in range(1, len(stations)):
        if stations[j].u <= stations[j - 1].u:
            raise ValueError(f"stations: two stations share the same u ({stations[j].u})")
    unit = doc.get("unit")
    if unit not in UNIT_MM:
        raise ValueError(f"unit must be one of {', '.join(UNIT_MM)}")
    waterline = doc.get("waterline")
    waterline = float(waterline) if isinstance(waterline, (int, float)) and math.isfinite(waterline) else 0.0
    angle = doc.get("deckTrimDeg", doc.get("deckRakeDeg"))
    angle = float(angle) if isinstance(angle, (int, float)) and math.isfinite(angle) else 0.0
    return CamberHull(
        name=doc.get("name") if isinstance(doc.get("name"), str) else "",
        unit=unit,
        sheer_plan=plan,
        sheer_trim=trim,
        transom=transom,
        stations=stations,
        waterline=waterline,
        deck_trim=math.radians(angle),
    )


# ----------------------------------------------------------------------------
# Version 1 (legacy/v1/convert.ts)
# ----------------------------------------------------------------------------
def _norm_simplex(w):
    c = [x if x > 0 else 0.0 for x in w]
    s = sum(c)
    return [x / s for x in c] if s > 0 else [1.0 / len(c)] * len(c)


def _local_maxima(f, n=400):
    w = [f(i / n) for i in range(n + 1)]
    out = []
    i = 0
    while i <= n:
        rise = i == 0 or w[i] > w[i - 1] + 1e-12
        if not rise:
            i += 1
            continue
        j = i
        while j < n and abs(w[j + 1] - w[i]) <= 1e-12:
            j += 1
        fall = j == n or w[j + 1] < w[i] - 1e-12
        if fall:
            if j > i:
                out.append((i + j) / 2 / n)
            else:
                out.append(_refine_max(f, max(0, i - 1) / n, min(n, i + 1) / n))
        i = j + 1
    return out


def _refine_max(f, a, b):
    for _ in range(60):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if f(m1) < f(m2):
            a = m1
        else:
            b = m2
    return (a + b) / 2


def convert_v1_to_v2(doc: dict[str, Any]) -> dict[str, Any]:
    """camber's ``convertV1ToV2``: templates blended by a weight curve → stations."""
    x = 0.0
    plan = []
    for i, p in enumerate(doc["sheerPlan"]):
        x = p["dx"] if i == 0 else x + p["dx"]
        plan.append((x, p["y"], _norm_simplex(p["w"])))
    x = 0.0
    trim = []
    for i, p in enumerate(doc["sheerTrim"]):
        x = p["dx"] if i == 0 else x + p["dx"]
        trim.append((x, -p["depth"], _knuckle(p.get("k"))))
    templates = []
    for pts in doc["templates"]:
        d = 0.0
        tpl = []
        for i, p in enumerate(pts):
            d = 0.0 if i == 0 else d + p["dd"]
            tpl.append((p["n"], d, _knuckle(p.get("k"))))
        templates.append(tpl)
    K = len(templates)

    if len(plan) <= 1:
        w0 = _norm_simplex(plan[0][2]) if plan else [1.0]

        def weight_at(_x):
            return list(w0)
    else:
        xs = [c[0] for c in plan]
        comps = [Pchip(xs, [c[2][j] for c in plan]) for j in range(len(plan[0][2]))]

        def weight_at(xx):
            return _norm_simplex([c(xx) for c in comps])

    curve = PlanCurve([(p[0], p[1]) for p in plan])
    keel_k = doc.get("keelK") or []
    stations = []
    for j in range(K):

        def f(u, j=j):
            w = weight_at(curve.at(u)[0])
            return w[j] if j < len(w) else 0.0

        peaks = _local_maxima(f) or [_refine_max(f, 0.0, 1.0)]
        best = max(f(u) for u in peaks)
        keep = [u for u in peaks if f(u) >= best * 0.5]
        for u in keep or peaks:
            stations.append(
                {
                    "u": _clamp(u, 0.0, 1.0),
                    "keelK": _clamp(_knuckle(keel_k[j] if j < len(keel_k) else 0), 0.0, 1.0),
                    "points": [
                        {"z": -d, "n": n, "k": station_knuckle(len(templates[j]), i, k)}
                        for i, (n, d, k) in enumerate(templates[j])
                    ],
                }
            )
    stations.sort(key=lambda s: s["u"])
    for i in range(1, len(stations)):
        if stations[i]["u"] - stations[i - 1]["u"] < 1e-3:
            stations[i]["u"] = stations[i - 1]["u"] + 1e-3
    over = stations[-1]["u"] - 1 if stations else 0.0
    if over > 0:
        for s in stations:
            s["u"] = _clamp(s["u"] - over, 0.0, 1.0)
    t = doc["transom"]
    top = {"x": t["x"], "z": -t["depthTop"]}
    z_bot = -(t["depthTop"] + t["dDepthBot"])
    bot = {"x": t["x"] + (z_bot - top["z"]) * t["transomRake"], "z": z_bot}
    return {
        "version": VERSION,
        "name": doc.get("name", ""),
        "unit": "mm",
        "waterline": doc.get("waterline", 0),
        "deckTrimDeg": doc.get("deckRakeDeg", 0),
        "sheerPlan": [{"x": p[0], "y": p[1]} for p in plan],
        "sheerTrim": [{"x": p[0], "z": p[1], "k": p[2]} for p in trim],
        "transom": [top, bot],
        "stations": stations,
    }
