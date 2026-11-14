"""
camber's sweep (``src/core/{model,mesh}.ts``): a section carried along the sheer
plan, its shape lofted between the stations, and the sheet trimmed by the sheer
trim, the centreline and the transom plane.

The trimmed half-sections are the regularized path of camber's ``mesh.ts``
(``sweptSection``: each column trimmed on its own to a constant row count), the
one camber's STEP export samples. Its topology is fixed, so the swept points move
continuously with the authored numbers, which is what a design variable needs.
"""

from __future__ import annotations

import math

import numpy as _onp

from hydrosandbox.hull.camber.curves import (
    Pchip,
    PlanCurve,
    _clamp,
    centripetal_params,
    cr_chain,
    eval_chain,
    eval_chain_d,
    hermite_eval,
    pchip_slopes,
)
from hydrosandbox.hull.camber.document import CamberHull

#: Sub-steps per section segment (camber's ``R_DEFAULT``).
R_DEFAULT = 16


class Loft:
    """Point ``i`` of every station traced along the hull (camber's ``buildLoft``)."""

    def __init__(self, hull: CamberHull):
        sts = hull.stations
        self.K = len(sts)
        self.S = len(sts[0].points)
        self.us = [st.u for st in sts]
        if self.K == 1:
            self._pts = _onp.array([[p.n, p.z] for p in sts[0].points])
            self._ks = [p.k for p in sts[0].points]
            self._keel_k = sts[0].keel_k
            return
        zeros = [0.0] * self.K
        self.chains = [
            cr_chain([[st.points[i].n, st.points[i].z] for st in sts], self.us, zeros) for i in range(self.S)
        ]
        self.k_curves = []
        for i in range(self.S):
            ys = [st.points[i].k for st in sts]
            self.k_curves.append((ys, pchip_slopes(self.us, ys)))
        keel_ys = [st.keel_k for st in sts]
        self.keel = (keel_ys, pchip_slopes(self.us, keel_ys))

    def param(self, u: float) -> float:
        us, K = self.us, self.K
        c = _clamp(u, us[0], us[-1])
        j = 0
        while j < K - 2 and c > us[j + 1]:
            j += 1
        return j + (c - us[j]) / ((us[j + 1] - us[j]) or 1.0)

    def at(self, u: float) -> tuple[_onp.ndarray, list[float], float]:
        """The lofted ``(S, 2)`` station points ``(n, z)``, their knuckles, and the keel knuckle."""
        if self.K == 1:
            return self._pts.copy(), list(self._ks), self._keel_k
        t = self.param(u)
        uc = _clamp(u, self.us[0], self.us[-1])
        pts = _onp.array([eval_chain(ch, t) for ch in self.chains])
        ks = [_clamp(hermite_eval(self.us, ys, m, uc), 0.0, 1.0) for ys, m in self.k_curves]
        keel_k = _clamp(hermite_eval(self.us, self.keel[0], self.keel[1], uc), 0.0, 1.0)
        return pts, ks, keel_k


class Section:
    """The section at some ``u``: a knuckled centripetal Catmull-Rom in ``(n, z)``, point ``i`` at ``v = i``."""

    def __init__(self, pts: _onp.ndarray, ks: list[float], keel_k: float):
        self.pts = pts
        self.ks = ks
        self.keel_k = keel_k
        self.segs = cr_chain(pts, centripetal_params(pts), ks)
        self.vmax = len(self.segs)

    def at(self, v):
        return eval_chain(self.segs, v)

    def d(self, v):
        return eval_chain_d(self.segs, v)


class SweptHull:
    """
    A camber hull assembled for evaluation (camber's ``Model``): the plan curve, the
    sheer-trim graph and the loft, and the sweep and trims built on them. All in the
    document's own frame and unit.
    """

    def __init__(self, hull: CamberHull):
        self.hull = hull
        self.plan = PlanCurve(hull.sheer_plan)
        self.trim_z = Pchip([p[0] for p in hull.sheer_trim], [p[1] for p in hull.sheer_trim])
        self.loft = Loft(hull)
        (self._tx_a, self._tz_a), (self._tx_b, self._tz_b) = hull.transom

    # -- frame and section ---------------------------------------------------
    def frame(self, u: float) -> tuple[_onp.ndarray, _onp.ndarray, _onp.ndarray]:
        """Origin ``p``, heading ``T`` and inboard normal ``n`` of the station plane at ``u``."""
        x, y = self.plan.at(u)
        dx, dy = self.plan.d(u)
        l = math.hypot(dx, dy) or 1.0
        T = _onp.array([dx / l, dy / l, 0.0])
        return _onp.array([x, y, 0.0]), T, _onp.array([T[1], -T[0], 0.0])

    def section(self, u: float) -> Section:
        return Section(*self.loft.at(u))

    @staticmethod
    def station_world(frame, nz) -> _onp.ndarray:
        """World points of station-plane offsets ``nz[..., (n, z)]``."""
        p, _, nh = frame
        nz = _onp.asarray(nz, dtype=float)
        return _onp.stack(
            [p[0] + nz[..., 0] * nh[0], p[1] + nz[..., 0] * nh[1], nz[..., 1]], axis=-1
        )

    def x_transom(self, z):
        """``x`` of the transom plane at height ``z``; the hull keeps ``x >= x_transom``."""
        return self._tx_a + (self._tx_b - self._tx_a) * ((z - self._tz_a) / ((self._tz_b - self._tz_a) or 1.0))

    def keep(self, world: _onp.ndarray) -> _onp.ndarray:
        """How far inside the hull world points are: the min of the three trims (>= 0 is kept)."""
        world = _onp.atleast_2d(world)
        trim = _onp.array([self.trim_z(x) for x in world[:, 0]])
        return _onp.minimum(_onp.minimum(trim - world[:, 2], world[:, 1]), world[:, 0] - self.x_transom(world[:, 2]))

    # -- trimming --------------------------------------------------------------
    def kept_span(self, frame, sec: Section, fn: int = 96) -> tuple[float, float] | None:
        """camber's ``keptSpan``: the trimmed ``[v_top, v_bot]`` of a section, or None."""

        def g(v):
            return float(self.keep(self.station_world(frame, sec.at(v)))[0])

        vs = sec.vmax * _onp.arange(fn + 1) / fn
        gs = self.keep(self.station_world(frame, sec.at(vs)))
        inside = _onp.nonzero(gs >= 0)[0]
        if inside.size == 0:
            return None
        lo, hi = int(inside[0]), int(inside[-1])
        v_top = 0.0 if lo == 0 else _bisect_root(g, vs[lo - 1], vs[lo], gs[lo - 1])
        v_bot = sec.vmax if hi == fn else _bisect_root(g, vs[hi + 1], vs[hi], gs[hi + 1])
        return (v_top, v_bot) if v_bot > v_top + 1e-9 else None

    def swept_section(self, u: float, R: int = R_DEFAULT, trim: bool = True) -> dict:
        """
        camber's ``sweptSection``: the trimmed starboard half-section at ``u`` as
        ``(S - 1) R + 1`` world points from the sheer down to the keel (``pts`` is
        None where nothing survives the trims).
        """
        frame = self.frame(u)
        sec = self.section(u)
        S = sec.vmax + 1
        span = self.kept_span(frame, sec) if trim else (0.0, float(sec.vmax))
        if span is None:
            return {"pts": None, "keel": False}
        anchors = _anchors(S, *span)
        r = _onp.arange(1, R + 1) / R
        vs = [anchors[0]] + [v for i in range(S - 1) for v in anchors[i] + (anchors[i + 1] - anchors[i]) * r]
        pts = self.station_world(frame, sec.at(_onp.asarray(vs)))
        keel = trim and abs(pts[-1, 1]) < 1e-6
        if keel:
            pts[-1, 1] = 0.0
        return {"pts": pts, "keel": keel}

    # -- extent ----------------------------------------------------------------
    def _exists(self, u: float) -> bool:
        return self.kept_span(self.frame(u), self.section(u)) is not None

    def forward_limit(self) -> float:
        """The last ``u`` with a section (camber's ``forwardLimit``)."""
        if self._exists(1.0):
            return 1.0
        lo, hi = 0.5, 1.0
        if not self._exists(lo):
            return 1.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if self._exists(m):
                lo = m
            else:
                hi = m
        return lo

    def aft_limit(self) -> float:
        """The first ``u`` with a section (camber's ``aftLimit``)."""
        if self._exists(0.0):
            return 0.0
        lo, hi = 0.0, 0.5
        if not self._exists(hi):
            return 0.0
        for _ in range(40):
            m = 0.5 * (lo + hi)
            if self._exists(m):
                hi = m
            else:
                lo = m
        return hi

    def surface_grid(self, n_columns: int = 121, R: int = 8) -> _onp.ndarray:
        """
        The trimmed starboard half-surface as a ``(n_columns, (S - 1) R + 1, 3)``
        grid of world points (document frame and unit), sheer to keel.

        Columns run uniformly in ``u`` between the aft and forward limits rather than
        over all of ``[0, 1]``, so every column has a section and the grid's
        topology never changes as the authored numbers move; the end columns are
        pulled a hair inside the limits, where the section has just vanished.
        """
        u0, u1 = self.aft_limit(), self.forward_limit()
        pad = 1e-6 * (u1 - u0)
        us = _onp.linspace(u0 + pad, u1 - pad, n_columns)
        rows = []
        for u in us:
            s = self.swept_section(float(u), R)
            if s["pts"] is None:
                raise RuntimeError(f"camber sweep: no section at u = {u:.6f} between the hull's limits")
            rows.append(s["pts"])
        return _onp.stack(rows)


def _bisect_root(g, a: float, b: float, ga: float) -> float:
    for _ in range(40):
        m = 0.5 * (a + b)
        if (g(m) < 0) == (ga < 0):
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def _anchors(S: int, v_top: float, v_bot: float) -> list[float]:
    margin = (v_bot - v_top) * 1e-3
    return [_clamp(float(i), v_top + i * margin, v_bot - (S - 1 - i) * margin) for i in range(S)]
