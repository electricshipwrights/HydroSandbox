"""
camber's curve primitives, ported from its ``src/core/{pchip,bspline,spline}.ts``:

- monotone cubic Hermite (PCHIP), in camber's C2-Fritsch-Carlson variant, whose
  slopes are C2 in the knots (so curves through blended knots stay smooth), with
  per-point knuckles;
- the clamped cubic B-spline that carries the sheer plan, read as a parametric
  curve ``P(u)`` with its exact hodograph;
- the non-uniform (centripetal) Catmull-Rom chain with per-point knuckles that
  draws every other curve: the stations' longitudinal loft and each section.

Each function mirrors its camber namesake; the comments there explain the
choices, and only what differs here is noted.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as _onp


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


# ----------------------------------------------------------------------------
# PCHIP (pchip.ts)
# ----------------------------------------------------------------------------
MID_EASE = 0.25
END_EASE = 0.25


def _c2_ease_in(q: float) -> float:
    return q * q * q * (6 + q * (3 * q - 8))


def _c2_ease_in_out(q: float) -> float:
    return q * q * q * (10 + q * (6 * q - 15))


def _pchip_end(h0: float, h1: float, d0: float, d1: float) -> float:
    m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    s = 3 * d0
    if s == 0:
        return 0.0
    t = m / s
    if t <= 0:
        return 0.0
    if t >= 1:
        return s
    if t < END_EASE:
        return s * END_EASE * _c2_ease_in(t / END_EASE)
    if t > 1 - END_EASE:
        return s * (1 - END_EASE * _c2_ease_in((1 - t) / END_EASE))
    return m


def pchip_slopes(xs: Sequence[float], ys: Sequence[float]) -> list[float]:
    """Knot slopes of camber's C2-Fritsch-Carlson monotone cubic through ``(xs, ys)``."""
    n = len(xs)
    if n == 1:
        return [0.0]
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    d = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    m = [0.0] * n
    for i in range(1, n - 1):
        a, b = d[i - 1], d[i]
        if a * b <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            mi = (w1 + w2) / (w1 / a + w2 / b)
            r = min(abs(a), abs(b)) / max(abs(a), abs(b))
            if r < MID_EASE:
                mi *= _c2_ease_in_out(r / MID_EASE)
            m[i] = mi
    m[0] = _pchip_end(h[0], h[1], d[0], d[1])
    m[n - 1] = _pchip_end(h[n - 2], h[n - 3], d[n - 2], d[n - 3])
    return m


def knuckle_slopes(
    xs: Sequence[float], ys: Sequence[float], ks: Sequence[float]
) -> tuple[list[float], list[float]]:
    """Left and right knot slopes of a knuckled PCHIP (``k = 1`` takes the secants)."""
    n = len(xs)
    m = pchip_slopes(xs, ys)
    left, right = list(m), list(m)
    for i in range(n):
        k = _clamp(ks[i] if i < len(ks) else 0.0, 0.0, 1.0)
        if k == 0:
            continue
        if i > 0:
            left[i] = m[i] + ((ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]) - m[i]) * k
        if i < n - 1:
            right[i] = m[i] + ((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) - m[i]) * k
    return left, right


def hermite_eval_lr(xs, ys, left, right, tt: float) -> float:
    """Hermite evaluation with independent left/right slopes at each knot."""
    i = 0
    while i < len(xs) - 2 and tt > xs[i + 1]:
        i += 1
    h = xs[i + 1] - xs[i]
    t = (tt - xs[i]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * ys[i]
        + (t3 - 2 * t2 + t) * h * right[i]
        + (-2 * t3 + 3 * t2) * ys[i + 1]
        + (t3 - t2) * h * left[i + 1]
    )


def hermite_eval(xs, ys, m, tt: float) -> float:
    return hermite_eval_lr(xs, ys, m, m, tt)


class Pchip:
    """A monotone cubic ``y(x)`` through the knots, clamped to their span."""

    def __init__(self, xs: Sequence[float], ys: Sequence[float]):
        self.xs = [float(x) for x in xs]
        self.ys = [float(y) for y in ys]
        self.m = pchip_slopes(self.xs, self.ys) if len(xs) > 1 else [0.0]

    def __call__(self, x: float) -> float:
        if len(self.xs) < 2:
            return self.ys[0] if self.ys else 0.0
        return hermite_eval(self.xs, self.ys, self.m, _clamp(x, self.xs[0], self.xs[-1]))


# ----------------------------------------------------------------------------
# Clamped B-spline plan curve (bspline.ts)
# ----------------------------------------------------------------------------
def uniform_knots(count: int, p: int) -> list[float]:
    """Clamped uniform knot vector for ``count`` control points of degree ``p``."""
    interior = count - p - 1
    return [0.0] * (p + 1) + [i / (interior + 1) for i in range(1, interior + 1)] + [1.0] * (p + 1)


def find_span(n: int, p: int, u: float, knots: Sequence[float]) -> int:
    """The knot span containing ``u`` (The NURBS Book, A2.1)."""
    if u >= knots[n + 1]:
        return n
    if u <= knots[p]:
        return p
    lo, hi = p, n + 1
    mid = (lo + hi) >> 1
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) >> 1
    return mid


def de_boor(pts: Sequence[Sequence[float]], knots, p: int, u: float) -> list[float]:
    """A point of the B-spline with control polygon ``pts`` at ``u``."""
    n = len(pts) - 1
    span = find_span(n, p, u, knots)
    d = [list(pts[span - p + j]) for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = span - p + j
            den = knots[i + p - r + 1] - knots[i]
            a = (u - knots[i]) / den if den > 0 else 0.0
            d[j] = [(1 - a) * d[j - 1][c] + a * d[j][c] for c in range(len(d[j]))]
    return d[p]


class PlanCurve:
    """
    The sheer plan ``P(u) = (x, y)``, ``u`` in [0, 1]: a clamped B-spline over a
    clamped uniform knot vector, cubic unless the polygon is too short for it.
    """

    def __init__(self, pts: Sequence[Sequence[float]]):
        self.pts = [(float(x), float(y)) for x, y in pts]
        n_cp = len(self.pts)
        self.degenerate = n_cp < 2
        if self.degenerate:
            return
        p = min(3, n_cp - 1)
        knots = uniform_knots(n_cp, p)
        self.p, self.knots = p, knots
        d_pts = []
        for i in range(n_cp - 1):
            den = knots[i + p + 1] - knots[i + 1]
            d_pts.append(
                (
                    p * (self.pts[i + 1][0] - self.pts[i][0]) / den,
                    p * (self.pts[i + 1][1] - self.pts[i][1]) / den,
                )
                if den > 0
                else (0.0, 0.0)
            )
        self.d_pts, self.d_knots, self.dp = d_pts, knots[1:-1], p - 1
        self.x0 = self.at(0.0)[0]
        self.x1 = self.at(1.0)[0]

    def at(self, u: float) -> tuple[float, float]:
        if self.degenerate:
            return tuple(self.pts[0]) if self.pts else (0.0, 0.0)
        x, y = de_boor(self.pts, self.knots, self.p, _clamp(u, 0.0, 1.0))
        return x, y

    def d(self, u: float) -> tuple[float, float]:
        """``dP/du``, exactly (the hodograph)."""
        if self.degenerate:
            return (1.0, 0.0)
        if self.dp < 1:
            return self.d_pts[0]
        x, y = de_boor(self.d_pts, self.d_knots, self.dp, _clamp(u, 0.0, 1.0))
        return x, y

    def u_at_x(self, x: float) -> float:
        """``u`` of a given ``x``, inverting the monotone ``x(u)`` by bisection."""
        if self.degenerate:
            return 0.0
        xc = _clamp(x, self.x0, self.x1)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.at(mid)[0] < xc:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# Catmull-Rom chains (spline.ts)
# ----------------------------------------------------------------------------
def centripetal_params(pts: Sequence[Sequence[float]]) -> list[float]:
    """Knots spaced by the square root of the chord, each step floored at 1e-6."""
    t = [0.0]
    for i in range(1, len(pts)):
        d = math.hypot(*(a - b for a, b in zip(pts[i], pts[i - 1])))
        t.append(t[-1] + max(math.sqrt(d), 1e-6))
    return t


def _tangent(p0, p1, p2, t0, t1, t2):
    p0, p1, p2 = _onp.asarray(p0), _onp.asarray(p1), _onp.asarray(p2)
    return (p1 - p0) / (t1 - t0) - (p2 - p0) / (t2 - t0) + (p2 - p1) / (t2 - t1)


def cr_chain(vals: Sequence[Sequence[float]], t: Sequence[float], ks: Sequence[float]) -> _onp.ndarray:
    """
    The Bezier segments ``(n - 1, 4, dim)`` of the Catmull-Rom chain through
    ``vals`` over knots ``t``, creased by the knuckles ``ks`` (the end points
    always count as ``k = 1``).
    """
    vals = _onp.asarray(vals, dtype=float)
    n = len(vals)
    segs = _onp.empty((max(n - 1, 0), 4, vals.shape[1]))
    for j in range(n - 1):
        p, q = vals[j], vals[j + 1]
        chord = q - p
        dt = t[j + 1] - t[j]
        i1 = _tangent(vals[j - 1], p, q, t[j - 1], t[j], t[j + 1]) * dt if j > 0 else None
        i2 = _tangent(p, q, vals[j + 2], t[j], t[j + 1], t[j + 2]) * dt if j + 2 < n else None
        e1 = 2 * chord - i2 if i2 is not None else chord
        e2 = 2 * chord - i1 if i1 is not None else chord
        kp = 1.0 if j == 0 else (ks[j] if j < len(ks) else 0.0)
        kq = 1.0 if j + 2 == n else (ks[j + 1] if j + 1 < len(ks) else 0.0)
        m1 = i1 if i1 is not None else e1
        m2 = i2 if i2 is not None else e2
        cp, cq = _clamp(kp, 0.0, 1.0), _clamp(kq, 0.0, 1.0)
        t1 = e1 + (chord - e1) * cq
        t2 = e2 + (chord - e2) * cp
        segs[j, 0] = p
        segs[j, 1] = p + (m1 + (t1 - m1) * cp) / 3
        segs[j, 2] = q - (m2 + (t2 - m2) * cq) / 3
        segs[j, 3] = q
    return segs


def eval_chain(segs: _onp.ndarray, v):
    """
    The chain at parameter ``v`` in ``[0, n_segments]`` (knot ``j`` at ``v = j``,
    clamped to the ends); ``v`` may be an array, giving ``(len(v), dim)``.
    """
    n = len(segs)
    v_arr = _onp.asarray(v, dtype=float)
    c = _onp.clip(v_arr, 0.0, n)
    j = _onp.minimum(_onp.floor(c), n - 1).astype(int)
    s = (c - j)[..., None]
    b = segs[j]
    w = 1 - s
    out = w**3 * b[..., 0, :] + 3 * w * w * s * b[..., 1, :] + 3 * w * s * s * b[..., 2, :] + s**3 * b[..., 3, :]
    return out


def eval_chain_d(segs: _onp.ndarray, v):
    """``dP/dv`` of the chain, in the parameter of ``eval_chain``."""
    n = len(segs)
    c = _onp.clip(_onp.asarray(v, dtype=float), 0.0, n)
    j = _onp.minimum(_onp.floor(c), n - 1).astype(int)
    s = (c - j)[..., None]
    b = segs[j]
    w = 1 - s
    return (
        3 * w * w * (b[..., 1, :] - b[..., 0, :])
        + 6 * w * s * (b[..., 2, :] - b[..., 1, :])
        + 3 * s * s * (b[..., 3, :] - b[..., 2, :])
    )
