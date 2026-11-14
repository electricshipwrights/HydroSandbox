"""
Hull families and their blends (camber's ``src/core/promote.ts`` and
``src/interpolate/blend.ts``).

A blend of hulls ``V_1 ... V_n`` with weights ``w_i >= 0, sum w_i = 1`` is
``sum w_i V_i``, taken number by number over a shared topology: every authored
coordinate, knuckle and station position. camber encodes a hull so that the valid
ones form a convex set, so every blend is a valid hull. Hulls whose control-point
counts differ are first promoted to a common topology (``promote_family``):
plans refitted to the largest count, trim points and section points inserted
where the gaps are widest, and stations matched by position along the hull, with
the missing ones read off each hull's own loft.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as _onp

from hydrosandbox.hull.camber.curves import PlanCurve, _clamp, de_boor, find_span, uniform_knots
from hydrosandbox.hull.camber.document import CamberHull, Station, StationPoint, station_knuckle
from hydrosandbox.hull.camber.sweep import SweptHull


# ----------------------------------------------------------------------------
# Blending
# ----------------------------------------------------------------------------
def blend(
    hulls: Sequence[CamberHull],
    weights: Sequence[float],
    waterline: float | None = None,
    deck_trim: float | None = None,
    name: str = "",
) -> CamberHull:
    """
    camber's ``blendState``: the convex combination of a promoted family (one
    topology, one unit). Weights are normalized to sum to one. The waterline and
    deck trim are not blended; they default to the first hull's.
    """
    if len(hulls) != len(weights):
        raise ValueError("one weight per hull")
    topo = hulls[0].topology
    if any(h.topology != topo or h.unit != hulls[0].unit for h in hulls):
        raise ValueError("blend needs a promoted family (one topology and unit); see promote_family")
    total = float(sum(weights)) or 1.0
    w = [float(x) / total for x in weights]

    def mix(f):
        return sum(wk * f(h) for wk, h in zip(w, hulls))

    first = hulls[0]
    return CamberHull(
        name=name,
        unit=first.unit,
        sheer_plan=[
            (mix(lambda h, i=i: h.sheer_plan[i][0]), mix(lambda h, i=i: h.sheer_plan[i][1]))
            for i in range(len(first.sheer_plan))
        ],
        sheer_trim=[
            tuple(mix(lambda h, i=i, c=c: h.sheer_trim[i][c]) for c in range(3))
            for i in range(len(first.sheer_trim))
        ],
        transom=[
            (mix(lambda h, i=i: h.transom[i][0]), mix(lambda h, i=i: h.transom[i][1]))
            for i in range(2)
        ],
        stations=[
            Station(
                u=mix(lambda h, j=j: h.stations[j].u),
                keel_k=mix(lambda h, j=j: h.stations[j].keel_k),
                points=[
                    StationPoint(
                        n=mix(lambda h, j=j, i=i: h.stations[j].points[i].n),
                        z=mix(lambda h, j=j, i=i: h.stations[j].points[i].z),
                        k=mix(lambda h, j=j, i=i: h.stations[j].points[i].k),
                    )
                    for i in range(len(st0.points))
                ],
            )
            for j, st0 in enumerate(first.stations)
        ],
        waterline=first.waterline if waterline is None else waterline,
        deck_trim=first.deck_trim if deck_trim is None else deck_trim,
    )


def mean_value_weights(point: Sequence[float], vertices: Sequence[Sequence[float]]) -> list[float]:
    """
    Mean-value coordinates of a point inside a convex polygon: how camber's blend
    pad maps a position among three or more hulls to their weights.
    """
    p = _onp.asarray(point, dtype=float)
    V = _onp.asarray(vertices, dtype=float)
    n = len(V)
    d = V - p
    r = _onp.hypot(d[:, 0], d[:, 1])
    for i in range(n):
        if r[i] < 1e-12:
            w = [0.0] * n
            w[i] = 1.0
            return w
    w = _onp.zeros(n)
    for i in range(n):
        a, b, c = d[i - 1], d[i], d[(i + 1) % n]
        ang_prev = math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)
        ang_next = math.atan2(b[0] * c[1] - b[1] * c[0], b @ c)
        w[i] = (math.tan(ang_prev / 2) + math.tan(ang_next / 2)) / r[i]
    return list(w / w.sum())


# ----------------------------------------------------------------------------
# Promotion
# ----------------------------------------------------------------------------
def _basis_funs(span: int, u: float, p: int, U: Sequence[float]) -> list[float]:
    N = [0.0] * (p + 1)
    left = [0.0] * (p + 1)
    right = [0.0] * (p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            t = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * t
            saved = left[j - r] * t
        N[j] = saved
    return N


def _gauss_solve(A, b):
    n = len(A)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        M[i][i] += 1e-9
    for c in range(n):
        pv = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[pv][c]):
                pv = r
        M[c], M[pv] = M[pv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [row[n] / M[i][i] for i, row in enumerate(M)]


def _clamped_bspline_y_of_x(pts):
    """camber's ``clampedBSplineSamplerX``: the plan's ``y`` at ``x`` (36-step bisection of ``x(u)``)."""
    n_cp = len(pts)
    p = min(3, n_cp - 1)
    U = uniform_knots(n_cp, p)
    x0, x1 = pts[0][0], pts[-1][0]

    def y_of(x):
        x = _clamp(x, x0, x1)
        lo, hi, mid = 0.0, 1.0, 0.5
        for _ in range(36):
            mid = 0.5 * (lo + hi)
            if de_boor(pts, U, p, mid)[0] < x:
                lo = mid
            else:
                hi = mid
        return de_boor(pts, U, p, mid)[1]

    return y_of


def _bspline_refit(pts, N: int):
    """Least-squares refit of a plan polygon to ``N`` control points, ends kept (camber's ``bsplineRefit``)."""
    yf = _clamped_bspline_y_of_x(pts)
    x0, x1 = pts[0][0], pts[-1][0]
    S = 200
    Q = [(x0 + (x1 - x0) * i / S, 0.0) for i in range(S + 1)]
    Q = [(x, yf(x)) for x, _ in Q]
    d = [0.0]
    for i in range(1, S + 1):
        d.append(d[-1] + math.hypot(Q[i][0] - Q[i - 1][0], Q[i][1] - Q[i - 1][1]))
    tot = d[S] or 1.0
    t = [v / tot for v in d]
    p = min(3, N - 1)
    U = uniform_knots(N, p)
    n = N - 1
    rows = []
    for k in range(S + 1):
        sp = find_span(n, p, t[k], U)
        bf = _basis_funs(sp, t[k], p, U)
        r = [0.0] * N
        for j in range(p + 1):
            r[sp - p + j] = bf[j]
        rows.append(r)
    free = N - 2
    if free <= 0:
        return [Q[0], Q[S]]
    ATA = [[0.0] * free for _ in range(free)]
    Rx = [0.0] * free
    Ry = [0.0] * free
    for k in range(1, S):
        rx = Q[k][0] - rows[k][0] * Q[0][0] - rows[k][N - 1] * Q[S][0]
        ry = Q[k][1] - rows[k][0] * Q[0][1] - rows[k][N - 1] * Q[S][1]
        for i in range(free):
            Ni = rows[k][i + 1]
            Rx[i] += Ni * rx
            Ry[i] += Ni * ry
            for j in range(free):
                ATA[i][j] += Ni * rows[k][j + 1]
    Px = _gauss_solve([list(r) for r in ATA], Rx)
    Py = _gauss_solve([list(r) for r in ATA], Ry)
    return [Q[0]] + [(Px[i], Py[i]) for i in range(free)] + [Q[S]]


def _promote_plan(h: CamberHull, N: int) -> None:
    if len(h.sheer_plan) >= N:
        return
    before = PlanCurve(h.sheer_plan)
    xs = [before.at(st.u)[0] for st in h.stations]
    fit = _bspline_refit(h.sheer_plan, N)
    h.sheer_plan = [(float(x), float(y)) for x, y in fit]
    after = PlanCurve(h.sheer_plan)
    for st, x in zip(h.stations, xs):
        st.u = after.u_at_x(x)


def _promote_trim(h: CamberHull, T: int) -> None:
    if len(h.sheer_trim) >= T:
        return
    zf = SweptHull(h).trim_z
    pts = list(h.sheer_trim)
    while len(pts) < T:
        bi, bg = 0, -1.0
        for i in range(len(pts) - 1):
            g = pts[i + 1][0] - pts[i][0]
            if g > bg:
                bg, bi = g, i
        xm = 0.5 * (pts[bi][0] + pts[bi + 1][0])
        pts.insert(bi + 1, (xm, zf(xm), 0.0))
    h.sheer_trim = pts


def _promote_section_on_curve(h: CamberHull, S: int) -> None:
    """
    Insert section points ON each station's own section curve, at the middle (in
    the curve's parameter) of the segment with station 0's longest chord, with no
    knuckle: the section passes through its old points and a point it already
    passed through, so its shape barely changes.
    """
    from hydrosandbox.hull.camber.sweep import Section

    while len(h.stations[0].points) < S:
        ref = h.stations[0].points
        bi = max(
            range(len(ref) - 1),
            key=lambda i: math.hypot(ref[i + 1].n - ref[i].n, ref[i + 1].z - ref[i].z),
        )
        for st in h.stations:
            sec = Section(_onp.array([[p.n, p.z] for p in st.points]), [p.k for p in st.points], st.keel_k)
            n, z = sec.at(bi + 0.5)
            st.points.insert(bi + 1, StationPoint(float(n), float(z), 0.0))


def _promote_section(h: CamberHull, S: int) -> None:
    """Insert section points at the midpoint of station 0's longest segment (camber's ``addStationPoint``)."""
    while len(h.stations[0].points) < S:
        ref = h.stations[0].points
        bi, bg = 0, -1.0
        for i in range(len(ref) - 1):
            g = math.hypot(ref[i + 1].n - ref[i].n, ref[i + 1].z - ref[i].z)
            if g > bg:
                bg, bi = g, i
        cn, cz = 0.5 * (ref[bi].n + ref[bi + 1].n), 0.5 * (ref[bi].z + ref[bi + 1].z)
        best, bt, bd = 1, 0.5, math.inf
        for i in range(len(ref) - 1):
            an, az = ref[i].n, ref[i].z
            vn, vz = ref[i + 1].n - an, ref[i + 1].z - az
            l2 = vn * vn + vz * vz or 1.0
            t = _clamp(((cn - an) * vn + (cz - az) * vz) / l2, 0.0, 1.0)
            dist = math.hypot(cn - (an + vn * t), cz - (az + vz * t))
            if dist < bd:
                bd, best, bt = dist, i + 1, t
        for j, st in enumerate(h.stations):
            a, c = st.points[best - 1], st.points[best]
            if j == 0:
                new = StationPoint(cn, _clamp(cz, min(a.z, c.z), max(a.z, c.z)), 0.0)
            else:
                new = StationPoint(a.n + (c.n - a.n) * bt, a.z + (c.z - a.z) * bt, a.k + (c.k - a.k) * bt)
            st.points.insert(best, new)


def _ref_u(c):
    vals = [u for u in c if u is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _monotone_match(a: Sequence[float], b: Sequence[float]) -> list[int]:
    m, n = len(a), len(b)
    if m == 0:
        return []
    if m == 1:
        return [0]
    if m == 2:
        return [0, n - 1]
    ai, bi = list(a[1 : m - 1]), list(b[1 : n - 1])
    p, q = len(ai), len(bi)
    g = [[math.inf] * (q + 1) for _ in range(p + 1)]
    for j in range(q + 1):
        g[p][j] = 0.0
    for i in range(p - 1, -1, -1):
        for j in range(q - 1, -1, -1):
            g[i][j] = min(abs(ai[i] - bi[j]) + g[i + 1][j + 1], g[i][j + 1])
    f = [0]
    i = j = 0
    while i < p:
        if j >= q:
            f.append(min(j + 1, n - 2))
            i += 1
            j += 1
            continue
        if abs(ai[i] - bi[j]) + g[i + 1][j + 1] <= g[i][j + 1]:
            f.append(j + 1)
            i += 1
            j += 1
        else:
            j += 1
    f.append(n - 1)
    return f


def auto_correspondence(hulls: Sequence[CamberHull]) -> list[list[float | None]]:
    """Which station of each hull is the same place on the boat (camber's ``autoCorrespondence``)."""
    H = len(hulls)
    if H == 0:
        return []

    def new_corr(h, u):
        c = [None] * H
        c[h] = u
        return c

    merged = [new_corr(0, st.u) for st in hulls[0].stations]
    for h in range(1, H):
        us = [st.u for st in hulls[h].stations]
        refs = [_ref_u(c) for c in merged]
        nxt = []
        if len(us) <= len(refs):
            f = _monotone_match(us, refs)
            taken = {j: i for i, j in enumerate(f)}
            for j, c in enumerate(merged):
                if j in taken:
                    c[h] = us[taken[j]]
                nxt.append(c)
        else:
            f = _monotone_match(refs, us)
            taken_u = set(f)
            i = 0
            for j, c in enumerate(merged):
                while i < f[j]:
                    if i not in taken_u:
                        nxt.append(new_corr(h, us[i]))
                    i += 1
                c[h] = us[f[j]]
                nxt.append(c)
                i = f[j] + 1
            while i < len(us):
                if i not in taken_u:
                    nxt.append(new_corr(h, us[i]))
                i += 1
        merged = nxt
    return merged


def _remap_u(corr, j: int, h: int) -> float:
    ref = _ref_u(corr[j])
    lo = next((k for k in range(j - 1, -1, -1) if corr[k][h] is not None), -1)
    hi = next((k for k in range(j + 1, len(corr)) if corr[k][h] is not None), -1)
    if lo < 0 and hi < 0:
        return ref
    if lo < 0:
        return min(ref, corr[hi][h])
    if hi < 0:
        return max(ref, corr[lo][h])
    a, b = _ref_u(corr[lo]), _ref_u(corr[hi])
    t = (ref - a) / (b - a) if b > a else 0.5
    return corr[lo][h] + (corr[hi][h] - corr[lo][h]) * t


def _promote_stations(hulls: Sequence[CamberHull], corr) -> None:
    for h, hull in enumerate(hulls):
        loft = SweptHull(hull).loft
        out = []
        for j, c in enumerate(corr):
            if c[h] is not None:
                found = next((st for st in hull.stations if st.u == c[h]), None)
                if found is not None:
                    out.append(found)
                    continue
            u = c[h] if c[h] is not None else _remap_u(corr, j, h)
            pts, ks, keel_k = loft.at(u)
            out.append(
                Station(
                    u=u,
                    keel_k=keel_k,
                    points=[
                        StationPoint(float(p[0]), float(p[1]), station_knuckle(len(pts), i, ks[i]))
                        for i, p in enumerate(pts)
                    ],
                )
            )
        for j in range(1, len(out)):
            if out[j].u - out[j - 1].u < 1e-4:
                out[j].u = out[j - 1].u + 1e-4
        over = out[-1].u - 1 if out else 0.0
        if over > 0:
            for st in out:
                st.u = max(0.0, st.u - over)
        hull.stations = out


def promote_family(
    hulls: Sequence[CamberHull],
    correspondence: list[list[float | None]] | None = None,
    section_points: str = "camber",
) -> list[CamberHull]:
    """
    camber's ``promoteFamily``: copies of ``hulls`` lifted to one topology and the
    first hull's unit, so that they blend.

    ``section_points`` chooses how points are added to sections with fewer than the
    family's most. ``"camber"`` is camber's rule: on the chord between two points,
    with knuckles interpolated from them (so next to an end point, a crease of 0.5
    or more). That changes the section's shape, noticeably where it is curved: it
    took 13% of the displacement off a Cal 20 being promoted from 4 to 8 points.
    ``"curve"`` puts them on each section's own curve instead, uncreased.
    """
    out = [h.converted(hulls[0].unit) for h in hulls]
    if len(out) < 2:
        return out
    N = max(len(h.sheer_plan) for h in out)
    T = max(len(h.sheer_trim) for h in out)
    S = max(len(h.stations[0].points) for h in out)
    for h in out:
        _promote_plan(h, N)
        _promote_trim(h, T)
        if section_points == "curve":
            _promote_section_on_curve(h, S)
        elif section_points == "camber":
            _promote_section(h, S)
        else:
            raise ValueError(f"section_points must be 'camber' or 'curve', not {section_points!r}")
    corr = correspondence if correspondence is not None else auto_correspondence(out)
    _promote_stations(out, corr)
    return out
