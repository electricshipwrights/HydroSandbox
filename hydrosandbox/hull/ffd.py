"""
Free-form deformation (Sederberg & Parry 1986) of a hull's offsets, as a design space.

``FFDLattice`` embeds a ``Hull``'s offset surface (the points ``(x, f, z)`` of its
table) in a box of control points: ``nx`` along the length, two across (the
centreplane and outboard, so the demihull stays symmetric) and ``nz`` up. A
point at box coordinates ``(s, t, u)`` moves by the trivariate Bernstein
combination of the control points' moves,

    dP(s, t, u) = sum_ijk B_i(s) B_j(t) B_k(u) dP_ijk,

which is zero when nothing moves (the lattice reproduces the hull) and linear
in the moves. Centreplane points move only fore-aft and up-down.

``FFDLattice.hull(moves)`` deforms the surface and cuts it back into an offset
table over its own extent (bow to transom, keel to deck), with the base table's
station and level spacing: each target level is found on every deformed station
column, and the waterline-like curve this makes is read at the target stations.
The transom, the last column, is projected onto the plane of its submerged
part's mean ``x`` (``Hull`` takes the last station as a vertical transom).

``FFDFamily`` makes the moves of chosen control points design variables of an
optimization (a ``DesignFamily``): exact deformed hulls at the nodes of a design
around a centre, interpolated between. A ``"linear"`` design (n + 1 hulls) is
exact for everything linear in the half-breadths when only sideways moves vary
(the wave amplitude spectrum, volume and moments at fixed attitude; not ratios
such as the centre of buoyancy, nor products such as the dynamic force); re-centre
it as a solve moves (a trust region) for the rest.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

import numpy as _onp

from hydrosandbox.hull.transformed import DesignFamily

AXES = "xyz"
_NAME = re.compile(r"^p(\d+)([io])(\d+)([xyz])$")


def _bernstein(n: int, t: _onp.ndarray) -> _onp.ndarray:
    """``(len(t), n + 1)`` Bernstein basis of degree ``n``."""
    t = _onp.asarray(t, dtype=float)[:, None]
    i = _onp.arange(n + 1)[None, :]
    c = _onp.array([math.comb(n, k) for k in range(n + 1)])[None, :]
    return c * t**i * (1.0 - t) ** (n - i)


class FFDLattice:
    """
    A Bernstein FFD lattice around ``hull`` (see the module docstring).

    Parameters
    ----------
    hull : The base ``Hull``.
    nx, nz : Control points along the length and up (two across).
    margin : Clearance of the box around the hull [fraction of length fore and
        aft; metres below the keel and above the deck]; the box spans
        ``1.2`` times the largest half-breadth across.

    Control-point moves are named ``p<i><side><k><axis>``: ``i`` along (0 at the
    bow), ``side`` ``i`` (centreplane) or ``o`` (outboard), ``k`` up (0 at the
    bottom), ``axis`` ``x`` (aft), ``y`` (outboard) or ``z`` (up), in metres.
    """

    def __init__(self, hull, nx: int = 4, nz: int = 3, margin: Sequence[float] = (0.04, 0.06, 0.05)):
        if nx < 2 or nz < 2:
            raise ValueError("a lattice needs at least two control points along and up")
        self.base = hull
        self.nx, self.nz = nx, nz
        x = _onp.asarray(hull.stations, dtype=float)
        z = _onp.asarray(hull.levels, dtype=float)
        f = _onp.asarray(hull.half_breadths, dtype=float)
        length = x[-1] - x[0]
        self.box = dict(
            x0=x[0] - margin[0] * length, x1=x[-1] + margin[0] * length,
            y1=1.2 * float(f.max()), z0=z[0] - margin[1], z1=z[-1] + margin[2],
        )
        b = self.box
        self._bx = _bernstein(nx - 1, (x - b["x0"]) / (b["x1"] - b["x0"]))  # (NI, nx)
        self._bz = _bernstein(nz - 1, (z - b["z0"]) / (b["z1"] - b["z0"]))  # (NJ, nz)
        self._t = f / b["y1"]  # (NI, NJ)
        # Points on the hull: its offsets, dilated by one so the keel and stem
        # (zero half-breadth, but part of the outline) count.
        on = f > 0.0
        grown = on.copy()
        grown[1:] |= on[:-1]; grown[:-1] |= on[1:]
        grown[:, 1:] |= on[:, :-1]; grown[:, :-1] |= on[:, 1:]
        self._on_hull = grown

    def __repr__(self) -> str:
        return f"FFDLattice({self.nx} x 2 x {self.nz} around '{self.base.name}')"

    # ------------------------------------------------------------------
    def names(self, axes: str = "xyz", sides: str = "io", along: Sequence[int] | None = None,
              up: Sequence[int] | None = None) -> list[str]:
        """Names of the control-point moves, filtered (centreplane sideways moves never appear)."""
        out = []
        for i in along if along is not None else range(self.nx):
            for side in sides:
                for k in up if up is not None else range(self.nz):
                    for a in axes:
                        if side == "i" and a == "y":
                            continue
                        out.append(f"p{i}{side}{k}{a}")
        return out

    def moves(self, values: Mapping[str, float] | None = None) -> _onp.ndarray:
        """The ``(nx, 2, nz, 3)`` array of control-point moves from named ``values``."""
        d = _onp.zeros((self.nx, 2, self.nz, 3))
        for name, v in (values or {}).items():
            m = _NAME.match(name)
            if not m:
                raise ValueError(f"bad control-point move name {name!r}")
            i, side, k, axis = int(m[1]), m[2], int(m[3]), m[4]
            if i >= self.nx or k >= self.nz:
                raise ValueError(f"{name!r} is outside the {self.nx} x 2 x {self.nz} lattice")
            if side == "i" and axis == "y":
                raise ValueError("centreplane control points can't move sideways")
            d[i, 0 if side == "i" else 1, k, AXES.index(axis)] = float(v)
        return d

    def control_points(self, values: Mapping[str, float] | None = None) -> _onp.ndarray:
        """``(nx, 2, nz, 3)`` control-point positions after ``values``' moves."""
        b = self.box
        p = _onp.zeros((self.nx, 2, self.nz, 3))
        p[..., 0] = _onp.linspace(b["x0"], b["x1"], self.nx)[:, None, None]
        p[:, 1, :, 1] = b["y1"]
        p[..., 2] = _onp.linspace(b["z0"], b["z1"], self.nz)[None, None, :]
        return p + self.moves(values)

    def deform(self, values: Mapping[str, float] | None = None):
        """The deformed offset surface: ``(x, y, z)`` arrays shaped like the base half-breadths."""
        d = self.moves(values)
        h = self.base
        x0 = _onp.asarray(h.stations, dtype=float)[:, None]
        z0 = _onp.asarray(h.levels, dtype=float)[None, :]
        f0 = _onp.asarray(h.half_breadths, dtype=float)
        t = self._t

        def field(side, axis):
            return self._bx @ d[:, side, :, axis] @ self._bz.T

        dx = (1.0 - t) * field(0, 0) + t * field(1, 0)
        dy = t * field(1, 1)
        dz = (1.0 - t) * field(0, 2) + t * field(1, 2)
        return x0 + dx, _onp.maximum(f0 + dy, 0.0), z0 + dz

    def surface_area(self, values: Mapping[str, float] | None = None, waterline: float = 0.0) -> float:
        """
        Wetted area [m^2, both sides] of the deformed offset surface itself, below
        ``waterline``: its cells' quads in 3D, each by the fraction of its height
        below the water.
        """
        X, Y, Z = self.deform(values)
        P = _onp.stack([X, Y, Z], axis=-1)
        a, b, c, d = P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]
        tri = lambda p, q, r: 0.5 * _onp.linalg.norm(_onp.cross(q - p, r - p), axis=-1)
        area = tri(a, b, c) + tri(a, c, d)
        f = _onp.asarray(self.base.half_breadths)
        on = (f[:-1, :-1] > 0) | (f[1:, :-1] > 0) | (f[:-1, 1:] > 0) | (f[1:, 1:] > 0)
        zs = _onp.stack([a[..., 2], b[..., 2], c[..., 2], d[..., 2]])
        lo, hi = zs.min(axis=0), zs.max(axis=0)
        wet = _onp.clip((waterline - lo) / _onp.fmax(hi - lo, 1e-12), 0.0, 1.0)
        return float(2.0 * _onp.sum(area * wet * on))

    def extent(self, values: Mapping[str, float] | None = None, surface=None) -> dict[str, float]:
        """The deformed hull's bow ``x``, transom plane ``x``, keel ``z`` and deck ``z``."""
        h = self.base
        X, _, Z = surface if surface is not None else self.deform(values)
        on = self._on_hull
        z_base = _onp.asarray(h.levels, dtype=float)
        transom = h.half_breadths[-1] > 0.0
        wet = transom & (z_base < 0.0)
        pick = wet if wet.any() else (transom if transom.any() else _onp.ones_like(transom))
        return dict(x_bow=float(X[on].min()), x_end=float(X[-1, pick].mean()),
                    z_keel=float(Z[on].min()), z_deck=float(Z[on].max()))

    def hull(self, values: Mapping[str, float] | None = None, name: str | None = None,
             extent: Mapping[str, float] | None = None):
        """
        The deformed hull, cut back into an offset table over its own extent, or
        over a given ``extent`` (``extent()``'s keys: hulls cut over the same one
        share their grid, and so their table kernels). Its own ``draft`` and
        ``freeboard`` (keel and deck relative to the design waterline) are kept
        as attributes either way.
        """
        from hydrosandbox.hull.hull import Hull

        h = self.base
        X, Y, Z = self.deform(values)
        x_base = _onp.asarray(h.stations, dtype=float)
        z_base = _onp.asarray(h.levels, dtype=float)
        own = self.extent(surface=(X, Y, Z))
        e = dict(own, **(extent or {}))
        x_bow, x_end, z_keel, z_deck = e["x_bow"], e["x_end"], e["z_keel"], e["z_deck"]

        # The base spacing, mapped onto the new extent (below and above the waterline
        # separately, so a level at z = 0 stays there).
        stations = x_bow + (x_base - x_base[0]) * (x_end - x_bow) / (x_base[-1] - x_base[0])
        lo, hi = z_base[0], z_base[-1]
        levels = _onp.where(
            z_base < 0.0,
            z_base * (z_keel / lo) if lo < 0.0 else z_base,
            z_base * (z_deck / hi) if hi > 0.0 else z_base,
        )
        if _onp.any(_onp.diff(stations) <= 0) or _onp.any(_onp.diff(levels) <= 0):
            raise ValueError("the deformation leaves no extent to cut the hull over")

        n_x, n_z = X.shape
        out = _onp.zeros((len(stations), len(levels)))
        for b, z in enumerate(levels):
            xs = _onp.empty(n_x)
            ys = _onp.empty(n_x)
            for a in range(n_x):
                zc = Z[a]
                if _onp.any(_onp.diff(zc) <= 0.0):
                    raise ValueError("the lattice folds a station over itself")
                xs[a] = _onp.interp(z, zc, X[a])
                ys[a] = _onp.interp(z, zc, Y[a], left=0.0, right=0.0)
            if _onp.any(_onp.diff(xs) <= 0.0):
                raise ValueError("the lattice folds the hull fore and aft")
            row = _onp.interp(stations, xs, ys, left=0.0, right=ys[-1])
            row[-1] = ys[-1]  # the transom, projected onto its plane
            out[:, b] = row
        result = Hull(
            stations, levels, out,
            name=name or f"{h.name} FFD",
            form_factor=h.form_factor,
            roughness_delta_cf=h.roughness_delta_cf,
            transom_closure=h.transom_closure,
            waterline_smoothing=h.waterline_smoothing,
        )
        # The table's outline (keel, stem) no longer lies on grid lines, which adds
        # shell along it: carry the deformed surface's own change of wetted area
        # instead (relative to the base, whose table area stands).
        if values:
            if not hasattr(self, "_area0"):
                self._area0 = (self.surface_area({}), float(_onp.ravel(h.hydrostatics()["wetted_surface"])[0]))
            direct0, table0 = self._area0
            table = float(_onp.ravel(Hull.hydrostatics(result)["wetted_surface"])[0])
            result.wetted_surface_scale = h.wetted_surface_scale * (self.surface_area(values) / direct0) / (table / table0)
        else:
            result.wetted_surface_scale = h.wetted_surface_scale
        result.draft = -own["z_keel"]
        result.freeboard = own["z_deck"]
        return result


def _fold_rows(lattice: FFDLattice, centre: Mapping[str, float], names: Sequence[str], margin: float):
    """
    Linear no-fold conditions ``G d + c >= 0`` on the moves ``d`` (from ``centre``)
    of ``names``: each gap up a station column and along a level, on the hull, at
    least ``margin`` of the base table's.
    """
    x0, _, z0 = lattice.deform(centre)
    grads = []
    for n in names:
        xs, _, zs = lattice.deform({**centre, n: centre.get(n, 0.0) + 1.0})
        grads.append((xs - x0, zs - z0))
    on = lattice._on_hull
    zbase = _onp.asarray(lattice.base.levels, dtype=float)
    xbase = _onp.asarray(lattice.base.stations, dtype=float)
    col = on[:, 1:] | on[:, :-1]
    row = on[1:, :] | on[:-1, :]
    g_col = _onp.stack([(gz[:, 1:] - gz[:, :-1])[col] for _, gz in grads], axis=1)
    g_row = _onp.stack([(gx[1:, :] - gx[:-1, :])[row] for gx, _ in grads], axis=1)
    c_col = (z0[:, 1:] - z0[:, :-1])[col] - margin * _onp.diff(zbase)[None, :].repeat(len(xbase), 0)[col]
    c_row = (x0[1:, :] - x0[:-1, :])[row] - margin * _onp.diff(xbase)[:, None].repeat(len(zbase), 1)[row]
    return _onp.vstack([g_col, g_row]), _onp.concatenate([c_col, c_row])


class FFDFamily(DesignFamily):
    """
    Moves of an ``FFDLattice``'s control points as design variables.

    Parameters
    ----------
    lattice : The ``FFDLattice``.
    names : The moves to vary (``lattice.names(...)``).
    half_width : Each move's range around ``centre`` [m]: a number, or a dict by name.
    centre : Moves held by every hull of the family, and the centre of each
        varied move's range (default: none).
    design : ``"linear"`` (default; ``n + 1`` hulls), ``"diagonal"`` (``2n + 1``)
        or ``"quadratic"``.
    name : The family's name.
    limit : Bound on every move [m], whatever the centre (ranges are clipped to it).
    shared_grid : Cut every node over one common extent (default), so their
        attitude tables share kernels and build together much faster; ``exact``
        still cuts over the hull's own extent.

    ``exact(values)`` and the nodes are ``lattice.hull(centre | values)``.
    """

    def __init__(self, lattice: FFDLattice, names: Sequence[str], half_width: float | Mapping[str, float] = 0.05,
                 centre: Mapping[str, float] | None = None, design: str = "linear", name: str | None = None,
                 limit: float | None = None, shared_grid: bool = True):
        self.lattice = lattice
        self.limit = limit
        self.centre = dict(centre or {})
        lattice.moves({**self.centre, **{n: 0.0 for n in names}})  # validates the names
        hw = dict(half_width) if isinstance(half_width, Mapping) else {n: float(half_width) for n in names}
        # Shrink any move whose range would fold the hull at either end, so every
        # node can be built (a centre at the edge of what offsets can represent).
        G, c = _fold_rows(lattice, self.centre, list(names), 0.1)
        for i, n in enumerate(names):
            g = G[:, i]
            up = c[g < 0] / -g[g < 0]  # largest step up each row allows
            down = c[g > 0] / g[g > 0]  # ... and down
            room = min(up.min() if up.size else _onp.inf, down.min() if down.size else _onp.inf)
            hw[n] = max(min(hw[n], 0.9 * max(room, 0.0)), 1e-4)
        ranges = {}
        for n in names:
            c = self.centre.get(n, 0.0)
            lo, hi = c - hw[n], c + hw[n]
            if limit is not None:
                lo, hi = max(lo, -limit), min(hi, limit)
                if hi - lo < 1e-6:  # at the limit: keep a sliver of range, inside it
                    lo, hi = (hi - 1e-6, hi) if c > 0 else (lo, lo + 1e-6)
            ranges[n] = (lo, hi)
        start = {n: self.centre.get(n, 0.0) for n in names}
        self.shared_grid = shared_grid
        self._extent = None

        def make_hull(values, node_name):
            if not shared_grid:
                return lattice.hull({**self.centre, **values}, name=node_name)
            if self._extent is None:
                # One grid for every node: the family's furthest bow, deepest keel
                # and highest deck, and the centre's transom plane.
                each = [lattice.extent({**self.centre, **p}) for p in self.node_parameters]
                self._extent = dict(
                    x_bow=min(e["x_bow"] for e in each), z_keel=min(e["z_keel"] for e in each),
                    z_deck=max(e["z_deck"] for e in each), x_end=lattice.extent(self.centre)["x_end"],
                )
            return lattice.hull({**self.centre, **values}, name=node_name, extent=self._extent)

        super().__init__(make_hull, ranges, design, name or f"{lattice.base.name} FFD", start)

    def no_fold(self, values, margin: float = 0.25) -> list:
        """
        Constraints (on ``values``, numbers or CasADi expressions, in ``names``
        order) that keep the deformed hull representable as offsets: along every
        station column the levels stay in order, and along every level the
        stations do, each gap at least ``margin`` of the base table's. The surface
        is linear in the moves, so these are linear.
        """
        import casadi as cas

        lat = self.lattice
        if not hasattr(self, "_fold_rows"):
            self._fold_rows = _fold_rows(lat, self.centre, self.names, margin)
        G, c = self._fold_rows
        values = self._values(values)
        d = cas.vertcat(*[v - self.centre.get(n, 0.0) for n, v in zip(self.names, values)])
        if not any(isinstance(v, (cas.MX, cas.SX)) for v in values):
            return list(_onp.asarray(G @ _onp.array([float(v) - self.centre.get(n, 0.0) for n, v in zip(self.names, values)]) + c) >= 0.0)
        return [cas.mtimes(cas.DM(G), d) + cas.DM(c) >= 0]

    def transom_depth(self, values):
        """
        Equivalent-rectangle depth [m] of the immersed transom at rest, as
        ``Hull.transom`` computes it, but straight from the deformed transom
        section, which is linear in the moves: exact (not interpolated between the
        nodes) and smooth, for numbers or CasADi expressions in ``names`` order.
        """
        import aerosandbox.numpy as np
        from hydrosandbox.library.free_surface import smooth_submergence_fraction

        lat = self.lattice
        if not hasattr(self, "_transom_map"):
            _, y0, z0 = lat.deform(self.centre)
            jy, jz = [], []
            for n in self.names:
                _, ys, zs = lat.deform({**self.centre, n: self.centre.get(n, 0.0) + 1.0})
                jy.append(ys[-1] - y0[-1])
                jz.append(zs[-1] - z0[-1])
            self._transom_map = (y0[-1], _onp.array(jy).T, z0[-1], _onp.array(jz).T)
        y0, jy, z0, jz = self._transom_map
        values = self._values(values)
        d = [v - self.centre.get(n, 0.0) for n, v in zip(self.names, values)]
        if any(np.is_casadi_type(v, recursive=False) for v in d):
            import casadi as cas

            dv = cas.vertcat(*d)
            y = cas.DM(y0) + cas.mtimes(cas.DM(jy), dv)
            z = cas.DM(z0) + cas.mtimes(cas.DM(jz), dv)
        else:
            dv = _onp.array([float(v) for v in d])
            y, z = y0 + jy @ dv, z0 + jz @ dv
        f_band = 0.5 * (y[:-1] + y[1:])
        z_lo, z_hi = z[:-1], z[1:]
        f_sub, _, _ = smooth_submergence_fraction(z_lo, z_hi, 0.0, transition_thickness=lat.base.waterline_smoothing)
        area = np.sum(2.0 * f_band * (z_hi - z_lo) * f_sub)
        softness = 0.02 * max(float(_onp.max(y0)), 1e-6)
        peak = softness * np.log(np.sum(np.exp(f_band * f_sub / softness)))
        return area / (2.0 * np.fmax(peak, 1e-9))

    def exact(self, values, **kwargs):
        """The deformed hull at ``values``, cut over its own extent."""
        values = self._values(values)
        return self.lattice.hull(self.moves(dict(zip(self.names, values))),
                                 name=kwargs.pop("name", f"{self.name} exact"))

    def moves(self, values: Mapping[str, float]) -> dict[str, float]:
        """All the lattice's moves at ``values`` of this family's variables."""
        return {**self.centre, **{k: float(v) for k, v in values.items()}}

    def recentred(self, values: Mapping[str, float], half_width: float | Mapping[str, float] | None = None,
                  design: str | None = None) -> "FFDFamily":
        """The same family centred on ``values`` (a trust-region step)."""
        hw = half_width if half_width is not None else {n: 0.5 * (hi - lo) for n, (lo, hi) in self.ranges.items()}
        return FFDFamily(self.lattice, self.names, hw, self.moves(values), design or self.design, self.name,
                         self.limit, self.shared_grid)
