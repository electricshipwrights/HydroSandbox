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

    def hull(self, values: Mapping[str, float] | None = None, name: str | None = None):
        """The deformed hull, cut back into an offset table over its own extent."""
        from hydrosandbox.hull.hull import Hull

        h = self.base
        X, Y, Z = self.deform(values)
        x_base = _onp.asarray(h.stations, dtype=float)
        z_base = _onp.asarray(h.levels, dtype=float)
        on = self._on_hull

        # Extent: bow to the transom's plane, keel to deck.
        x_bow = float(X[on].min())
        transom = h.half_breadths[-1] > 0.0
        wet = transom & (z_base < 0.0)
        pick = wet if wet.any() else (transom if transom.any() else _onp.ones_like(transom))
        x_end = float(X[-1, pick].mean())
        z_keel, z_deck = float(Z[on].min()), float(Z[on].max())

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
        return Hull(
            stations, levels, out,
            name=name or f"{h.name} FFD",
            form_factor=h.form_factor,
            roughness_delta_cf=h.roughness_delta_cf,
            transom_closure=h.transom_closure,
            waterline_smoothing=h.waterline_smoothing,
        )


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

    ``exact(values)`` and the nodes are ``lattice.hull(centre | values)``.
    """

    def __init__(self, lattice: FFDLattice, names: Sequence[str], half_width: float | Mapping[str, float] = 0.05,
                 centre: Mapping[str, float] | None = None, design: str = "linear", name: str | None = None):
        self.lattice = lattice
        self.centre = dict(centre or {})
        lattice.moves({**self.centre, **{n: 0.0 for n in names}})  # validates the names
        hw = half_width if isinstance(half_width, Mapping) else {n: float(half_width) for n in names}
        ranges = {n: (self.centre.get(n, 0.0) - hw[n], self.centre.get(n, 0.0) + hw[n]) for n in names}
        start = {n: self.centre.get(n, 0.0) for n in names}
        super().__init__(
            lambda values, node_name: lattice.hull({**self.centre, **values}, name=node_name),
            ranges, design, name or f"{lattice.base.name} FFD", start,
        )

    def moves(self, values: Mapping[str, float]) -> dict[str, float]:
        """All the lattice's moves at ``values`` of this family's variables."""
        return {**self.centre, **{k: float(v) for k, v in values.items()}}

    def recentred(self, values: Mapping[str, float], half_width: float | Mapping[str, float] | None = None,
                  design: str | None = None) -> "FFDFamily":
        """The same family centred on ``values`` (a trust-region step)."""
        hw = half_width if half_width is not None else {n: 0.5 * (hi - lo) for n, (lo, hi) in self.ranges.items()}
        return FFDFamily(self.lattice, self.names, hw, self.moves(values), design or self.design, self.name)
