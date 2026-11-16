"""
One hull's offsets under a few smooth transformations, as a design space.

``transform_hull`` maps any ``Hull``'s offset table (stations, levels,
half-breadths) to a transformed one:

- ``beam``: half-breadths scaled by this factor.
- ``draft``: levels below the design waterline (``z < 0``) scaled by this factor;
  the topsides are unchanged.
- ``length``: stations scaled about the bow by this factor.
- ``cp``, ``lcb``: Lackenby's transformation. Sections are moved fore and aft (the
  bow, the transom and the largest section stay put) by a parabolic shift in each
  half, sized so that the prismatic coefficient of the sectional-area curve below
  the waterline changes by ``cp`` and its centroid moves by ``lcb`` (a fraction
  of the length, positive aft).
- ``bow_rake``: the forebody sheared in ``x`` in proportion to height above the
  waterline: a point at height ``z`` moves forward by ``bow_rake * z * w(x)``,
  with ``w`` going smoothly from 1 at the bow to 0 at the largest section. So
  positive values rake the stem forward above water and cut the forefoot away
  below; the waterline and the afterbody (and transom) are unchanged.

``TransformedFamily`` makes these numbers design variables the way ``HullFamily``
makes blend weights ones: the transformed hull is computed exactly at the nodes
of a design in the chosen parameters, and anything a ``Hull`` computes is the
quadratic (or linear) interpolant of the nodes' values (see
``hydrosandbox.hull.camber.blended``). The design is the minimal one for a full
quadratic in ``n`` parameters, ``(n + 1)(n + 2) / 2`` nodes: the centre of the
box of ranges, a step to each end of each range, and one corner per pair.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as _onp

#: Each transformation's parameter and its neutral (identity) value.
NEUTRAL = {"beam": 1.0, "draft": 1.0, "length": 1.0, "cp": 0.0, "lcb": 0.0, "bow_rake": 0.0}


def _section_areas(stations, levels, half_breadths) -> _onp.ndarray:
    """Area of each station's section below ``z = 0`` (both sides) [m^2]."""
    z = _onp.asarray(levels, dtype=float)
    out = []
    for f in _onp.asarray(half_breadths, dtype=float):
        if z[0] >= 0.0:
            out.append(0.0)
            continue
        zs = _onp.concatenate([z[z < 0.0], [0.0]])
        fs = _onp.interp(zs, z, f)
        out.append(2.0 * _onp.trapezoid(fs, zs))
    return _onp.array(out)


def area_curve_coefficients(stations, levels, half_breadths) -> dict[str, float]:
    """Prismatic coefficient and centroid (fraction of length from the bow) of the sectional-area curve below the waterline."""
    x = _onp.asarray(stations, dtype=float)
    area = _section_areas(x, levels, half_breadths)
    volume = _onp.trapezoid(area, x)
    length = x[-1] - x[0]
    return {
        "cp": volume / (area.max() * length),
        "lcb": (_onp.trapezoid(area * x, x) / volume - x[0]) / length,
        "volume": volume,
    }


def _lackenby_shift(x: _onp.ndarray, x_mid: float, a_fore: float, a_aft: float) -> _onp.ndarray:
    """``dx`` of each station: ``a`` times a parabola in each half, zero at the ends and at ``x_mid``."""
    x_bow, x_end = x[0], x[-1]
    dx = _onp.zeros_like(x)
    fore = x < x_mid
    xi = (x_mid - x[fore]) / (x_mid - x_bow)
    dx[fore] = -a_fore * (x_mid - x_bow) * xi * (1.0 - xi)
    aft = x > x_mid
    xi = (x[aft] - x_mid) / (x_end - x_mid)
    dx[aft] = a_aft * (x_end - x_mid) * xi * (1.0 - xi)
    return dx


def lackenby(stations, levels, half_breadths, cp: float = 0.0, lcb: float = 0.0,
             iterations: int = 8, tolerance: float = 1e-9) -> _onp.ndarray:
    """
    Stations moved by Lackenby's transformation so that the sectional-area curve's
    prismatic coefficient changes by ``cp`` and its centroid by ``lcb`` (fraction of
    the length, positive aft). Positive shift amplitudes fill the ends (more ``cp``);
    the fore and aft amplitudes are found by Newton's method.
    """
    x = _onp.asarray(stations, dtype=float)
    if cp == 0.0 and lcb == 0.0:
        return x.copy()
    area = _section_areas(x, levels, half_breadths)
    i_mid = int(_onp.argmax(area))
    if i_mid in (0, len(x) - 1):
        raise ValueError("Lackenby's transformation needs the largest section between the ends")
    x_mid = x[i_mid]
    base = area_curve_coefficients(x, levels, half_breadths)
    target = _onp.array([base["cp"] + cp, base["lcb"] + lcb])

    def coefficients(a):
        c = area_curve_coefficients(x + _lackenby_shift(x, x_mid, *a), levels, half_breadths)
        return _onp.array([c["cp"], c["lcb"]])

    a = _onp.zeros(2)
    for _ in range(iterations):
        r = coefficients(a) - target
        if _onp.max(_onp.abs(r)) < tolerance:
            break
        h = 1e-6
        jac = _onp.column_stack([(coefficients(a + h * e) - coefficients(a)) / h for e in _onp.eye(2)])
        a = a - _onp.linalg.solve(jac, r)
    if _onp.max(_onp.abs(coefficients(a) - target)) > 1e-6 or _onp.max(_onp.abs(a)) >= 0.95:
        raise ValueError(
            f"Lackenby's transformation can't reach cp {cp:+.3f}, lcb {lcb:+.3f} "
            f"(shift amplitudes {a[0]:+.2f}, {a[1]:+.2f}; stations must stay in order)"
        )
    return x + _lackenby_shift(x, x_mid, *a)


def _smoothstep(t):
    t = _onp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def raked(stations, levels, half_breadths, bow_rake: float):
    """Offsets with the forebody sheared by ``bow_rake`` (see the module docstring); returns new stations and half-breadths."""
    x = _onp.asarray(stations, dtype=float)
    z = _onp.asarray(levels, dtype=float)
    f = _onp.asarray(half_breadths, dtype=float)
    if bow_rake == 0.0:
        return x.copy(), f.copy()
    i_mid = int(_onp.argmax(_section_areas(x, z, f)))
    x_bow, x_mid, x_end = x[0], x[max(i_mid, 1)], x[-1]
    w = 1.0 - _smoothstep((x - x_bow) / (x_mid - x_bow))
    # Where each original station's point at each level ends up.
    moved = x[:, None] - bow_rake * z[None, :] * w[:, None]
    if _onp.any(_onp.diff(moved, axis=0) <= 0.0):
        raise ValueError(f"bow_rake {bow_rake} folds the forebody over itself")
    new_bow = min(x_bow, float(moved[0].min()))
    # Same station spacing, stretched to the new extent.
    x_new = new_bow + (x - x_bow) * (x_end - new_bow) / (x_end - x_bow)
    f_new = _onp.column_stack(
        [_onp.interp(x_new, moved[:, j], f[:, j], left=0.0, right=f[-1, j]) for j in range(len(z))]
    )
    return x_new, f_new


def transform_hull(hull, name: str | None = None, **parameters):
    """
    ``hull`` (any ``Hull``) with its offsets transformed (see the module docstring
    for each parameter; omitted ones are neutral). Its viscous and closure settings
    and waterline smoothing carry over.
    """
    from hydrosandbox.hull.hull import Hull

    unknown = set(parameters) - set(NEUTRAL)
    if unknown:
        raise ValueError(f"unknown transformation(s) {sorted(unknown)}; expected some of {list(NEUTRAL)}")
    p = {**NEUTRAL, **{k: float(v) for k, v in parameters.items()}}
    x = _onp.array(hull.stations, dtype=float)
    z = _onp.array(hull.levels, dtype=float)
    f = _onp.array(hull.half_breadths, dtype=float)

    x, f = raked(x, z, f, p["bow_rake"])
    x = lackenby(x, z, f, p["cp"], p["lcb"])
    f = f * p["beam"]
    z = _onp.where(z < 0.0, z * p["draft"], z)
    x = x[0] + (x - x[0]) * p["length"]

    return Hull(
        x, z, f,
        name=name or f"{hull.name} transformed",
        form_factor=hull.form_factor,
        roughness_delta_cf=hull.roughness_delta_cf,
        transom_closure=hull.transom_closure,
        waterline_smoothing=hull.waterline_smoothing,
    )


def box_design(n: int, degree: int = 2) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]]]:
    """
    Nodes (in ``[-1, 1]^n``) and monomial exponents of the minimal interpolation
    design for a full polynomial of total ``degree`` 1 or 2: the centre, a step to
    the upper end of each axis (and, for degree 2, the lower end and one corner per
    pair of axes).
    """
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    eye = _onp.eye(n, dtype=int)
    nodes = [tuple([0.0] * n)] + [tuple(map(float, e)) for e in eye]
    monomials = [tuple([0] * n)] + [tuple(e) for e in eye]
    if degree == 2:
        nodes += [tuple(map(float, -e)) for e in eye]
        monomials += [tuple(2 * e) for e in eye]
        for i, j in itertools.combinations(range(n), 2):
            nodes.append(tuple(map(float, eye[i] + eye[j])))
            monomials.append(tuple(eye[i] + eye[j]))
    return nodes, monomials


class TransformedFamily:
    """
    Transformations of one hull (see the module docstring) as design variables.

    Parameters
    ----------
    hull : The base ``Hull`` (any: from camber, a mesh, offsets, ...).
    ranges : ``{parameter: (low, high)}`` for each transformation to vary, from
        ``NEUTRAL``'s keys. The others stay neutral.
    degree : 2 (default; ``(n + 1)(n + 2) / 2`` nodes) or 1 (``n + 1`` nodes).
    name : The family's name.

    ``at(values)`` is the interpolated hull at a ``{parameter: value}`` dict or a
    sequence in ``names`` order (numbers or CasADi expressions); ``exact(values)``
    the transformed hull itself, to check a solution against.
    """

    def __init__(self, hull, ranges: Mapping[str, tuple[float, float]], degree: int = 2,
                 name: str | None = None):
        unknown = set(ranges) - set(NEUTRAL)
        if unknown:
            raise ValueError(f"unknown transformation(s) {sorted(unknown)}; expected some of {list(NEUTRAL)}")
        if not ranges:
            raise ValueError("vary at least one transformation")
        self.base = hull
        self.names = list(ranges)
        self.ranges = {k: (float(lo), float(hi)) for k, (lo, hi) in ranges.items()}
        for k, (lo, hi) in self.ranges.items():
            if not hi > lo:
                raise ValueError(f"range of {k} must be (low, high) with high > low")
        self.degree = degree
        self.name = name or f"{hull.name} transformed"
        unit_nodes, self._monomials = box_design(len(self.names), degree)
        self.node_parameters = [self._from_unit(u) for u in unit_nodes]
        vandermonde = _onp.array([[self._monomial(u, m) for m in self._monomials] for u in unit_nodes])
        self._inverse = _onp.linalg.inv(vandermonde)
        self.nodes = [
            transform_hull(hull, name=f"{self.name} node {i}", **p)
            for i, p in enumerate(self.node_parameters)
        ]

    def __repr__(self) -> str:
        return (
            f"TransformedFamily '{self.name}' ({', '.join(self.names)}; degree {self.degree}, "
            f"{len(self.nodes)} nodes)"
        )

    def _from_unit(self, u) -> dict[str, float]:
        return {
            k: 0.5 * (lo + hi) + 0.5 * (hi - lo) * u_k
            for (k, (lo, hi)), u_k in zip(self.ranges.items(), u)
        }

    def _to_unit(self, values: Sequence[Any]) -> list[Any]:
        return [
            (v - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
            for v, (lo, hi) in zip(values, self.ranges.values())
        ]

    @staticmethod
    def _monomial(u, exponents):
        value = 1.0
        for u_k, e in zip(u, exponents):
            for _ in range(e):
                value = value * u_k
        return value

    def _values(self, values) -> list[Any]:
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.names)
            if unknown:
                raise ValueError(f"{sorted(unknown)} are not varied by this family ({self.names})")
            return [values.get(k, NEUTRAL[k]) for k in self.names]
        values = list(values)
        if len(values) != len(self.names):
            raise ValueError(f"expected {len(self.names)} values ({self.names})")
        return values

    def basis(self, values) -> list[Any]:
        """The Lagrange basis of the design at ``values`` (numbers or CasADi expressions)."""
        u = self._to_unit(self._values(values))
        monomials = [self._monomial(u, m) for m in self._monomials]
        out = []
        for k in range(len(self.nodes)):
            total = 0.0
            for j, m in enumerate(monomials):
                c = float(self._inverse[j, k])
                if abs(c) > 1e-14:
                    total = total + c * m
            out.append(total)
        return out

    def at(self, values):
        """The interpolated hull at ``values`` (a dict, or a sequence in ``names`` order)."""
        from hydrosandbox.hull.camber.blended import BlendedHull

        return BlendedHull(self, self._values(values))

    def exact(self, values, **kwargs):
        """The transformed hull itself at ``values``."""
        values = self._values(values)
        return transform_hull(
            self.base, name=kwargs.pop("name", f"{self.name} exact"),
            **{k: float(v) for k, v in zip(self.names, values)},
        )
