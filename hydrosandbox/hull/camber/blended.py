"""
A camber hull family as a design space: the blend weights become variables of an
optimization.

camber blends a family number by number over its authored control points, then
sweeps the result (``family.blend``). That map from weights to hull runs through
the sweep and its trims and is evaluated numerically, so it can't sit inside a
gradient-based solve directly. ``HullFamily`` stands in for it with an
interpolant that can:

1. Blends are taken exactly, by camber's rule, at the nodes of a lattice on the
   weight simplex (``degree + 1`` blends along a two-hull slider, 10 for three hulls
   at degree 3, ...), and each is swept and cut into an offset table
   (``Hull.from_camber``) over its own extent, so that its transom is its last
   station, with the same numbers of stations and levels as every other node.
2. Anything a ``Hull`` computes for a blend (hydrostatics, wave amplitude spectra,
   dynamic force tables, sections for hull-foil interaction, keel height) is
   then the polynomial interpolant, in the weights, of what each node hull computes:
   ``Q(w) = sum_a L_a(w) Q_a``, with ``L_a`` the Lagrange basis of the lattice. The
   interpolant is exact at every node, smooth in the weights, and differentiable
   when they are CasADi variables. Thin-ship quantities are linear in the offsets
   (the wave amplitude spectrum, hydrostatics) or smooth functions of them, so they
   are well resolved by a polynomial in the weights once the offsets themselves vary
   smoothly with them, as camber's C2 construction makes them.

``HullFamily.exact(weights)`` builds the blend itself, to check a solution against.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Sequence

import numpy as _onp
import aerosandbox.numpy as np

from hydrosandbox.hull.camber.document import CamberHull
from hydrosandbox.hull.camber.family import blend, promote_family


def simplex_lattice(n_hulls: int, degree: int) -> list[tuple[int, ...]]:
    """Multi-indices ``a`` with ``sum(a) = degree``: the nodes ``a / degree`` of the weight simplex."""
    return [
        a
        for a in itertools.product(range(degree + 1), repeat=n_hulls)
        if sum(a) == degree
    ][::-1]


def lattice_basis(weights: Sequence[Any], lattice: Sequence[tuple[int, ...]], degree: int) -> list[Any]:
    """
    The Lagrange basis of the simplex lattice at barycentric ``weights`` (numbers or
    CasADi expressions): ``L_a(w) = prod_i prod_{j < a_i} (degree w_i - j) / (j + 1)``,
    which is 1 at node ``a`` and 0 at every other node.
    """
    out = []
    for a in lattice:
        value = 1.0
        for w_i, a_i in zip(weights, a):
            for j in range(a_i):
                value = value * (degree * w_i - j) / (j + 1)
        out.append(value)
    return out


def _is_zero(b) -> bool:
    return isinstance(b, (int, float)) and b == 0.0


class _Combined:
    """The polynomial combination of the same object (a table, a field) across the node hulls."""

    def __init__(self, objects: Sequence[Any], basis: Sequence[Any]):
        self._objects = list(objects)
        self._basis = list(basis)

    def __call__(self, *args, **kwargs):
        return _mix([o(*args, **kwargs) for o in self._objects], self._basis)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        values = [getattr(o, name) for o in self._objects]
        if callable(values[0]):
            return _Combined(values, self._basis)
        if name == "length_scales":  # set by the caller, shared by every node
            return values[0]
        return _mix(values, self._basis)

    def rebind(self, basis: Sequence[Any]) -> "_Combined":
        """The same objects combined with another basis (other weights)."""
        return _Combined(self._objects, basis)


def _mix(values: Sequence[Any], basis: Sequence[Any]):
    """``sum_k basis[k] values[k]``, through dicts, tuples and callables."""
    v0 = values[0]
    if isinstance(v0, dict):
        keys = list(v0)
        if any(set(v) != set(keys) for v in values):
            raise ValueError(
                "the family's node hulls disagree in structure (e.g. some have an "
                "immersed transom and some do not); a blend can't interpolate them"
            )
        return {k: _mix([v[k] for v in values], basis) for k in keys}
    if isinstance(v0, (tuple, list)):
        return type(v0)(_mix(list(z), basis) for z in zip(*values))
    if v0 is None or isinstance(v0, (str, bool)):
        return v0
    if isinstance(v0, int) and all(v == v0 for v in values):
        return v0  # a count, not a quantity
    if callable(v0) or not _is_quantity(v0):
        return _Combined(values, basis)
    total = 0.0
    for b, v in zip(basis, values):
        if _is_zero(b):
            continue
        total = total + b * v
    return total


def _is_quantity(v) -> bool:
    return isinstance(v, (int, float, _onp.ndarray, _onp.generic)) or np.is_casadi_type(v, recursive=False)


class BlendedHull:
    """
    A ``Hull`` stand-in for one blend of a ``HullFamily``: every method and table is
    the polynomial interpolant across the family's node hulls (see the module
    docstring). Built by ``HullFamily.at``; ``weights`` may be CasADi expressions.
    """

    #: The same for every node hull, so taken from the first.
    _SHARED = {"to_earth", "form_factor", "roughness_delta_cf", "transom_closure"}
    #: Methods that, given the blend's own ``sections``, read nothing else of the hull.
    _FROM_SECTIONS = {"induced_velocity", "pressure_load"}

    def __init__(self, family: "HullFamily", weights: Sequence[Any]):
        self.family = family
        self.weights = list(weights)
        # Any family with ``nodes``, ``name`` and a ``basis(weights)`` (the
        # Lagrange basis of its lattice) can stand behind a ``BlendedHull``.
        self.basis = family.basis(self.weights)
        self.name = family.name

    def __repr__(self) -> str:
        return f"BlendedHull of {self.family!r}"

    @property
    def nodes(self):
        return self.family.nodes

    @property
    def has_transom(self) -> bool:
        return any(n.has_transom for n in self.nodes)

    def __getattr__(self, name):
        if name.startswith("__") or name in ("family", "weights", "basis"):
            raise AttributeError(name)
        if name in self._SHARED:
            return getattr(self.nodes[0], name)
        first = getattr(self.nodes[0], name)
        if not callable(first):
            return _mix([getattr(n, name) for n in self.nodes], self.basis)

        def method(*args, **kwargs):
            if name in self._FROM_SECTIONS and kwargs.get("sections") is not None:
                return first(*args, **kwargs)
            # Tables are built once and rebound to other weights, so every node needs one.
            every = name.endswith("_table")
            active = [(b, n) for b, n in zip(self.basis, self.nodes) if every or not _is_zero(b)]
            return _mix([getattr(n, name)(*args, **kwargs) for _, n in active], [b for b, _ in active])

        method.__name__ = name
        return method

    def rebind(self, value):
        """A ``_Combined`` made for another blend (a table, say), recombined for this one."""
        return value.rebind(self.basis) if isinstance(value, _Combined) else value


class HullFamily:
    """
    A family of camber hulls, promoted to one topology and tabulated for
    optimization over its blend weights.

    Parameters
    ----------
    hulls : The family's members (``CamberHull``, or paths / JSON / dicts of camber
        documents). The waterline and deck trim of the first apply to every blend.
    degree : Polynomial degree of the interpolant in the weights (default: 4 for
        two hulls, 3 for three, 2 beyond), so ``degree + 1`` swept blends along a
        two-hull slider and ``(K + degree - 1)! / ((K - 1)! degree!)`` for ``K`` hulls.
    n_stations, n_levels : Each node's offset grid (over its own extent).
    n_columns, rows_per_segment, level_oversampling : Sampling of each sweep and of
        its offset table (``Hull.from_camber``).
    length : If given [m], every member is first scaled uniformly (waterline depth
        included) to this overall length, so that the blend compares shapes rather
        than sizes.
    correspondence, section_points : For ``promote_family`` (default: camber's
        automatic station matching, and camber's rule for added section points).
    hull_kwargs : Further ``Hull`` arguments (``form_factor``, ...).
    """

    def __init__(
        self,
        hulls: Sequence[Any],
        degree: int | None = None,
        n_stations: int = 41,
        n_levels: int = 25,
        n_columns: int = 121,
        rows_per_segment: int = 8,
        level_oversampling: int = 8,
        length: float | None = None,
        correspondence=None,
        section_points: str = "camber",
        name: str | None = None,
        **hull_kwargs,
    ):
        from hydrosandbox.hull.hull import Hull

        docs = [_as_document(h) for h in hulls]
        if length is not None:
            docs = [d.scaled(length / (d.loa * d.unit_m)) for d in docs]
        if len(docs) < 2:
            raise ValueError("a family needs at least two hulls")
        self.members = promote_family(docs, correspondence, section_points)
        self.member_names = [d.name for d in docs]
        self.name = name or " / ".join(n or f"hull {i}" for i, n in enumerate(self.member_names))
        K = len(self.members)
        self.degree = degree if degree is not None else (4 if K == 2 else 3 if K == 3 else 2)
        self.lattice = simplex_lattice(K, self.degree)
        self._hull_kwargs = dict(
            n_stations=n_stations, n_levels=n_levels, n_columns=n_columns,
            rows_per_segment=rows_per_segment, level_oversampling=level_oversampling,
            **hull_kwargs,
        )
        self.node_weights = [tuple(a_i / self.degree for a_i in a) for a in self.lattice]
        self.node_documents = [self.document(w) for w in self.node_weights]
        self.nodes = [
            Hull.from_camber(d, **{"name": f"{self.name} node {i}", **self._hull_kwargs})
            for i, d in enumerate(self.node_documents)
        ]

    def __repr__(self) -> str:
        return (
            f"HullFamily '{self.name}' ({len(self.members)} hulls, degree {self.degree}, "
            f"{len(self.nodes)} nodes)"
        )

    @property
    def n_hulls(self) -> int:
        return len(self.members)

    def basis(self, weights: Sequence[Any]) -> list[Any]:
        """The Lagrange basis of the family's lattice at ``weights``."""
        return lattice_basis(weights, self.lattice, self.degree)

    def document(self, weights: Sequence[float], name: str = "") -> CamberHull:
        """The blend itself, as a camber document (camber's ``blendState``)."""
        return blend(self.members, [float(w) for w in weights], name=name)

    def at(self, weights: Sequence[Any]) -> BlendedHull:
        """The interpolated hull at barycentric ``weights`` (numbers or CasADi expressions)."""
        if len(weights) != self.n_hulls:
            raise ValueError(f"expected {self.n_hulls} weights")
        return BlendedHull(self, weights)

    def exact(self, weights: Sequence[float], **kwargs):
        """The blend swept and cut as the nodes are: the hull the interpolant stands in for."""
        from hydrosandbox.hull.hull import Hull

        return Hull.from_camber(
            self.document(weights), **{"name": f"{self.name} blend", **self._hull_kwargs, **kwargs}
        )


def _as_document(h) -> CamberHull:
    if isinstance(h, CamberHull):
        return h
    if isinstance(h, dict):
        return CamberHull.from_dict(h)
    return CamberHull.from_json(h)

