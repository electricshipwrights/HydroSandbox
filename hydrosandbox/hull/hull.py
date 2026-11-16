"""
A displacement hull as a table of offsets, with the thin-ship (Michell) physics of
the ``boatmath`` project (crates ``hullgeom`` and ``thinship``) ported to
``aerosandbox.numpy`` so that a hull's attitude can be an optimization variable.
"""

import collections
import hashlib
from typing import Any, Literal, Sequence
import numpy as _onp
from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.library.free_surface import smooth_submergence_fraction

#: Free-fall hollow-length coefficient (boatmath ``thinship::BALLISTIC_COEFF``).
BALLISTIC_COEFF = float(_onp.sqrt(2.0))
#: Dry-transom hollow length, in transverse wavelengths (boatmath's ``DRY_HOLLOW_FRACTION``,
#: used by the ``"boatmath"`` closure).
BOATMATH_DRY_HOLLOW_FRACTION = 0.5
#: Dry-transom hollow length of the ``"auto"`` closure, in transverse wavelengths
#: ``2 pi U^2 / g``: fitted to the wave-pattern resistance of the Southampton
#: catamaran series (Molland, Wellicome & Couser 1994, models 4b, 5b, 6b,
#: monohull and S/L 0.2-0.5, Fn 0.4-1.0). The physical hollow closes much sooner
#: (about 0.1 L behind the transom); this is the length a linear source closure
#: needs to radiate what the tank measured.
DRY_HOLLOW_FRACTION = 0.3
#: Transom Froude numbers between which the transom goes from wet to dry (and the
#: ``"auto"`` closure fades in). Fitted to the same data: no closure fits best up
#: to Fn 0.4, the full closure from Fn 0.45.
AUTO_WET_FN_T = 2.0
AUTO_DRY_FN_T = 2.5


#: The most recent expensive numeric kernels (depth exponentials, strip phases),
#: keyed by their inputs. They depend on the offset grid and the attitude but not
#: on the half-breadths, so hulls sharing a grid (a design family's nodes, built
#: attitude by attitude) and repeated contractions within one evaluation reuse them.
_KERNELS: "collections.OrderedDict[bytes, Any]" = collections.OrderedDict()
_KERNELS_KEPT = 64
_KERNELS_BYTES = 600e6  # per process


def _nbytes(value) -> int:
    if isinstance(value, _onp.ndarray):
        return value.nbytes
    if isinstance(value, (tuple, list)):
        return sum(_nbytes(v) for v in value)
    return 0


def _kernel(tag: str, arrays, compute):
    """``compute()``, or its cached value for the same ``tag`` and input ``arrays``."""
    h = hashlib.blake2b(tag.encode(), digest_size=16)
    for a in arrays:
        a = _onp.ascontiguousarray(a, dtype=float)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    key = h.digest()
    if key in _KERNELS:
        _KERNELS.move_to_end(key)
        return _KERNELS[key]
    value = compute()
    _KERNELS[key] = value
    while len(_KERNELS) > 1 and (
        len(_KERNELS) > _KERNELS_KEPT or sum(_nbytes(v) for v in _KERNELS.values()) > _KERNELS_BYTES
    ):
        _KERNELS.popitem(last=False)
    return value


def ittc57_cf(reynolds: Vectorizable) -> Vectorizable:
    """ITTC-57 model-ship correlation line, ``C_F = 0.075 / (log10(Re) - 2)^2``."""
    return 0.075 / (np.log10(reynolds) - 2.0) ** 2


class Hull(AeroSandboxObject):
    """
    A port/starboard-symmetric hull described by half-breadths ``f(x, z)`` on a grid
    of stations and levels, in geometry axes: ``+x`` aft (bow at the low-``x`` end),
    ``+z`` up, with the design waterline at ``z = 0``. Levels should run from the
    keel to above any waterline the hull will see, so it can be re-posed.

    An attitude is a ``sinkage`` [m, downward] of a pivot point and a ``trim``
    [deg, bow up] about it; the still water surface is then the plane ``z = 0`` in
    earth axes. All attitude-dependent methods are smooth in both, and accept
    CasADi variables.

    The flow model is thin-ship theory, as in boatmath: a slender hull
    (``|df/dx| << 1``) in deep water, represented by centreplane sources of
    strength ``2 U df/dx``. It gives wave resistance (with any other wave-maker
    sharing the spectrum), but no dynamic lift, sinkage or trim of the hull itself.
    """

    #: Factor on the wetted surface the offset table gives (and so on viscous drag):
    #: set by constructions that know the true surface better than the table can
    #: (``FFDLattice.hull``, whose deformed keel and stem lines fall between grid lines).
    wetted_surface_scale: float = 1.0

    def __init__(
        self,
        stations: Sequence[float],
        levels: Sequence[float],
        half_breadths: Sequence[Sequence[float]],
        name: str = "Hull",
        form_factor: float = 0.0,
        roughness_delta_cf: float = 0.0,
        transom_closure: Literal["auto", "boatmath", "none"] | float = "auto",
        waterline_smoothing: float | None = None,
    ):
        """
        Parameters
        ----------
        stations : Station positions ``x`` [m], increasing from bow to stern.
        levels : Level heights ``z`` [m], increasing from keel to deck.
        half_breadths : ``(n_stations, n_levels)`` half-breadths [m].
        form_factor : Viscous form factor ``k``, as ``(1 + k) * C_F``.
        roughness_delta_cf : Roughness allowance added to ``C_F`` outside ``1 + k``.
        transom_closure : How the wave integral closes an immersed transom:
            ``"auto"`` (calibrated against tank data: no closure while the transom
            is wet, the closing hollow fading in as it runs dry, over
            ``DRY_HOLLOW_FRACTION`` of a transverse wavelength), ``"boatmath"``
            (boatmath's ``TransomClosure::Auto``: ballistic hollow while wet, half
            a transverse wavelength once dry), ``"none"`` (an open body), or a
            fixed hollow length [m].
        waterline_smoothing : Thickness [m] over which a cell's wetting is blended
            as the water surface passes through it (keeps everything
            differentiable in attitude). Defaults to 1% of the depth of the
            offset table below the design waterline, at most 5 mm.
        """
        self.name = name
        self.stations = _onp.asarray(stations, dtype=float)
        self.levels = _onp.asarray(levels, dtype=float)
        self.half_breadths = _onp.asarray(half_breadths, dtype=float)
        self.form_factor = form_factor
        self.roughness_delta_cf = roughness_delta_cf
        self.transom_closure = transom_closure
        if waterline_smoothing is None:
            waterline_smoothing = min(0.005, 0.01 * max(float(-_onp.min(levels)), 1e-3))
        self.waterline_smoothing = waterline_smoothing

        n_x, n_z = len(self.stations), len(self.levels)
        if self.half_breadths.shape != (n_x, n_z):
            raise ValueError(
                f"`half_breadths` must have shape (n_stations, n_levels) = "
                f"({n_x}, {n_z}); got {self.half_breadths.shape}."
            )
        if n_x < 2 or n_z < 2:
            raise ValueError("A hull needs at least two stations and two levels.")
        if _onp.any(_onp.diff(self.stations) <= 0) or _onp.any(_onp.diff(self.levels) <= 0):
            raise ValueError("`stations` and `levels` must be strictly increasing.")
        if _onp.any(self.half_breadths < 0):
            raise ValueError("`half_breadths` must be non-negative.")

        ### Cell tables: one cell per (station strip, level band), flattened.
        f = self.half_breadths
        x, z = self.stations, self.levels
        dx = _onp.diff(x)[:, None] * _onp.ones((1, n_z - 1))
        dz = _onp.ones((n_x - 1, 1)) * _onp.diff(z)[None, :]
        self._cell_x = (0.5 * (x[:-1] + x[1:])[:, None] * _onp.ones((1, n_z - 1))).reshape(-1)
        self._cell_z_lo = (_onp.ones((n_x - 1, 1)) * z[None, :-1]).reshape(-1)
        self._cell_z_hi = (_onp.ones((n_x - 1, 1)) * z[None, 1:]).reshape(-1)
        self._cell_dx = dx.reshape(-1)
        self._cell_dz = dz.reshape(-1)
        # Mean half-breadth, and its mean jumps across the cell along x and z.
        self._cell_f = (0.25 * (f[:-1, :-1] + f[1:, :-1] + f[:-1, 1:] + f[1:, 1:])).reshape(-1)
        self._cell_dfx = (
            0.5 * ((f[1:, :-1] - f[:-1, :-1]) + (f[1:, 1:] - f[:-1, 1:]))
        ).reshape(-1)
        self._cell_dfz = (
            0.5 * ((f[:-1, 1:] - f[:-1, :-1]) + (f[1:, 1:] - f[1:, :-1]))
        ).reshape(-1)

        ### Bow & transom end-station sections (stations[0] and stations[-1]).
        f_bow = f[0, :]
        self._bow_f = 0.5 * (f_bow[:-1] + f_bow[1:])
        self._bow_half_beam = float(_onp.max(f_bow))
        self.has_bow_step = self._bow_half_beam > 1e-6 * max(
            float(_onp.max(f)), 1e-12
        )
        f_transom = f[-1, :]
        self._transom_f = 0.5 * (f_transom[:-1] + f_transom[1:])
        self._transom_z_lo = z[:-1]
        self._transom_z_hi = z[1:]
        self._transom_half_beam = float(_onp.max(f_transom))
        self.has_transom = self._transom_half_beam > 1e-6 * max(
            float(_onp.max(f)), 1e-12
        )

    def __repr__(self) -> str:
        return (
            f"Hull '{self.name}' (L = {self.length:.3f} m, "
            f"{len(self.stations)} stations x {len(self.levels)} levels"
            f"{', transom' if self.has_transom else ''})"
        )

    @property
    def length(self) -> float:
        """Overall length of the offset table [m]."""
        return float(self.stations[-1] - self.stations[0])

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def wigley(
        cls,
        length: float = 10.0,
        beam: float = 1.0,
        draft: float = 0.625,
        freeboard: float | None = None,
        n_stations: int = 41,
        n_levels: int = 21,
        **kwargs,
    ) -> "Hull":
        """
        The Wigley parabolic hull ``f = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)``, centred on
        ``x = 0`` and wall-sided above the design waterline up to ``freeboard``.
        """
        if freeboard is None:
            freeboard = 0.5 * draft
        stations = 0.5 * length * -_onp.cos(_onp.pi * _onp.linspace(0.0, 1.0, n_stations))
        n_below = max(2, int(round((n_levels - 1) * draft / (draft + freeboard))))
        levels = _onp.concatenate(
            [
                _onp.linspace(-draft, 0.0, n_below + 1),
                _onp.linspace(0.0, freeboard, n_levels - n_below)[1:],
            ]
        )
        depth_shape = 1.0 - (_onp.minimum(levels, 0.0) / draft) ** 2
        half_breadths = (
            0.5 * beam * (1.0 - (2.0 * stations / length) ** 2)[:, None] * depth_shape[None, :]
        )
        kwargs.setdefault("name", "Wigley")
        return cls(stations, levels, _onp.maximum(half_breadths, 0.0), **kwargs)

    @classmethod
    def from_mesh(
        cls,
        vertices: Sequence[Sequence[float]],
        triangles: Sequence[Sequence[int]],
        n_stations: int = 41,
        n_levels: int = 25,
        x_forward: bool = False,
        centerplane_y: float | None = None,
        stations: Sequence[float] | None = None,
        levels: Sequence[float] | None = None,
        level_oversampling: int = 1,
        **kwargs,
    ) -> "Hull":
        """
        Cut a triangle mesh (metres; ``z`` up, design waterline at ``z = 0``) into an
        offset table. The half-breadth at each (station, level) is the largest
        distance from the hull's centreplane at which the line through it, parallel
        to ``y``, meets the mesh; either one side or both sides may be meshed.

        By default the grid spans the mesh (``n_stations`` cosine-clustered
        stations, ``n_levels`` levels split at the waterline). Pass ``stations`` and
        ``levels`` (in this class's axes, ``x`` aft) to cut on a given grid instead,
        e.g. one shared by every member of a hull family.

        With ``level_oversampling > 1`` each level's half-breadth is instead the
        mean over its band (the levels either side, weighted by a hat), sampled that
        many times per gap: smoother as the geometry moves through the levels (a
        flat bottom, a hard chine, a raked transom), and area-conserving.

        ``centerplane_y`` is the hull's own centreplane (a demihull drawn off the
        centreline, say). If None, it is taken as the median mid-point of the lines
        that cross both sides, or ``y = 0`` if none do.

        Set ``x_forward=True`` for meshes whose bow is at the high-``x`` end (the
        convention of boatmath hull files); the table is stored bow-first.
        """
        v = _onp.asarray(vertices, dtype=float)
        tri = _onp.asarray(triangles, dtype=int)
        if x_forward:
            v = v * _onp.array([-1.0, 1.0, 1.0])

        x_min, x_max = v[:, 0].min(), v[:, 0].max()
        z_min, z_max = v[:, 2].min(), v[:, 2].max()
        if stations is not None:
            stations = _onp.asarray(stations, dtype=float)
            n_stations = len(stations)
        else:
            # Cosine-clustered stations, pulled a hair inside the ends.
            s = 0.5 * (1.0 - _onp.cos(_onp.pi * _onp.linspace(0.0, 1.0, n_stations)))
            stations = x_min + (x_max - x_min) * (1e-6 + (1.0 - 2e-6) * s)
        if levels is not None:
            levels = _onp.asarray(levels, dtype=float)
            n_levels = len(levels)
        elif z_min < 0.0 < z_max:
            # Resolve the underwater body: half the levels below the design
            # waterline, half above it.
            n_below = n_levels // 2
            levels = _onp.concatenate(
                [
                    z_min * _onp.linspace(1.0 - 1e-6, 0.0, n_below + 1),
                    z_max * _onp.linspace(0.0, 1.0 - 1e-6, n_levels - n_below)[1:],
                ]
            )
        else:
            levels = z_min + (z_max - z_min) * _onp.linspace(1e-6, 1.0 - 1e-6, n_levels)

        if level_oversampling > 1:
            # Each level's half-breadth is the hat-weighted mean over its band (the
            # levels either side), sampled `level_oversampling` times per gap. That
            # keeps the table continuous as a flat bottom, a chine or a raked transom
            # moves through a level, and conserves each section's area.
            sub = _onp.linspace(0.0, 1.0, level_oversampling + 1)[:-1]
            fine = _onp.concatenate(
                [levels[j] + (levels[j + 1] - levels[j]) * sub for j in range(n_levels - 1)]
                + [levels[-1:]]
            )
        else:
            fine = levels
        y_low, y_high = _mesh_y_extent(v, tri, stations, fine)
        hit = ~_onp.isnan(y_low)
        if centerplane_y is None:
            both = hit & ((y_high - y_low) > 1e-6)
            centerplane_y = (
                float(_onp.median(0.5 * (y_high + y_low)[both])) if _onp.any(both) else 0.0
            )
        half_breadths = _onp.where(
            hit,
            _onp.maximum(
                _onp.abs(_onp.nan_to_num(y_low) - centerplane_y),
                _onp.abs(_onp.nan_to_num(y_high) - centerplane_y),
            ),
            0.0,
        )
        if level_oversampling > 1:
            # Trapezoid weights on the fine levels, times each level's hat.
            dz = _onp.diff(fine)
            trap = _onp.zeros(len(fine))
            trap[:-1] += 0.5 * dz
            trap[1:] += 0.5 * dz
            hats = _onp.zeros((n_levels, len(fine)))
            for j in range(n_levels):
                lo = levels[j - 1] if j > 0 else levels[0]
                hi = levels[j + 1] if j < n_levels - 1 else levels[-1]
                rise = (fine - lo) / max(levels[j] - lo, 1e-300) if j > 0 else _onp.ones_like(fine)
                fall = (hi - fine) / max(hi - levels[j], 1e-300) if j < n_levels - 1 else _onp.ones_like(fine)
                hat = _onp.where(fine <= levels[j], rise, fall)
                hat = _onp.where((fine >= lo) & (fine <= hi), _onp.clip(hat, 0.0, 1.0), 0.0)
                hats[j] = hat * trap
            hats /= hats.sum(axis=1, keepdims=True)
            half_breadths = half_breadths @ hats.T
        hull = cls(stations, levels, half_breadths, **kwargs)
        hull.source_centerplane_y = float(centerplane_y)
        return hull

    @classmethod
    def from_camber(
        cls,
        document,
        n_stations: int = 41,
        n_levels: int = 25,
        stations: Sequence[float] | None = None,
        levels: Sequence[float] | None = None,
        n_columns: int = 121,
        rows_per_segment: int = 8,
        level_oversampling: int = 8,
        **kwargs,
    ) -> "Hull":
        """
        Build a hull from a camber document: a ``CamberHull``, its JSON text, a path
        to it, or the decoded dict. The document's sweep is evaluated (see
        ``hydrosandbox.hull.camber``), floated at its deck trim with its design
        waterline at ``z = 0``, turned bow-first (``x`` aft) and cut into an offset
        table over the hull's own extent, from the bow to where the transom plane
        crosses the design waterline (or on given ``stations`` and ``levels``),
        each level averaged over its band (``level_oversampling``; see
        ``from_mesh``) so the table moves smoothly with the authored numbers.

        ``n_columns`` and ``rows_per_segment`` set how finely the sweep is sampled
        along the hull and around each section.
        """
        from hydrosandbox.hull.camber import CamberHull, SweptHull

        if isinstance(document, dict):
            document = CamberHull.from_dict(document)
        elif not isinstance(document, CamberHull):
            document = CamberHull.from_json(document)
        swept = SweptHull(document)
        vertices, triangles = swept.hull_mesh(n_columns=n_columns, R=rows_per_segment)
        if stations is None:
            # Bow to the transom plane's crossing of the design waterline, so the
            # transom is the last station (where this class looks for it) however
            # it is raked; only above-water overhang aft of that point is left out.
            x_bow = float(vertices[:, 0].min())
            x_end = min(float(vertices[:, 0].max()), swept.transom_waterline_x())
            s = 0.5 * (1.0 - _onp.cos(_onp.pi * _onp.linspace(0.0, 1.0, n_stations)))
            stations = x_bow + (x_end - x_bow) * (1e-6 + (1.0 - 2e-6) * s)
        kwargs.setdefault("name", document.name or "camber hull")
        result = cls.from_mesh(
            vertices, triangles, n_stations=n_stations, n_levels=n_levels,
            centerplane_y=0.0, stations=stations, levels=levels,
            level_oversampling=level_oversampling, **kwargs,
        )
        result.camber_document = document
        return result

    @classmethod
    def from_boatmath_geometry(
        cls,
        geometry: dict[str, Any],
        hull_index: int = 0,
        n_stations: int = 41,
        n_levels: int = 25,
        tessellation: int = 48,
        centerplane_y: float | None = None,
        **kwargs,
    ) -> "Hull":
        """
        Build a hull from the ``geometry`` of a boatmath ``hull`` record (its native
        JSON, in metres with the design waterline at ``z = 0`` and the bow at high
        ``x``): either ``kind: "mesh"`` or ``kind: "nurbs"`` (polynomial B-spline
        patches, which are tessellated ``tessellation`` x ``tessellation``).
        """
        hull = geometry["hulls"][hull_index]
        if geometry["kind"] == "mesh":
            vertices, triangles = hull["vertices"], hull["triangles"]
        elif geometry["kind"] == "nurbs":
            from scipy.interpolate import BSpline

            vertex_blocks, triangle_blocks, offset = [], [], 0
            for patch in hull["patches"]:
                (deg_u, deg_v), (n_u, n_v) = patch["degree"], patch["n_ctrl"]
                ctrl = _onp.asarray(patch["ctrl"], dtype=float).reshape(n_u, n_v, 3)
                knots_u = _onp.asarray(patch["knots_u"], dtype=float)
                knots_v = _onp.asarray(patch["knots_v"], dtype=float)
                box = patch.get("trim_uv") or [
                    knots_u[deg_u], knots_u[-deg_u - 1],
                    knots_v[deg_v], knots_v[-deg_v - 1],
                ]
                u = _onp.linspace(box[0], box[1], tessellation + 1)
                w = _onp.linspace(box[2], box[3], tessellation + 1)
                along_u = BSpline(knots_u, ctrl, deg_u, axis=0)(u)  # (U, n_v, 3)
                pts = BSpline(knots_v, along_u, deg_v, axis=1)(w)  # (U, V, 3)
                m = tessellation + 1
                grid = _onp.arange(m * m).reshape(m, m) + offset
                quads = _onp.stack(
                    [grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], axis=-1
                ).reshape(-1, 4)
                vertex_blocks.append(pts.reshape(-1, 3))
                triangle_blocks += [quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]
                offset += m * m
            vertices = _onp.concatenate(vertex_blocks)
            triangles = _onp.concatenate(triangle_blocks)
        else:
            raise ValueError(f"Unknown boatmath geometry kind {geometry['kind']!r}.")
        result = cls.from_mesh(
            vertices, triangles, n_stations=n_stations, n_levels=n_levels,
            x_forward=True, centerplane_y=centerplane_y, **kwargs,
        )
        if geometry["kind"] == "nurbs":
            # Keep the designer's surfaces (for CAD export), in this class's axes:
            # x aft, and y measured from the hull's own centreplane. Bounded patches
            # (a transom plane cut to the hull's outline) are left out: the record
            # carries their parameter box, not the curve that trims them.
            result.source_patches = [
                {
                    "degree": tuple(patch["degree"]),
                    "knots_u": [float(k) for k in patch["knots_u"]],
                    "knots_v": [float(k) for k in patch["knots_v"]],
                    "ctrl": (
                        _onp.asarray(patch["ctrl"], dtype=float).reshape(
                            patch["n_ctrl"][0], patch["n_ctrl"][1], 3
                        )
                        * _onp.array([-1.0, 1.0, 1.0])
                        - _onp.array([0.0, result.source_centerplane_y, 0.0])
                    ),
                }
                for patch in hull["patches"]
                if not patch.get("trim_uv")
            ]
        return result

    def __getstate__(self):
        # The cached CasADi keel interpolant cannot be pickled; it is rebuilt on use.
        state = dict(self.__dict__)
        state.pop("_keel_interpolant", None)
        return state

    def keel_height(self, x: Vectorizable) -> Vectorizable:
        """
        Height ``z`` [m] of the hull's keel at station ``x`` (hull axes, negative
        below the design waterline): the lowest level of the offsets table at which
        the hull has breadth, taken one level lower so it bounds the hull from
        below. Interpolated linearly along the hull; numeric or symbolic in ``x``.

        Useful for hanging things under the hull: where a strut meets it, and how
        much clearance a foil has.
        """
        import casadi as cas

        if not hasattr(self, "_keel_interpolant"):
            threshold = 1e-4 * float(_onp.max(self.half_breadths))
            keel = _onp.full(len(self.stations), _onp.nan)
            for i, row in enumerate(self.half_breadths):
                wet = _onp.nonzero(row > threshold)[0]
                if wet.size:
                    keel[i] = self.levels[max(int(wet[0]) - 1, 0)]
            valid = ~_onp.isnan(keel)
            keel = _onp.interp(self.stations, self.stations[valid], keel[valid])
            self._keel_z = keel
            self._keel_interpolant = cas.interpolant(
                "keel_height", "linear", [self.stations.tolist()], keel.tolist()
            )
        x_clamped = np.clip(x, self.stations[0], self.stations[-1])
        value = self._keel_interpolant(x_clamped)
        return value if np.is_casadi_type(x_clamped, recursive=False) else float(value)

    def surface_patches(self) -> list[dict[str, Any]]:
        """
        The hull's surfaces as B-spline patches in its own axes (x aft, y from the
        centreplane, z up), each a dict of ``degree``, ``knots_u``, ``knots_v`` and
        an ``(n_u, n_v, 3)`` control net ``ctrl``.

        A hull read from B-spline geometry returns the designer's own surfaces, and
        one read from a camber document B-splines fitted through its sweep (see
        ``SweptHull.surface_patches``). Any other hull returns its offsets table as
        two bilinear patches, one per side.
        """
        if getattr(self, "source_patches", None):
            return self.source_patches
        if getattr(self, "camber_document", None) is not None:
            # A camber hull: B-splines through its sweep, computed on first use.
            from hydrosandbox.hull.camber import SweptHull

            self.source_patches = SweptHull(self.camber_document).surface_patches()
            return self.source_patches
        n_x, n_z = self.half_breadths.shape
        patches = []
        for side in (1.0, -1.0):
            ctrl = _onp.stack(
                [
                    self.stations[:, None] * _onp.ones((1, n_z)),
                    side * self.half_breadths,
                    _onp.ones((n_x, 1)) * self.levels[None, :],
                ],
                axis=-1,
            )
            patches.append(
                {
                    "degree": (1, 1),
                    "knots_u": [0.0] + list(_onp.linspace(0.0, 1.0, n_x)) + [1.0],
                    "knots_v": [0.0] + list(_onp.linspace(0.0, 1.0, n_z)) + [1.0],
                    "ctrl": ctrl,
                }
            )
        return patches

    # ------------------------------------------------------------------
    # Attitude
    # ------------------------------------------------------------------
    @staticmethod
    def to_earth(
        x: Vectorizable,
        z: Vectorizable,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
    ) -> tuple[Vectorizable, Vectorizable]:
        """
        Map geometry-axes points to earth axes (``x`` downstream, ``z`` up from the
        still water surface) for a hull whose ``pivot_xz`` point has sunk by
        ``sinkage`` [m] and which is trimmed ``trim`` [deg] bow-up about it.
        """
        dx = x - pivot_xz[0]
        dz = z - pivot_xz[1]
        cos, sin = np.cosd(trim), np.sind(trim)
        x_e = pivot_xz[0] + dx * cos + dz * sin
        z_e = pivot_xz[1] - sinkage - dx * sin + dz * cos
        return x_e, z_e

    def _cells(self, sinkage, trim, pivot_xz) -> dict[str, Vectorizable]:
        """Submerged fraction and the submerged part of each cell's vertical edge."""
        xa, za = self.to_earth(self._cell_x, self._cell_z_lo, sinkage, trim, pivot_xz)
        xb, zb = self.to_earth(self._cell_x, self._cell_z_hi, sinkage, trim, pivot_xz)
        f_sub, s_sub, _ = smooth_submergence_fraction(
            za, zb, 0.0, transition_thickness=self.waterline_smoothing
        )
        t_lo = s_sub - 0.5 * f_sub
        t_hi = s_sub + 0.5 * f_sub
        return {
            "f_sub": f_sub,
            "x_lo": xa + t_lo * (xb - xa),
            "z_lo": za + t_lo * (zb - za),
            "x_hi": xa + t_hi * (xb - xa),
            "z_hi": za + t_hi * (zb - za),
        }

    # ------------------------------------------------------------------
    # Hydrostatics and friction
    # ------------------------------------------------------------------
    def hydrostatics(
        self,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
    ) -> dict[str, Vectorizable]:
        """
        Displaced volume [m^3], centre of buoyancy (earth axes ``x``, ``z``) [m] and
        wetted surface [m^2] at an attitude.
        """
        cells = self._cells(sinkage, trim, pivot_xz)
        d_volume = 2.0 * self._cell_f * self._cell_dx * self._cell_dz * cells["f_sub"]
        volume = np.sum(d_volume)
        volume_safe = np.fmax(volume, 1e-12)
        x_mid = 0.5 * (cells["x_lo"] + cells["x_hi"])
        z_mid = 0.5 * (cells["z_lo"] + cells["z_hi"])

        d_area = 2.0 * _onp.sqrt(
            (self._cell_dx * self._cell_dz) ** 2
            + (self._cell_dfx * self._cell_dz) ** 2
            + (self._cell_dfz * self._cell_dx) ** 2
        )
        # Cells off the hull (no breadth anywhere) carry no shell.
        d_area = _onp.where(
            (self._cell_f > 0) | (_onp.abs(self._cell_dfz) > 0), d_area, 0.0
        )
        # Wetted length: strips count by how much of their largest section is wet.
        strip_volume = self._sum_groups(np.reshape(d_volume, (-1, 1)), len(self.stations) - 1)
        strip_scale = 1e-3 * float(
            _onp.max(
                (2.0 * self._cell_f * self._cell_dx * self._cell_dz)
                .reshape(len(self.stations) - 1, -1)
                .sum(axis=1)
            )
        )
        wetted_length = np.sum(
            np.reshape(_onp.diff(self.stations), (-1, 1))
            * strip_volume / (strip_volume + strip_scale)
        )
        return {
            "volume": volume,
            "wetted_length": wetted_length,
            "x_cb": np.sum(d_volume * x_mid) / volume_safe,
            "z_cb": np.sum(d_volume * z_mid) / volume_safe,
            "wetted_surface": self.wetted_surface_scale * np.sum(d_area * cells["f_sub"]),
        }

    def viscous_resistance(
        self,
        velocity: Vectorizable,
        density: Vectorizable,
        kinematic_viscosity: Vectorizable,
        wetted_surface: Vectorizable,
        wetted_length: Vectorizable | None = None,
    ) -> Vectorizable:
        """
        ITTC-57 friction on the wetted surface, ``((1 + k) C_F + dC_F) q S``, with
        the Reynolds number on the wetted length (the overall length if not given),
        as boatmath's ``thinship::viscous_resistance_for``.
        """
        if wetted_length is None:
            wetted_length = self.length
        reynolds = np.fmax(velocity * wetted_length / kinematic_viscosity, 1e4)
        c_v = (1.0 + self.form_factor) * ittc57_cf(reynolds) + self.roughness_delta_cf
        return c_v * 0.5 * density * velocity**2 * wetted_surface

    def _end_station_submergence(
        self,
        x_station: float,
        sinkage: Vectorizable,
        trim: Vectorizable,
        pivot_xz: Sequence[float],
    ) -> dict[str, Vectorizable]:
        """Submerged waterplane bands of an end station ``x = x_station``."""
        x_earth, _ = self.to_earth(x_station, 0.0, sinkage, trim, pivot_xz)
        xa, za = self.to_earth(x_station, self._transom_z_lo, sinkage, trim, pivot_xz)
        xb, zb = self.to_earth(x_station, self._transom_z_hi, sinkage, trim, pivot_xz)
        f_sub, s_sub, h_mid = smooth_submergence_fraction(
            za, zb, 0.0, transition_thickness=self.waterline_smoothing
        )
        t_lo, t_hi = s_sub - 0.5 * f_sub, s_sub + 0.5 * f_sub
        return {
            "x_earth": x_earth,
            "f_sub": f_sub,
            "h_mid": h_mid,
            "x_lo": xa + t_lo * (xb - xa),
            "x_hi": xa + t_hi * (xb - xa),
            "z_lo": za + t_lo * (zb - za),
            "z_hi": za + t_hi * (zb - za),
        }

    # ------------------------------------------------------------------
    # Transom
    # ------------------------------------------------------------------
    def transom(
        self,
        velocity: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
    ) -> dict[str, Vectorizable]:
        """
        The immersed transom at an attitude: its ``area`` [m^2, both sides], its
        first moment about the waterline ``depth_moment`` [m^3], its
        equivalent-rectangle ``depth`` [m], its ``dryness`` (0 wet .. 1 dry, by
        transom Froude number), the closing ``hollow_length`` [m] and the
        ``closure_strength`` (the fraction of the transom's flux the hollow
        closes: ``dryness`` for the ``"auto"`` closure, else 1).
        """
        end = self._end_station_submergence(self.stations[-1], sinkage, trim, pivot_xz)
        f_sub = end["f_sub"]
        d_area = 2.0 * self._transom_f * (self._transom_z_hi - self._transom_z_lo) * f_sub
        area = np.sum(d_area)
        depth_moment = np.sum(d_area * end["h_mid"])
        # Equivalent rectangle on the widest *immersed* half-breadth (boatmath
        # `SectionalHull::detect_transom`), smoothly.
        immersed = self._transom_f * f_sub
        # (a smooth maximum, so numeric and symbolic evaluations agree exactly)
        softness = 0.02 * max(self._transom_half_beam, 1e-6)
        peak = softness * np.log(np.sum(np.exp(immersed / softness)))
        depth = area / (2.0 * np.fmax(peak, 1e-9))

        nu = gravity / velocity**2
        depth_safe = np.fmax(depth, 0.0) + 1e-9
        froude_t = 1.0 / np.sqrt(nu * depth_safe)
        t = np.clip((froude_t - AUTO_WET_FN_T) / (AUTO_DRY_FN_T - AUTO_WET_FN_T), 0.0, 1.0)
        dryness = t * t * (3.0 - 2.0 * t)

        strength = 1.0
        if self.transom_closure == "none" or not self.has_transom:
            hollow = None
        elif self.transom_closure == "auto":
            # A wet transom is an open body (no closure fits the tank data best);
            # the closure fades in as the transom runs dry.
            hollow = DRY_HOLLOW_FRACTION * 2.0 * np.pi / nu
            strength = dryness
        elif self.transom_closure == "boatmath":
            wet = BALLISTIC_COEFF * np.sqrt(depth_safe / nu)
            dry = BOATMATH_DRY_HOLLOW_FRACTION * 2.0 * np.pi / nu
            hollow = wet + dryness * (dry - wet)
        else:
            hollow = float(self.transom_closure)
        return {
            "area": area,
            "depth_moment": depth_moment,
            "depth": depth,
            "dryness": dryness,
            "hollow_length": hollow,
            "closure_strength": strength,
            "f_sub": f_sub,
            "x_earth": end["x_earth"],
            # Submerged part of each band of the transom section, in earth z.
            "z_lo": end["z_lo"],
            "z_hi": end["z_hi"],
        }

    # ------------------------------------------------------------------
    # Wave-making
    # ------------------------------------------------------------------
    def wave_sources(
        self,
        velocity: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        n_hollow: int = 12,
    ) -> dict[str, Any]:
        """
        The hull's thin-ship sources at an attitude, as line elements (in earth
        axes) for ``hydrosandbox.library.kelvin_wave_resistance``.

        Every cell of the offset table carries the flux
        ``2 U (cos(trim) df/dx + sin(trim) df/dz)`` over its submerged centreplane
        area, on a vertical line at the cell's mid-station. An immersed transom is
        closed by boatmath's virtual appendage: the transom section carried aft
        over the hollow length, decaying as ``(1 - s)^2 (1 + 2 s)``.
        """
        cells = self._cells(sinkage, trim, pivot_xz)
        cos, sin = np.cosd(trim), np.sind(trim)
        flux = (
            2.0
            * velocity
            * cells["f_sub"]
            * (cos * self._cell_dfx * self._cell_dz + sin * self._cell_dfz * self._cell_dx)
        )
        n = len(self._cell_x)
        sets = {
            "xa": [cells["x_lo"]], "za": [cells["z_lo"]],
            "xb": [cells["x_hi"]], "zb": [cells["z_hi"]],
            "flux": [flux], "smear": [self._cell_dx * _onp.ones(n)],
        }

        dz = self._transom_z_hi - self._transom_z_lo
        if _onp.any(self._bow_f > 0.0):
            bow = self._end_station_submergence(self.stations[0], sinkage, trim, pivot_xz)
            sets["xa"].append(bow["x_lo"])
            sets["za"].append(bow["z_lo"])
            sets["xb"].append(bow["x_hi"])
            sets["zb"].append(bow["z_hi"])
            sets["flux"].append(2.0 * velocity * cos * bow["f_sub"] * self._bow_f * dz)
            sets["smear"].append(_onp.zeros(len(dz)))

        transom = self.transom(velocity, gravity, sinkage, trim, pivot_xz)
        hollow = transom["hollow_length"]
        if hollow is not None:
            s_edges = _onp.linspace(0.0, 1.0, n_hollow + 1)
            shape = (1.0 - s_edges) ** 2 * (1.0 + 2.0 * s_edges)
            for k in range(n_hollow):
                x_k = transom["x_earth"] + 0.5 * (s_edges[k] + s_edges[k + 1]) * hollow
                ones = _onp.ones(len(dz))
                sets["xa"].append(x_k * ones)
                sets["za"].append(transom["z_lo"])
                sets["xb"].append(x_k * ones)
                sets["zb"].append(transom["z_hi"])
                sets["flux"].append(
                    2.0
                    * velocity
                    * cos
                    * transom["f_sub"]
                    * self._transom_f
                    * dz
                    * (shape[k + 1] - shape[k])
                    * transom["closure_strength"]
                )
                sets["smear"].append((hollow / n_hollow) * ones)

        def stack(pieces):
            return np.concatenate([np.reshape(p, (-1, 1)) for p in pieces], axis=0)

        x_a, z_a, x_b, z_b = (stack(sets[k]) for k in ("xa", "za", "xb", "zb"))
        zeros = _onp.zeros((x_a.shape[0], 1))
        return {
            "xyz_a": (x_a, zeros, z_a),
            "xyz_b": (x_b, zeros, z_b),
            "source_flux": stack(sets["flux"]),
            "smear_length": stack(sets["smear"]),
        }

    # ------------------------------------------------------------------
    # Station-contracted transforms (fast wave amplitude; near-field force)
    # ------------------------------------------------------------------
    @staticmethod
    def _depth_average(z_lo, z_hi, kappa_row):
        """
        Mean of ``exp(kappa z)`` along each cell's submerged edge, (N, M):
        ``(e^{kappa z_hi} - e^{kappa z_lo}) / (kappa (z_hi - z_lo))``.
        """
        a = np.reshape(np.fmin(z_lo, 0.0), (-1, 1)) @ kappa_row
        b = np.reshape(np.fmin(z_hi, 0.0), (-1, 1)) @ kappa_row
        d = b - a
        small = np.fabs(d) < 1e-6
        d_safe = np.where(small, 1.0, d)
        return np.where(small, np.exp(0.5 * (a + b)), (np.exp(b) - np.exp(a)) / d_safe)

    @staticmethod
    def _sum_groups(matrix, n_groups: int):
        """Sum an (n_groups * n_per, M) matrix over consecutive row groups."""
        n_per = matrix.shape[0] // n_groups
        if np.is_casadi_type(matrix, recursive=False):
            import casadi as cas

            rows = _onp.repeat(_onp.arange(n_groups), n_per)
            cols = _onp.arange(n_groups * n_per)
            aggregate = cas.DM.triplet(
                rows.tolist(), cols.tolist(), [1.0] * len(cols), n_groups, n_groups * n_per
            )
            return aggregate @ matrix
        return _onp.asarray(matrix).reshape(n_groups, n_per, -1).sum(axis=1)

    @staticmethod
    def _scale_rows(column, matrix):
        """Multiply each row of an (N, M) matrix by the matching entry of a column."""
        column = np.reshape(column, (-1, 1))
        if np.is_casadi_type(column, recursive=False) or np.is_casadi_type(
            matrix, recursive=False
        ):
            import casadi as cas

            return cas.diag(column) @ matrix
        return column * matrix

    def _strips(self, velocity, gravity, sinkage, trim, pivot_xz, n_hollow: int = 12):
        """
        The hull, and its transom-closing hollow, as station strips of cells:
        per-cell slopes ``(df/dx)`` x area and submerged edges, per-strip earth
        ``x`` and width.
        """
        cells = self._cells(sinkage, trim, pivot_xz)
        cos, sin = np.cosd(trim), np.sind(trim)
        n_strips = len(self.stations) - 1
        x_mid = 0.5 * (self.stations[:-1] + self.stations[1:])
        x_strip, _ = self.to_earth(x_mid, 0.0 * x_mid, sinkage, trim, pivot_xz)
        bow = self._end_station_submergence(self.stations[0], sinkage, trim, pivot_xz)

        out = {
            "n_hull": n_strips,
            "x_hull": x_strip,
            "dx_hull": _onp.diff(self.stations),
            # d f / d x (aft) times the cell's submerged centreplane area [m^2]
            "slope_hull": cells["f_sub"]
            * (cos * self._cell_dfx * self._cell_dz + sin * self._cell_dfz * self._cell_dx),
            # half the cell's submerged volume [m^3]
            "half_volume": cells["f_sub"] * self._cell_f * self._cell_dx * self._cell_dz,
            "z_lo_hull": cells["z_lo"],
            "z_hi_hull": cells["z_hi"],
            "bow": bow,
            "n_bow": 0,
            "n_hollow": 0,
        }

        dz = self._transom_z_hi - self._transom_z_lo
        if _onp.any(self._bow_f > 0.0):
            out.update(
                n_bow=1,
                x_bow=np.reshape(bow["x_earth"], (1,)),
                dx_bow=0.0,
                slope_bow=cos * bow["f_sub"] * self._bow_f * dz,
                z_lo_bow=bow["z_lo"],
                z_hi_bow=bow["z_hi"],
            )

        transom = self.transom(velocity, gravity, sinkage, trim, pivot_xz)
        out["transom"] = transom
        hollow = transom["hollow_length"]
        if hollow is not None:
            s_edges = _onp.linspace(0.0, 1.0, n_hollow + 1)
            shape = (1.0 - s_edges) ** 2 * (1.0 + 2.0 * s_edges)
            # The hollow is a feature of the water surface behind the transom: it
            # carries the transom's immersed section straight aft (it does not
            # pitch with the hull).
            x_list, slope, z_lo, z_hi = [], [], [], []
            for k in range(n_hollow):
                x_list.append(
                    transom["x_earth"] + 0.5 * (s_edges[k] + s_edges[k + 1]) * hollow
                )
                slope.append(
                    cos * transom["f_sub"] * self._transom_f * dz * (shape[k + 1] - shape[k])
                    * transom["closure_strength"]
                )
                z_lo.append(transom["z_lo"])
                z_hi.append(transom["z_hi"])

            def stack(pieces):
                return np.concatenate([np.reshape(v, (-1, 1)) for v in pieces], axis=0)

            out.update(
                n_hollow=n_hollow,
                x_hollow=stack(x_list),
                dx_hollow=hollow / n_hollow,
                slope_hollow=stack(slope),
                z_lo_hollow=stack(z_lo),
                z_hi_hollow=stack(z_hi),
            )
        return out

    @staticmethod
    def _has_casadi(*values) -> bool:
        for v in values:
            if v is None:
                continue
            if isinstance(v, _onp.ndarray) and v.dtype != object:
                continue
            if np.is_casadi_type(v, recursive=True):
                return True
        return False

    def _contract_groups(self, weights, z_lo, z_hi, k_row, n_groups: int):
        """
        Compute ``_sum_groups(_scale_rows(weights, _depth_average(z_lo, z_hi, k_row)), n_groups)``.
        Uses a wet-only chunked NumPy fast path when numeric to avoid allocating
        the full dry+wet cell-by-wavenumber exponential matrix.
        """
        if self._has_casadi(weights, z_lo, z_hi, k_row):
            return self._sum_groups(
                self._scale_rows(weights, self._depth_average(z_lo, z_hi, k_row)),
                n_groups,
            )
        w = _onp.asarray(weights, dtype=float).ravel()
        zlo = _onp.minimum(_onp.asarray(z_lo, dtype=float).ravel(), 0.0)
        zhi = _onp.minimum(_onp.asarray(z_hi, dtype=float).ravel(), 0.0)
        k = _onp.asarray(k_row, dtype=float).reshape(1, -1)
        m = k.shape[1]
        n_per = len(w) // n_groups
        nz = _onp.flatnonzero(w != 0.0)
        if len(nz) == 0:
            return _onp.zeros((n_groups, m), dtype=float)

        def depth_average(rows):
            out = _onp.empty((len(rows), m), dtype=float)
            zl, zh = zlo[rows, None], zhi[rows, None]
            for s0 in range(0, m, 512):
                kc = k[:, s0 : s0 + 512]
                a, b = zl * kc, zh * kc
                d = b - a
                sm = _onp.abs(d) < 1e-6
                d_safe = _onp.where(sm, 1.0, d)
                out[:, s0 : s0 + 512] = _onp.where(
                    sm, _onp.exp(0.5 * (a + b)), (_onp.exp(b) - _onp.exp(a)) / d_safe
                )
            return out

        # Every cell's depth average, shared by whatever weights come next.
        rows, avg = _kernel(
            "depth_average", (zlo, zhi, k),
            lambda: (lambda r: (r, depth_average(r)))(_onp.arange(len(w))),
        )
        where = _onp.full(len(w), -1)
        where[rows] = _onp.arange(len(rows))
        pos = where[nz]
        missing = pos < 0
        # Each cell belongs to one group: a weighted sum of rows per group.
        def grouped(cells, rows_of_cells):
            out = _onp.zeros((n_groups, m), dtype=float)
            if len(cells) == 0:
                return out
            groups = cells // n_per  # ascending: cells are numbered strip by strip
            starts = _onp.flatnonzero(_onp.r_[True, groups[1:] != groups[:-1]])
            out[groups[starts]] = _onp.add.reduceat(w[cells, None] * rows_of_cells, starts, axis=0)
            return out

        hit = nz[~missing]
        out = grouped(hit, avg[pos[~missing]])
        if missing.any():  # (cells above the water only through smoothing)
            lost = nz[missing]
            out = out + grouped(lost, depth_average(lost))
        return out

    @staticmethod
    def _strip_transform(weights, x_strip, dx_strip, kx_row, sign: float):
        """
        ``sum_s weights[s, m] * sinc(kx dx_s / 2) * exp(sign * i kx x_s)`` over strips,
        as (real, imag) rows.
        """
        if not Hull._has_casadi(weights, x_strip, dx_strip, kx_row):
            w_arr = _onp.asarray(weights, dtype=float)
            x_col = _onp.asarray(x_strip, dtype=float).reshape(-1, 1)
            kx_arr = _onp.asarray(kx_row, dtype=float).reshape(1, -1)
            dx_arr = _onp.asarray(dx_strip, dtype=float)

            def kernel():
                phase = x_col @ kx_arr
                if dx_arr.ndim == 0:
                    half = (0.5 * float(dx_arr)) * kx_arr
                else:
                    half = 0.5 * dx_arr.reshape(-1, 1) @ kx_arr
                half_safe = _onp.where(_onp.fabs(half) < 1e-6, 1.0, half)
                smear = _onp.where(
                    _onp.fabs(half) < 1e-6, 1.0, _onp.sin(half_safe) / half_safe
                )
                return smear * _onp.cos(phase), smear * _onp.sin(phase)

            ker_c, ker_s = _kernel("strip_transform", (x_col, dx_arr, kx_arr), kernel)
            re = _onp.sum(w_arr * ker_c, axis=0, keepdims=True)
            im = sign * _onp.sum(w_arr * ker_s, axis=0, keepdims=True)
            return re, im
        x_col = np.reshape(x_strip, (-1, 1))
        phase = x_col @ kx_row
        if np.is_casadi_type(dx_strip, recursive=False) or _onp.ndim(dx_strip) == 0:
            half = 0.5 * (_onp.ones((x_col.shape[0], 1)) * dx_strip) @ kx_row
        else:
            half = 0.5 * _onp.reshape(dx_strip, (-1, 1)) @ kx_row
        half_safe = np.where(np.fabs(half) < 1e-6, 1.0, half)
        smear = np.where(np.fabs(half) < 1e-6, 1.0, np.sin(half_safe) / half_safe)
        weighted = weights * smear
        re = np.reshape(np.sum(weighted * np.cos(phase), axis=0), (1, -1))
        im = sign * np.reshape(np.sum(weighted * np.sin(phase), axis=0), (1, -1))
        return re, im

    def wave_amplitude(
        self,
        velocity: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        demihull_spacing: float | None = None,
    ):
        """
        The hull's source-flux transform ``sum Q exp(kappa z - i kx x)`` as a
        callable ``(kx, kappa, ky) -> (re, im)``, for the ``extra_source_amplitude``
        of ``kelvin_wave_resistance``.

        With ``demihull_spacing`` [m], this hull is one of a catamaran's two
        demihulls, centred ``spacing / 2`` either side of ``y = 0``: their two
        source systems add with the placement phases ``exp(-/+ i ky spacing / 2)``,
        i.e. a factor ``2 cos(ky spacing / 2)``, which carries their interference.

        Equivalent to ``wave_sources`` (same cells, same transom closure) but
        integrates each station's depth first, as boatmath's sectional kernel
        does, so each wavenumber costs one exponential per cell and one phase per
        station instead of a full line-segment transform per cell.
        """
        strips = self._strips(velocity, gravity, sinkage, trim, pivot_xz)
        base_symbolic = self._has_casadi(
            velocity, sinkage, trim, pivot_xz[0], pivot_xz[1]
        )

        def _eval_chunk(kx_c, kappa_c, ky_c=None):
            g_hull = self._contract_groups(
                strips["slope_hull"],
                strips["z_lo_hull"],
                strips["z_hi_hull"],
                kappa_c,
                strips["n_hull"],
            )
            re, im = self._strip_transform(
                g_hull, strips["x_hull"], strips["dx_hull"], kx_c, sign=-1.0
            )
            if strips["n_bow"] > 0:
                g_bow = self._contract_groups(
                    strips["slope_bow"],
                    strips["z_lo_bow"],
                    strips["z_hi_bow"],
                    kappa_c,
                    strips["n_bow"],
                )
                re_b, im_b = self._strip_transform(
                    g_bow, strips["x_bow"], strips["dx_bow"], kx_c, sign=-1.0
                )
                re, im = re + re_b, im + im_b
            if strips["n_hollow"] > 0:
                g_hollow = self._contract_groups(
                    strips["slope_hollow"],
                    strips["z_lo_hollow"],
                    strips["z_hi_hollow"],
                    kappa_c,
                    strips["n_hollow"],
                )
                re_h, im_h = self._strip_transform(
                    g_hollow, strips["x_hollow"], strips["dx_hollow"], kx_c, sign=-1.0
                )
                re, im = re + re_h, im + im_h
            if demihull_spacing is not None:
                if ky_c is None:
                    raise ValueError("A demihull pair's amplitude needs `ky`.")
                pair = 2.0 * np.cos(0.5 * demihull_spacing * ky_c)
                re, im = pair * re, pair * im
            return 2.0 * velocity * re, 2.0 * velocity * im

        def amplitude(kx_row, kappa_row, ky_row=None):
            kx_row = np.reshape(kx_row, (1, -1))
            kappa_row = np.reshape(kappa_row, (1, -1))
            ky_reshaped = None if ky_row is None else np.reshape(ky_row, (1, -1))
            m = kx_row.shape[1]
            is_sym = base_symbolic or self._has_casadi(kx_row, kappa_row, ky_reshaped)
            chunk = 256
            if is_sym and m > chunk:
                re_parts, im_parts = [], []
                for s0 in range(0, m, chunk):
                    s1 = min(s0 + chunk, m)
                    re_c, im_c = _eval_chunk(
                        kx_row[:, s0:s1],
                        kappa_row[:, s0:s1],
                        None if ky_reshaped is None else ky_reshaped[:, s0:s1],
                    )
                    re_parts.append(re_c)
                    im_parts.append(im_c)
                return np.concatenate(re_parts, axis=1), np.concatenate(im_parts, axis=1)
            return _eval_chunk(kx_row, kappa_row, ky_reshaped)

        return amplitude

    def _waterline_half_breadths(self, sinkage, trim, pivot_xz):
        """Half-breadth of each station at the water surface, (n_stations, 1)."""
        cos, sin = np.cosd(trim), np.sind(trim)
        # Hull-frame height of the water surface at each station: where
        # z_earth = z_p - sinkage - (x - x_p) sin + (z - z_p) cos vanishes.
        z_w = pivot_xz[1] + (
            sinkage + (self.stations - pivot_xz[0]) * sin - pivot_xz[1]
        ) / cos
        z_w_clip = np.reshape(np.clip(z_w, self.levels[0], self.levels[-1]), (-1, 1))
        n_x, n_z = self.half_breadths.shape
        offset = z_w_clip @ _onp.ones((1, n_z)) - _onp.ones((n_x, 1)) @ self.levels[None, :]
        big = 1e12
        below = _onp.concatenate([[big], _onp.diff(self.levels)])[None, :] * _onp.ones((n_x, 1))
        above = _onp.concatenate([_onp.diff(self.levels), [big]])[None, :] * _onp.ones((n_x, 1))
        hat = np.fmax(np.fmin(1.0 + offset / below, 1.0 - offset / above), 0.0)
        wl = np.reshape(np.sum(hat * self.half_breadths, axis=1), (-1, 1))
        clear = np.reshape(z_w - self.levels[0], (-1, 1))
        s_scale = max(float(self.waterline_smoothing), 1e-6)
        in_water = 0.5 * (1.0 + np.tanh(clear / s_scale))
        return wl * in_water

    def dynamic_force(
        self,
        velocity: Vectorizable,
        density: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        x_ref: Vectorizable | None = None,
        reference_velocity: float | None = None,
        level: float = 1,
        demihull_spacing: float | None = None,
    ) -> dict[str, Vectorizable]:
        """
        Thin-ship **dynamic sinkage force and trim moment**: the vertical force and
        pitching moment of the hull's own steady near-field pressure, a port of
        boatmath's ``thinship::squat``:

            F_up  = -(rho U^2 / 2 pi^2) PV iint d^2k [ A Re(q* q) - ((A - 1)/k) Re(w* q) ]
            M_pv  = the same with r, r_wl in place of the weights q, w
            M_res = -(4 rho U^2 nu / pi) int dlambda lambda / sqrt(lambda^2 - 1)
                        Im[ nu lambda^2 r* q - r_wl* q ]      on k = kx^2 / nu

        with ``A = (kx^2 + nu k) / (kx^2 - nu k)`` the free-surface image amplitude,
        ``q`` the transform of ``df/dx`` (sources include the transom's closing
        hollow; weights carry the transom step instead), ``w`` its waterline
        counterpart, and ``r`` the transform of ``d/dx[(x - x_ref) f]``. See that
        module, and ``docs/dynamic-squat-derivation.md`` in boatmath, for the theory.

        Parameters
        ----------
        x_ref : Earth-axes ``x`` of the moment reference [m] (default: the pivot).
        reference_velocity : Numeric speed for laying out the quadrature when
            ``velocity`` is symbolic.
        level : Quadrature refinement (panel counts scale with it; boatmath starts
            at 1 and doubles until converged).
        demihull_spacing : If given [m], the force and moment on a catamaran of two
            such demihulls this far apart, each in the other's pressure field as
            well as its own: every pair of hulls contributes with boatmath's
            placement factor ``cos(ky dy)``, so the bilinear forms scale by
            ``2 (1 + cos(ky spacing))``.

        Returns
        -------
        Dict with ``force_up`` [N] (negative: suction, the hull sinks),
        ``moment_bow_up`` [N m] about ``x_ref``, its ``moment_local`` and
        ``moment_wave`` parts, and ``moment_slope`` [N m / m]
        (``d(moment_bow_up)/dx_ref``).
        """
        v_ref = reference_velocity
        if v_ref is None:
            if np.is_casadi_type(velocity, recursive=False):
                raise ValueError(
                    "`velocity` is symbolic; pass a numeric `reference_velocity`."
                )
            v_ref = float(velocity)
        nu = gravity / velocity**2
        nu_ref = gravity / v_ref**2
        if x_ref is None:
            x_ref = pivot_xz[0]

        strips = self._strips(velocity, gravity, sinkage, trim, pivot_xz)
        n_hull = strips["n_hull"]
        cos = np.cosd(trim)
        # boatmath's frame for these transforms: X forward (bow at high X).
        big_x_hull = -np.reshape(strips["x_hull"], (-1, 1))
        arm_hull = big_x_hull + x_ref  # X - X_ref, with X_ref = -x_ref
        x_t_earth = strips["transom"]["x_earth"]
        arm_transom = -x_t_earth + x_ref
        x_b_earth = strips["bow"]["x_earth"]
        arm_bow = -x_b_earth + x_ref

        f_wl = self._waterline_half_breadths(sinkage, trim, pivot_xz)
        d_f_wl = -(f_wl[1:, :] - f_wl[:-1, :])  # d f / dX per strip, times dx
        f_wl_mid = 0.5 * (f_wl[1:, :] + f_wl[:-1, :]) * np.reshape(strips["dx_hull"], (-1, 1))
        f_wl_transom = f_wl[-1, :]
        f_wl_bow = f_wl[0, :]
        transom_f_sub = strips["transom"]["f_sub"]
        bow_f_sub = strips["bow"]["f_sub"]
        transom_dz = self._transom_z_hi - self._transom_z_lo

        def tile(matrix, repeats: int):
            if repeats == 1:
                return matrix
            if np.is_casadi_type(matrix, recursive=False):
                import casadi as cas

                return cas.repmat(matrix, 1, repeats)
            return _onp.tile(matrix, (1, repeats))

        def contract(k_row):
            """
            Everything that depends on the decay rate ``k`` alone: each station's
            depth integrals (boatmath's ``contract_z``). Done once per k-node and
            shared by every direction theta.
            """
            k_row = np.reshape(k_row, (1, -1))
            out = {
                # Weights of the real hull: -slope is d f / dX.
                "g_q": -self._contract_groups(
                    strips["slope_hull"],
                    strips["z_lo_hull"],
                    strips["z_hi_hull"],
                    k_row,
                    n_hull,
                ),
                "g_p": self._contract_groups(
                    strips["half_volume"],
                    strips["z_lo_hull"],
                    strips["z_hi_hull"],
                    k_row,
                    n_hull,
                ),
            }
            if strips["n_hollow"] > 0:
                out["g_h"] = -self._contract_groups(
                    strips["slope_hollow"],
                    strips["z_lo_hollow"],
                    strips["z_hi_hollow"],
                    k_row,
                    strips["n_hollow"],
                )
            out["z_t"] = self._contract_groups(
                cos * transom_f_sub * self._transom_f * transom_dz,
                strips["transom"]["z_lo"],
                strips["transom"]["z_hi"],
                k_row,
                1,
            )
            out["z_b"] = self._contract_groups(
                cos * bow_f_sub * self._bow_f * transom_dz,
                strips["bow"]["z_lo"],
                strips["bow"]["z_hi"],
                k_row,
                1,
            )
            return out

        def forms(
            kx_row, k_row, contraction=None, repeats: int = 1, include_q_im: bool = False
        ):
            """
            The bilinear forms q*q, w*q, r*q, r_wl*q at paired (kx, k) rows. With
            ``contraction`` (of the distinct k) and ``repeats``, the k-row is that
            contraction's k repeated ``repeats`` times.
            """
            kx_row = np.reshape(kx_row, (1, -1))
            if contraction is None:
                contraction = contract(k_row)
            numeric = not self._has_casadi(
                big_x_hull, arm_hull, d_f_wl, f_wl_mid, contraction["g_q"], contraction["g_p"], kx_row
            )
            if numeric and repeats > 1:
                g_q = g_p = None  # (contracted per k below, never tiled)
            else:
                g_q = tile(contraction["g_q"], repeats)
                g_p = tile(contraction["g_p"], repeats)

            if numeric:
                x_col = _onp.asarray(-big_x_hull, dtype=float).reshape(-1, 1)
                kx_arr = _onp.asarray(kx_row, dtype=float).reshape(1, -1)
                dx_col = _onp.asarray(strips["dx_hull"], dtype=float).reshape(-1, 1)

                def strip_kernel():
                    phase_h = x_col @ kx_arr
                    half_h = 0.5 * dx_col @ kx_arr
                    half_safe = _onp.where(_onp.fabs(half_h) < 1e-6, 1.0, half_h)
                    smear_h = _onp.where(
                        _onp.fabs(half_h) < 1e-6, 1.0, _onp.sin(half_safe) / half_safe
                    )
                    return smear_h * _onp.cos(phase_h), smear_h * _onp.sin(phase_h)

                ker_re, ker_im = _kernel("strip_kernel", (x_col, dx_col, kx_arr), strip_kernel)
                arm_col = _onp.asarray(arm_hull, dtype=float).reshape(-1, 1)
                df_row = _onp.asarray(d_f_wl, dtype=float).reshape(1, -1)
                fwm_row = _onp.asarray(f_wl_mid, dtype=float).reshape(1, -1)
                adf_row = (arm_col.ravel() * df_row.ravel()).reshape(1, -1)
                # The three weightings that don't depend on k, as one product.
                rows3 = _onp.vstack([df_row, fwm_row, adf_row])
                r_re, r_im = rows3 @ ker_re, rows3 @ ker_im
                w_re, pw_re, q1w_re = r_re[0:1], r_re[1:2], r_re[2:3]
                w_im, pw_im, q1w_im = r_im[0:1], r_im[1:2], r_im[2:3]
                if repeats > 1:
                    # The k-row is the contraction's k repeated per theta (theta-major):
                    # sum over strips as one batched product per k, with the kernel
                    # laid out (k, theta, strip), instead of tiling the weights.
                    n_s = x_col.shape[0]
                    n_kk = kx_arr.shape[1] // repeats
                    kt_re, kt_im = _kernel(
                        "strip_kernel_by_k", (x_col, dx_col, kx_arr),
                        lambda: tuple(
                            _onp.ascontiguousarray(k.reshape(n_s, repeats, n_kk).transpose(2, 1, 0))
                            for k in (ker_re, ker_im)
                        ),
                    )
                    g0_q = _onp.asarray(contraction["g_q"], dtype=float)
                    g0_p = _onp.asarray(contraction["g_p"], dtype=float)
                    stacked = _onp.stack([g0_q, g0_p, arm_col * g0_q], axis=-1).transpose(1, 0, 2)  # (k, s, 3)
                    t_re = _onp.matmul(kt_re, stacked)  # (k, theta, 3)
                    t_im = _onp.matmul(kt_im, stacked)

                    def flat(t, c):
                        return t[:, :, c].T.reshape(1, -1)  # theta-major row

                    q_re, p_re, q1_re = (flat(t_re, c) for c in range(3))
                    q_im, p_im, q1_im = (flat(t_im, c) for c in range(3))
                else:
                    g_q_arr = _onp.asarray(g_q, dtype=float)
                    g_p_arr = _onp.asarray(g_p, dtype=float)
                    g_q1_arr = arm_col * g_q_arr
                    q_re = _onp.sum(g_q_arr * ker_re, axis=0, keepdims=True)
                    q_im = _onp.sum(g_q_arr * ker_im, axis=0, keepdims=True)
                    p_re = _onp.sum(g_p_arr * ker_re, axis=0, keepdims=True)
                    p_im = _onp.sum(g_p_arr * ker_im, axis=0, keepdims=True)
                    q1_re = _onp.sum(g_q1_arr * ker_re, axis=0, keepdims=True)
                    q1_im = _onp.sum(g_q1_arr * ker_im, axis=0, keepdims=True)
            else:
                phase_h = np.reshape(-big_x_hull, (-1, 1)) @ kx_row
                half_h = 0.5 * _onp.reshape(strips["dx_hull"], (-1, 1)) @ kx_row
                half_safe = np.where(np.fabs(half_h) < 1e-6, 1.0, half_h)
                smear_h = np.where(np.fabs(half_h) < 1e-6, 1.0, np.sin(half_safe) / half_safe)
                ker_re = smear_h * np.cos(phase_h)
                ker_im = smear_h * np.sin(phase_h)

                def _t_mat(weights_mat):
                    return (
                        np.reshape(np.sum(weights_mat * ker_re, axis=0), (1, -1)),
                        np.reshape(np.sum(weights_mat * ker_im, axis=0), (1, -1)),
                    )

                def _t_row(weight_vec):
                    w_r = np.reshape(weight_vec, (1, -1))
                    return w_r @ ker_re, w_r @ ker_im

                q_re, q_im = _t_mat(g_q)
                p_re, p_im = _t_mat(g_p)
                q1_re, q1_im = _t_mat(self._scale_rows(arm_hull, g_q))
                w_re, w_im = _t_row(d_f_wl)
                pw_re, pw_im = _t_row(f_wl_mid)
                q1w_re, q1w_im = _t_row(arm_hull * d_f_wl)

            # Sources: the hull's, the bow step (-z_b), plus the closing hollow.
            src_re, src_im = q_re, q_im
            z_b = tile(contraction["z_b"], repeats)
            phase_b = x_b_earth * kx_row
            cos_b, sin_b = np.cos(phase_b), np.sin(phase_b)
            src_re, src_im = src_re - z_b * cos_b, src_im - z_b * sin_b
            if strips["n_hollow"] > 0:
                g_h = tile(contraction["g_h"], repeats)
                h_re, h_im = self._strip_transform(
                    g_h, strips["x_hollow"], strips["dx_hollow"], kx_row, sign=1.0
                )
                src_re, src_im = src_re + h_re, src_im + h_im

            # Weights: stern (0 -> +f_T) and bow (+f_B -> 0) end-station steps,
            # included unconditionally so near-field force stays smooth as breadth -> 0.
            z_t = tile(contraction["z_t"], repeats)
            phase_t = x_t_earth * kx_row
            cos_t, sin_t = np.cos(phase_t), np.sin(phase_t)
            q_re = q_re + z_t * cos_t - z_b * cos_b
            q_im = q_im + z_t * sin_t - z_b * sin_b
            q1_re = q1_re + arm_transom * z_t * cos_t - arm_bow * z_b * cos_b
            q1_im = q1_im + arm_transom * z_t * sin_t - arm_bow * z_b * sin_b
            w_re = w_re + f_wl_transom * cos_t - f_wl_bow * cos_b
            w_im = w_im + f_wl_transom * sin_t - f_wl_bow * sin_b
            q1w_re = q1w_re + arm_transom * f_wl_transom * cos_t - arm_bow * f_wl_bow * cos_b
            q1w_im = q1w_im + arm_transom * f_wl_transom * sin_t - arm_bow * f_wl_bow * sin_b

            r_re, r_im = p_re + q1_re, p_im + q1_im
            rw_re, rw_im = pw_re + q1w_re, pw_im + q1w_im

            def conj_times_source(a_re, a_im):
                return a_re * src_re + a_im * src_im, a_re * src_im - a_im * src_re

            qq, qq_im = conj_times_source(q_re, q_im)
            wq, wq_im = conj_times_source(w_re, w_im)
            rq_re, rq_im = conj_times_source(r_re, r_im)
            rwq_re, rwq_im = conj_times_source(rw_re, rw_im)
            if demihull_spacing is not None:
                k_all = np.reshape(k_row, (1, -1))
                if repeats > 1:
                    k_all = tile(k_all, repeats)
                ky = np.sqrt(np.fmax(k_all**2 - kx_row**2, 0.0))
                pair = 2.0 * (1.0 + np.cos(demihull_spacing * ky))
                qq, qq_im, wq, wq_im, rq_re, rq_im, rwq_re, rwq_im = (
                    pair * v
                    for v in (qq, qq_im, wq, wq_im, rq_re, rq_im, rwq_re, rwq_im)
                )
            if include_q_im:
                return qq, wq, rq_re, rq_im, rwq_re, rwq_im, qq_im, wq_im
            return qq, wq, rq_re, rq_im, rwq_re, rwq_im

        ### Local (principal-value) part: boatmath `Fleet::local_integral`.
        length = self.length
        draft = max(float(-self.levels[0]), 1e-6)
        gx, gw = _onp.polynomial.legendre.leggauss(8)

        def panels(edges):
            half = 0.5 * _onp.diff(edges)
            mid = 0.5 * (edges[1:] + edges[:-1])
            return (
                (mid[:, None] + half[:, None] * gx[None, :]).reshape(-1),
                (half[:, None] * gw[None, :]).reshape(-1),
            )

        k_lo = 0.05 / length
        k_max = max(1.0e4 / length, 60.0 / draft)
        n_log = max(int(round(12 * level)), 2)
        k_nodes, k_weights = panels(
            _onp.concatenate([[0.0], k_lo * (k_max / k_lo) ** (_onp.arange(n_log + 1) / n_log)])
        )
        q_cap = 400.0 / length
        theta_c = min(_onp.arccos(min(q_cap / k_max, 1.0)), _onp.pi / 2 - 1e-7)
        theta_nodes, theta_weights = panels(_onp.linspace(0.0, theta_c, max(int(round(24 * level)), 2) + 1))
        # The sliver (theta_c, pi/2), at its theta -> pi/2 limit.
        theta_nodes = _onp.append(theta_nodes, theta_c)
        theta_weights = _onp.append(theta_weights, _onp.pi / 2 - theta_c)

        force_integral = 0.0
        moment_integral = 0.0
        cos_t = _onp.cos(theta_nodes)
        k0_all = nu / cos_t**2  # pole, per theta (symbolic if velocity is)
        k0_ref = nu_ref / cos_t**2
        pole = forms(np.reshape(k0_all * cos_t, (1, -1)), np.reshape(k0_all, (1, -1)))
        two_nu_sec2 = 2.0 * nu / cos_t**2

        n_k, n_t = len(k_nodes), len(theta_nodes)
        kx_grid = (cos_t[:, None] * k_nodes[None, :]).reshape(1, -1)  # theta-major
        qq, wq, rq_re, _, rwq_re, _ = forms(
            kx_grid, k_nodes, contraction=contract(k_nodes), repeats=n_t
        )

        def per_theta(row):
            """A theta-major (1, n_t * n_k) row as an (n_k, n_t) matrix: column = theta."""
            if np.is_casadi_type(row, recursive=False):
                import casadi as cas

                return cas.reshape(row, n_k, n_t)  # column-major
            return _onp.asarray(row).reshape(n_t, n_k).T

        # (n_k, n_t) matrices: column = theta.
        qq, wq, rq_re, rwq_re = (per_theta(v) for v in (qq, wq, rq_re, rwq_re))
        k_col = k_nodes.reshape(-1, 1)
        wk_col = k_weights.reshape(-1, 1)
        ones_k = _onp.ones((n_k, 1))

        def row(v):
            return np.reshape(v, (1, -1))

        k0_row = row(k0_all)
        coef_row = row(two_nu_sec2)
        k_mat = k_col @ _onp.ones((1, n_t))
        denom = k_mat - ones_k @ k0_row
        eps_d = 1e-12 * _onp.maximum(k_mat, 1.0)
        inv_denom = denom / (denom * denom + eps_d * eps_d)
        h_f = (ones_k @ coef_row) * (k_mat * qq - wq)
        h_m = (ones_k @ coef_row) * (k_mat * rq_re - rwq_re)
        h0_f = coef_row * (k0_row * row(pole[0]) - row(pole[1]))
        h0_m = coef_row * (k0_row * row(pole[2]) - row(pole[4]))
        inner_f = np.reshape(
            np.sum((wk_col @ _onp.ones((1, n_t))) * (k_mat * qq + (h_f - ones_k @ h0_f) * inv_denom), axis=0),
            (1, -1),
        )
        inner_m = np.reshape(
            np.sum((wk_col @ _onp.ones((1, n_t))) * (k_mat * rq_re + (h_m - ones_k @ h0_m) * inv_denom), axis=0),
            (1, -1),
        )

        # Where the pole lies beyond the shared grid, extend the range for that theta.
        k_hi = _onp.where(k0_ref >= k_max, 4.0 * k0_ref, k_max)
        extension_f = [0.0] * n_t
        extension_m = [0.0] * n_t
        for i in _onp.nonzero(k_hi > k_max)[0]:
            n_ext = max(int(round(4 * level)), 1)
            ext_nodes, ext_weights = panels(
                k_max * (k_hi[i] / k_max) ** (_onp.arange(n_ext + 1) / n_ext)
            )
            e_qq, e_wq, e_rq, _, e_rwq, _ = forms(ext_nodes * cos_t[i], ext_nodes)
            ext_k = ext_nodes.reshape(1, -1)
            ext_w = ext_weights.reshape(1, -1)
            k0_i, coef_i = k0_all[i], two_nu_sec2[i]
            e_hf = coef_i * (ext_k * e_qq - e_wq)
            e_hm = coef_i * (ext_k * e_rq - e_rwq)
            ext_d = ext_k - k0_i
            eps_ext = 1e-12 * _onp.maximum(ext_k, 1.0)
            inv_ext_d = ext_d / (ext_d * ext_d + eps_ext * eps_ext)
            extension_f[i] = np.sum(ext_w * (ext_k * e_qq + (e_hf - h0_f[0, i]) * inv_ext_d))
            extension_m[i] = np.sum(ext_w * (ext_k * e_rq + (e_hm - h0_m[0, i]) * inv_ext_d))

        for i in range(n_t):
            diff_k = k_hi[i] - k0_all[i]
            eps_k = 1e-12 * np.fmax(k_hi[i], k0_all[i])
            log_term = 0.5 * np.log(
                (diff_k * diff_k + eps_k * eps_k) / (k0_all[i] * k0_all[i])
            )
            force_integral = force_integral + theta_weights[i] * (
                inner_f[0, i] + extension_f[i] + h0_f[0, i] * log_term
            )
            moment_integral = moment_integral + theta_weights[i] * (
                inner_m[0, i] + extension_m[i] + h0_m[0, i] * log_term
            )
        prefactor = -(density * velocity**2) / (2.0 * _onp.pi**2)
        force_up = prefactor * 4.0 * force_integral
        moment_local = prefactor * 4.0 * moment_integral

        ### Wave (residue) part of the moment: `Fleet::wave_moment_integral`.
        kx_cap = max(400.0 / length, 2.0 * nu_ref)
        theta_max = min(_onp.arccos(min(nu_ref / kx_cap, 1.0)), _onp.pi / 2 - 1e-7)
        wave_nodes, wave_weights = panels(_onp.linspace(0.0, theta_max, max(int(round(48 * level)), 2) + 1))
        lam = 1.0 / _onp.cos(wave_nodes)
        k_wave = np.reshape(nu * lam**2, (1, -1))
        _, _, _, rq_im, _, rwq_im, qq_im, wq_im = forms(
            np.reshape(nu * lam, (1, -1)), k_wave, include_q_im=True
        )
        wave_wt_row = (wave_weights * lam**2).reshape(1, -1)
        wave_integral = np.sum((k_wave * rq_im - rwq_im) * wave_wt_row)
        wave_slope_integral = np.sum((k_wave * qq_im - wq_im) * wave_wt_row)
        moment_wave = -(4.0 * density * velocity**2 * nu / _onp.pi) * wave_integral
        moment_wave_slope = -(4.0 * density * velocity**2 * nu / _onp.pi) * wave_slope_integral

        return {
            "force_up": force_up,
            "moment_bow_up": moment_local + moment_wave,
            "moment_local": moment_local,
            "moment_wave": moment_wave,
            "moment_slope": force_up + moment_wave_slope,
        }

    # ------------------------------------------------------------------
    # Interaction with things near the hull (foils, propellers)
    # ------------------------------------------------------------------
    def sections(
        self,
        velocity: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
    ) -> dict[str, Vectorizable]:
        """
        The hull reduced to its stations at an attitude, as columns over the
        station strips (and, where there is one, the strips of the transom's closing
        hollow): earth-axes ``x``, strip length ``dx``, immersed section ``area``
        [m^2], its change along the strip ``d_area``, the depth of its centroid
        ``z_centroid`` (negative below the surface), the wetted ``half_beam`` at the
        waterline and the ``mean_depth`` (area over beam) of the section.
        """
        strips = self._strips(velocity, gravity, sinkage, trim, pivot_xz)
        n_hull = strips["n_hull"]
        half_area_x_dx = self._sum_groups(np.reshape(strips["half_volume"], (-1, 1)), n_hull)
        z_mid = 0.5 * (strips["z_lo_hull"] + strips["z_hi_hull"])
        moment = self._sum_groups(np.reshape(strips["half_volume"] * z_mid, (-1, 1)), n_hull)
        dx = np.reshape(strips["dx_hull"], (-1, 1))
        area = 2.0 * half_area_x_dx / dx
        f_wl = self._waterline_half_breadths(sinkage, trim, pivot_xz)
        half_beam = 0.5 * (f_wl[1:, :] + f_wl[:-1, :])
        out = {
            "x": np.reshape(strips["x_hull"], (-1, 1)),
            "dx": dx,
            "area": area,
            "d_area": 2.0 * self._sum_groups(np.reshape(strips["slope_hull"], (-1, 1)), n_hull),
            "z_centroid": moment / (half_area_x_dx + 1e-12),
            "half_beam": half_beam,
            "mean_depth": 0.5 * area / (half_beam + 1e-6),
            "n_hull": n_hull,
        }
        if strips["n_hollow"] > 0:
            n_h = strips["n_hollow"]
            slope = np.reshape(strips["slope_hollow"], (-1, 1))
            z_h = 0.5 * (strips["z_lo_hollow"] + strips["z_hi_hollow"])
            weight = self._sum_groups(np.fabs(slope), n_h)
            out.update(
                x_hollow=np.reshape(strips["x_hollow"], (-1, 1)),
                d_area_hollow=2.0 * self._sum_groups(slope, n_h),
                z_hollow=self._sum_groups(np.fabs(slope) * np.reshape(z_h, (-1, 1)), n_h)
                / (weight + 1e-12),
            )
        return out

    def induced_velocity(
        self,
        x: Vectorizable,
        y: Vectorizable,
        z: Vectorizable,
        velocity: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        demihull_spacing: float | None = None,
        free_surface: Literal["zero_potential", "rigid_lid", "none"] = "zero_potential",
        sections: dict[str, Vectorizable] | None = None,
    ) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
        """
        The flow the hull's thickness induces at field points, in earth axes
        (``x`` aft, ``z`` up from the still water surface): ``(u, v, w)`` [m/s],
        columns over the points.

        Slender-body form of the thin-ship sources: each station strip is a point
        source of strength ``U dA`` (the change in immersed section area along it)
        at the section's centroid, softened over the section's own size, with an
        image above the surface: of opposite sign for ``"zero_potential"``
        (``phi = 0``, the high-speed limit used for the foils' own images) or the
        same sign for ``"rigid_lid"``. The transom's closing hollow is included, so
        the sources sum to zero. ``y`` is measured from the hull's centreplane; with
        ``demihull_spacing`` it is measured from the craft's centreline and both
        demihulls contribute.

        This is the local (double-body-like) flow only: the hull's wave system at
        the field points is not included.
        """
        if sections is None:
            sections = self.sections(velocity, gravity, sinkage, trim, pivot_xz)
        px, py, pz = (np.reshape(v, (-1, 1)) for v in (x, y, z))
        n_points = px.shape[0]
        image = {"zero_potential": -1.0, "rigid_lid": 1.0, "none": 0.0}[free_surface]

        groups = [(sections["x"], sections["z_centroid"], sections["d_area"],
                   sections["area"] / _onp.pi + 1e-8)]
        if "x_hollow" in sections:
            groups.append((sections["x_hollow"], sections["z_hollow"],
                           sections["d_area_hollow"], 1e-6))
        y_hulls = [0.0] if demihull_spacing is None else [-0.5 * demihull_spacing, 0.5 * demihull_spacing]

        u = v = w = 0.0
        for xs, zs, d_area, core2 in groups:
            n_s = xs.shape[0]
            ones_p, ones_s = _onp.ones((n_points, 1)), _onp.ones((1, n_s))
            rx = px @ ones_s - ones_p @ np.reshape(xs, (1, -1))
            strength = ones_p @ np.reshape(velocity * d_area, (1, -1)) / (4.0 * _onp.pi)
            core = ones_p @ np.reshape(0.25 * core2, (1, -1))
            for y_hull in y_hulls:
                ry = (py - y_hull) @ ones_s
                for sign, weight in ((1.0, 1.0), (-1.0, image)):
                    if weight == 0.0:
                        continue
                    rz = pz @ ones_s - sign * (ones_p @ np.reshape(zs, (1, -1)))
                    inv_r3 = weight * strength / (rx**2 + ry**2 + rz**2 + core) ** 1.5
                    u = u + np.reshape(np.sum(inv_r3 * rx, axis=1), (-1, 1))
                    v = v + np.reshape(np.sum(inv_r3 * ry, axis=1), (-1, 1))
                    w = w + np.reshape(np.sum(inv_r3 * rz, axis=1), (-1, 1))
        return u, v, w

    def wave_field(
        self,
        velocity: float,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        demihull_spacing: float | None = None,
        k_cap: float = 100.0,
        level: int = 1,
    ) -> "HullWaveField":
        """
        The hull's steady **wave field**: the wave part of its thin-ship flow, which
        ``induced_velocity`` leaves out. A port of the wave part of boatmath's
        ``thinship::nearfield``.

        The Kelvin source splits (Havelock) into a Rankine source minus its negative
        image, which is ``induced_velocity(free_surface="zero_potential")``, and a
        regular wave part. The wave part is a two-dimensional wavenumber integral of
        the hull's own source transform ``q``: ``k`` on a log grid, the direction
        ``theta`` by Gauss panels, the pole on the dispersion curve
        ``k0 = nu sec^2(theta)`` by subtraction (principal value), and the radiation
        condition as a half-residue on that curve, which puts the waves astern. It
        becomes a sum over wavenumber nodes,

            phi_x = sum_n Re(c_n exp(i kx_n X)) cos(ky_n Y) exp(-k_n depth),

        from which the returned ``HullWaveField`` gives the wave elevation and the
        wave-induced velocity anywhere at or below the surface.

        Parameters
        ----------
        velocity : Speed [m/s]; numeric (it lays out the wavenumber nodes).
        k_cap : Wavenumbers are tapered away between ``0.6 k_cap`` and ``k_cap``
            [1/m]. Features shorter than about ``2 pi / k_cap`` are smoothed; at a
            depth ``d`` everything above ``~30 / d`` is negligible anyway.
        level : Quadrature refinement (panel counts scale with it).
        """
        velocity = float(velocity)
        nu = gravity / velocity**2
        strips = self._strips(velocity, gravity, sinkage, trim, pivot_xz)
        groups = [
            (strips["slope_hull"], strips["z_lo_hull"], strips["z_hi_hull"],
             strips["n_hull"], strips["x_hull"], strips["dx_hull"])
        ]
        if strips["n_bow"] > 0:
            groups.append((strips["slope_bow"], strips["z_lo_bow"], strips["z_hi_bow"],
                           strips["n_bow"], strips["x_bow"], strips["dx_bow"]))
        if strips["n_hollow"] > 0:
            groups.append((strips["slope_hollow"], strips["z_lo_hollow"], strips["z_hi_hollow"],
                           strips["n_hollow"], strips["x_hollow"], strips["dx_hollow"]))

        def tile(matrix, repeats: int):
            if repeats == 1:
                return matrix
            if np.is_casadi_type(matrix, recursive=False):
                import casadi as cas

                return cas.repmat(matrix, 1, repeats)
            return _onp.tile(matrix, (1, repeats))

        def source_transform(kx_row, k_row, repeats: int = 1):
            """
            ``q = iint (df/dX) exp(-k depth) exp(-i kx X)`` in boatmath's frame (X
            forward), for the distinct ``k_row`` repeated ``repeats`` times against
            ``kx_row``: minus the conjugate of ``sum slope exp(k z - i kx x)``.
            """
            kx_row = _onp.reshape(kx_row, (1, -1))
            k_row = _onp.reshape(k_row, (1, -1))
            re = im = 0.0
            for slope, z_lo, z_hi, n, x, dx in groups:
                g = self._sum_groups(
                    self._scale_rows(slope, self._depth_average(z_lo, z_hi, k_row)), n
                )
                g_re, g_im = self._strip_transform(tile(g, repeats), x, dx, kx_row, sign=-1.0)
                re, im = re + g_re, im + g_im
            return -re, im

        return _wave_field_from_transform(
            lambda kx, k, ky, repeats: source_transform(kx, k, repeats),
            velocity=velocity,
            gravity=gravity,
            k_lo=0.05 / self.length,
            k_cap=k_cap,
            level=level,
            y_hulls=[0.0] if demihull_spacing is None
            else [-0.5 * demihull_spacing, 0.5 * demihull_spacing],
        )

    def pressure_load(
        self,
        perturbation_u,
        velocity: Vectorizable,
        density: Vectorizable,
        gravity: float = 9.80665,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        pivot_xz: Sequence[float] = (0.0, 0.0),
        x_ref: Vectorizable | None = None,
        z_ref: Vectorizable = 0.0,
        y_hull: float = 0.0,
        sections: dict[str, Vectorizable] | None = None,
    ) -> dict[str, Vectorizable]:
        """
        The load on the hull from an external flow disturbance, such as a foil's
        field: the linearized pressure ``p = -rho U u`` it brings to the hull's
        bottom, integrated.

        ``perturbation_u(x, y, z)`` returns the disturbance's streamwise velocity
        [m/s] at earth-axes points (columns). Each station's pressure is sampled at
        three points across its wetted beam, at the section's mean depth, and acts
        upward on the beam's plan area and aft on the growth of the section's area
        along the strip. ``y_hull`` places the hull off the craft's centreline.

        Returns ``force_up`` and ``force_aft`` [N] and ``moment_bow_up`` [N m] about
        ``(x_ref, z_ref)`` (earth axes; default the pivot's x).
        """
        if sections is None:
            sections = self.sections(velocity, gravity, sinkage, trim, pivot_xz)
        if x_ref is None:
            x_ref = pivot_xz[0]
        xs, half_beam, depth = sections["x"], sections["half_beam"], sections["mean_depth"]
        n_s = xs.shape[0]
        nodes = (-_onp.sqrt(0.6), 0.0, _onp.sqrt(0.6))
        gauss = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)
        pressure = 0.0
        for node, weight in zip(nodes, gauss):
            u = perturbation_u(xs, y_hull + node * half_beam, -depth)
            pressure = pressure + weight * (-density * velocity * np.reshape(u, (-1, 1)))
        f_up = pressure * 2.0 * half_beam * sections["dx"]
        f_aft = pressure * sections["d_area"]
        return {
            "force_up": np.sum(f_up),
            "force_aft": np.sum(f_aft),
            "moment_bow_up": np.sum((-depth - z_ref) * f_aft - (xs - x_ref) * f_up),
            "pressure": pressure,
        }

    def dynamic_force_table(
        self,
        velocity: float,
        density: float,
        gravity: float = 9.80665,
        sinkage_range: tuple[float, float] = (-0.15, 0.05),
        trim_range: tuple[float, float] = (-2.0, 4.0),
        n_sinkage: int = 9,
        n_trim: int = 13,
        pivot_xz: Sequence[float] | None = None,
        demihull_spacing: float | None = None,
        level: float = 1,
        jobs: int | None = 1,
    ) -> "DynamicForceTable":
        """
        Tabulate ``dynamic_force`` over a grid of attitudes at one speed, for reuse.

        The hull's dynamic sinkage force and trim moment depend only on its own
        attitude and speed, not on anything attached to it. Inside a design
        optimization they are therefore the same smooth two-variable function at
        every iteration, and by far the most expensive thing to rebuild and
        differentiate. The returned table evaluates them by cubic B-spline
        interpolation (numerically or symbolically), at any pivot and moment
        reference; pass it to ``CraftBuildup(hull_dynamics=...)``.

        Parameters
        ----------
        sinkage_range, trim_range : The attitudes covered [m], [deg]. Evaluations
            outside are clamped to the edge, so make them generous.
        pivot_xz : The point whose sinkage the table is indexed by (default: the
            middle of the hull at the design waterline).
        jobs : Worker processes for filling the table (``None``: all cores).
        """
        from hydrosandbox.parallel import parallel_map

        if pivot_xz is None:
            pivot_xz = (0.5 * float(self.stations[0] + self.stations[-1]), 0.0)
        sinkages = _onp.linspace(sinkage_range[0], sinkage_range[1], n_sinkage)
        trims = _onp.linspace(trim_range[0], trim_range[1], n_trim)
        tasks = [
            (self, float(velocity), float(density), gravity, float(s), float(t),
             tuple(pivot_xz), level, demihull_spacing)
            for t in trims
            for s in sinkages
        ]
        values = parallel_map(_dynamic_force_point, tasks, jobs=jobs)
        return DynamicForceTable(
            sinkages=sinkages,
            trims=trims,
            force=_onp.array([v[0] for v in values]),
            moment=_onp.array([v[1] for v in values]),
            moment_slope=_onp.array([v[2] for v in values]),
            pivot_xz=tuple(float(v) for v in pivot_xz),
            velocity=float(velocity),
        )

    def default_wave_length_scales(
        self, demihull_spacing: float | None = None
    ) -> tuple[float, float, float]:
        """Default ``(y_extent, x_extent, min_depth)`` [m] for wave quadrature."""
        level_spacing = float(_onp.min(_onp.diff(self.levels)))
        beam = 2.0 * float(_onp.max(self.half_breadths)) + (demihull_spacing or 0.0)
        return (
            max(beam, 1.0),
            1.5 * self.length,
            min(0.05, max(0.5 * level_spacing, 0.01)),
        )

    def wave_amplitude_table(
        self,
        velocity: float,
        gravity: float = 9.80665,
        length_scales: tuple[float, float, float] | None = None,
        sinkage_range: tuple[float, float] = (-0.15, 0.05),
        trim_range: tuple[float, float] = (-2.0, 4.0),
        n_sinkage: int = 9,
        n_trim: int = 13,
        pivot_xz: Sequence[float] | None = None,
        demihull_spacing: float | None = None,
        phase_per_panel: float = 2.0,
        nodes_per_panel: int = 6,
        jobs: int | None = 1,
    ) -> "WaveAmplitudeTable":
        """
        Tabulate ``wave_amplitude`` over a grid of attitudes at one speed and
        quadrature layout, for reuse in ``CraftBuildup(hull_wave_amplitude=...)``.
        """
        from hydrosandbox.library.wave_resistance import wave_quadrature_nodes
        from hydrosandbox.parallel import parallel_map

        if length_scales is None:
            length_scales = self.default_wave_length_scales(demihull_spacing)
        min_d = max(float(length_scales[2]), 1e-3)
        y_ext = max(float(length_scales[0]), min_d)
        x_ext = max(float(length_scales[1]), min_d)
        vel = float(velocity)
        nu = gravity / vel**2
        sigma, _ = wave_quadrature_nodes(
            nu=nu,
            y_extent=y_ext,
            x_extent=x_ext,
            min_depth=min_d,
            phase_per_panel=phase_per_panel,
            nodes_per_panel=nodes_per_panel,
        )
        kappa = (nu + sigma**2).reshape(1, -1)
        kx = _onp.sqrt(nu * kappa)
        ky = sigma.reshape(1, -1) * _onp.sqrt(kappa)

        if pivot_xz is None:
            pivot_xz = (0.5 * float(self.stations[0] + self.stations[-1]), 0.0)
        pivot_tuple = (float(pivot_xz[0]), float(pivot_xz[1]))
        sinkages = _onp.linspace(sinkage_range[0], sinkage_range[1], n_sinkage)
        trims = _onp.linspace(trim_range[0], trim_range[1], n_trim)
        tasks = [
            (self, vel, gravity, float(s), trims, pivot_tuple, demihull_spacing, kx, kappa, ky)
            for s in sinkages
        ]
        rows = parallel_map(_wave_amplitude_row, tasks, jobs=jobs)
        re_grid = _onp.array([r[0] for r in rows], dtype=float)
        im_grid = _onp.array([r[1] for r in rows], dtype=float)
        return WaveAmplitudeTable(
            sinkages=sinkages,
            trims=trims,
            re=re_grid,
            im=im_grid,
            kx=kx.ravel(),
            kappa=kappa.ravel(),
            pivot_xz=pivot_tuple,
            velocity=vel,
            length_scales=tuple(float(v) for v in length_scales),
        )


def _mesh_y_extent(v, tri, stations, levels):
    """
    Least and greatest ``y`` at which the line through each (station, level),
    parallel to ``y``, meets the triangle mesh (NaN where it misses).
    """
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    # Barycentric coordinates of each (x, z) query in each triangle's xz-projection.
    v0 = (b - a)[:, [0, 2]]
    v1 = (c - a)[:, [0, 2]]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = _onp.abs(det) > 1e-14
    inv = _onp.where(ok, 1.0 / _onp.where(ok, det, 1.0), 0.0)
    x_lo = _onp.minimum(a[:, 0], _onp.minimum(b[:, 0], c[:, 0]))
    x_hi = _onp.maximum(a[:, 0], _onp.maximum(b[:, 0], c[:, 0]))
    levels = _onp.asarray(levels, dtype=float)
    y_low = _onp.full((len(stations), len(levels)), _onp.nan)
    y_high = _onp.full((len(stations), len(levels)), _onp.nan)
    for i, xs in enumerate(stations):
        idx = _onp.nonzero((x_lo <= xs) & (x_hi >= xs) & ok)[0]
        if idx.size == 0:
            continue
        px = (xs - a[idx, 0])[:, None]
        pz = levels[None, :] - a[idx, 2][:, None]  # (triangles, levels)
        u = (px * v1[idx, 1][:, None] - pz * v1[idx, 0][:, None]) * inv[idx][:, None]
        w = (pz * v0[idx, 0][:, None] - px * v0[idx, 1][:, None]) * inv[idx][:, None]
        hit = (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
        y = a[idx, 1][:, None] + u * (b[idx, 1] - a[idx, 1])[:, None] + w * (c[idx, 1] - a[idx, 1])[:, None]
        any_hit = hit.any(axis=0)
        y_low[i] = _onp.where(any_hit, _onp.where(hit, y, _onp.inf).min(axis=0), _onp.nan)
        y_high[i] = _onp.where(any_hit, _onp.where(hit, y, -_onp.inf).max(axis=0), _onp.nan)
    return y_low, y_high


def _wave_amplitude_row(hull, velocity, gravity, sinkage, trims, pivot_xz, spacing, kx, kappa, ky):
    """Evaluate one sinkage row across all trims for ``Hull.wave_amplitude_table``."""
    re_row = _onp.empty((len(trims), kx.size), dtype=float)
    im_row = _onp.empty((len(trims), kx.size), dtype=float)
    for j, trim in enumerate(trims):
        amp = hull.wave_amplitude(
            velocity, gravity, sinkage, float(trim), pivot_xz, demihull_spacing=spacing
        )
        re_v, im_v = amp(kx, kappa, ky)
        re_row[j, :] = _onp.asarray(re_v, dtype=float).ravel()
        im_row[j, :] = _onp.asarray(im_v, dtype=float).ravel()
    return re_row, im_row


def _dynamic_force_point(hull, velocity, density, gravity, sinkage, trim, pivot_xz, level, spacing):
    """
    One table entry: (force_up, moment_bow_up about the pivot, and the moment's
    change per metre the reference is moved aft).
    """
    here = hull.dynamic_force(
        velocity,
        density,
        gravity,
        sinkage,
        trim,
        pivot_xz,
        x_ref=pivot_xz[0],
        level=level,
        demihull_spacing=spacing,
    )
    return (
        float(here["force_up"]),
        float(here["moment_bow_up"]),
        float(here["moment_slope"]),
    )


# ----------------------------------------------------------------------
# Tables for many hulls at once (a design family's nodes)
# ----------------------------------------------------------------------
_REBUILT: dict = {}


def _hull_spec(h) -> tuple:
    """What a worker needs to rebuild ``h`` (much lighter to send than the hull)."""
    return (
        _onp.asarray(h.stations, dtype=float), _onp.asarray(h.levels, dtype=float),
        _onp.asarray(h.half_breadths, dtype=float),
        dict(name=h.name, form_factor=h.form_factor, roughness_delta_cf=h.roughness_delta_cf,
             transom_closure=h.transom_closure, waterline_smoothing=h.waterline_smoothing),
    )


def _hulls_from_specs(specs) -> list:
    """The hulls of ``specs``, rebuilt once per worker process."""
    digest = hashlib.blake2b(digest_size=16)
    for st, lv, hb, kw in specs:
        for a in (st, lv, hb):
            digest.update(a.tobytes())
        digest.update(repr(sorted(kw.items())).encode())
    key = digest.digest()
    if key not in _REBUILT:
        _REBUILT.clear()
        _REBUILT[key] = [Hull(st, lv, hb, **kw) for st, lv, hb, kw in specs]
    return _REBUILT[key]


def _dynamic_force_point_batch(specs, velocity, density, gravity, sinkage, trim, pivots, level, spacing):
    """One attitude for every hull in turn, so they share its kernels."""
    return [
        _dynamic_force_point(h, velocity, density, gravity, sinkage, trim, p, level, spacing)
        for h, p in zip(_hulls_from_specs(specs), pivots)
    ]


def _wave_amplitude_point_batch(specs, velocity, gravity, sinkage, trim, pivots, spacing, kx, kappa, ky):
    """One attitude's amplitude spectrum for every hull in turn."""
    out = []
    for h, p in zip(_hulls_from_specs(specs), pivots):
        re_v, im_v = h.wave_amplitude(velocity, gravity, sinkage, trim, p, demihull_spacing=spacing)(kx, kappa, ky)
        out.append((_onp.asarray(re_v, dtype=float).ravel(), _onp.asarray(im_v, dtype=float).ravel()))
    return out


def dynamic_force_tables(
    hulls: Sequence["Hull"], velocity: float, density: float, gravity: float = 9.80665,
    sinkage_range: tuple[float, float] = (-0.15, 0.05), trim_range: tuple[float, float] = (-2.0, 4.0),
    n_sinkage: int = 9, n_trim: int = 13, pivot_xz: Sequence[float] | None = None,
    demihull_spacing: float | None = None, level: float = 1, jobs: int | None = 1,
) -> list["DynamicForceTable"]:
    """
    ``Hull.dynamic_force_table`` for many hulls at once: each worker takes an
    attitude and evaluates every hull there, so hulls on the same offset grid (a
    design family's nodes) share its expensive kernels.
    """
    from hydrosandbox.parallel import parallel_map

    pivots = [
        tuple(float(v) for v in pivot_xz) if pivot_xz is not None
        else (0.5 * float(h.stations[0] + h.stations[-1]), 0.0)
        for h in hulls
    ]
    specs = [_hull_spec(h) for h in hulls]
    sinkages = _onp.linspace(sinkage_range[0], sinkage_range[1], n_sinkage)
    trims = _onp.linspace(trim_range[0], trim_range[1], n_trim)
    tasks = [
        (specs, float(velocity), float(density), gravity, float(s), float(t), pivots, level, demihull_spacing)
        for t in trims
        for s in sinkages
    ]
    values = parallel_map(_dynamic_force_point_batch, tasks, jobs=jobs)
    return [
        DynamicForceTable(
            sinkages=sinkages, trims=trims,
            force=_onp.array([v[n][0] for v in values]),
            moment=_onp.array([v[n][1] for v in values]),
            moment_slope=_onp.array([v[n][2] for v in values]),
            pivot_xz=pivots[n], velocity=float(velocity),
        )
        for n in range(len(hulls))
    ]


def wave_amplitude_tables(
    hulls: Sequence["Hull"], velocity: float, gravity: float = 9.80665,
    length_scales: tuple[float, float, float] | None = None,
    sinkage_range: tuple[float, float] = (-0.15, 0.05), trim_range: tuple[float, float] = (-2.0, 4.0),
    n_sinkage: int = 9, n_trim: int = 13, pivot_xz: Sequence[float] | None = None,
    demihull_spacing: float | None = None, phase_per_panel: float = 2.0, nodes_per_panel: int = 6,
    jobs: int | None = 1,
) -> list["WaveAmplitudeTable"]:
    """``Hull.wave_amplitude_table`` for many hulls at once (see ``dynamic_force_tables``)."""
    from hydrosandbox.library.wave_resistance import wave_quadrature_nodes
    from hydrosandbox.parallel import parallel_map

    if length_scales is None:
        length_scales = hulls[0].default_wave_length_scales(demihull_spacing)
    min_d = max(float(length_scales[2]), 1e-3)
    y_ext = max(float(length_scales[0]), min_d)
    x_ext = max(float(length_scales[1]), min_d)
    vel = float(velocity)
    nu = gravity / vel**2
    sigma, _ = wave_quadrature_nodes(
        nu=nu, y_extent=y_ext, x_extent=x_ext, min_depth=min_d,
        phase_per_panel=phase_per_panel, nodes_per_panel=nodes_per_panel,
    )
    kappa = (nu + sigma**2).reshape(1, -1)
    kx = _onp.sqrt(nu * kappa)
    ky = sigma.reshape(1, -1) * _onp.sqrt(kappa)
    pivots = [
        tuple(float(v) for v in pivot_xz) if pivot_xz is not None
        else (0.5 * float(h.stations[0] + h.stations[-1]), 0.0)
        for h in hulls
    ]
    specs = [_hull_spec(h) for h in hulls]
    sinkages = _onp.linspace(sinkage_range[0], sinkage_range[1], n_sinkage)
    trims = _onp.linspace(trim_range[0], trim_range[1], n_trim)
    tasks = [
        (specs, vel, gravity, float(s), float(t), pivots, demihull_spacing, kx, kappa, ky)
        for s in sinkages
        for t in trims
    ]
    values = parallel_map(_wave_amplitude_point_batch, tasks, jobs=jobs)
    n_s, n_t, m = len(sinkages), len(trims), kx.size
    out = []
    for n in range(len(hulls)):
        re = _onp.array([v[n][0] for v in values]).reshape(n_s, n_t, m)
        im = _onp.array([v[n][1] for v in values]).reshape(n_s, n_t, m)
        out.append(WaveAmplitudeTable(
            sinkages=sinkages, trims=trims, re=re, im=im, kx=kx.ravel(), kappa=kappa.ravel(),
            pivot_xz=pivots[n], velocity=vel, length_scales=tuple(float(v) for v in length_scales),
        ))
    return out


class DynamicForceTable:
    """
    A hull's dynamic sinkage force and trim moment at one speed, tabulated over
    attitude (see ``Hull.dynamic_force_table``).
    """

    def __init__(self, sinkages, trims, force, moment, moment_slope, pivot_xz, velocity):
        import casadi as cas

        self.sinkages = _onp.asarray(sinkages, dtype=float)
        self.trims = _onp.asarray(trims, dtype=float)
        self.force = _onp.asarray(force, dtype=float)
        self.moment = _onp.asarray(moment, dtype=float)
        self.moment_slope = _onp.asarray(moment_slope, dtype=float)
        self.pivot_xz = pivot_xz
        self.velocity = velocity
        grid = [self.sinkages.tolist(), self.trims.tolist()]
        # Values are ordered with sinkage varying fastest, as CasADi expects.
        self._force = cas.interpolant("hull_dynamic_force", "bspline", grid, self.force.tolist())
        self._moment = cas.interpolant("hull_dynamic_moment", "bspline", grid, self.moment.tolist())
        self._slope = cas.interpolant(
            "hull_dynamic_moment_slope", "bspline", grid, self.moment_slope.tolist()
        )

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __setstate__(self, state):
        self.__init__(**state)

    def __call__(
        self,
        sinkage: Vectorizable,
        trim: Vectorizable,
        pivot_xz: Sequence[Vectorizable] | None = None,
        x_ref: Vectorizable | None = None,
    ) -> dict[str, Vectorizable]:
        """
        ``force_up`` [N] and ``moment_bow_up`` [N m] for a hull whose ``pivot_xz``
        has sunk by ``sinkage`` and which is trimmed ``trim`` [deg] about it, the
        moment taken about earth-axes ``x_ref`` (default: the pivot).
        """
        import casadi as cas

        x0, z0 = self.pivot_xz
        if pivot_xz is None:
            pivot_xz = self.pivot_xz
        if x_ref is None:
            x_ref = pivot_xz[0]
        cos, sin = np.cosd(trim), np.sind(trim)
        # The table's own pivot, carried by this pose: how far it has sunk, and where
        # it now is along the track.
        sinkage_table = sinkage + (x0 - pivot_xz[0]) * sin + (z0 - pivot_xz[1]) * (1.0 - cos)
        x_table = pivot_xz[0] + (x0 - pivot_xz[0]) * cos + (z0 - pivot_xz[1]) * sin
        s_clamped = np.clip(sinkage_table, self.sinkages[0], self.sinkages[-1])
        t_clamped = np.clip(trim, self.trims[0], self.trims[-1])
        symbolic = np.is_casadi_type(s_clamped, recursive=False) or np.is_casadi_type(
            t_clamped, recursive=False
        )
        point = cas.vertcat(s_clamped, t_clamped)
        force, moment, slope = self._force(point), self._moment(point), self._slope(point)
        if not symbolic:
            force, moment, slope = float(force), float(moment), float(slope)
        # The moment is exactly linear in its reference. Its slope is the force for a
        # hull that closes aft; with an immersed transom the wave part of the moment
        # adds to it (the closing hollow radiates but carries no pressure), so the
        # slope is tabulated rather than assumed.
        return {
            "force_up": force,
            "moment_bow_up": moment + slope * (x_ref - x_table),
        }


def _wave_field_from_transform(transform, velocity, gravity, k_lo, k_cap, level, y_hulls):
    """
    The wave part of the steady flow of a wave-maker, given the transform of its
    sources (see ``Hull.wave_field``, whose quadrature this is).

    ``transform(kx, k, ky, repeats) -> (re, im)`` returns
    ``q = iint (df/dX) exp(-k depth) exp(-i kx X)`` in boatmath's frame (X forward)
    as ``1 x M`` rows, at the wavenumbers ``kx``, ``ky`` (length ``M``) and ``k``
    tiled ``repeats`` times. ``y_hulls`` are the centreplanes the field is the sum
    over.
    """
    nu = gravity / velocity**2
    gx, gw = _onp.polynomial.legendre.leggauss(8)

    def panels(edges):
        half = 0.5 * _onp.diff(edges)
        mid = 0.5 * (edges[1:] + edges[:-1])
        return (
            (mid[:, None] + half[:, None] * gx[None, :]).reshape(-1),
            (half[:, None] * gw[None, :]).reshape(-1),
        )

    def taper(k):
        a = 0.6 * k_cap
        s_ = _onp.clip((k - a) / (k_cap - a), 0.0, 1.0)
        return _onp.where(k >= k_cap, 0.0, _onp.cos(0.5 * _onp.pi * s_) ** 2)

    ### Principal-value part: every (theta, k) node, and the pole's node per theta.
    n_log = 12 * level
    k_nodes, k_weights = panels(
        _onp.concatenate([[0.0], k_lo * (k_cap / k_lo) ** (_onp.arange(n_log + 1) / n_log)])
    )
    theta, theta_w = panels(_onp.linspace(0.0, _onp.pi / 2, 24 * level + 1))
    sec, sin_t, cos_t = 1.0 / _onp.cos(theta), _onp.sin(theta), _onp.cos(theta)
    k0 = nu * sec**2
    pref = 2.0 * velocity * nu / _onp.pi**2
    n_k, n_t = len(k_nodes), len(theta)

    d = k_nodes[None, :] - k0[:, None]  # (theta, k)
    live = _onp.abs(d) > 1e-9 * k0[:, None]
    inv_d = _onp.where(live, 1.0 / _onp.where(live, d, 1.0), 0.0)
    w_grid = (
        theta_w[:, None] * pref * sec[:, None]
        * (k_nodes * taper(k_nodes) * k_weights)[None, :] * inv_d
    )
    kx_grid = cos_t[:, None] * k_nodes[None, :]
    ky_grid = sin_t[:, None] * k_nodes[None, :]
    q_re, q_im = transform(kx_grid.reshape(-1), k_nodes, ky_grid.reshape(-1), n_t)
    w_flat = w_grid.reshape(1, -1)
    # c = (-i q) w
    sets = [(kx_grid.reshape(-1), ky_grid.reshape(-1), _onp.tile(k_nodes, n_t),
             q_im * w_flat, -q_re * w_flat)]

    pole = (taper(k0) > 0.0) & (k0 < k_cap)
    if _onp.any(pole):
        sum_w = _onp.sum(k_weights[None, :] * inv_d, axis=1)
        w0 = (
            theta_w * pref * sec * k0 * taper(k0)
            * (_onp.log(_onp.where(pole, (k_cap - k0) / k0, 1.0)) - sum_w)
        )[pole]
        q_re, q_im = transform((k0 * cos_t)[pole], k0[pole], (k0 * sin_t)[pole], 1)
        sets.append(((k0 * cos_t)[pole], (k0 * sin_t)[pole], k0[pole],
                     q_im * w0.reshape(1, -1), -q_re * w0.reshape(1, -1)))

    ### Half-residue on the dispersion curve: the radiated waves, astern.
    theta_r = min(_onp.arccos(_onp.sqrt(min(nu / k_cap, 1.0))), _onp.pi / 2 - 1e-7)
    rt, rw = panels(_onp.linspace(0.0, theta_r, 48 * level + 1))
    r_sec = 1.0 / _onp.cos(rt)
    r_k0 = nu * r_sec**2
    w_res = (rw * (-2.0 * velocity * nu / _onp.pi) * r_sec * r_k0 * taper(r_k0)).reshape(1, -1)
    q_re, q_im = transform(r_k0 * _onp.cos(rt), r_k0, r_k0 * _onp.sin(rt), 1)
    sets.append((r_k0 * _onp.cos(rt), r_k0 * _onp.sin(rt), r_k0, q_re * w_res, q_im * w_res))

    def row(pieces):
        return np.concatenate([np.reshape(v, (1, -1)) for v in pieces], axis=1)

    return HullWaveField(
        kx=_onp.concatenate([s_[0] for s_ in sets]),
        ky=_onp.concatenate([s_[1] for s_ in sets]),
        k=_onp.concatenate([s_[2] for s_ in sets]),
        c_re=row([s_[3] for s_ in sets]),
        c_im=row([s_[4] for s_ in sets]),
        velocity=velocity,
        gravity=gravity,
        y_hulls=y_hulls,
    )


def wave_field_from_source_amplitude(
    amplitude,
    velocity: float,
    gravity: float = 9.80665,
    length: float = 1.0,
    k_cap: float = 100.0,
    level: int = 1,
) -> "HullWaveField":
    """
    The wave field (see ``Hull.wave_field``) of any wave-maker that is symmetric
    about the centreplane ``y = 0``, from its source-flux transform
    ``amplitude(kx, kappa, ky) -> (re, im)`` of ``sum Q exp(kappa z - i kx x)``
    [m^3/s] (x aft): the same function ``kelvin_wave_resistance`` takes as
    ``extra_source_amplitude``, e.g. ``Hull.wave_amplitude`` or
    ``hydrosandbox.library.wave_resistance.element_source_amplitude``.
    ``length`` [m] sets the longest waves resolved.
    """
    velocity = float(velocity)

    def transform(kx, k, ky, repeats):
        kx, ky = (_onp.reshape(v, (1, -1)) for v in (kx, ky))
        k = _onp.tile(_onp.reshape(k, (1, -1)), (1, repeats))
        re, im = amplitude(kx, k, ky)
        scale = 1.0 / (2.0 * velocity)
        return -_onp.reshape(re, (1, -1)) * scale, _onp.reshape(im, (1, -1)) * scale

    return _wave_field_from_transform(
        transform, velocity, gravity, 0.05 / length, k_cap, level, y_hulls=[0.0]
    )



class HullWaveField:
    """
    The wave part of a hull's steady flow as a sum over wavenumber nodes (see
    ``Hull.wave_field``). Points are in earth axes: ``x`` aft, ``y`` from the craft's
    centreline, ``z`` up from the still water surface.
    """

    def __init__(self, kx, ky, k, c_re, c_im, velocity, gravity, y_hulls):
        self.kx, self.ky, self.k = (_onp.asarray(v, dtype=float).reshape(1, -1) for v in (kx, ky, k))
        self.c_re, self.c_im = c_re, c_im
        self.velocity, self.gravity = velocity, gravity
        self.y_hulls = list(y_hulls)

    def _terms(self, x, y, z):
        x, y, z = (np.reshape(v, (-1, 1)) for v in (x, y, z))
        n_points = x.shape[0]
        ones = _onp.ones((n_points, 1))
        # boatmath's frame: X forward, depth down.
        phase = (-x) @ self.kx
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        decay = np.exp(np.fmin(z, 0.0) @ self.k)
        c_re, c_im = ones @ self.c_re, ones @ self.c_im
        across = across_slope = 0.0
        for y_hull in self.y_hulls:
            arg = (y - y_hull) @ self.ky
            across = across + np.cos(arg)
            across_slope = across_slope - (ones @ self.ky) * np.sin(arg)
        phi_x = (c_re * cos_p - c_im * sin_p) * decay  # d(phi)/dX, per unit `across`
        phi = (c_re * sin_p + c_im * cos_p) * decay / (ones @ self.kx)
        return phi_x, phi, across, across_slope, ones

    #: Numeric evaluations are done this many points at a time, to bound memory
    #: (each point meets every wavenumber node).
    chunk = 32

    def _in_chunks(self, method, *coordinates):
        """Evaluate a numeric many-point request piecewise; None if not applicable."""
        if any(np.is_casadi_type(c, recursive=False) for c in coordinates) or np.is_casadi_type(
            self.c_re, recursive=False
        ):
            return None
        columns = [_onp.reshape(_onp.asarray(c, dtype=float), (-1, 1)) for c in coordinates]
        n_points = max(c.shape[0] for c in columns)
        if n_points <= self.chunk:
            return None
        columns = [c if c.shape[0] == n_points else _onp.full((n_points, 1), c[0, 0]) for c in columns]
        pieces = [
            method(*(c[i:i + self.chunk] for c in columns))
            for i in range(0, n_points, self.chunk)
        ]
        if isinstance(pieces[0], tuple):
            return tuple(_onp.concatenate([p[k] for p in pieces]) for k in range(len(pieces[0])))
        return _onp.concatenate(pieces)

    def elevation(self, x: Vectorizable, y: Vectorizable) -> Vectorizable:
        """Wave elevation [m], positive up: ``(U / g) phi_X`` on the still surface."""
        chunked = self._in_chunks(self.elevation, x, y)
        if chunked is not None:
            return chunked
        phi_x, _, across, _, _ = self._terms(x, y, 0.0 * np.reshape(x, (-1, 1)))
        return (self.velocity / self.gravity) * np.reshape(
            np.sum(phi_x * across, axis=1), (-1, 1)
        )

    def induced_velocity(
        self, x: Vectorizable, y: Vectorizable, z: Vectorizable
    ) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
        """
        Wave-induced flow ``(u, v, w)`` [m/s] at points at or below the surface: aft,
        to starboard, and up.
        """
        chunked = self._in_chunks(self.induced_velocity, x, y, z)
        if chunked is not None:
            return chunked
        phi_x, phi, across, across_slope, ones = self._terms(x, y, z)

        def total(matrix):
            return np.reshape(np.sum(matrix, axis=1), (-1, 1))

        u = -total(phi_x * across)  # X runs forward, x aft
        v = total(phi * across_slope)
        w = total((ones @ self.k) * phi * across)  # d/d(depth) = -k; z up = -depth
        return u, v, w


class WaveAmplitudeTable:
    """
    A hull's free-surface source-flux transform ``(re, im)`` at one speed and
    quadrature layout, tabulated over attitude using separable cubic B-splines
    (see ``Hull.wave_amplitude_table``).
    """

    def __init__(
        self,
        sinkages,
        trims,
        re,
        im,
        kx,
        kappa,
        pivot_xz,
        velocity,
        length_scales,
    ):
        import casadi as cas

        self.sinkages = _onp.asarray(sinkages, dtype=float)
        self.trims = _onp.asarray(trims, dtype=float)
        self.re = _onp.asarray(re, dtype=float)
        self.im = _onp.asarray(im, dtype=float)
        self.kx = _onp.asarray(kx, dtype=float).ravel()
        self.kappa = _onp.asarray(kappa, dtype=float).ravel()
        self.pivot_xz = tuple(float(v) for v in pivot_xz)
        self.velocity = float(velocity)
        self.length_scales = tuple(float(v) for v in length_scales)

        n_s, n_t, m = self.re.shape
        if self.im.shape != (n_s, n_t, m) or n_s != len(self.sinkages) or n_t != len(self.trims):
            raise ValueError("WaveAmplitudeTable grid shapes do not match.")
        if self.kx.size != m or self.kappa.size != m:
            raise ValueError("WaveAmplitudeTable wavenumber vector length mismatch.")

        self._basis_s = cas.interpolant(
            "hull_wave_basis_s",
            "bspline",
            [self.sinkages.tolist()],
            _onp.eye(n_s).ravel().tolist(),
        )
        self._basis_t = cas.interpolant(
            "hull_wave_basis_t",
            "bspline",
            [self.trims.tolist()],
            _onp.eye(n_t).ravel().tolist(),
        )
        packed = _onp.concatenate(
            [
                self.re.reshape(n_s * n_t, m, order="C"),
                self.im.reshape(n_s * n_t, m, order="C"),
            ],
            axis=1,
        )
        self._packed_np = packed
        self._packed_dm = cas.DM(packed)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def __setstate__(self, state):
        self.__init__(**state)

    def __call__(
        self,
        sinkage: Vectorizable,
        trim: Vectorizable,
        pivot_xz: Sequence[Vectorizable] | None = None,
    ):
        """
        Return a callable ``(kx, kappa, ky) -> (re, im)`` for ``extra_source_amplitude``
        of ``kelvin_wave_resistance`` at this attitude and pivot.
        """
        import casadi as cas

        x0, z0 = self.pivot_xz
        if pivot_xz is None:
            pivot_xz = self.pivot_xz
        cos, sin = np.cosd(trim), np.sind(trim)
        sinkage_table = (
            sinkage + (x0 - pivot_xz[0]) * sin + (z0 - pivot_xz[1]) * (1.0 - cos)
        )
        x_table = pivot_xz[0] + (x0 - pivot_xz[0]) * cos + (z0 - pivot_xz[1]) * sin
        dx_shift = x_table - x0
        s_clamped = np.clip(sinkage_table, self.sinkages[0], self.sinkages[-1])
        t_clamped = np.clip(trim, self.trims[0], self.trims[-1])
        symbolic = np.is_casadi_type(s_clamped, recursive=False) or np.is_casadi_type(
            t_clamped, recursive=False
        )
        n_s, n_t, m = self.re.shape
        if symbolic:
            ws = self._basis_s(s_clamped)
            wt = self._basis_t(t_clamped)
            w2d = cas.reshape(wt @ ws.T, 1, n_s * n_t)
            both = w2d @ self._packed_dm
            re_0, im_0 = both[:, :m], both[:, m:]
        else:
            ws = _onp.asarray(self._basis_s(float(s_clamped)), dtype=float).reshape(-1, 1)
            wt = _onp.asarray(self._basis_t(float(t_clamped)), dtype=float).reshape(-1, 1)
            w2d = (wt @ ws.T).ravel(order="F").reshape(1, -1)
            both = w2d @ self._packed_np
            re_0, im_0 = both[:, :m], both[:, m:]

        def amplitude(kx_row, kappa_row, ky_row=None):
            kx_r = np.reshape(kx_row, (1, -1))
            kappa_r = np.reshape(kappa_row, (1, -1))
            if kx_r.shape[1] != m:
                raise ValueError(
                    f"WaveAmplitudeTable expects {m} wavenumbers, got {kx_r.shape[1]} "
                    "(check velocity and wave_length_scales)."
                )
            if not np.is_casadi_type(kx_r, recursive=False) and not np.is_casadi_type(
                kappa_r, recursive=False
            ):
                if not (
                    _onp.allclose(_onp.asarray(kx_r, dtype=float).ravel(), self.kx, rtol=1e-5, atol=1e-8)
                    and _onp.allclose(_onp.asarray(kappa_r, dtype=float).ravel(), self.kappa, rtol=1e-5, atol=1e-8)
                ):
                    raise ValueError(
                        "WaveAmplitudeTable wavenumber grid mismatch "
                        "(check velocity and wave_length_scales)."
                    )
            if not np.is_casadi_type(dx_shift, recursive=False) and abs(float(dx_shift)) == 0.0:
                return re_0, im_0
            phi = dx_shift * kx_r
            c_p, s_p = np.cos(phi), np.sin(phi)
            return re_0 * c_p + im_0 * s_p, im_0 * c_p - re_0 * s_p

        return amplitude
