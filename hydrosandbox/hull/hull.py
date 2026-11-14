"""
A displacement hull as a table of offsets, with the thin-ship (Michell) physics of
the ``boatmath`` project (crates ``hullgeom`` and ``thinship``) ported to
``aerosandbox.numpy`` so that a hull's attitude can be an optimization variable.
"""

from typing import Any, Literal, Sequence
import numpy as _onp
from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.library.free_surface import smooth_submergence_fraction

#: Free-fall hollow-length coefficient (boatmath ``thinship::BALLISTIC_COEFF``).
BALLISTIC_COEFF = float(_onp.sqrt(2.0))
#: Dry-transom hollow length, in transverse wavelengths (``DRY_HOLLOW_FRACTION``).
DRY_HOLLOW_FRACTION = 0.5
#: Transom Froude numbers between which the transom goes from wet to dry.
AUTO_WET_FN_T = 2.5
AUTO_DRY_FN_T = 3.5


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

    def __init__(
        self,
        stations: Sequence[float],
        levels: Sequence[float],
        half_breadths: Sequence[Sequence[float]],
        name: str = "Hull",
        form_factor: float = 0.0,
        roughness_delta_cf: float = 0.0,
        transom_closure: Literal["auto", "none"] | float = "auto",
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
        transom_closure : How the wave integral closes an immersed transom, as in
            boatmath's ``TransomClosure``: ``"auto"`` (ballistic hollow while the
            transom is wet, half a transverse wavelength once dry), ``"none"`` (an
            open body), or a fixed hollow length [m].
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
        **kwargs,
    ) -> "Hull":
        """
        Cut a triangle mesh (metres; ``z`` up, design waterline at ``z = 0``) into an
        offset table. The half-breadth at each (station, level) is the largest
        distance from the hull's centreplane at which the line through it, parallel
        to ``y``, meets the mesh; either one side or both sides may be meshed.

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
        # Cosine-clustered stations, pulled a hair inside the ends.
        s = 0.5 * (1.0 - _onp.cos(_onp.pi * _onp.linspace(0.0, 1.0, n_stations)))
        stations = x_min + (x_max - x_min) * (1e-6 + (1.0 - 2e-6) * s)
        if z_min < 0.0 < z_max:
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

        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        # Barycentric coordinates of each (x, z) query in each triangle's xz-projection.
        v0 = (b - a)[:, [0, 2]]
        v1 = (c - a)[:, [0, 2]]
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = _onp.abs(det) > 1e-14
        inv = _onp.where(ok, 1.0 / _onp.where(ok, det, 1.0), 0.0)

        y_low = _onp.full((n_stations, n_levels), _onp.nan)
        y_high = _onp.full((n_stations, n_levels), _onp.nan)
        for i, xs in enumerate(stations):
            px = xs - a[:, 0]
            near = (a[:, 0].clip(max=_onp.minimum(b[:, 0], c[:, 0])) <= xs) & (
                _onp.maximum(a[:, 0], _onp.maximum(b[:, 0], c[:, 0])) >= xs
            )
            idx = _onp.nonzero(near & ok)[0]
            if idx.size == 0:
                continue
            for j, zs in enumerate(levels):
                pz = zs - a[idx, 2]
                u = (px[idx] * v1[idx, 1] - pz * v1[idx, 0]) * inv[idx]
                w = (pz * v0[idx, 0] - px[idx] * v0[idx, 1]) * inv[idx]
                hit = (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1.0 + 1e-9)
                if _onp.any(hit):
                    k = idx[hit]
                    y = a[k, 1] + u[hit] * (b[k, 1] - a[k, 1]) + w[hit] * (c[k, 1] - a[k, 1])
                    y_low[i, j], y_high[i, j] = _onp.min(y), _onp.max(y)

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
        return cls(stations, levels, half_breadths, **kwargs)

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
        return cls.from_mesh(
            vertices, triangles, n_stations=n_stations, n_levels=n_levels,
            x_forward=True, centerplane_y=centerplane_y, **kwargs,
        )

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
            "wetted_surface": np.sum(d_area * cells["f_sub"]),
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
        transom Froude number) and the closing ``hollow_length`` [m].
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

        if self.transom_closure == "none" or not self.has_transom:
            hollow = None
        elif self.transom_closure == "auto":
            wet = BALLISTIC_COEFF * np.sqrt(depth_safe / nu)
            dry = DRY_HOLLOW_FRACTION * 2.0 * np.pi / nu
            hollow = wet + dryness * (dry - wet)
        else:
            hollow = float(self.transom_closure)
        return {
            "area": area,
            "depth_moment": depth_moment,
            "depth": depth,
            "dryness": dryness,
            "hollow_length": hollow,
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
            return (column @ _onp.ones((1, matrix.shape[1]))) * matrix
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
    def _strip_transform(weights, x_strip, dx_strip, kx_row, sign: float):
        """
        ``sum_s weights[s, m] * sinc(kx dx_s / 2) * exp(sign * i kx x_s)`` over strips,
        as (real, imag) rows.
        """
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

        def amplitude(kx_row, kappa_row, ky_row=None):
            kx_row = np.reshape(kx_row, (1, -1))
            kappa_row = np.reshape(kappa_row, (1, -1))
            g_hull = self._sum_groups(
                self._scale_rows(
                    strips["slope_hull"],
                    self._depth_average(strips["z_lo_hull"], strips["z_hi_hull"], kappa_row),
                ),
                strips["n_hull"],
            )
            re, im = self._strip_transform(
                g_hull, strips["x_hull"], strips["dx_hull"], kx_row, sign=-1.0
            )
            if strips["n_bow"] > 0:
                g_bow = self._sum_groups(
                    self._scale_rows(
                        strips["slope_bow"],
                        self._depth_average(
                            strips["z_lo_bow"], strips["z_hi_bow"], kappa_row
                        ),
                    ),
                    strips["n_bow"],
                )
                re_b, im_b = self._strip_transform(
                    g_bow, strips["x_bow"], strips["dx_bow"], kx_row, sign=-1.0
                )
                re, im = re + re_b, im + im_b
            if strips["n_hollow"] > 0:
                g_hollow = self._sum_groups(
                    self._scale_rows(
                        strips["slope_hollow"],
                        self._depth_average(
                            strips["z_lo_hollow"], strips["z_hi_hollow"], kappa_row
                        ),
                    ),
                    strips["n_hollow"],
                )
                re_h, im_h = self._strip_transform(
                    g_hollow, strips["x_hollow"], strips["dx_hollow"], kx_row, sign=-1.0
                )
                re, im = re + re_h, im + im_h
            if demihull_spacing is not None:
                if ky_row is None:
                    raise ValueError("A demihull pair's amplitude needs `ky`.")
                pair = 2.0 * np.cos(0.5 * demihull_spacing * np.reshape(ky_row, (1, -1)))
                re, im = pair * re, pair * im
            return 2.0 * velocity * re, 2.0 * velocity * im

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
        ``moment_bow_up`` [N m] about ``x_ref``, and its ``moment_local`` and
        ``moment_wave`` parts.
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

        def cols(matrix_or_col, m):
            """Broadcast an (n, 1) column across m wavenumber columns."""
            return np.reshape(matrix_or_col, (-1, 1)) @ _onp.ones((1, m))

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
            depth_hull = self._depth_average(strips["z_lo_hull"], strips["z_hi_hull"], k_row)
            out = {
                # Weights of the real hull: -slope is d f / dX.
                "g_q": -self._sum_groups(
                    self._scale_rows(strips["slope_hull"], depth_hull), n_hull
                ),
                "g_p": self._sum_groups(
                    self._scale_rows(strips["half_volume"], depth_hull), n_hull
                ),
            }
            if strips["n_hollow"] > 0:
                depth_hollow = self._depth_average(
                    strips["z_lo_hollow"], strips["z_hi_hollow"], k_row
                )
                out["g_h"] = -self._sum_groups(
                    self._scale_rows(strips["slope_hollow"], depth_hollow),
                    strips["n_hollow"],
                )
            depth_t = self._depth_average(
                strips["transom"]["z_lo"], strips["transom"]["z_hi"], k_row
            )
            out["z_t"] = np.reshape(
                np.sum(
                    self._scale_rows(cos * transom_f_sub * self._transom_f * transom_dz, depth_t),
                    axis=0,
                ),
                (1, -1),
            )
            depth_b = self._depth_average(
                strips["bow"]["z_lo"], strips["bow"]["z_hi"], k_row
            )
            out["z_b"] = np.reshape(
                np.sum(
                    self._scale_rows(cos * bow_f_sub * self._bow_f * transom_dz, depth_b),
                    axis=0,
                ),
                (1, -1),
            )
            return out

        def forms(kx_row, k_row, contraction=None, repeats: int = 1):
            """
            The bilinear forms q*q, w*q, r*q, r_wl*q at paired (kx, k) rows. With
            ``contraction`` (of the distinct k) and ``repeats``, the k-row is that
            contraction's k repeated ``repeats`` times.
            """
            kx_row = np.reshape(kx_row, (1, -1))
            m = kx_row.shape[1]
            if contraction is None:
                contraction = contract(k_row)
            g_q = tile(contraction["g_q"], repeats)
            g_p = tile(contraction["g_p"], repeats)

            def transform(weights):
                return self._strip_transform(
                    weights, -big_x_hull, strips["dx_hull"], kx_row, sign=1.0
                )  # exp(-i kx X) = exp(+i kx x_earth)

            q_re, q_im = transform(g_q)
            p_re, p_im = transform(g_p)
            q1_re, q1_im = transform(cols(arm_hull, m) * g_q)
            w_re, w_im = transform(cols(d_f_wl, m))
            pw_re, pw_im = transform(cols(f_wl_mid, m))
            q1w_re, q1w_im = transform(cols(arm_hull * d_f_wl, m))

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

            qq, _ = conj_times_source(q_re, q_im)
            wq, _ = conj_times_source(w_re, w_im)
            rq_re, rq_im = conj_times_source(r_re, r_im)
            rwq_re, rwq_im = conj_times_source(rw_re, rw_im)
            if demihull_spacing is not None:
                k_all = np.reshape(k_row, (1, -1))
                if repeats > 1:
                    k_all = tile(k_all, repeats)
                ky = np.sqrt(np.fmax(k_all**2 - kx_row**2, 0.0))
                pair = 2.0 * (1.0 + np.cos(demihull_spacing * ky))
                qq, wq, rq_re, rq_im, rwq_re, rwq_im = (
                    pair * v for v in (qq, wq, rq_re, rq_im, rwq_re, rwq_im)
                )
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
        k_grid = _onp.tile(k_nodes, n_t).reshape(1, -1)
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
        _, _, _, rq_im, _, rwq_im = forms(np.reshape(nu * lam, (1, -1)), k_wave)
        wave_integral = np.sum((k_wave * rq_im - rwq_im) * (wave_weights * lam**2).reshape(1, -1))
        moment_wave = -(4.0 * density * velocity**2 * nu / _onp.pi) * wave_integral

        return {
            "force_up": force_up,
            "moment_bow_up": moment_local + moment_wave,
            "moment_local": moment_local,
            "moment_wave": moment_wave,
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


def _dynamic_force_point(hull, velocity, density, gravity, sinkage, trim, pivot_xz, level, spacing):
    """
    One table entry: (force_up, moment_bow_up about the pivot, and the moment's
    change per metre the reference is moved aft).
    """
    def at(x_ref):
        return hull.dynamic_force(
            velocity, density, gravity, sinkage, trim, pivot_xz, x_ref=x_ref,
            level=level, demihull_spacing=spacing,
        )

    here, aft = at(pivot_xz[0]), at(pivot_xz[0] + 1.0)
    return (
        float(here["force_up"]),
        float(here["moment_bow_up"]),
        float(aft["moment_bow_up"]) - float(here["moment_bow_up"]),
    )


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
