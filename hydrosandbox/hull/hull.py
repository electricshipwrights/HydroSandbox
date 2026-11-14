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
        """
        self.name = name
        self.stations = _onp.asarray(stations, dtype=float)
        self.levels = _onp.asarray(levels, dtype=float)
        self.half_breadths = _onp.asarray(half_breadths, dtype=float)
        self.form_factor = form_factor
        self.roughness_delta_cf = roughness_delta_cf
        self.transom_closure = transom_closure

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

        ### Transom: the aft station's section, if it has any breadth.
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
        **kwargs,
    ) -> "Hull":
        """
        Cut a triangle mesh (metres; ``z`` up, design waterline at ``z = 0``) into an
        offset table. The half-breadth at each (station, level) is the largest
        ``|y|`` at which the line through it, parallel to ``y``, meets the mesh;
        either one side or both sides of the hull may be meshed.

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
        levels = z_min + (z_max - z_min) * _onp.linspace(1e-6, 1.0 - 1e-6, n_levels)

        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        # Barycentric coordinates of each (x, z) query in each triangle's xz-projection.
        v0 = (b - a)[:, [0, 2]]
        v1 = (c - a)[:, [0, 2]]
        det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = _onp.abs(det) > 1e-14
        inv = _onp.where(ok, 1.0 / _onp.where(ok, det, 1.0), 0.0)

        half_breadths = _onp.zeros((n_stations, n_levels))
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
                    half_breadths[i, j] = _onp.max(_onp.abs(y))
        return cls(stations, levels, half_breadths, **kwargs)

    @classmethod
    def from_boatmath_geometry(
        cls,
        geometry: dict[str, Any],
        hull_index: int = 0,
        n_stations: int = 41,
        n_levels: int = 25,
        tessellation: int = 48,
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
            x_forward=True, **kwargs,
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
        f_sub, s_sub, _ = smooth_submergence_fraction(za, zb, 0.0)
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
        return {
            "volume": volume,
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
    ) -> Vectorizable:
        """
        ITTC-57 friction on the wetted surface, ``((1 + k) C_F + dC_F) q S``, with
        the Reynolds number on the hull's overall length (boatmath
        ``thinship::viscous_resistance_for``).
        """
        reynolds = np.fmax(velocity * self.length / kinematic_viscosity, 1e4)
        c_v = (1.0 + self.form_factor) * ittc57_cf(reynolds) + self.roughness_delta_cf
        return c_v * 0.5 * density * velocity**2 * wetted_surface

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
        x_t = self.stations[-1]
        _, za = self.to_earth(x_t, self._transom_z_lo, sinkage, trim, pivot_xz)
        _, zb = self.to_earth(x_t, self._transom_z_hi, sinkage, trim, pivot_xz)
        f_sub, _, h_mid = smooth_submergence_fraction(za, zb, 0.0)
        d_area = 2.0 * self._transom_f * (self._transom_z_hi - self._transom_z_lo) * f_sub
        area = np.sum(d_area)
        depth_moment = np.sum(d_area * h_mid)
        depth = area / (2.0 * max(self._transom_half_beam, 1e-12))

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

        transom = self.transom(velocity, gravity, sinkage, trim, pivot_xz)
        hollow = transom["hollow_length"]
        if hollow is not None:
            s_edges = _onp.linspace(0.0, 1.0, n_hollow + 1)
            shape = (1.0 - s_edges) ** 2 * (1.0 + 2.0 * s_edges)
            x_t = self.stations[-1]
            dz = self._transom_z_hi - self._transom_z_lo
            for k in range(n_hollow):
                x_mid = x_t + 0.5 * (s_edges[k] + s_edges[k + 1]) * hollow
                xa, za = self.to_earth(x_mid, self._transom_z_lo, sinkage, trim, pivot_xz)
                xb, zb = self.to_earth(x_mid, self._transom_z_hi, sinkage, trim, pivot_xz)
                f_sub, s_sub, _ = smooth_submergence_fraction(za, zb, 0.0)
                t_lo, t_hi = s_sub - 0.5 * f_sub, s_sub + 0.5 * f_sub
                sets["xa"].append(xa + t_lo * (xb - xa))
                sets["za"].append(za + t_lo * (zb - za))
                sets["xb"].append(xa + t_hi * (xb - xa))
                sets["zb"].append(za + t_hi * (zb - za))
                sets["flux"].append(
                    2.0 * velocity * f_sub * self._transom_f * dz * (shape[k + 1] - shape[k])
                )
                sets["smear"].append(
                    (hollow / n_hollow) * _onp.ones(len(dz))
                )

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
