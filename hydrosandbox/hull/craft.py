from typing import Any, Literal, Sequence
import numpy as _onp
from aerosandbox import ExplicitAnalysis
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import HydrofoilAssembly
from hydrosandbox.hull.hull import Hull
from hydrosandbox.hydrodynamics.hydro_buildup import HydroBuildup
from hydrosandbox.library.wave_resistance import kelvin_wave_resistance
from hydrosandbox.performance import HydroOperatingPoint


def _stack(pieces: list) -> Vectorizable:
    """Stack scalars / lists / columns into one (N, 1) column (NumPy or CasADi)."""
    columns = []
    for piece in pieces:
        if isinstance(piece, (list, tuple)):
            if len(piece) == 0:
                continue
            # Numeric section polars come back as length-1 arrays; flatten them.
            piece = np.array(
                [
                    v
                    if np.is_casadi_type(v, recursive=False)
                    else float(_onp.reshape(_onp.asarray(v, dtype=float), -1)[0])
                    for v in piece
                ]
            )
        columns.append(np.reshape(piece, (-1, 1)))
    return np.concatenate(columns, axis=0)


class CraftBuildup(ExplicitAnalysis):
    """
    A hull and a hydrofoil assembly as one craft at speed: calm-water resistance with
    hull-foil wave interference, and the vertical-force and pitching-moment balance
    that sets sinkage and trim.

    The hull (``Hull``, thin-ship theory) and the assembly (``HydroBuildup``) share
    geometry axes (``+x`` aft, ``+z`` up, design waterline at ``z = 0``) and move as
    one rigid body: the centre of gravity sinks by ``sinkage`` [m] and the craft
    trims ``trim`` [deg, bow up] about it. Both may be ``asb.Opti`` variables;
    equilibrium is then two constraints,
    ``F_z_net == 0`` and ``M_y_net == 0``, alongside whatever is being designed.
    ``CraftBuildup.solve_equilibrium`` does exactly that for a fixed design.

    What is modelled:
    - Hull: hydrostatic buoyancy at the attitude, ITTC-57 friction on the wetted
      surface, thin-ship wave sources (with boatmath's transom closure).
    - Assembly: everything in ``HydroBuildup``, at the craft's trim and at the
      true depth of the foil beneath the trimmed waterplane.
    - Waves: the hull's sources and the assembly's lift dipoles and thickness
      sources radiate into one spectrum, so ``D_wave`` includes their interference
      (reported separately as ``D_wave_interference``).
    - Thrust, equal to the drag, along the hull's x-axis through ``xyz_thrust``.

    What is not: the hull's own dynamic lift, suction, sinkage and trim (boatmath's
    ``squat`` and 2D+t lift are not ported), the hull's near-field flow at the foil,
    and the foil's pressure field on the hull.
    """

    def __init__(
        self,
        hull: Hull,
        velocity: Vectorizable,
        mass: Vectorizable,
        xyz_cg: Sequence[Vectorizable],
        assembly: HydrofoilAssembly | None = None,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        water: Water | None = None,
        xyz_thrust: Sequence[Vectorizable] | None = None,
        model_size: str = "small",
        free_surface_mode: Literal[
            "high_froude", "froude_blended", "rigid_lid"
        ] = "high_froude",
        include_transom_drag: bool = False,
        include_interference_breakdown: bool = True,
        wave_length_scales: tuple[float, float, float] | None = None,
        wave_reference_velocity: float | None = None,
        buildup_options: dict[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        hull, assembly : The hull and (optionally) the foil assembly, in shared
            geometry axes. With no assembly this is the bare hull.
        velocity : Speed through the water [m/s].
        mass : All-up mass [kg].
        xyz_cg : Centre of gravity, in geometry axes [m]. Also the pivot of the
            attitude.
        sinkage : Downward displacement of the centre of gravity from its design
            position [m].
        trim : Bow-up rotation about the centre of gravity [deg].
        xyz_thrust : A point on the thrust line (which runs along the hull's
            x-axis), in geometry axes. If None, thrust acts through the centre of
            gravity along the direction of travel: no moment, no vertical force.
        include_transom_drag : Add the hydrostatic thrust a dry transom loses
            (``rho g`` times the transom's first moment, scaled by its dryness), as
            boatmath's ``study --transom-drag``.
        include_interference_breakdown : Also evaluate the hull's and the
            assembly's wave resistance alone (two more wave integrals), to report
            the interference between them.
        wave_length_scales : ``(y_extent, x_extent, min_depth)`` [m] for the wave
            quadrature. Defaults to the hull's length and the shallower of its
            level spacing and 5 cm.
        wave_reference_velocity : Numeric speed for sizing the wave quadrature
            when ``velocity`` is symbolic.
        buildup_options : Extra keyword arguments for ``HydroBuildup``.
        """
        super().__init__()
        self.hull = hull
        self.assembly = assembly
        self.velocity = velocity
        self.mass = mass
        self.xyz_cg = xyz_cg
        self.sinkage = sinkage
        self.trim = trim
        self.water = Water.seawater() if water is None else water
        self.xyz_thrust = xyz_thrust
        self.model_size = model_size
        self.free_surface_mode = free_surface_mode
        self.include_transom_drag = include_transom_drag
        self.include_interference_breakdown = include_interference_breakdown
        self.wave_length_scales = wave_length_scales
        self.wave_reference_velocity = wave_reference_velocity
        self.buildup_options = buildup_options or {}

    def _earth(self, x, z):
        return self.hull.to_earth(
            x, z, self.sinkage, self.trim, (self.xyz_cg[0], self.xyz_cg[2])
        )

    def _wave_resistance(self, element_sets: list[dict[str, Any]]) -> Vectorizable:
        """One wave integral over several element sets, all in earth axes."""
        if not element_sets:
            return 0.0

        def strengths(key):
            pieces = []
            for elements in element_sets:
                n = _stack([elements["xyz_a"][0]]).shape[0]
                pieces.append(elements[key] if key in elements else _onp.zeros(n))
            return _stack(pieces)

        scales = self.wave_length_scales
        if scales is None:
            level_spacing = float(_onp.min(_onp.diff(self.hull.levels)))
            beam = 2.0 * float(_onp.max(self.hull.half_breadths))
            scales = (
                max(beam, 1.0),
                1.5 * self.hull.length,
                min(0.05, max(0.5 * level_spacing, 0.01)),
            )
        water = self.water
        return kelvin_wave_resistance(
            xyz_a=tuple(_stack([e["xyz_a"][k] for e in element_sets]) for k in range(3)),
            xyz_b=tuple(_stack([e["xyz_b"][k] for e in element_sets]) for k in range(3)),
            source_flux=strengths("source_flux"),
            force_y=strengths("force_y"),
            force_z=strengths("force_z"),
            smear_length=strengths("smear_length"),
            velocity=self.velocity,
            density=water.density(),
            surface_z=0.0,
            gravity=water.gravity,
            length_scales=scales,
            reference_velocity=self.wave_reference_velocity,
        )

    def run(self) -> dict[str, Any]:
        """
        Compute the craft's resistance breakdown and its net vertical force and
        pitching moment about the centre of gravity at the given attitude.
        """
        hull, water = self.hull, self.water
        rho, g = water.density(), water.gravity
        velocity, trim = self.velocity, self.trim
        x_cg, z_cg = self.xyz_cg[0], self.xyz_cg[2]
        pivot = (x_cg, z_cg)
        cos, sin = np.cosd(trim), np.sind(trim)
        _, z_cg_earth = self._earth(x_cg, z_cg)

        def moment_about_cg(x_e, z_e, f_x, f_z):
            """Bow-up moment of an earth-axes force (aft, up) applied at a point."""
            return (z_e - z_cg_earth) * f_x - (x_e - x_cg) * f_z

        ### Hull
        statics = hull.hydrostatics(self.sinkage, trim, pivot)
        buoyancy_hull = rho * g * statics["volume"]
        D_hull_viscous = hull.viscous_resistance(
            velocity, rho, water.kinematic_viscosity(), statics["wetted_surface"]
        )
        transom = hull.transom(velocity, g, self.sinkage, trim, pivot)
        D_transom = (
            rho * g * transom["depth_moment"] * transom["dryness"]
            if self.include_transom_drag
            else 0.0
        )
        hull_sources = hull.wave_sources(velocity, g, self.sinkage, trim, pivot)

        F_z = -self.mass * g + buoyancy_hull
        M_y = moment_about_cg(statics["x_cb"], statics["z_cb"], 0.0, buoyancy_hull)

        ### Foil assembly, riding with the hull
        foil: dict[str, Any] | None = None
        foil_sources: dict[str, Any] | None = None
        D_foil = 0.0
        if self.assembly is not None:
            # HydroBuildup's free surface is flat in geometry axes, so place it at
            # the true depth of the main foil beneath the trimmed waterplane.
            lifting = self.assembly.foils or self.assembly.wings
            anchor = lifting[0].xsecs[0].xyz_le
            _, z_anchor_earth = self._earth(anchor[0], anchor[2])
            op_point = HydroOperatingPoint(
                water=water,
                velocity=velocity,
                alpha=trim,
                surface_z=anchor[2] - z_anchor_earth,
            )
            options = dict(
                model_size=self.model_size,
                free_surface_mode=self.free_surface_mode,
                wave_length_scales=self.wave_length_scales,
                wave_reference_velocity=self.wave_reference_velocity,
            )
            options.update(self.buildup_options)
            foil = HydroBuildup(
                assembly=self.assembly,
                op_point=op_point,
                xyz_ref=list(self.xyz_cg),
                include_wave_drag="elements_only",
                **options,
            ).run()
            D_foil = foil["D"]

            # Forces in geometry axes (x aft, z up), buoyancy included -> earth axes.
            f_g = foil["F_g_with_buoyancy"]
            f_x_earth = f_g[0] * cos + f_g[2] * sin
            f_z_earth = -f_g[0] * sin + f_g[2] * cos
            F_z = F_z + f_z_earth
            M_y = M_y + foil["M_g_with_buoyancy"][1]

            elements = foil["wave_elements"]
            if elements is not None and len(elements["source_flux"]) > 0:
                xa, za = self._earth(_stack([elements["xyz_a"][0]]), _stack([elements["xyz_a"][2]]))
                xb, zb = self._earth(_stack([elements["xyz_b"][0]]), _stack([elements["xyz_b"][2]]))
                foil_sources = {
                    "xyz_a": (xa, _stack([elements["xyz_a"][1]]), za),
                    "xyz_b": (xb, _stack([elements["xyz_b"][1]]), zb),
                    "source_flux": _stack([elements["source_flux"]]),
                    "force_y": _stack([elements["force_y"]]),
                    "force_z": _stack([elements["force_z"]]) * cos,
                    "smear_length": _stack([elements["smear_length"]]),
                }

        ### Waves: one spectrum for everything
        sets = [hull_sources] + ([foil_sources] if foil_sources is not None else [])
        D_wave = self._wave_resistance(sets)
        if self.include_interference_breakdown and foil_sources is not None:
            D_wave_hull = self._wave_resistance([hull_sources])
            D_wave_foil = self._wave_resistance([foil_sources])
            D_wave_interference = D_wave - D_wave_hull - D_wave_foil
        else:
            D_wave_hull = D_wave if foil_sources is None else None
            D_wave_foil = 0.0 if foil_sources is None else None
            D_wave_interference = 0.0 if foil_sources is None else None

        ### Drag, thrust, and their part in the balance
        D_hull = D_hull_viscous + D_transom
        D = D_hull + D_foil + D_wave
        # Hull friction and all wave drag are taken to act at the centre of buoyancy;
        # the foil's own drag is already in its moment.
        M_y = M_y + moment_about_cg(
            statics["x_cb"], statics["z_cb"], D_hull + D_wave, 0.0
        )
        if self.xyz_thrust is not None:
            thrust = D / cos
            x_t, z_t = self._earth(self.xyz_thrust[0], self.xyz_thrust[2])
            F_z = F_z + thrust * sin
            M_y = M_y + moment_about_cg(x_t, z_t, -thrust * cos, thrust * sin)

        return {
            "F_z_net": F_z,
            "M_y_net": M_y,
            "D": D,
            "P_drag": D * velocity,
            "D_hull_viscous": D_hull_viscous,
            "D_transom": D_transom,
            "D_foil": D_foil,
            "D_wave": D_wave,
            "D_wave_hull_alone": D_wave_hull,
            "D_wave_foil_alone": D_wave_foil,
            "D_wave_interference": D_wave_interference,
            "L_foil": foil["L"] if foil is not None else 0.0,
            "buoyancy_hull": buoyancy_hull,
            "displaced_volume_hull": statics["volume"],
            "wetted_surface_hull": statics["wetted_surface"],
            "x_cb_hull": statics["x_cb"],
            "z_cb_hull": statics["z_cb"],
            "transom": transom,
            "sinkage": self.sinkage,
            "trim": trim,
            "foil": foil,
        }

    @classmethod
    def solve_equilibrium(
        cls,
        *args,
        sinkage_guess: float = 0.0,
        trim_guess: float = 0.0,
        max_trim: float = 15.0,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Find the sinkage and trim at which a fixed craft is in vertical-force and
        pitching-moment balance, and return its results there (numeric).

        Takes the constructor's arguments (without ``sinkage`` and ``trim``).
        """
        import aerosandbox as asb

        hull = kwargs.get("hull", args[0] if args else None)
        depth_scale = float(hull.levels[-1] - hull.levels[0])
        opti = asb.Opti()
        sinkage = opti.variable(
            init_guess=sinkage_guess, lower_bound=-depth_scale, upper_bound=depth_scale
        )
        trim = opti.variable(
            init_guess=trim_guess, lower_bound=-max_trim, upper_bound=max_trim
        )
        res = cls(*args, sinkage=sinkage, trim=trim, **kwargs).run()
        weight = kwargs["mass"] * 9.80665
        opti.subject_to(
            [
                res["F_z_net"] / weight == 0,
                res["M_y_net"] / (weight * hull.length) == 0,
            ]
        )
        sol = opti.solve(verbose=verbose)
        return cls(
            *args, sinkage=float(sol(sinkage)), trim=float(sol(trim)), **kwargs
        ).run()
