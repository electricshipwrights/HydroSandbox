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

    - The hull's own dynamic sinkage force and trim moment (``Hull.dynamic_force``,
      boatmath's ``thinship::squat``), when ``include_hull_dynamics``.

    What is not: the hull's planing lift (boatmath's 2D+t ``lift``), the hull's
    near-field flow at the foil, the foil's pressure field on the hull, and the
    foil's wave field in the hull's dynamic force.
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
        shaft_angle: Vectorizable = 0.0,
        point_masses: Sequence[tuple[Vectorizable, Sequence[Vectorizable]]] | None = None,
        model_size: str = "small",
        free_surface_mode: Literal[
            "high_froude", "froude_blended", "rigid_lid"
        ] = "high_froude",
        demihull_spacing: float | None = None,
        include_transom_drag: bool = False,
        include_hull_dynamics: bool = True,
        hull_dynamics: Any = None,
        hull_wave_amplitude: Any = None,
        pitch_rate: Vectorizable = 0.0,
        include_interference_breakdown: bool = True,
        wave_length_scales: tuple[float, float, float] | None = None,
        wave_reference_velocity: float | None = None,
        buildup_options: dict[str, Any] | None = None,
        thrust_deduction: Vectorizable = 0.0,
        squat_level: float = 1,
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
        mass, xyz_cg : Mass [kg] and centre of gravity [m, geometry axes] of
            everything not listed in ``point_masses``.
        point_masses : Further items as ``(mass, xyz)`` pairs (a motor, a battery,
            crew...). They add to the mass and shift the centre of gravity; the
            attitude's pivot is the combined centre of gravity.
        xyz_thrust : The point the thrust acts at (the propeller), in geometry
            axes. If None, thrust acts through the centre of gravity along the
            direction of travel: no moment, no vertical force.
        shaft_angle : Inclination of the thrust line to the hull's x-axis [deg],
            positive for a shaft running down and aft (thrust then has an upward
            component). The thrust is sized so its horizontal part equals the drag.
        thrust_deduction : Hull thrust-deduction fraction ``t`` in [0, 1).
            When ``xyz_thrust`` is set, propeller suction adds aft drag
            ``t * thrust * cos(trim + shaft_angle)`` at the hull's centre of
            buoyancy, so ``thrust = D / ((1 - t) * cos(trim + shaft_angle))``.
        demihull_spacing : If given [m], the craft is a catamaran: ``hull`` is one
            demihull, and there are two, centred this far apart either side of the
            craft's centreline ``y = 0``. Buoyancy, friction and transom drag
            double; the two hulls' waves interfere with each other (and with the
            foils'); and each hull's dynamic force includes the other's pressure
            field. ``mass`` is the whole craft's.
        include_transom_drag : Add the hydrostatic thrust a dry transom loses
            (``rho g`` times the transom's first moment, scaled by its dryness), as
            boatmath's ``study --transom-drag``.
        include_hull_dynamics : Include the thin-ship dynamic sinkage force and trim
            moment of the hull's own pressure field in the balance.
        hull_dynamics : A ``Hull.dynamic_force_table(...)`` for this speed (and
            demihull spacing). The hull's dynamic force is then interpolated from it
            instead of being integrated afresh, which makes each evaluation, and
            above all each ``Opti`` solve, far cheaper in time and memory.
        hull_wave_amplitude : A ``Hull.wave_amplitude_table(...)`` for this speed
            and ``wave_length_scales`` (and demihull spacing). The hull's free-surface
            source spectrum is then interpolated over ``(sinkage, trim)`` instead of
            being rebuilt across thousands of wavenumbers per iteration.
        pitch_rate : Bow-up pitch rate about the centre of gravity [rad/s]. Seen by
            the foils (quasi-steadily, as a change in local inflow), not by the hull.
            Used by ``run_with_stability`` to find the foils' pitch damping.
        squat_level : Quadrature refinement level passed to ``Hull.dynamic_force``.
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
        self.base_mass = mass
        self.base_xyz_cg = xyz_cg
        self.point_masses = list(point_masses or [])
        if self.point_masses:
            total = mass + sum(m for m, _ in self.point_masses)
            xyz_cg = [
                (mass * xyz_cg[k] + sum(m * xyz[k] for m, xyz in self.point_masses)) / total
                for k in range(3)
            ]
            mass = total
        self.mass = mass
        self.xyz_cg = xyz_cg
        self.shaft_angle = shaft_angle
        self.sinkage = sinkage
        self.trim = trim
        self.water = Water.seawater() if water is None else water
        self.xyz_thrust = xyz_thrust
        self.model_size = model_size
        self.free_surface_mode = free_surface_mode
        self.demihull_spacing = demihull_spacing
        self.include_transom_drag = include_transom_drag
        self.include_hull_dynamics = include_hull_dynamics
        self.hull_dynamics = hull_dynamics
        self.hull_wave_amplitude = hull_wave_amplitude
        self.pitch_rate = pitch_rate
        self.include_interference_breakdown = include_interference_breakdown
        self.wave_length_scales = wave_length_scales
        self.wave_reference_velocity = wave_reference_velocity
        self.buildup_options = buildup_options or {}
        self.thrust_deduction = thrust_deduction
        self.squat_level = squat_level

    def _earth(self, x, z):
        return self.hull.to_earth(
            x, z, self.sinkage, self.trim, (self.xyz_cg[0], self.xyz_cg[2])
        )

    def _wave_resistance(
        self, element_sets: list[dict[str, Any]], hull_amplitude=None
    ) -> Vectorizable:
        """
        One wave integral over element sets (earth axes) and, optionally, the
        hull's station-contracted source amplitude.
        """
        if not element_sets and hull_amplitude is None:
            return 0.0

        def strengths(key):
            pieces = []
            for elements in element_sets:
                n = _stack([elements["xyz_a"][0]]).shape[0]
                pieces.append(elements[key] if key in elements else _onp.zeros(n))
            return _stack(pieces)

        scales = self.wave_length_scales
        if scales is None:
            if self.hull_wave_amplitude is not None and hasattr(
                self.hull_wave_amplitude, "length_scales"
            ):
                scales = self.hull_wave_amplitude.length_scales
            else:
                scales = self.hull.default_wave_length_scales(self.demihull_spacing)
        water = self.water
        if element_sets:
            element_kwargs = dict(
                xyz_a=tuple(_stack([e["xyz_a"][k] for e in element_sets]) for k in range(3)),
                xyz_b=tuple(_stack([e["xyz_b"][k] for e in element_sets]) for k in range(3)),
                source_flux=strengths("source_flux"),
                force_y=strengths("force_y"),
                force_z=strengths("force_z"),
                smear_length=strengths("smear_length"),
            )
        else:
            element_kwargs = dict(xyz_a=([], [], []), xyz_b=([], [], []))
        return kelvin_wave_resistance(
            **element_kwargs,
            extra_source_amplitude=hull_amplitude,
            velocity=self.velocity,
            density=water.density(),
            surface_z=0.0,
            gravity=water.gravity,
            length_scales=scales,
            reference_velocity=self.wave_reference_velocity,
        )

    def _hull_state(self) -> dict[str, Any]:
        """Compute hull hydrostatics, viscous/transom drag, dynamic force, and wave amplitude."""
        hull, water = self.hull, self.water
        rho, g = water.density(), water.gravity
        velocity, trim = self.velocity, self.trim
        x_cg, z_cg = self.xyz_cg[0], self.xyz_cg[2]
        pivot = (x_cg, z_cg)
        n_hulls = 1 if self.demihull_spacing is None else 2
        statics = dict(hull.hydrostatics(self.sinkage, trim, pivot))
        statics["volume"] = n_hulls * statics["volume"]
        statics["wetted_surface"] = n_hulls * statics["wetted_surface"]
        buoyancy_hull = rho * g * statics["volume"]
        D_hull_viscous = hull.viscous_resistance(
            velocity,
            rho,
            water.kinematic_viscosity(),
            statics["wetted_surface"],
            statics["wetted_length"],
        )
        transom = hull.transom(velocity, g, self.sinkage, trim, pivot)
        D_transom = (
            n_hulls * rho * g * transom["depth_moment"] * transom["dryness"]
            if self.include_transom_drag
            else 0.0
        )
        if self.hull_wave_amplitude is not None:
            hull_amplitude = self.hull_wave_amplitude(
                self.sinkage, trim, pivot_xz=pivot
            )
        else:
            hull_amplitude = hull.wave_amplitude(
                velocity, g, self.sinkage, trim, pivot, demihull_spacing=self.demihull_spacing
            )
        dynamic = None
        if self.include_hull_dynamics and self.hull_dynamics is not None:
            dynamic = self.hull_dynamics(self.sinkage, trim, pivot_xz=pivot, x_ref=x_cg)
        elif self.include_hull_dynamics:
            dynamic = hull.dynamic_force(
                velocity,
                rho,
                g,
                self.sinkage,
                trim,
                pivot,
                x_ref=x_cg,
                reference_velocity=self.wave_reference_velocity,
                level=self.squat_level,
                demihull_spacing=self.demihull_spacing,
            )
        return {
            "statics": statics,
            "buoyancy_hull": buoyancy_hull,
            "D_hull_viscous": D_hull_viscous,
            "transom": transom,
            "D_transom": D_transom,
            "hull_amplitude": hull_amplitude,
            "dynamic": dynamic,
        }

    def run(self, _hull_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Compute the craft's resistance breakdown and its net vertical force and
        pitching moment about the centre of gravity at the given attitude.
        """
        hull, water = self.hull, self.water
        rho, g = water.density(), water.gravity
        velocity, trim = self.velocity, self.trim
        x_cg, z_cg = self.xyz_cg[0], self.xyz_cg[2]
        cos, sin = np.cosd(trim), np.sind(trim)
        _, z_cg_earth = self._earth(x_cg, z_cg)

        def moment_about_cg(x_e, z_e, f_x, f_z):
            """Bow-up moment of an earth-axes force (aft, up) applied at a point."""
            return (z_e - z_cg_earth) * f_x - (x_e - x_cg) * f_z

        ### Hull
        hull_state = self._hull_state() if _hull_state is None else _hull_state
        statics = hull_state["statics"]
        buoyancy_hull = hull_state["buoyancy_hull"]
        D_hull_viscous = hull_state["D_hull_viscous"]
        transom = hull_state["transom"]
        D_transom = hull_state["D_transom"]
        hull_amplitude = hull_state["hull_amplitude"]
        dynamic = hull_state["dynamic"]

        F_z = -self.mass * g + buoyancy_hull
        M_y = moment_about_cg(statics["x_cb"], statics["z_cb"], 0.0, buoyancy_hull)
        if dynamic is not None:
            F_z = F_z + dynamic["force_up"]
            M_y = M_y + dynamic["moment_bow_up"]

        ### Foil assembly, riding with the hull
        foil: dict[str, Any] | None = None
        foil_sources: dict[str, Any] | None = None
        D_foil = 0.0
        if self.assembly is not None:
            # Map geometry (x, z) to height above the undisturbed waterplane
            # (-z_earth is local depth), so swept, aft, and trimmed surfaces
            # all see the true slanted waterplane.
            op_point = HydroOperatingPoint(
                water=water,
                velocity=velocity,
                alpha=trim,
                q=self.pitch_rate,
                surface_z=lambda x_g, z_g: self._earth(x_g, z_g)[1],
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
                    "force_z": _stack([elements["force_z"]]),
                    "smear_length": _stack([elements["smear_length"]]),
                }

        ### Waves: one spectrum for everything
        sets = [foil_sources] if foil_sources is not None else []
        D_wave = self._wave_resistance(sets, hull_amplitude)
        if self.include_interference_breakdown and foil_sources is not None:
            D_wave_hull = self._wave_resistance([], hull_amplitude)
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
            # Thrust along the shaft, inclined (trim + shaft angle) above horizontal.
            # With thrust deduction t, propeller suction adds t * T * cos(shaft) aft drag
            # at the hull's centre of buoyancy, so T = D / ((1 - t) * cos(shaft)).
            cos_shaft = np.cosd(trim + self.shaft_angle)
            sin_shaft = np.sind(trim + self.shaft_angle)
            thrust = D / np.fmax((1.0 - self.thrust_deduction) * cos_shaft, 1e-6)
            suction_drag = self.thrust_deduction * thrust * cos_shaft
            x_t, z_t = self._earth(self.xyz_thrust[0], self.xyz_thrust[2])
            F_z = F_z + thrust * sin_shaft
            M_y = M_y + moment_about_cg(
                x_t, z_t, -thrust * cos_shaft, thrust * sin_shaft
            )
            M_y = M_y + moment_about_cg(
                statics["x_cb"], statics["z_cb"], suction_drag, 0.0
            )
        else:
            thrust = D / np.fmax(1.0 - self.thrust_deduction, 1e-6)

        return {
            "F_z_net": F_z,
            "M_y_net": M_y,
            "D": D,
            "thrust": thrust,
            "mass": self.mass,
            "xyz_cg": self.xyz_cg,
            "P_drag": D * velocity,
            "D_hull_viscous": D_hull_viscous,
            "D_transom": D_transom,
            "D_foil": D_foil,
            "D_wave": D_wave,
            "D_wave_hull_alone": D_wave_hull,
            "D_wave_foil_alone": D_wave_foil,
            "D_wave_interference": D_wave_interference,
            "L_foil": foil["L"] if foil is not None else 0.0,
            "F_z_hull_dynamic": dynamic["force_up"] if dynamic is not None else 0.0,
            "M_y_hull_dynamic": dynamic["moment_bow_up"] if dynamic is not None else 0.0,
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

    def run_with_stability(
        self,
        d_sinkage: float = 1e-3,
        d_trim: float = 0.05,
        d_pitch_rate: float = 0.01,
        include_pitch_damping: bool = True,
    ) -> dict[str, Any]:
        """
        ``run()``, plus the craft's static heave and pitch stiffness at this attitude
        and speed. Uses exact CasADi algorithmic differentiation when symbolic
        (``asb.Opti``) and central finite differences when numeric.

        Pass ``include_pitch_damping=False`` to skip the finite-difference evaluation
        for ``c_pitch_foils`` (which will then be reported as ``None``).

        Meaningful at an equilibrium attitude. Works symbolically, so the stiffnesses
        can be constrained or optimized in ``asb.Opti``. Added to the results:

        - ``k_heave`` [N/m]: ``dF_z/d(sinkage)``. Positive restores.
        - ``k_pitch_fixed_heave`` [N m/rad]: ``-dM_y/d(trim)`` with heave held.
        - ``k_pitch`` [N m/rad]: the pitch stiffness with **heave free** (the
          vertical force kept in balance as the craft pitches),
          ``-(M_t - M_s F_t / F_s)``. This is the one that decides static pitch
          stability: positive is stable.
        - ``GM_L`` [m]: ``k_pitch / (m g)``, the effective longitudinal metacentric
          height at speed (at rest, the hydrostatic ``GM_L``).
        - ``c_pitch_foils`` [N m s/rad]: the **foils' pitch damping**,
          ``-dM_y/d(pitch rate)`` about the centre of gravity: the moment opposing a
          pitching motion, from the change in each foil's inflow as it is carried up
          and down (quasi-steady; it grows with the square of a foil's distance
          from the centre of gravity, on either side of it).

        Limits: the stiffnesses are static. The damping is the foils' share only:
        the hull's own pitch damping (wave radiation, and its lift at speed) and its
        added inertia are not modelled, so this is what the foils *add*, not a
        total, and no damping ratio or porpoising boundary can be derived from it.
        """
        import copy

        hull_state = self._hull_state()
        base = self.run(_hull_state=hull_state)
        is_symbolic = np.is_casadi_type(
            base["F_z_net"], recursive=False
        ) or np.is_casadi_type(base["M_y_net"], recursive=False)

        if is_symbolic:
            import casadi as ca

            rad_per_deg = _onp.pi / 180.0
            direct_attitude = (
                isinstance(self.sinkage, ca.MX)
                and self.sinkage.is_symbolic()
                and self.sinkage.numel() == 1
                and isinstance(self.trim, ca.MX)
                and self.trim.is_symbolic()
                and self.trim.numel() == 1
                and not ca.is_equal(self.sinkage, self.trim)
            )
            if direct_attitude:
                vars_self = ca.vertcat(self.sinkage, self.trim)
                externals = [
                    self.velocity,
                    self.mass,
                    self.xyz_cg[0],
                    self.xyz_cg[1],
                    self.xyz_cg[2],
                    self.shaft_angle,
                    self.pitch_rate,
                    self.thrust_deduction,
                ]
                if self.xyz_thrust is not None:
                    externals.extend(self.xyz_thrust)
                if any(
                    isinstance(x, ca.MX) and ca.depends_on(x, vars_self)
                    for x in externals
                ):
                    direct_attitude = False

            if direct_attitude:
                vec = ca.vertcat(base["F_z_net"], base["M_y_net"])
                df_ds = ca.jtimes(vec, self.sinkage, 1.0)
                df_dt = ca.jtimes(vec, self.trim, 1.0)
                f_s, m_s = df_ds[0], df_ds[1]
                f_t, m_t = df_dt[0] / rad_per_deg, df_dt[1] / rad_per_deg
            else:
                s_var = ca.MX.sym("_stab_sinkage")
                t_var = ca.MX.sym("_stab_trim")
                probe = copy.copy(self)
                probe.sinkage = s_var
                probe.trim = t_var
                probe.include_interference_breakdown = False
                probed = probe.run()

                vars_st = ca.vertcat(s_var, t_var)
                vals_st = ca.vertcat(self.sinkage, self.trim)
                vec = ca.vertcat(probed["F_z_net"], probed["M_y_net"])
                df_ds = ca.jtimes(vec, s_var, 1.0)
                df_dt = ca.jtimes(vec, t_var, 1.0)
                derivs = ca.substitute(ca.vertcat(df_ds, df_dt), vars_st, vals_st)
                f_s, m_s = derivs[0], derivs[1]
                f_t, m_t = derivs[2] / rad_per_deg, derivs[3] / rad_per_deg
        else:
            def nudged(ds: float, dt: float) -> dict[str, Any]:
                other = copy.copy(self)
                other.sinkage = self.sinkage + ds
                other.trim = self.trim + dt
                other.include_interference_breakdown = False
                return other.run()

            sunk_p = nudged(d_sinkage, 0.0)
            sunk_m = nudged(-d_sinkage, 0.0)
            pitch_p = nudged(0.0, d_trim)
            pitch_m = nudged(0.0, -d_trim)
            d_trim_rad = d_trim * _onp.pi / 180.0

            f_s = (sunk_p["F_z_net"] - sunk_m["F_z_net"]) / (2.0 * d_sinkage)
            m_s = (sunk_p["M_y_net"] - sunk_m["M_y_net"]) / (2.0 * d_sinkage)
            f_t = (pitch_p["F_z_net"] - pitch_m["F_z_net"]) / (2.0 * d_trim_rad)
            m_t = (pitch_p["M_y_net"] - pitch_m["M_y_net"]) / (2.0 * d_trim_rad)

        # The foils' pitch damping, from one further evaluation at a small pitch rate
        # (the response is linear in it), numeric or symbolic alike. Pitch rate only
        # enters HydroBuildup, so the hull's state and wave spectrum are reused.
        if not include_pitch_damping:
            c_pitch = None
        elif self.assembly is not None:
            pitching = copy.copy(self)
            pitching.pitch_rate = self.pitch_rate + d_pitch_rate
            pitching.include_interference_breakdown = False
            c_pitch = -(
                pitching.run(_hull_state=hull_state)["M_y_net"] - base["M_y_net"]
            ) / d_pitch_rate
        else:
            c_pitch = 0.0

        k_pitch = -(m_t - m_s * f_t / f_s)
        base.update(
            k_heave=f_s,
            k_pitch_fixed_heave=-m_t,
            k_pitch=k_pitch,
            GM_L=k_pitch / (self.mass * self.water.gravity),
            c_pitch_foils=c_pitch,
            dFz_dtrim=f_t,
            dMy_dsinkage=m_s,
        )
        return base

    @classmethod
    def solve_equilibrium(
        cls,
        *args,
        sinkage_guess: float = 0.0,
        trim_guess: float = 0.0,
        max_trim: float = 15.0,
        tolerance: float = 1e-6,
        max_iterations: int = 40,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Find the sinkage and trim at which a fixed (numeric) craft is in
        vertical-force and pitching-moment balance, and return its results there.

        Takes the constructor's arguments (without ``sinkage`` and ``trim``). Solved
        by a damped Newton iteration with a finite-difference Jacobian, residuals
        scaled by the weight and by weight x hull length; converged when both are
        below ``tolerance``. Raises ``RuntimeError`` if no balance is found (a craft
        with no stable attitude, e.g. a lifting foil far forward at speed).
        """
        fast = dict(kwargs, include_interference_breakdown=False)
        probe = cls(*args, sinkage=0.0, trim=0.0, **fast)
        hull = probe.hull
        weight = float(probe.mass) * float(probe.water.gravity)
        depth_scale = float(hull.levels[-1] - hull.levels[0])
        scale = _onp.array([weight, weight * hull.length])

        def residual(state):
            res = cls(*args, sinkage=float(state[0]), trim=float(state[1]), **fast).run()
            return _onp.array(
                [
                    float(_onp.ravel(res["F_z_net"])[0]),
                    float(_onp.ravel(res["M_y_net"])[0]),
                ]
            ) / scale

        def clamp(state):
            return _onp.array(
                [
                    _onp.clip(state[0], -depth_scale, depth_scale),
                    _onp.clip(state[1], -max_trim, max_trim),
                ]
            )

        state = clamp(_onp.array([sinkage_guess, trim_guess], dtype=float))
        steps = _onp.array([1e-4 * max(depth_scale, 1e-3), 1e-3])
        value = residual(state)
        for iteration in range(max_iterations):
            norm = float(_onp.max(_onp.abs(value)))
            if verbose:
                print(
                    f"  iter {iteration}: sinkage {state[0]:+.5f} m, trim {state[1]:+.4f} deg, "
                    f"residual {norm:.2e}"
                )
            if norm < tolerance:
                break
            jacobian = _onp.column_stack(
                [
                    (residual(state + _onp.eye(2)[k] * steps[k]) - value) / steps[k]
                    for k in range(2)
                ]
            )
            try:
                step = _onp.linalg.solve(jacobian, -value)
            except _onp.linalg.LinAlgError:
                break
            if not _onp.all(_onp.isfinite(step)):
                break
            # Limit the step, then backtrack until the residual falls.
            limit = _onp.array([0.25 * depth_scale, 3.0])
            step = step * min(1.0, float(_onp.min(limit / _onp.maximum(_onp.abs(step), 1e-300))))
            damping = 1.0
            accepted = False
            while damping > 1e-3:
                trial = clamp(state + damping * step)
                trial_value = residual(trial)
                if _onp.max(_onp.abs(trial_value)) < norm:
                    state, value = trial, trial_value
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                break
        final_norm = float(_onp.max(_onp.abs(value)))
        if not (final_norm < max(tolerance, 1e-12) * 10):
            raise RuntimeError(
                "No equilibrium attitude found "
                f"(last sinkage {state[0]:+.4f} m, trim {state[1]:+.3f} deg, "
                f"scaled residual {final_norm:.2e})."
            )
        return cls(*args, sinkage=float(state[0]), trim=float(state[1]), **kwargs).run()
