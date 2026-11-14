import copy
from typing import Any, Callable, Literal
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import Airplane, ControlSurface
import aerosandbox.geometry.mesh_utilities as mesh_utils
import aerosandbox.numpy as np
from aerosandbox.performance import OperatingPoint
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import HydrofoilAssembly
from hydrosandbox.hydrodynamics.hydro_buildup import HydroBuildup
from hydrosandbox.hydrodynamics.singularities import (
    calculate_induced_velocity_horseshoe_hydro,
)
from hydrosandbox.library.cavitation import (
    cavitation_drag_increment,
    cavitation_inception_speed,
    cavitation_number,
)
from hydrosandbox.library.free_surface import (
    froude_image_weight,
    spray_drag_strut,
)


def tall(array):
    """Reshape an array into a tall (Nx1) column vector."""
    return np.reshape(array, (-1, 1))


def wide(array):
    """Reshape an array into a wide (1xN) row vector."""
    return np.reshape(array, (1, -1))


class HydroLiftingLine(ExplicitAnalysis):
    """
    Nonlinear viscous Lifting Line hydrodynamics solver for lifting hydrofoil assemblies.

    Combines 3D Prandtl lifting-line / horseshoe-vortex induced downwash (with
    free-surface method-of-images reflection and surface-piercing clipping across
    ``op_point.surface_z``) with viscous 2D section polars and suction-peak
    ``Cpmin_0`` from NeuralFoil at ``mach = 0.0``.

    Supports:
    - Free-surface image reflection (``"high_froude"``, ``"froude_blended"``,
      ``"rigid_lid"``, or ``"none"``).
    - Surface-piercing geometry clipping and Hoerner spray drag on piercing struts.
    - Spanwise cavitation diagnostics (``panel_Cpmin``, ``panel_sigma``,
      ``cavitation_margin``, ``V_cavitation_inception``) and optional cavitation
      drag rise.
    - Incompressible hull/pod fuselage hydrodynamics and Archimedes buoyancy.
    - Finite-difference stability derivatives including vertical heave stiffness
      ``CLh`` and ``L_h``.
    """

    def __init__(
        self,
        assembly: HydrofoilAssembly | Airplane | None = None,
        op_point: OperatingPoint | None = None,
        xyz_ref: list[float] | np.ndarray | None = None,
        model_size: Literal[
            "xxsmall", "xsmall", "small", "medium", "large", "xlarge"
        ] = "medium",
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 4,
        spanwise_spacing_function: Callable[
            [float, float, int], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        free_surface_mode: Literal[
            "high_froude", "froude_blended", "rigid_lid", "none"
        ] = "high_froude",
        clip_to_free_surface: bool = True,
        include_spray_drag: bool = True,
        include_cavitation_penalty: bool = False,
        include_fuselages: bool = True,
        airplane: Airplane | HydrofoilAssembly | None = None,
    ):
        super().__init__()

        if assembly is None:
            assembly = airplane
        if assembly is None:
            raise ValueError(
                "Must provide `assembly` (a HydrofoilAssembly or Airplane)."
            )
        if op_point is None:
            raise ValueError(
                "Must provide `op_point` (a HydroOperatingPoint or OperatingPoint)."
            )

        if xyz_ref is None:
            xyz_ref = assembly.xyz_ref

        self.assembly = assembly
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.model_size = model_size
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        self.free_surface_mode = free_surface_mode
        self.clip_to_free_surface = clip_to_free_surface
        self.include_spray_drag = include_spray_drag
        self.include_cavitation_penalty = include_cavitation_penalty
        self.include_fuselages = include_fuselages

        if run_symmetric_if_possible:
            raise NotImplementedError(
                "HydroLiftingLine with symmetry detection is not yet implemented."
            )
        self.run_symmetric = False

    @property
    def airplane(self) -> HydrofoilAssembly | Airplane:
        """Alias ``assembly`` as ``airplane`` for AeroSandbox interoperability."""
        return self.assembly

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\tassembly={self.assembly},\n"
            f"\top_point={self.op_point},\n"
            f"\tfree_surface_mode={self.free_surface_mode!r},\n"
            f"\txyz_ref={self.xyz_ref}\n)"
        )

    def _compute_image_weight(
        self,
        surface_z: float | None,
        vortex_centers: np.ndarray,
    ) -> float:
        if surface_z is None or self.free_surface_mode == "none":
            return 0.0
        if self.free_surface_mode == "froude_blended":
            g = getattr(self.op_point.atmosphere, "gravity", 9.80665)
            h_mean = float(
                np.mean(np.maximum(surface_z - vortex_centers[:, 2], 1e-3))
            )
            fr_h = float(self.op_point.velocity) / float(np.sqrt(g * h_mean))
            return float(froude_image_weight(fr_h, mode="froude_blended"))
        return float(froude_image_weight(mode=self.free_surface_mode))

    @staticmethod
    def _mirror_control_surfaces(
        control_surfaces: list[ControlSurface], mirror: bool
    ) -> tuple[ControlSurface, ...]:
        if not control_surfaces:
            return ()
        mirrored: list[ControlSurface] = []
        for surf in control_surfaces:
            if mirror and not surf.symmetric:
                s_copy = surf.copy()
                s_copy.deflection = -s_copy.deflection
                mirrored.append(s_copy)
            else:
                mirrored.append(surf)
        return tuple(mirrored)

    def _evaluate_section_polars(
        self,
        airfoils_a: list,
        airfoils_b: list,
        weights_a: np.ndarray,
        control_surfaces_list: list[tuple[ControlSurface, ...]],
        alphas_deg: np.ndarray,
        Re_panels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate NeuralFoil incompressible 2D section data (``CL``, ``CD``, ``CM``,
        ``Cpmin_0``) batched across unique ``(airfoil, control_surfaces)`` pairs.
        """
        n_panels = len(alphas_deg)

        def _batch_eval(airfoil_seq: list) -> dict[str, np.ndarray]:
            out = {
                "CL": np.zeros(n_panels),
                "CD": np.zeros(n_panels),
                "CM": np.zeros(n_panels),
                "Cpmin_0": np.zeros(n_panels),
            }
            groups: dict[tuple[int, tuple], tuple[Any, tuple, list[int]]] = {}
            for idx, (af, cs) in enumerate(zip(airfoil_seq, control_surfaces_list)):
                cs_key = tuple(
                    (c.name, float(c.deflection), float(c.hinge_point), bool(c.symmetric))
                    for c in cs
                )
                key = (id(af), cs_key)
                if key not in groups:
                    groups[key] = (af, cs, [])
                groups[key][2].append(idx)

            for af, cs, indices in groups.values():
                idx_arr = np.array(indices, dtype=int)
                aero = af.get_aero_from_neuralfoil(
                    alpha=alphas_deg[idx_arr],
                    Re=Re_panels[idx_arr],
                    mach=0.0,
                    control_surfaces=list(cs),
                    model_size=self.model_size,
                )
                out["CL"][idx_arr] = np.asarray(aero["CL"], dtype=float).reshape(-1)
                out["CD"][idx_arr] = np.asarray(aero["CD"], dtype=float).reshape(-1)
                out["CM"][idx_arr] = np.asarray(aero["CM"], dtype=float).reshape(-1)
                out["Cpmin_0"][idx_arr] = np.asarray(
                    aero["Cpmin_0"], dtype=float
                ).reshape(-1)
            return out

        eval_a = _batch_eval(airfoils_a)

        same_all = all(a is b for a, b in zip(airfoils_a, airfoils_b))
        if same_all:
            return eval_a

        eval_b = _batch_eval(airfoils_b)
        wb = 1.0 - weights_a
        return {
            k: weights_a * eval_a[k] + wb * eval_b[k]
            for k in ("CL", "CD", "CM", "Cpmin_0")
        }

    def _compute_spray_drag_force_and_moment(
        self,
        surface_z: float | None,
        steady_freestream_direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute Hoerner spray drag on unclipped surface-piercing sections."""
        if (
            not self.include_spray_drag
            or surface_z is None
            or self.free_surface_mode in ("none", "rigid_lid")
        ):
            return np.zeros(3), np.zeros(3), 0.0

        fluid = self.op_point.atmosphere
        g = getattr(fluid, "gravity", 9.80665)
        q = float(self.op_point.dynamic_pressure())
        V = float(self.op_point.velocity)
        xyz_ref = np.asarray(self.xyz_ref, dtype=float)

        f_spray_total = np.zeros(3)
        m_spray_total = np.zeros(3)
        d_spray_total = 0.0

        for wing in self.assembly.wings:
            n_xsecs = len(wing.xsecs)
            if n_xsecs < 2:
                continue
            qcs = np.stack(
                [
                    np.asarray(
                        wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0.0),
                        dtype=float,
                    )
                    for i in range(n_xsecs)
                ]
            )
            tcs = np.array(
                [float(xs.airfoil.max_thickness()) for xs in wing.xsecs], dtype=float
            )

            for i in range(n_xsecs - 1):
                za = float(qcs[i, 2])
                zb = float(qcs[i + 1, 2])
                if za == zb:
                    continue
                crosses = (za < surface_z <= zb) or (zb < surface_z <= za)
                if not crosses:
                    continue

                t = (float(surface_z) - za) / (zb - za)
                c_wl = (1.0 - t) * float(wing.xsecs[i].chord) + t * float(
                    wing.xsecs[i + 1].chord
                )
                tc_wl = (1.0 - t) * tcs[i] + t * tcs[i + 1]
                fr_c = V / float(np.sqrt(g * max(c_wl, 1e-4)))
                d_sec = float(
                    spray_drag_strut(
                        waterline_chord=c_wl,
                        thickness_to_chord=tc_wl,
                        dynamic_pressure=q,
                        froude_chord=fr_c,
                    )
                )
                wl_xyz = (1.0 - t) * qcs[i] + t * qcs[i + 1]

                sides = [wl_xyz]
                if wing.symmetric and abs(float(wl_xyz[1])) > 1e-12:
                    sides.append(np.array([wl_xyz[0], -wl_xyz[1], wl_xyz[2]]))

                for xyz_hit in sides:
                    f_vec = d_sec * steady_freestream_direction
                    m_vec = np.cross(xyz_hit - xyz_ref, f_vec)
                    f_spray_total = f_spray_total + f_vec
                    m_spray_total = m_spray_total + m_vec
                    d_spray_total += d_sec

        return f_spray_total, m_spray_total, d_spray_total

    def run(self) -> dict[str, Any]:
        """Compute 3D lifting-line hydrodynamic forces, moments, and cavitation metrics."""
        surface_z = getattr(self.op_point, "surface_z", None)
        self.surface_z = surface_z

        if (
            surface_z is not None
            and self.clip_to_free_surface
            and hasattr(self.assembly, "clipped_to_free_surface")
        ):
            active_assembly = self.assembly.clipped_to_free_surface(surface_z)
        else:
            active_assembly = self.assembly
        self.active_assembly = active_assembly

        if self.verbose:
            print("Meshing submerged hydrofoil lifting-line panels...")

        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        airfoils_a = []
        airfoils_b = []
        weights_a = []
        control_surfaces_list: list[tuple[ControlSurface, ...]] = []

        for wing in active_assembly.wings:
            if len(wing.xsecs) < 2:
                continue
            if self.spanwise_resolution > 1:
                wing = wing.subdivide_sections(
                    ratio=self.spanwise_resolution,
                    spacing_function=self.spanwise_spacing_function,
                )

            points, faces = wing.mesh_thin_surface(
                method="quad",
                chordwise_resolution=1,
                add_camber=False,
            )
            if len(faces) == 0:
                continue

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])

            n_sects = len(wing.xsecs) - 1
            for mirror in (False, True) if wing.symmetric else (False,):
                for idx in range(n_sects):
                    xa = wing.xsecs[idx]
                    xb = wing.xsecs[idx + 1]
                    airfoils_a.append(xa.airfoil)
                    airfoils_b.append(xb.airfoil)
                    c_sum = float(xa.chord + xb.chord)
                    weights_a.append(
                        float(xa.chord) / c_sum if c_sum > 0 else 0.5
                    )
                    control_surfaces_list.append(
                        self._mirror_control_surfaces(xa.control_surfaces, mirror)
                    )

        steady_freestream_velocity = (
            self.op_point.compute_freestream_velocity_geometry_axes()
        )
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction

        fluid = self.op_point.atmosphere
        rho = fluid.density()
        nu = fluid.kinematic_viscosity()
        q = self.op_point.dynamic_pressure()
        s_ref = self.assembly.s_ref
        b_ref = self.assembly.b_ref
        c_ref = self.assembly.c_ref

        f_spray_g, m_spray_g, D_spray = self._compute_spray_drag_force_and_moment(
            surface_z=surface_z,
            steady_freestream_direction=steady_freestream_direction,
        )

        f_fuse_g = np.zeros(3)
        m_fuse_g = np.zeros(3)
        D_fuselage = 0.0
        if self.include_fuselages and len(self.assembly.fuselages) > 0:
            hb_fuse = HydroBuildup(
                assembly=self.assembly,
                op_point=self.op_point,
                xyz_ref=self.xyz_ref,
                free_surface_mode=self.free_surface_mode,
            )
            for fuse in self.assembly.fuselages:
                f_res = hb_fuse.fuselage_hydrodynamics(
                    fuselage=fuse, include_induced_drag=True
                )
                f_fuse_g = f_fuse_g + np.asarray(f_res.F_g, dtype=float)
                m_fuse_g = m_fuse_g + np.asarray(f_res.M_g, dtype=float)
            f_fuse_w = self.op_point.convert_axes(
                f_fuse_g[0], f_fuse_g[1], f_fuse_g[2],
                from_axes="geometry", to_axes="wind",
            )
            D_fuselage = -float(f_fuse_w[0])

        if not front_left_vertices:
            self.front_left_vertices = np.zeros((0, 3))
            self.back_left_vertices = np.zeros((0, 3))
            self.back_right_vertices = np.zeros((0, 3))
            self.front_right_vertices = np.zeros((0, 3))
            self.vortex_strengths = np.array([])
            self.image_weight = 0.0
            self.panel_Cpmin = np.array([])
            force_geometry = f_spray_g + f_fuse_g
            moment_geometry = m_spray_g + m_fuse_g
            force_body = self.op_point.convert_axes(
                force_geometry[0], force_geometry[1], force_geometry[2],
                from_axes="geometry", to_axes="body",
            )
            force_wind = self.op_point.convert_axes(
                force_body[0], force_body[1], force_body[2],
                from_axes="body", to_axes="wind",
            )
            moment_body = self.op_point.convert_axes(
                moment_geometry[0], moment_geometry[1], moment_geometry[2],
                from_axes="geometry", to_axes="body",
            )
            moment_wind = self.op_point.convert_axes(
                moment_body[0], moment_body[1], moment_body[2],
                from_axes="body", to_axes="wind",
            )
            L = -force_wind[2]
            D = -force_wind[0]
            Y = force_wind[1]
            l_b, m_b, n_b = moment_body
            output: dict[str, Any] = {
                "F_g": force_geometry,
                "F_b": force_body,
                "F_w": force_wind,
                "M_g": moment_geometry,
                "M_b": moment_body,
                "M_w": moment_wind,
                "L": L,
                "D": D,
                "Y": Y,
                "l_b": l_b,
                "m_b": m_b,
                "n_b": n_b,
                "CL": L / q / s_ref,
                "CD": D / q / s_ref,
                "CY": Y / q / s_ref,
                "Cl": l_b / q / s_ref / b_ref,
                "Cm": m_b / q / s_ref / c_ref,
                "Cn": n_b / q / s_ref / b_ref,
                "D_induced": 0.0,
                "D_profile": 0.0,
                "D_spray": D_spray,
                "D_fuselage": D_fuselage,
                "D_cavitation": 0.0,
                "cavitation_margin": np.inf,
                "Cpmin_global": 0.0,
                "sigma_min": np.inf,
            }
            if isinstance(self.assembly, HydrofoilAssembly) and isinstance(
                fluid, Water
            ):
                output.update(
                    self.assembly.hydrostatic_forces(
                        water=fluid, surface_z=surface_z, xyz_ref=self.xyz_ref
                    )
                )
            return output

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        weights_a_arr = np.asarray(weights_a, dtype=float)

        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2.0

        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = 0.5 * (left_vortex_vertices + right_vortex_vertices)
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        span_lengths = np.linalg.norm(vortex_bound_leg, axis=1)
        span_directions = vortex_bound_leg / tall(span_lengths)

        chord_vectors = 0.5 * (
            (back_left_vertices - front_left_vertices)
            + (back_right_vertices - front_right_vertices)
        )
        chords = np.linalg.norm(chord_vectors, axis=1)
        local_downstream_directions = chord_vectors / tall(chords)

        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.normal_directions = normal_directions
        self.areas = areas
        self.chords = chords
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg

        self.image_weight = self._compute_image_weight(surface_z, vortex_centers)

        rot_vel_panels = self.op_point.compute_rotation_velocity_geometry_axes(
            vortex_centers
        )
        freestream_velocities = np.add(wide(steady_freestream_velocity), rot_vel_panels)
        freestream_speeds = np.linalg.norm(freestream_velocities, axis=1)
        freestream_directions = freestream_velocities / tall(freestream_speeds)
        self.freestream_velocities = freestream_velocities

        vi_cross_li_fs = np.cross(freestream_velocities, vortex_bound_leg, axis=1)
        vi_cross_li_fs_norm = np.linalg.norm(vi_cross_li_fs, axis=1)

        V_perp = np.linalg.norm(
            np.cross(freestream_velocities, span_directions, axis=1), axis=1
        )
        V_perp = np.maximum(V_perp, 1e-6)
        Re_panels = V_perp * chords / nu

        cos_normals = np.clip(
            np.sum(freestream_directions * normal_directions, axis=1), -1.0, 1.0
        )
        alpha_geometrics = 90.0 - np.arccosd(cos_normals)

        polar_geo = self._evaluate_section_polars(
            airfoils_a=airfoils_a,
            airfoils_b=airfoils_b,
            weights_a=weights_a_arr,
            control_surfaces_list=control_surfaces_list,
            alphas_deg=alpha_geometrics,
            Re_panels=Re_panels,
        )
        CLs_geo = polar_geo["CL"]

        u_col_unit, v_col_unit, w_col_unit = calculate_induced_velocity_horseshoe_hydro(
            x_field=tall(vortex_centers[:, 0]),
            y_field=tall(vortex_centers[:, 1]),
            z_field=tall(vortex_centers[:, 2]),
            x_left=wide(left_vortex_vertices[:, 0]),
            y_left=wide(left_vortex_vertices[:, 1]),
            z_left=wide(left_vortex_vertices[:, 2]),
            x_right=wide(right_vortex_vertices[:, 0]),
            y_right=wide(right_vortex_vertices[:, 1]),
            z_right=wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1.0, 0.0, 0.0])
            ),
            gamma=1.0,
            vortex_core_radius=self.vortex_core_radius,
            surface_z=self.surface_z,
            image_weight=self.image_weight,
        )

        AIC = (
            u_col_unit * tall(normal_directions[:, 0])
            + v_col_unit * tall(normal_directions[:, 1])
            + w_col_unit * tall(normal_directions[:, 2])
        )
        CLas = 2.0 * np.pi * np.ones_like(areas)

        alpha_influence_matrix = AIC / tall(V_perp)
        diag_terms = (2.0 * vi_cross_li_fs_norm) / (V_perp**2 * areas)
        A = alpha_influence_matrix * tall(CLas) - np.diag(diag_terms)
        self.vortex_strengths = np.linalg.solve(A, -CLs_geo)

        v_ind_at_centers = self.get_induced_velocity_at_points(vortex_centers)
        v_centers = freestream_velocities + v_ind_at_centers
        v_perp_total = np.linalg.norm(
            np.cross(v_centers, span_directions, axis=1), axis=1
        )
        v_perp_total = np.maximum(v_perp_total, 1e-6)

        alpha_induced = np.degrees(
            np.sum(v_ind_at_centers * normal_directions, axis=1) / V_perp
        )
        alphas_effective = alpha_geometrics + alpha_induced
        self.alpha_geometrics = alpha_geometrics
        self.alpha_induced = alpha_induced
        self.alphas_effective = alphas_effective

        polar_eff = self._evaluate_section_polars(
            airfoils_a=airfoils_a,
            airfoils_b=airfoils_b,
            weights_a=weights_a_arr,
            control_surfaces_list=control_surfaces_list,
            alphas_deg=alphas_effective,
            Re_panels=Re_panels,
        )
        panel_CLs = (2.0 * vi_cross_li_fs_norm * self.vortex_strengths) / (
            V_perp**2 * areas
        )
        panel_CDs = polar_eff["CD"]
        panel_CMs = polar_eff["CM"]
        panel_Cpmin = polar_eff["Cpmin_0"]

        self.panel_CLs = panel_CLs
        self.panel_CDs = panel_CDs
        self.panel_CMs = panel_CMs
        self.panel_Cpmin = panel_Cpmin

        # Inviscid Kutta-Joukowski force (gives Lift + Induced Drag)
        vi_cross_li = np.cross(v_centers, vortex_bound_leg, axis=1)
        forces_inviscid_g = rho * vi_cross_li * tall(self.vortex_strengths)

        # Viscous profile drag force along local flow direction
        q_perp = 0.5 * rho * v_perp_total**2
        v_center_dirs = v_centers / tall(np.linalg.norm(v_centers, axis=1))
        forces_profile_g = tall(q_perp * areas * panel_CDs) * v_center_dirs

        # Section pitching moment around quarter-chord (right-hand axis = normal x downstream)
        pitch_axes = np.cross(normal_directions, local_downstream_directions, axis=1)
        moments_pitch_g = tall(q_perp * areas * chords * panel_CMs) * pitch_axes

        # Cavitation diagnostics and optional cavitation drag penalty
        panel_cav_drag_g = np.zeros_like(forces_profile_g)
        D_cavitation = 0.0
        cav_outputs: dict[str, Any] = {}
        if hasattr(self.op_point, "cavitation_number") or isinstance(fluid, Water):
            if hasattr(self.op_point, "cavitation_number"):
                panel_sigma = self.op_point.cavitation_number(vortex_centers[:, 2])
                panel_p_local = self.op_point.local_pressure(vortex_centers[:, 2])
            else:
                depth_panels = (
                    (surface_z - vortex_centers[:, 2])
                    if surface_z is not None
                    else (fluid.depth - vortex_centers[:, 2])
                )
                panel_p_local = fluid.pressure(depth=depth_panels)
                panel_sigma = cavitation_number(
                    panel_p_local, fluid.vapor_pressure(), q
                )

            panel_cav_margin = panel_sigma + panel_Cpmin
            min_idx = int(np.argmin(panel_cav_margin))
            v_inc = cavitation_inception_speed(
                Cpmin=panel_Cpmin[min_idx],
                local_pressure=panel_p_local[min_idx],
                vapor_pressure=fluid.vapor_pressure(),
                density=rho,
            )
            self.panel_sigma = panel_sigma
            self.panel_cavitation_margin = panel_cav_margin
            cav_outputs = {
                "panel_sigma": panel_sigma,
                "panel_cavitation_margin": panel_cav_margin,
                "sigma_min": np.min(panel_sigma),
                "cavitation_margin": np.min(panel_cav_margin),
                "V_cavitation_inception": v_inc,
            }
            if self.include_cavitation_penalty:
                delta_cd_cav = cavitation_drag_increment(panel_Cpmin, panel_sigma)
                panel_cav_drag_g = tall(q_perp * areas * delta_cd_cav) * v_center_dirs
                cav_force_w = self.op_point.convert_axes(
                    np.sum(panel_cav_drag_g[:, 0]),
                    np.sum(panel_cav_drag_g[:, 1]),
                    np.sum(panel_cav_drag_g[:, 2]),
                    from_axes="geometry",
                    to_axes="wind",
                )
                D_cavitation = -float(cav_force_w[0])

        forces_geometry = forces_inviscid_g + forces_profile_g + panel_cav_drag_g
        r_arm = np.add(vortex_centers, -wide(np.array(self.xyz_ref)))
        moments_geometry = np.cross(r_arm, forces_geometry) + moments_pitch_g

        inv_total_g = np.sum(forces_inviscid_g, axis=0)
        prof_total_g = np.sum(forces_profile_g, axis=0)
        D_induced = -float(
            self.op_point.convert_axes(
                inv_total_g[0], inv_total_g[1], inv_total_g[2],
                from_axes="geometry", to_axes="wind",
            )[0]
        )
        D_profile = -float(
            self.op_point.convert_axes(
                prof_total_g[0], prof_total_g[1], prof_total_g[2],
                from_axes="geometry", to_axes="wind",
            )[0]
        )

        force_geometry = np.sum(forces_geometry, axis=0) + f_spray_g + f_fuse_g
        moment_geometry = np.sum(moments_geometry, axis=0) + m_spray_g + m_fuse_g

        force_body = self.op_point.convert_axes(
            force_geometry[0], force_geometry[1], force_geometry[2],
            from_axes="geometry", to_axes="body",
        )
        force_wind = self.op_point.convert_axes(
            force_body[0], force_body[1], force_body[2],
            from_axes="body", to_axes="wind",
        )
        moment_body = self.op_point.convert_axes(
            moment_geometry[0], moment_geometry[1], moment_geometry[2],
            from_axes="geometry", to_axes="body",
        )
        moment_wind = self.op_point.convert_axes(
            moment_body[0], moment_body[1], moment_body[2],
            from_axes="body", to_axes="wind",
        )

        self.forces_geometry = forces_geometry
        self.moments_geometry = moments_geometry
        self.force_geometry = force_geometry
        self.force_body = force_body
        self.force_wind = force_wind
        self.moment_geometry = moment_geometry
        self.moment_body = moment_body
        self.moment_wind = moment_wind

        L = -force_wind[2]
        D = -force_wind[0]
        Y = force_wind[1]
        l_b, m_b, n_b = moment_body

        output: dict[str, Any] = {
            "F_g": force_geometry,
            "F_b": force_body,
            "F_w": force_wind,
            "M_g": moment_geometry,
            "M_b": moment_body,
            "M_w": moment_wind,
            "L": L,
            "D": D,
            "P_drag": D * self.op_point.velocity,
            "Y": Y,
            "l_b": l_b,
            "m_b": m_b,
            "n_b": n_b,
            "CL": L / q / s_ref,
            "CD": D / q / s_ref,
            "CY": Y / q / s_ref,
            "Cl": l_b / q / s_ref / b_ref,
            "Cm": m_b / q / s_ref / c_ref,
            "Cn": n_b / q / s_ref / b_ref,
            "D_induced": D_induced,
            "D_profile": D_profile,
            "D_spray": D_spray,
            "D_fuselage": D_fuselage,
            "D_cavitation": D_cavitation,
            "panel_CLs": panel_CLs,
            "panel_CDs": panel_CDs,
            "panel_CMs": panel_CMs,
            "panel_Cpmin": panel_Cpmin,
            "Cpmin_global": np.min(panel_Cpmin),
        }
        output.update(cav_outputs)

        if isinstance(self.assembly, HydrofoilAssembly) and isinstance(fluid, Water):
            output.update(
                self.assembly.hydrostatic_forces(
                    water=fluid, surface_z=surface_z, xyz_ref=self.xyz_ref
                )
            )

        return output

    def run_with_stability_derivatives(
        self,
        alpha: bool = True,
        beta: bool = True,
        p: bool = True,
        q: bool = True,
        r: bool = True,
        surface_z: bool = True,
    ) -> dict[str, Any]:
        """
        Compute lifting-line forces, moments, and finite-difference stability
        derivatives (including vertical heave derivatives ``CLh``, ``CDh``, ``Cmh``,
        and ``L_h`` when ``op_point.surface_z`` is specified).
        """
        do_analysis = {
            "alpha": alpha,
            "beta": beta,
            "p": p,
            "q": q,
            "r": r,
        }
        abbreviations = {
            "alpha": "a",
            "beta": "b",
            "p": "p",
            "q": "q",
            "r": "r",
        }
        finite_difference_amounts = {
            "alpha": 0.001,
            "beta": 0.001,
            "p": 0.001 * (2.0 * self.op_point.velocity) / self.assembly.b_ref,
            "q": 0.001 * (2.0 * self.op_point.velocity) / self.assembly.c_ref,
            "r": 0.001 * (2.0 * self.op_point.velocity) / self.assembly.b_ref,
        }
        scaling_factors = {
            "alpha": np.degrees(1),
            "beta": np.degrees(1),
            "p": (2.0 * self.op_point.velocity) / self.assembly.b_ref,
            "q": (2.0 * self.op_point.velocity) / self.assembly.c_ref,
            "r": (2.0 * self.op_point.velocity) / self.assembly.b_ref,
        }

        original_op_point = self.op_point
        run_base = self.run()

        for denom, active in do_analysis.items():
            if not active:
                continue
            inc_op = copy.copy(original_op_point)
            setattr(
                inc_op,
                denom,
                getattr(original_op_point, denom) + finite_difference_amounts[denom],
            )
            ll_inc = copy.copy(self)
            ll_inc.op_point = inc_op
            run_inc = ll_inc.run()

            for num in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                run_base[num + abbreviations[denom]] = (
                    (run_inc[num] - run_base[num])
                    / finite_difference_amounts[denom]
                    * scaling_factors[denom]
                )

            if denom == "alpha":
                cla = np.where(run_base["CLa"] == 0, np.nan, run_base["CLa"])
                run_base["x_np"] = self.xyz_ref[0] - (
                    run_base["Cma"] * (self.assembly.c_ref / cla)
                )
            if denom == "beta":
                cyb = np.where(run_base["CYb"] == 0, np.nan, run_base["CYb"])
                run_base["x_np_lateral"] = self.xyz_ref[0] - (
                    run_base["Cnb"] * (self.assembly.b_ref / cyb)
                )

        if surface_z and getattr(original_op_point, "surface_z", None) is not None:
            delta_h = 1e-3
            inc_op = copy.copy(original_op_point)
            inc_op.surface_z = original_op_point.surface_z + delta_h
            ll_inc = copy.copy(self)
            ll_inc.op_point = inc_op
            run_inc = ll_inc.run()

            for num in ("CL", "CD", "Cm"):
                run_base[num + "h"] = (run_inc[num] - run_base[num]) / delta_h
            run_base["L_h"] = (run_inc["L"] - run_base["L"]) / delta_h

        return run_base

    def get_induced_velocity_at_points(self, points: np.ndarray) -> np.ndarray:
        """Compute induced velocity [m/s] at an Nx3 array of field points."""
        if len(self.vortex_strengths) == 0:
            return np.zeros_like(points, dtype=float)

        u_ind, v_ind, w_ind = calculate_induced_velocity_horseshoe_hydro(
            x_field=tall(points[:, 0]),
            y_field=tall(points[:, 1]),
            z_field=tall(points[:, 2]),
            x_left=wide(self.left_vortex_vertices[:, 0]),
            y_left=wide(self.left_vortex_vertices[:, 1]),
            z_left=wide(self.left_vortex_vertices[:, 2]),
            x_right=wide(self.right_vortex_vertices[:, 0]),
            y_right=wide(self.right_vortex_vertices[:, 1]),
            z_right=wide(self.right_vortex_vertices[:, 2]),
            trailing_vortex_direction=(
                self.steady_freestream_direction
                if self.align_trailing_vortices_with_wind
                else np.array([1.0, 0.0, 0.0])
            ),
            gamma=wide(self.vortex_strengths),
            vortex_core_radius=self.vortex_core_radius,
            surface_z=self.surface_z,
            image_weight=self.image_weight,
        )
        return np.stack(
            [np.sum(u_ind, axis=1), np.sum(v_ind, axis=1), np.sum(w_ind, axis=1)],
            axis=1,
        )

    def get_velocity_at_points(self, points: np.ndarray) -> np.ndarray:
        """Compute total velocity (freestream + rotation + induced) at Nx3 points."""
        v_induced = self.get_induced_velocity_at_points(points)
        rot_vel = self.op_point.compute_rotation_velocity_geometry_axes(points)
        freestream = np.add(wide(self.steady_freestream_velocity), rot_vel)
        return v_induced + freestream

    def calculate_streamlines(
        self,
        seed_points: np.ndarray | None = None,
        n_steps: int = 300,
        length: float | None = None,
    ) -> np.ndarray:
        """Compute trailing streamlines starting from trailing-edge panel centers."""
        if length is None:
            length = self.assembly.c_ref * 5.0
        if seed_points is None:
            left_te = self.back_left_vertices
            right_te = self.back_right_vertices
            n_target = 100
            per_panel = max(1, n_target // max(1, len(left_te)))
            nodes = np.linspace(0.0, 1.0, per_panel + 1)
            locs = 0.5 * (nodes[1:] + nodes[:-1])
            seed_points = np.concatenate(
                [x * left_te + (1.0 - x) * right_te for x in locs]
            )

        streamlines = np.empty((len(seed_points), 3, n_steps))
        streamlines[:, :, 0] = seed_points
        for i in range(1, n_steps):
            vel = self.get_velocity_at_points(streamlines[:, :, i - 1])
            streamlines[:, :, i] = streamlines[:, :, i - 1] + (
                length / n_steps
            ) * vel / tall(np.linalg.norm(vel, axis=1))
        self.streamlines = streamlines
        return streamlines

    def draw(
        self,
        c: np.ndarray | None = None,
        cmap: str | None = "viridis",
        colorbar_label: str | None = None,
        show: bool = True,
        show_kwargs: dict | None = None,
        draw_streamlines: bool = True,
        recalculate_streamlines: bool = False,
        draw_free_surface: bool = True,
        backend: Literal["matplotlib", "pyvista", "plotly"] = "matplotlib",
    ):
        """Draw the 3D HydroLiftingLine mesh, solution scalars, streamlines, and free water surface."""
        if show_kwargs is None:
            show_kwargs = {}

        if c is None:
            c = self.vortex_strengths
            colorbar_label = "Vortex Strengths"

        if draw_streamlines and len(self.front_left_vertices) > 0:
            if (not hasattr(self, "streamlines")) or recalculate_streamlines:
                self.calculate_streamlines()

        surface_z = getattr(self, "surface_z", None)

        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            quads = [
                np.stack(
                    [
                        self.front_left_vertices[i],
                        self.back_left_vertices[i],
                        self.back_right_vertices[i],
                        self.front_right_vertices[i],
                    ]
                )
                for i in range(len(self.front_left_vertices))
            ]

            if quads:
                c_arr = np.asarray(c, dtype=float)
                norm = Normalize(vmin=float(np.min(c_arr)), vmax=float(np.max(c_arr)))
                cmap_obj = plt.get_cmap(cmap)
                facecolors = cmap_obj(norm(c_arr))
                poly = Poly3DCollection(
                    quads, facecolors=facecolors, edgecolors="k", linewidths=0.4
                )
                ax.add_collection3d(poly)
                sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
                sm.set_array(c_arr)
                fig.colorbar(sm, ax=ax, shrink=0.65, label=colorbar_label or "")

                all_pts = np.concatenate(quads, axis=0)
                mins = np.min(all_pts, axis=0)
                maxs = np.max(all_pts, axis=0)
                pad = max(0.3, 0.25 * float(np.max(maxs - mins)))
                xlim = (float(mins[0] - pad), float(maxs[0] + pad))
                ylim = (float(mins[1] - pad), float(maxs[1] + pad))
                zlim = (
                    float(mins[2] - pad),
                    float(
                        max(maxs[2], surface_z if surface_z is not None else maxs[2])
                        + pad
                    ),
                )

                if draw_free_surface and surface_z is not None:
                    sz = float(surface_z)
                    fs_quad = [
                        np.array(
                            [
                                [xlim[0], ylim[0], sz],
                                [xlim[1], ylim[0], sz],
                                [xlim[1], ylim[1], sz],
                                [xlim[0], ylim[1], sz],
                            ]
                        )
                    ]
                    ax.add_collection3d(
                        Poly3DCollection(
                            fs_quad,
                            facecolors=[(0.1, 0.65, 0.9, 0.22)],
                            edgecolors=[(0.0, 0.4, 0.7, 0.5)],
                            linewidths=0.6,
                        )
                    )

                if draw_streamlines and hasattr(self, "streamlines"):
                    for i in range(self.streamlines.shape[0]):
                        pts = self.streamlines[i]
                        ax.plot(pts[0], pts[1], pts[2], color="#7700FF", alpha=0.5, lw=0.7)

                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_zlim(*zlim)
                ax.set_box_aspect(
                    [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]]
                )

            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.set_zlabel("z [m]")
            ax.set_title("HydroLiftingLine")
            if show:
                plt.show(**show_kwargs)
            return fig, ax

        elif backend == "plotly":
            from aerosandbox.visualization.plotly_Figure3D import Figure3D

            fig = Figure3D()
            for i in range(len(self.front_left_vertices)):
                fig.add_quad(
                    points=[
                        self.front_left_vertices[i, :],
                        self.back_left_vertices[i, :],
                        self.back_right_vertices[i, :],
                        self.front_right_vertices[i, :],
                    ],
                    intensity=c[i],
                    outline=True,
                )
            if draw_streamlines and hasattr(self, "streamlines"):
                for i in range(self.streamlines.shape[0]):
                    fig.add_streamline(self.streamlines[i, :, :].T)
            return fig.draw(show=show, colorbar_title=colorbar_label, **show_kwargs)

        elif backend == "pyvista":
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.title = "HydroSandbox HydroLiftingLine"
            plotter.add_axes()
            plotter.show_grid(color="gray")

            points = np.concatenate(
                [
                    self.front_left_vertices,
                    self.back_left_vertices,
                    self.back_right_vertices,
                    self.front_right_vertices,
                ]
            )
            n = len(self.front_left_vertices)
            range_n = np.arange(n)
            faces = tall(range_n) + wide(np.array([0, 1, 2, 3]) * n)
            mesh = pv.PolyData(
                *mesh_utils.convert_mesh_to_polydata_format(points, faces)
            )
            scalar_bar_args = {}
            if colorbar_label is not None:
                scalar_bar_args["title"] = colorbar_label
            plotter.add_mesh(
                mesh=mesh,
                scalars=c,
                show_edges=True,
                show_scalar_bar=c is not None,
                scalar_bar_args=scalar_bar_args,
                cmap=cmap,
            )
            if draw_free_surface and surface_z is not None:
                mins = np.min(points, axis=0)
                maxs = np.max(points, axis=0)
                span = max(1.0, float(np.max(maxs - mins)))
                plane = pv.Plane(
                    center=(
                        0.5 * (mins[0] + maxs[0]),
                        0.5 * (mins[1] + maxs[1]),
                        float(surface_z),
                    ),
                    direction=(0, 0, 1),
                    i_size=2.0 * span,
                    j_size=2.0 * span,
                )
                plotter.add_mesh(plane, color="deepskyblue", opacity=0.25)
            if draw_streamlines and hasattr(self, "streamlines"):
                for i in range(self.streamlines.shape[0]):
                    plotter.add_mesh(
                        pv.Spline(self.streamlines[i, :, :].T),
                        color="#7700FF",
                        opacity=0.7,
                        line_width=1,
                    )
            if show:
                plotter.show(**show_kwargs)
            return plotter

        else:
            raise ValueError(
                f"Bad value of `backend` ({backend!r}). Expected 'matplotlib', 'pyvista', or 'plotly'."
            )
