import copy
from typing import Any, Callable, Literal
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import Airplane
import aerosandbox.geometry.mesh_utilities as mesh_utils
import aerosandbox.numpy as np
from aerosandbox.performance import OperatingPoint
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import HydrofoilAssembly
from hydrosandbox.hydrodynamics.singularities import (
    calculate_induced_velocity_horseshoe_hydro,
)
from hydrosandbox.library.cavitation import (
    cavitation_inception_speed,
    cavitation_number,
)
from hydrosandbox.library.free_surface import froude_image_weight


def tall(array):
    """Reshape an array into a tall (Nx1) column vector."""
    return np.reshape(array, (-1, 1))


def wide(array):
    """Reshape an array into a wide (1xN) row vector."""
    return np.reshape(array, (1, -1))


class HydroVLM(ExplicitAnalysis):
    """
    3D Vortex Lattice Method (VLM) hydrodynamics solver for lifting hydrofoil assemblies.

    Supports:
    - Automatic clipping of surface-piercing struts/foils at ``op_point.surface_z``.
    - Free-surface method-of-images boundary conditions across ``z = surface_z``
      (``"high_froude"``, ``"froude_blended"``, ``"rigid_lid"``, or ``"none"``).
    - Panel-level pressure difference ``Delta_Cp``, suction peak ``Cpmin``, local
      cavitation number ``sigma``, and cavitation margin diagnostics.
    - Hydrostatic buoyancy reporting when given a ``HydrofoilAssembly`` and ``Water``.
    - Stability derivatives including vertical heave stiffness ``CLh`` and ``L_h``.
    """

    def __init__(
        self,
        assembly: HydrofoilAssembly | Airplane | None = None,
        op_point: OperatingPoint | None = None,
        xyz_ref: list[float] | np.ndarray | None = None,
        run_symmetric_if_possible: bool = False,
        verbose: bool = False,
        spanwise_resolution: int = 10,
        spanwise_spacing_function: Callable[
            [float, float, int], np.ndarray
        ] = np.cosspace,
        chordwise_resolution: int = 10,
        chordwise_spacing_function: Callable[
            [float, float, int], np.ndarray
        ] = np.cosspace,
        vortex_core_radius: float = 1e-8,
        align_trailing_vortices_with_wind: bool = False,
        free_surface_mode: Literal[
            "high_froude", "froude_blended", "rigid_lid", "none"
        ] = "high_froude",
        clip_to_free_surface: bool = True,
        airplane: Airplane | HydrofoilAssembly | None = None,
    ):
        super().__init__()

        if assembly is None:
            assembly = airplane
        if assembly is None:
            raise ValueError("Must provide `assembly` (a HydrofoilAssembly or Airplane).")
        if op_point is None:
            raise ValueError("Must provide `op_point` (a HydroOperatingPoint or OperatingPoint).")

        if xyz_ref is None:
            xyz_ref = assembly.xyz_ref

        self.assembly = assembly
        self.op_point = op_point
        self.xyz_ref = xyz_ref
        self.verbose = verbose
        self.spanwise_resolution = spanwise_resolution
        self.spanwise_spacing_function = spanwise_spacing_function
        self.chordwise_resolution = chordwise_resolution
        self.chordwise_spacing_function = chordwise_spacing_function
        self.vortex_core_radius = vortex_core_radius
        self.align_trailing_vortices_with_wind = align_trailing_vortices_with_wind
        self.free_surface_mode = free_surface_mode
        self.clip_to_free_surface = clip_to_free_surface

        if run_symmetric_if_possible:
            raise NotImplementedError(
                "HydroVLM with symmetry detection is not yet implemented."
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

    def run(self) -> dict[str, Any]:
        """Compute 3D hydrodynamic forces, moments, and cavitation diagnostics."""
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
            print("Meshing submerged hydrofoil panels...")

        front_left_vertices = []
        back_left_vertices = []
        back_right_vertices = []
        front_right_vertices = []
        is_trailing_edge = []
        panel_t_over_c = []
        tc_cache: dict[int, float] = {}

        def _get_tc(af) -> float:
            key = id(af)
            if key not in tc_cache:
                tc_cache[key] = float(af.max_thickness())
            return tc_cache[key]

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
                chordwise_resolution=self.chordwise_resolution,
                chordwise_spacing_function=self.chordwise_spacing_function,
                add_camber=True,
            )
            if len(faces) == 0:
                continue

            front_left_vertices.append(points[faces[:, 0], :])
            back_left_vertices.append(points[faces[:, 1], :])
            back_right_vertices.append(points[faces[:, 2], :])
            front_right_vertices.append(points[faces[:, 3], :])
            is_trailing_edge.append(
                (np.arange(len(faces)) + 1) % self.chordwise_resolution == 0
            )

            sec_tc = np.array(
                [
                    0.5 * (_get_tc(xa.airfoil) + _get_tc(xb.airfoil))
                    for xa, xb in zip(wing.xsecs[:-1], wing.xsecs[1:])
                ]
            )
            tc_half = np.repeat(sec_tc, self.chordwise_resolution)
            if wing.symmetric:
                panel_t_over_c.append(np.concatenate([tc_half, tc_half]))
            else:
                panel_t_over_c.append(tc_half)

        if not front_left_vertices:
            self.vortex_strengths = np.array([])
            self.front_left_vertices = np.zeros((0, 3))
            self.back_left_vertices = np.zeros((0, 3))
            self.back_right_vertices = np.zeros((0, 3))
            self.front_right_vertices = np.zeros((0, 3))
            self.image_weight = 0.0
            output: dict[str, Any] = {
                "F_g": np.zeros(3),
                "F_b": (0.0, 0.0, 0.0),
                "F_w": (0.0, 0.0, 0.0),
                "M_g": np.zeros(3),
                "M_b": (0.0, 0.0, 0.0),
                "M_w": (0.0, 0.0, 0.0),
                "L": 0.0,
                "D": 0.0,
                "Y": 0.0,
                "l_b": 0.0,
                "m_b": 0.0,
                "n_b": 0.0,
                "CL": 0.0,
                "CD": 0.0,
                "CY": 0.0,
                "Cl": 0.0,
                "Cm": 0.0,
                "Cn": 0.0,
                "cavitation_margin": np.inf,
                "Cpmin_global": 0.0,
                "sigma_min": np.inf,
            }
            if isinstance(self.assembly, HydrofoilAssembly) and isinstance(
                self.op_point.atmosphere, Water
            ):
                output.update(
                    self.assembly.hydrostatic_forces(
                        water=self.op_point.atmosphere,
                        surface_z=surface_z,
                        xyz_ref=self.xyz_ref,
                    )
                )
            return output

        front_left_vertices = np.concatenate(front_left_vertices)
        back_left_vertices = np.concatenate(back_left_vertices)
        back_right_vertices = np.concatenate(back_right_vertices)
        front_right_vertices = np.concatenate(front_right_vertices)
        is_trailing_edge = np.concatenate(is_trailing_edge)
        panel_t_over_c = np.concatenate(panel_t_over_c)

        diag1 = front_right_vertices - back_left_vertices
        diag2 = front_left_vertices - back_right_vertices
        cross = np.cross(diag1, diag2)
        cross_norm = np.linalg.norm(cross, axis=1)
        normal_directions = cross / tall(cross_norm)
        areas = cross_norm / 2.0

        left_vortex_vertices = 0.75 * front_left_vertices + 0.25 * back_left_vertices
        right_vortex_vertices = 0.75 * front_right_vertices + 0.25 * back_right_vertices
        vortex_centers = (left_vortex_vertices + right_vortex_vertices) / 2.0
        vortex_bound_leg = right_vortex_vertices - left_vortex_vertices
        collocation_points = 0.5 * (
            0.25 * front_left_vertices + 0.75 * back_left_vertices
        ) + 0.5 * (0.25 * front_right_vertices + 0.75 * back_right_vertices)

        self.front_left_vertices = front_left_vertices
        self.back_left_vertices = back_left_vertices
        self.back_right_vertices = back_right_vertices
        self.front_right_vertices = front_right_vertices
        self.is_trailing_edge = is_trailing_edge
        self.normal_directions = normal_directions
        self.areas = areas
        self.panel_t_over_c = panel_t_over_c
        self.left_vortex_vertices = left_vortex_vertices
        self.right_vortex_vertices = right_vortex_vertices
        self.vortex_centers = vortex_centers
        self.vortex_bound_leg = vortex_bound_leg
        self.collocation_points = collocation_points

        self.image_weight = self._compute_image_weight(surface_z, vortex_centers)

        steady_freestream_velocity = (
            self.op_point.compute_freestream_velocity_geometry_axes()
        )
        steady_freestream_direction = steady_freestream_velocity / np.linalg.norm(
            steady_freestream_velocity
        )
        rotation_freestream_velocities = (
            self.op_point.compute_rotation_velocity_geometry_axes(collocation_points)
        )
        freestream_velocities = np.add(
            wide(steady_freestream_velocity), rotation_freestream_velocities
        )
        freestream_influences = np.sum(
            freestream_velocities * normal_directions, axis=1
        )

        self.steady_freestream_velocity = steady_freestream_velocity
        self.steady_freestream_direction = steady_freestream_direction
        self.freestream_velocities = freestream_velocities

        u_col_unit, v_col_unit, w_col_unit = calculate_induced_velocity_horseshoe_hydro(
            x_field=tall(collocation_points[:, 0]),
            y_field=tall(collocation_points[:, 1]),
            z_field=tall(collocation_points[:, 2]),
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
        self.vortex_strengths = np.linalg.solve(AIC, -freestream_influences)

        v_centers = self.get_velocity_at_points(vortex_centers)
        vi_cross_li = np.cross(v_centers, vortex_bound_leg, axis=1)

        forces_geometry = (
            self.op_point.atmosphere.density()
            * vi_cross_li
            * tall(self.vortex_strengths)
        )
        moments_geometry = np.cross(
            np.add(vortex_centers, -wide(np.array(self.xyz_ref))),
            forces_geometry,
        )

        force_geometry = np.sum(forces_geometry, axis=0)
        moment_geometry = np.sum(moments_geometry, axis=0)

        force_body = self.op_point.convert_axes(
            force_geometry[0],
            force_geometry[1],
            force_geometry[2],
            from_axes="geometry",
            to_axes="body",
        )
        force_wind = self.op_point.convert_axes(
            force_body[0],
            force_body[1],
            force_body[2],
            from_axes="body",
            to_axes="wind",
        )
        moment_body = self.op_point.convert_axes(
            moment_geometry[0],
            moment_geometry[1],
            moment_geometry[2],
            from_axes="geometry",
            to_axes="body",
        )
        moment_wind = self.op_point.convert_axes(
            moment_body[0],
            moment_body[1],
            moment_body[2],
            from_axes="body",
            to_axes="wind",
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
        l_b = moment_body[0]
        m_b = moment_body[1]
        n_b = moment_body[2]

        q = self.op_point.dynamic_pressure()
        s_ref = self.assembly.s_ref
        b_ref = self.assembly.b_ref
        c_ref = self.assembly.c_ref

        CL = L / q / s_ref
        CD = D / q / s_ref
        CY = Y / q / s_ref
        Cl = l_b / q / s_ref / b_ref
        Cm = m_b / q / s_ref / c_ref
        Cn = n_b / q / s_ref / b_ref

        # Panel pressure loading & cavitation diagnostics
        panel_Delta_Cp = np.sum(forces_geometry * normal_directions, axis=1) / (
            q * areas
        )
        panel_Cpmin = (
            1.0
            - (1.0 + 1.20 * panel_t_over_c + 0.8 * panel_t_over_c**2) ** 2
            - 0.50 * np.abs(panel_Delta_Cp)
        )
        self.panel_Delta_Cp = panel_Delta_Cp
        self.panel_Cpmin = panel_Cpmin

        output = {
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
            "CL": CL,
            "CD": CD,
            "CY": CY,
            "Cl": Cl,
            "Cm": Cm,
            "Cn": Cn,
            "panel_Delta_Cp": panel_Delta_Cp,
            "panel_Cpmin": panel_Cpmin,
            "Cpmin_global": np.min(panel_Cpmin),
        }

        fluid = self.op_point.atmosphere
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
                density=fluid.density(),
            )
            self.panel_sigma = panel_sigma
            self.panel_cavitation_margin = panel_cav_margin
            output["panel_sigma"] = panel_sigma
            output["panel_cavitation_margin"] = panel_cav_margin
            output["sigma_min"] = np.min(panel_sigma)
            output["cavitation_margin"] = np.min(panel_cav_margin)
            output["V_cavitation_inception"] = v_inc

        if isinstance(self.assembly, HydrofoilAssembly) and isinstance(fluid, Water):
            output.update(
                self.assembly.hydrostatic_forces(
                    water=fluid,
                    surface_z=surface_z,
                    xyz_ref=self.xyz_ref,
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
        Compute hydrodynamic forces, moments, and finite-difference stability derivatives
        (including heave derivatives ``CLh``, ``CDh``, ``Cmh``, and ``L_h`` when
        ``op_point.surface_z`` is specified).
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

        for denom in abbreviations:
            if not do_analysis[denom]:
                continue
            inc_op = copy.copy(original_op_point)
            setattr(
                inc_op,
                denom,
                getattr(original_op_point, denom) + finite_difference_amounts[denom],
            )
            vlm_inc = copy.copy(self)
            vlm_inc.op_point = inc_op
            run_inc = vlm_inc.run()

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
            vlm_inc = copy.copy(self)
            vlm_inc.op_point = inc_op
            run_inc = vlm_inc.run()

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
            left_te = self.back_left_vertices[self.is_trailing_edge.astype(bool)]
            right_te = self.back_right_vertices[self.is_trailing_edge.astype(bool)]
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
        """Draw the 3D HydroVLM mesh, solution scalars, streamlines, and free water surface."""
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
                    quads, facecolors=facecolors, edgecolors="k", linewidths=0.3
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
                    float(max(maxs[2], surface_z if surface_z is not None else maxs[2]) + pad),
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
            ax.set_title("HydroVLM")
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
            return fig.draw(
                show=show,
                colorbar_title=colorbar_label,
                **show_kwargs,
            )

        elif backend == "pyvista":
            import pyvista as pv

            plotter = pv.Plotter()
            plotter.title = "HydroSandbox HydroVLM"
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
