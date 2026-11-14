from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING
from aerosandbox import ExplicitAnalysis
from aerosandbox.geometry import Airplane, Fuselage, Wing
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels.fuselage_aerodynamics_utilities import (
    fuselage_base_drag_coefficient,
    fuselage_form_factor,
    jorgensen_eta,
)
from aerosandbox.aerodynamics.aero_3D.aero_buildup_submodels.softmax_scalefree import (
    softmax_scalefree,
)
import aerosandbox.library.aerodynamics as aerolib
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from aerosandbox.performance import OperatingPoint
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import HydrofoilAssembly
from hydrosandbox.library.cavitation import (
    cavitation_drag_increment,
    cavitation_inception_speed,
    cavitation_number,
    ventilation_margin_lifting_foil,
    ventilation_margin_strut,
)
from hydrosandbox.library.free_surface import (
    free_surface_induced_drag_factor,
    free_surface_lift_slope_factor,
    froude_image_weight,
    junction_interference_drag,
    smooth_submergence_fraction,
    smooth_surface_piercing_weight,
    spray_drag_strut,
)
from hydrosandbox.library.wave_resistance import (
    fuselage_wave_sources,
    kelvin_wave_resistance,
    merge_wave_elements,
    wing_thickness_wave_sources,
)

if TYPE_CHECKING:
    from aerosandbox.geometry import ControlSurface


def _safe_softmin(values: list, softness: float = 0.01):
    """Differentiable softmin handling lists of length 0, 1, or >=2."""
    if len(values) == 0:
        return np.inf
    if len(values) == 1:
        return values[0]
    return np.softmin(*values, softness=softness)


def _safe_softmax(values: list, softness: float = 0.01):
    """Differentiable softmax handling lists of length 0, 1, or >=2."""
    if len(values) == 0:
        return -np.inf
    if len(values) == 1:
        return values[0]
    return np.softmax(*values, softness=softness)


class HydroBuildup(ExplicitAnalysis):
    """
    Fast, vectorized, CasADi-differentiable 3D conceptual hydrodynamic buildup.

    Designed for multidisciplinary design optimization (MDO) of hydrofoil assemblies
    via ``asb.Opti``. Models:
    - Smooth submergence clipping across ``op_point.surface_z`` for foils, struts, and pods.
    - Free-surface lift-slope reduction and induced-drag penalty (``"high_froude"``,
      ``"froude_blended"``, ``"rigid_lid"``, or ``"none"``).
    - Incompressible 2D section polars and suction-peak ``Cpmin_0`` via NeuralFoil
      (``mach = 0.0``), local hydrostatic cavitation number ``sigma``, cavitation
      margin, and smooth cavitation drag rise.
    - Hoerner surface-piercing spray drag and T-junction interference drag.
    - Wave-making drag (``include_wave_drag``) by linear far-field (Michell / Kochin)
      theory: the lift and side force of every foil and strut (as pressure dipoles)
      and the thickness of every foil, strut and pod (as sources), with their mutual
      interference. When active, induced drag is evaluated with a rigid-lid image
      (``1 - sigma``) and ``D_wave`` carries the rest of the free-surface effect, so
      that their sum tends to the high-Froude image result as ``Fr -> infinity``.
    - Foil/strut ventilation safety margins and Archimedes hydrostatic buoyancy.
    - Finite-difference stability derivatives including vertical heave stiffness
      (``CLh`` and ``L_h``).
    """

    def __init__(
        self,
        assembly: HydrofoilAssembly | Airplane | None = None,
        op_point: OperatingPoint | None = None,
        xyz_ref: np.ndarray | list[float] | None = None,
        model_size: Literal[
            "xxsmall", "xsmall", "small", "medium", "large", "xlarge"
        ] = "small",
        free_surface_mode: Literal[
            "high_froude", "froude_blended", "rigid_lid", "none"
        ] = "high_froude",
        include_spray_drag: bool = True,
        include_junction_drag: bool = True,
        n_junctions: int | None = None,
        include_cavitation_penalty: bool = True,
        include_buoyancy_in_totals: bool = False,
        include_wave_drag: bool = True,
        wave_length_scales: tuple[float, float, float] | None = None,
        wave_reference_velocity: float | None = None,
        airplane: Airplane | HydrofoilAssembly | None = None,
    ):
        """
        Parameters beyond the obvious:

        include_wave_drag : Add wave-making drag (needs ``op_point.surface_z`` and
            ``free_surface_mode != "none"``).
        wave_length_scales : ``(y_extent, x_extent, min_depth)`` [m] of the submerged
            assembly, used only to size the wave-integral quadrature when the
            geometry is symbolic (default ``(2, 2, 0.05)``; inferred when numeric).
        wave_reference_velocity : Numeric speed [m/s] used to size that quadrature
            when ``op_point.velocity`` is symbolic.
        """
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
        self.model_size = model_size
        self.free_surface_mode = free_surface_mode
        self.include_spray_drag = include_spray_drag
        self.include_junction_drag = include_junction_drag
        self.n_junctions = n_junctions
        self.include_cavitation_penalty = include_cavitation_penalty
        self.include_buoyancy_in_totals = include_buoyancy_in_totals
        self.include_wave_drag = include_wave_drag
        self.wave_length_scales = wave_length_scales
        self.wave_reference_velocity = wave_reference_velocity

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

    @dataclass
    class HydroComponentResults:
        """Hydrodynamic forces, moments, and diagnostics on a single assembly component."""

        s_ref: float
        c_ref: float
        b_ref: float
        op_point: OperatingPoint
        F_g: list[Vectorizable]
        M_g: list[Vectorizable]
        span_L_effective: Vectorizable
        span_Y_effective: Vectorizable
        oswalds_efficiency: Vectorizable
        D_profile: Vectorizable = 0.0
        D_cavitation: Vectorizable = 0.0
        D_spray: Vectorizable = 0.0
        Cpmin_min: Vectorizable = 0.0
        sigma_min: Vectorizable = np.inf
        cavitation_margin: Vectorizable = np.inf
        ventilation_margin: Vectorizable = np.inf
        V_cavitation_inception: Vectorizable = np.inf
        submerged_weight: Vectorizable = 1.0
        junction_chord: Vectorizable = 0.1
        junction_tc: Vectorizable = 0.12
        junction_xyz: list[Vectorizable] | None = None
        wave_elements: dict[str, Any] | None = None

        @property
        def F_b(self) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
            return self.op_point.convert_axes(
                self.F_g[0], self.F_g[1], self.F_g[2], from_axes="geometry", to_axes="body"
            )

        @property
        def F_w(self) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
            return self.op_point.convert_axes(
                self.F_g[0], self.F_g[1], self.F_g[2], from_axes="geometry", to_axes="wind"
            )

        @property
        def M_b(self) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
            return self.op_point.convert_axes(
                self.M_g[0], self.M_g[1], self.M_g[2], from_axes="geometry", to_axes="body"
            )

        @property
        def M_w(self) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
            return self.op_point.convert_axes(
                self.M_g[0], self.M_g[1], self.M_g[2], from_axes="geometry", to_axes="wind"
            )

        @property
        def L(self) -> Vectorizable:
            return -self.F_w[2]

        @property
        def Y(self) -> Vectorizable:
            return self.F_w[1]

        @property
        def D(self) -> Vectorizable:
            return -self.F_w[0]

        @property
        def l_b(self) -> Vectorizable:
            return self.M_b[0]

        @property
        def m_b(self) -> Vectorizable:
            return self.M_b[1]

        @property
        def n_b(self) -> Vectorizable:
            return self.M_b[2]

    def _local_depth_and_pressure(
        self, z_g: Vectorizable
    ) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
        """Return ``(depth, p_local, p_vapor)`` at geometry elevation ``z_g``."""
        fluid = self.op_point.atmosphere
        surface_z = getattr(self.op_point, "surface_z", None)
        if surface_z is not None:
            depth = surface_z - z_g
        elif hasattr(fluid, "depth"):
            depth = fluid.depth - z_g
        else:
            depth = 0.0

        if isinstance(fluid, Water) or hasattr(fluid, "vapor_pressure"):
            p_local = fluid.pressure(depth=depth)
            p_vapor = fluid.vapor_pressure()
        else:
            p_local = fluid.pressure()
            p_vapor = 1705.0
        return depth, p_local, p_vapor

    def _wave_drag_active(self) -> bool:
        return (
            self.include_wave_drag
            and getattr(self.op_point, "surface_z", None) is not None
            and self.free_surface_mode != "none"
        )

    def _wave_resistance(self, elements: dict[str, Any]) -> Vectorizable:
        """
        Evaluate the far-field wave resistance of merged wave elements, looping over
        a numerically vectorized operating point if needed.
        """
        op_point = self.op_point
        fluid = op_point.atmosphere
        surface_z = op_point.surface_z
        strength_keys = ("source_flux", "force_y", "force_z")

        def size_of(value) -> int:
            if np.is_casadi_type(value, recursive=False):
                return 1
            return int(np.size(value))

        def pick(value, k: int):
            if np.is_casadi_type(value, recursive=False):
                return value
            flat = np.reshape(np.asarray(value), -1)
            return flat[0] if flat.size == 1 else flat[k]

        sizes = [size_of(v) for key in strength_keys for v in elements[key]]
        sizes += [size_of(op_point.velocity), size_of(surface_z)]
        n_points = max(sizes) if sizes else 1

        def evaluate(k: int):
            velocity = pick(op_point.velocity, k)
            return kelvin_wave_resistance(
                xyz_a=elements["xyz_a"],
                xyz_b=elements["xyz_b"],
                source_flux=[pick(v, k) for v in elements["source_flux"]],
                force_y=[pick(v, k) for v in elements["force_y"]],
                force_z=[pick(v, k) for v in elements["force_z"]],
                smear_length=elements["smear_length"],
                velocity=velocity,
                density=pick(fluid.density(), k),
                surface_z=pick(surface_z, k),
                gravity=getattr(fluid, "gravity", 9.80665),
                length_scales=self.wave_length_scales,
                reference_velocity=self.wave_reference_velocity,
            )

        if n_points == 1:
            return evaluate(0)
        return np.array([evaluate(k) for k in range(n_points)])

    def run(self) -> dict[str, Any]:
        """
        Compute 3D hydrodynamic forces, moments, drag breakdown, cavitation/ventilation
        margins, and hydrostatic buoyancy.
        """
        wing_components = [
            self.wing_hydrodynamics(wing=wing, include_induced_drag=False)
            for wing in self.assembly.wings
        ]
        fuse_components = [
            self.fuselage_hydrodynamics(fuselage=fuse, include_induced_drag=False)
            for fuse in self.assembly.fuselages
        ]

        components = wing_components + fuse_components

        if not components:
            F_g_total = [0.0, 0.0, 0.0]
            M_g_total = [0.0, 0.0, 0.0]
        else:
            F_g_total = [sum(comp.F_g[i] for comp in components) for i in range(3)]
            M_g_total = [sum(comp.M_g[i] for comp in components) for i in range(3)]

        Q = self.op_point.dynamic_pressure()
        _, sideforce, lift = self.op_point.convert_axes(
            F_g_total[0], F_g_total[1], F_g_total[2], from_axes="geometry", to_axes="wind"
        )

        if components:
            span_L_sq = softmax_scalefree(
                [
                    comp.span_L_effective**2 * comp.oswalds_efficiency
                    for comp in components
                ]
            )
            span_Y_sq = softmax_scalefree(
                [
                    comp.span_Y_effective**2 * comp.oswalds_efficiency
                    for comp in components
                ]
            )
            D_induced = lift**2 / (Q * np.pi * span_L_sq) + sideforce**2 / (
                Q * np.pi * span_Y_sq
            )
        else:
            D_induced = 0.0

        D_induced_g = self.op_point.convert_axes(
            -D_induced, 0.0, 0.0, from_axes="wind", to_axes="geometry"
        )
        for i in range(3):
            F_g_total[i] = F_g_total[i] + D_induced_g[i]

        # Junction interference drag
        D_junction = 0.0
        if self.include_junction_drag and wing_components:
            if self.n_junctions is not None:
                n_junc = self.n_junctions
                ref_comp = wing_components[0]
                d_junc_single = junction_interference_drag(
                    chord=ref_comp.junction_chord,
                    thickness_to_chord=ref_comp.junction_tc,
                    dynamic_pressure=Q,
                    n_junctions=1,
                )
                D_junction = n_junc * d_junc_single
                junc_arm = [
                    (ref_comp.junction_xyz[i] - self.xyz_ref[i])
                    if ref_comp.junction_xyz is not None
                    else 0.0
                    for i in range(3)
                ]
                f_junc_g = self.op_point.convert_axes(
                    -D_junction, 0.0, 0.0, from_axes="wind", to_axes="geometry"
                )
                m_junc_g = np.cross(junc_arm, f_junc_g, manual=True)
                for i in range(3):
                    F_g_total[i] = F_g_total[i] + f_junc_g[i]
                    M_g_total[i] = M_g_total[i] + m_junc_g[i]
            else:
                has_fuse = len(fuse_components) > 0
                fuse_w = (
                    _safe_softmax([fc.submerged_weight for fc in fuse_components])
                    if has_fuse
                    else 1.0
                )
                for wing, comp in zip(self.assembly.wings, wing_components):
                    is_strut = getattr(wing, "is_strut", False)
                    if has_fuse or is_strut:
                        w_eff = comp.submerged_weight * (fuse_w if has_fuse else 1.0)
                        d_j = w_eff * junction_interference_drag(
                            chord=comp.junction_chord,
                            thickness_to_chord=comp.junction_tc,
                            dynamic_pressure=Q,
                            n_junctions=1,
                        )
                        D_junction = D_junction + d_j
                        junc_arm = [
                            (comp.junction_xyz[i] - self.xyz_ref[i])
                            if comp.junction_xyz is not None
                            else 0.0
                            for i in range(3)
                        ]
                        f_junc_g = self.op_point.convert_axes(
                            -d_j, 0.0, 0.0, from_axes="wind", to_axes="geometry"
                        )
                        m_junc_g = np.cross(junc_arm, f_junc_g, manual=True)
                        for i in range(3):
                            F_g_total[i] = F_g_total[i] + f_junc_g[i]
                            M_g_total[i] = M_g_total[i] + m_junc_g[i]

        # Wave-making drag (lift/side-force dipoles + thickness sources, with interference)
        D_wave = 0.0
        if self._wave_drag_active():
            surface_z_wave = self.op_point.surface_z
            element_sets = [
                comp.wave_elements
                for comp in wing_components
                if comp.wave_elements is not None
            ]
            lift_elements = merge_wave_elements(*element_sets)
            for wing in self.assembly.wings:
                element_sets.append(
                    wing_thickness_wave_sources(
                        wing, velocity=self.op_point.velocity, surface_z=surface_z_wave
                    )
                )
            for fuse in self.assembly.fuselages:
                element_sets.append(
                    fuselage_wave_sources(
                        fuse, velocity=self.op_point.velocity, surface_z=surface_z_wave
                    )
                )
            elements = merge_wave_elements(*element_sets)
            if len(elements["source_flux"]) > 0:
                D_wave = self._wave_resistance(elements)

                # Apply at the load-weighted centroid of the lifting elements.
                n_lift = len(lift_elements["force_z"])
                if n_lift > 0:
                    weights = [
                        np.sqrt(
                            lift_elements["force_y"][j] ** 2
                            + lift_elements["force_z"][j] ** 2
                            + 1e-12
                        )
                        for j in range(n_lift)
                    ]
                    weight_sum = sum(weights)
                    wave_arm = [
                        sum(
                            weights[j]
                            * 0.5
                            * (lift_elements["xyz_a"][k][j] + lift_elements["xyz_b"][k][j])
                            for j in range(n_lift)
                        )
                        / weight_sum
                        - self.xyz_ref[k]
                        for k in range(3)
                    ]
                else:
                    wave_arm = [0.0, 0.0, 0.0]
                f_wave_g = self.op_point.convert_axes(
                    -D_wave, 0.0, 0.0, from_axes="wind", to_axes="geometry"
                )
                m_wave_g = np.cross(wave_arm, f_wave_g, manual=True)
                for i in range(3):
                    F_g_total[i] = F_g_total[i] + f_wave_g[i]
                    M_g_total[i] = M_g_total[i] + m_wave_g[i]

        # Hydrostatic buoyancy
        fluid = self.op_point.atmosphere
        surface_z = getattr(self.op_point, "surface_z", None)
        if isinstance(self.assembly, HydrofoilAssembly) and isinstance(fluid, Water):
            hydrostatics = self.assembly.hydrostatic_forces(
                water=fluid,
                surface_z=surface_z,
                xyz_ref=self.xyz_ref,
            )
        else:
            hydrostatics = {
                "V_displaced": 0.0,
                "xyz_cb": list(self.xyz_ref),
                "buoyancy_force": 0.0,
                "F_buoyancy_g": [0.0, 0.0, 0.0],
                "M_buoyancy_g": [0.0, 0.0, 0.0],
            }

        F_g_with_buoyancy = [
            F_g_total[i] + hydrostatics["F_buoyancy_g"][i] for i in range(3)
        ]
        M_g_with_buoyancy = [
            M_g_total[i] + hydrostatics["M_buoyancy_g"][i] for i in range(3)
        ]

        if self.include_buoyancy_in_totals:
            F_g_out = F_g_with_buoyancy
            M_g_out = M_g_with_buoyancy
        else:
            F_g_out = F_g_total
            M_g_out = M_g_total

        output: dict[str, Any] = {
            "F_g": F_g_out,
            "M_g": M_g_out,
        }
        output["F_b"] = self.op_point.convert_axes(
            F_g_out[0], F_g_out[1], F_g_out[2], from_axes="geometry", to_axes="body"
        )
        output["F_w"] = self.op_point.convert_axes(
            F_g_out[0], F_g_out[1], F_g_out[2], from_axes="geometry", to_axes="wind"
        )
        output["M_b"] = self.op_point.convert_axes(
            M_g_out[0], M_g_out[1], M_g_out[2], from_axes="geometry", to_axes="body"
        )
        output["M_w"] = self.op_point.convert_axes(
            M_g_out[0], M_g_out[1], M_g_out[2], from_axes="geometry", to_axes="wind"
        )

        output["L"] = -output["F_w"][2]
        output["Y"] = output["F_w"][1]
        output["D"] = -output["F_w"][0]
        output["P_drag"] = output["D"] * self.op_point.velocity
        output["l_b"] = output["M_b"][0]
        output["m_b"] = output["M_b"][1]
        output["n_b"] = output["M_b"][2]

        qS = self.op_point.dynamic_pressure() * self.assembly.s_ref
        c = self.assembly.c_ref
        b = self.assembly.b_ref

        output["CL"] = output["L"] / qS
        output["CY"] = output["Y"] / qS
        output["CD"] = output["D"] / qS
        output["Cl"] = output["l_b"] / qS / b
        output["Cm"] = output["m_b"] / qS / c
        output["Cn"] = output["n_b"] / qS / b

        # Drag breakdown (sums identically to output["D"] when include_buoyancy_in_totals=False)
        output["D_profile"] = sum(comp.D_profile for comp in components)
        output["D_induced"] = D_induced
        output["D_spray"] = sum(comp.D_spray for comp in wing_components)
        output["D_junction"] = D_junction
        output["D_cavitation"] = sum(comp.D_cavitation for comp in wing_components)
        output["D_wave"] = D_wave

        # Cavitation & ventilation summaries across submerged wing components
        if wing_components:
            output["Cpmin_global"] = _safe_softmin(
                [comp.Cpmin_min for comp in wing_components], softness=0.01
            )
            output["sigma_min"] = _safe_softmin(
                [comp.sigma_min for comp in wing_components], softness=0.01
            )
            output["cavitation_margin"] = _safe_softmin(
                [comp.cavitation_margin for comp in wing_components], softness=0.01
            )
            output["V_cavitation_inception"] = _safe_softmin(
                [comp.V_cavitation_inception for comp in wing_components], softness=0.05
            )
            output["ventilation_margin"] = _safe_softmin(
                [comp.ventilation_margin for comp in wing_components], softness=0.01
            )
        else:
            output["Cpmin_global"] = 0.0
            output["sigma_min"] = np.inf
            output["cavitation_margin"] = np.inf
            output["V_cavitation_inception"] = np.inf
            output["ventilation_margin"] = np.inf

        output.update(hydrostatics)
        output["F_g_with_buoyancy"] = F_g_with_buoyancy
        output["M_g_with_buoyancy"] = M_g_with_buoyancy
        output["wing_hydro_components"] = wing_components
        output["fuselage_hydro_components"] = fuse_components

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
        Compute hydrodynamic forces, moments, and stability derivatives, including
        heave derivatives ``CLh``, ``CDh``, ``Cmh``, and ``L_h`` when
        ``op_point.surface_z`` is specified.
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

        run_base = self.run()

        for d, active in do_analysis.items():
            if not active:
                continue
            inc_op = self.op_point.copy()
            setattr(
                inc_op,
                d,
                getattr(self.op_point, d) + finite_difference_amounts[d],
            )
            hb_inc = self.copy()
            hb_inc.op_point = inc_op
            run_inc = hb_inc.run()

            for num in ("CL", "CD", "CY", "Cl", "Cm", "Cn"):
                run_base[num + abbreviations[d]] = (
                    (run_inc[num] - run_base[num])
                    / finite_difference_amounts[d]
                    * scaling_factors[d]
                )

            if d == "alpha":
                cla = np.where(run_base["CLa"] == 0, np.nan, run_base["CLa"])
                run_base["x_np"] = self.xyz_ref[0] - (
                    run_base["Cma"] / cla * self.assembly.c_ref
                )
            if d == "beta":
                cyb = np.where(run_base["CYb"] == 0, np.nan, run_base["CYb"])
                run_base["x_np_lateral"] = self.xyz_ref[0] - (
                    run_base["Cnb"] / cyb * self.assembly.b_ref
                )

        if surface_z and getattr(self.op_point, "surface_z", None) is not None:
            delta_h = 1e-3
            inc_op = self.op_point.copy()
            inc_op.surface_z = self.op_point.surface_z + delta_h
            hb_inc = self.copy()
            hb_inc.op_point = inc_op
            run_inc = hb_inc.run()

            for num in ("CL", "CD", "Cm"):
                run_base[num + "h"] = (run_inc[num] - run_base[num]) / delta_h
            run_base["L_h"] = (run_inc["L"] - run_base["L"]) / delta_h

        return run_base

    def wing_hydrodynamics(
        self,
        wing: Wing,
        include_induced_drag: bool = True,
    ) -> HydroComponentResults:
        """
        Compute hydrodynamic forces, moments, cavitation, spray, and ventilation
        metrics on a ``Wing`` / ``Foil`` / ``Strut``.
        """
        op_point = self.op_point
        surface_z = getattr(op_point, "surface_z", None)
        fluid = op_point.atmosphere
        g = getattr(fluid, "gravity", 9.80665)

        wing_MAC = wing.mean_aerodynamic_chord()
        wing_taper = wing.taper_ratio()
        wing_sweep = wing.mean_sweep_angle()
        wing_dihedral = wing.mean_dihedral_angle()

        sectional_spans = wing.span(
            type="yz", include_centerline_distance=False, _sectional=True
        )
        xsec_chords = [xsec.chord for xsec in wing.xsecs]
        sectional_chords = [
            0.5 * (c_a + c_b) for c_a, c_b in zip(xsec_chords[:-1], xsec_chords[1:])
        ]
        sectional_areas = [
            span * chord for span, chord in zip(sectional_spans, sectional_chords)
        ]

        # Section-by-section submergence & surface-piercing weights
        n_sects = len(wing.xsecs) - 1
        sect_f_sub = []
        sect_s_sub = []
        sect_h_sub_mid = []
        sect_pierce_w = []
        sect_t_pierce = []

        for i in range(n_sects):
            z_a = wing.xsecs[i].xyz_le[2]
            z_b = wing.xsecs[i + 1].xyz_le[2]
            if surface_z is not None:
                f_sub, s_sub, h_sub_mid = smooth_submergence_fraction(
                    z_a, z_b, surface_z
                )
                pierce_w, t_pierce = smooth_surface_piercing_weight(
                    z_a, z_b, surface_z
                )
            else:
                f_sub = 1.0
                s_sub = 0.5
                z_mid = 0.5 * (z_a + z_b)
                h_sub_mid = (
                    fluid.depth - z_mid if hasattr(fluid, "depth") else 1.0
                )
                pierce_w = 0.0
                t_pierce = 0.5

            sect_f_sub.append(f_sub)
            sect_s_sub.append(s_sub)
            sect_h_sub_mid.append(h_sub_mid)
            sect_pierce_w.append(pierce_w)
            sect_t_pierce.append(t_pierce)

        half_span_sub = sum(s * f for s, f in zip(sectional_spans, sect_f_sub))
        half_area_sub = sum(a * f for a, f in zip(sectional_areas, sect_f_sub))

        if len(wing.xsecs) > 0:
            span_inboard_to_YZ = np.inf
            for i in range(len(wing.xsecs)):
                span_inboard_to_YZ = np.minimum(
                    span_inboard_to_YZ,
                    np.abs(
                        wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0)[1]
                    ),
                )
        else:
            span_inboard_to_YZ = 0.0

        root_f_sub = sect_f_sub[0] if sect_f_sub else 0.0
        area_inboard_to_YZ = span_inboard_to_YZ * wing_MAC * root_f_sub

        if wing.symmetric:
            span_0_dih = 2.0 * (half_span_sub + span_inboard_to_YZ * 0.5 * root_f_sub)
            span_90_dih = half_span_sub
            area_0_dih = 2.0 * (half_area_sub + area_inboard_to_YZ * 0.5)
            area_90_dih = half_area_sub
            dihedral_factor = np.sind(wing_dihedral) ** 2
            span_eff_raw = span_0_dih + (span_90_dih - span_0_dih) * dihedral_factor
            area_eff_raw = area_0_dih + (area_90_dih - area_0_dih) * dihedral_factor
        else:
            span_eff_raw = half_span_sub
            area_eff_raw = half_area_sub

        span_eff_safe = np.softmax(span_eff_raw, 1e-4, softness=1e-4)
        area_eff_safe = np.softmax(area_eff_raw, 1e-5, softness=1e-5)
        AR_sub = span_eff_safe**2 / area_eff_safe

        is_strut = getattr(wing, "is_strut", False)
        cos_dih_sq = np.cosd(wing_dihedral) ** 2
        horiz_weight = 0.0 if is_strut else cos_dih_sq

        AR_3D_factor = aerolib.CL_over_Cl(
            aspect_ratio=AR_sub,
            mach=0.0,
            sweep=wing_sweep,
            Cl_is_compressible=True,
        )
        oswalds_eff_inf = aerolib.oswalds_efficiency(
            taper_ratio=wing_taper,
            aspect_ratio=AR_sub,
            sweep=wing_sweep,
            fuselage_diameter_to_span_ratio=0.0,
        )

        # Free-surface corrections
        if surface_z is not None and self.free_surface_mode != "none":
            weight_sum = np.maximum(
                sum(a * f for a, f in zip(sectional_areas, sect_f_sub)), 1e-8
            )
            h_mean = (
                sum(
                    a * f * h
                    for a, f, h in zip(sectional_areas, sect_f_sub, sect_h_sub_mid)
                )
                / weight_sum
            )
            fr_h_wing = op_point.velocity / np.sqrt(
                g * np.softmax(h_mean, 1e-3, softness=1e-3)
            )
            kappa_wing = froude_image_weight(
                froude_depth=fr_h_wing, mode=self.free_surface_mode
            )

            K_L_fs = free_surface_lift_slope_factor(
                h_over_c=h_mean / np.maximum(wing_MAC, 1e-4),
                aspect_ratio=AR_sub,
                froude_depth=fr_h_wing,
                mode=self.free_surface_mode,
            )
            # With wave drag active, the trailing-vortex (induced) drag is that of a
            # rigid-lid image; the wave resistance supplies the rest (see
            # `hydrosandbox.library.wave_resistance`).
            wave_active = self._wave_drag_active()
            K_Di_fs = free_surface_induced_drag_factor(
                h_over_b=h_mean / span_eff_safe,
                froude_depth=fr_h_wing,
                mode="rigid_lid" if wave_active else self.free_surface_mode,
            )
            kappa_drag = -1.0 if wave_active else kappa_wing
            K_Di_strut = 0.75 + 0.25 * kappa_drag
            K_Di_eff = horiz_weight * K_Di_fs + (1.0 - horiz_weight) * K_Di_strut

            AR_3D_eff = AR_3D_factor * (
                horiz_weight * K_L_fs + (1.0 - horiz_weight) * 1.0
            )
            oswalds_eff = oswalds_eff_inf / K_Di_eff
        else:
            AR_3D_eff = AR_3D_factor
            oswalds_eff = oswalds_eff_inf

        areas = wing.area(_sectional=True)
        aerodynamic_centers = wing.aerodynamic_center(_sectional=True)

        a = AR_sub / (AR_sub + 2.0)
        s = np.radians(wing_sweep)
        t = np.exp(-wing_taper)
        neutral_point_shift = (
            -(
                (
                    (
                        (3.557726 ** (a**2.8443985))
                        * ((((s * a) + (t * 1.9149417)) + -1.4449639) * s)
                    )
                    + (a + -0.89228547)
                )
                * -0.16073418
            )
            * wing_MAC
        )
        aerodynamic_centers = [
            ac + np.array([neutral_point_shift, 0.0, 0.0])
            for ac in aerodynamic_centers
        ]

        xsec_quarter_chords = [
            wing._compute_xyz_of_WingXSec(index=i, x_nondim=0.25, z_nondim=0.0)
            for i in range(len(wing.xsecs))
        ]
        xsec_tcs = [xsec.airfoil.max_thickness() for xsec in wing.xsecs]

        wave_lift_records: list[dict[str, Any]] = []
        cav_margins = []
        cpmins = []
        sigmas = []
        v_incs = []
        vent_margins = []

        def compute_section_hydrodynamics(
            sect_id: int, mirror_across_XZ: bool = False
        ):
            xsec_a = wing.xsecs[sect_id]
            xsec_b = wing.xsecs[sect_id + 1]

            a_weight = xsec_a.chord / (xsec_a.chord + xsec_b.chord)
            b_weight = xsec_b.chord / (xsec_a.chord + xsec_b.chord)
            mean_chord = 0.5 * (xsec_a.chord + xsec_b.chord)

            f_sub = sect_f_sub[sect_id]
            s_sub = sect_s_sub[sect_id]
            h_sub_mid = sect_h_sub_mid[sect_id]
            pierce_w = sect_pierce_w[sect_id]
            t_pierce = sect_t_pierce[sect_id]

            xg_local, yg_local, zg_local = wing._compute_frame_of_section(sect_id)
            xg_local = [xg_local[0], xg_local[1], xg_local[2]]
            yg_local = [yg_local[0], yg_local[1], yg_local[2]]
            zg_local = [zg_local[0], zg_local[1], zg_local[2]]
            if mirror_across_XZ:
                xg_local[1] *= -1
                yg_local[1] *= -1
                zg_local[1] *= -1

            ac_full = aerodynamic_centers[sect_id]
            qc_a = xsec_quarter_chords[sect_id]
            qc_b = xsec_quarter_chords[sect_id + 1]
            ac_sub_linear = [
                (1.0 - s_sub) * qc_a[k] + s_sub * qc_b[k] for k in range(3)
            ]
            ac_sub_linear[0] = ac_sub_linear[0] + neutral_point_shift
            w_full = f_sub**4
            sect_AC_raw = [
                w_full * ac_full[k] + (1.0 - w_full) * ac_sub_linear[k]
                for k in range(3)
            ]
            if mirror_across_XZ:
                sect_AC_raw = [sect_AC_raw[0], -sect_AC_raw[1], sect_AC_raw[2]]

            sect_AC = [sect_AC_raw[k] - self.xyz_ref[k] for k in range(3)]

            vel_freestream_g = op_point.convert_axes(
                x_from=-op_point.velocity,
                y_from=0.0,
                z_from=0.0,
                from_axes="wind",
                to_axes="geometry",
            )
            vel_rot_g = np.cross(
                sect_AC,
                op_point.convert_axes(
                    op_point.p,
                    op_point.q,
                    op_point.r,
                    from_axes="body",
                    to_axes="geometry",
                ),
                manual=True,
            )
            vel_vector_g = [vel_freestream_g[k] + vel_rot_g[k] for k in range(3)]
            vel_mag_g = np.sqrt(sum(comp**2 for comp in vel_vector_g))
            vel_dir_g = [vel_vector_g[k] / vel_mag_g for k in range(3)]
            vel_dot_x = np.dot(vel_dir_g, xg_local, manual=True)
            vel_dot_z = np.dot(vel_dir_g, zg_local, manual=True)

            alpha_gen = np.where(
                vel_dot_x > 0,
                90.0 - np.arccosd(np.clip(vel_dot_z, -1.0, 1.0)),
                90.0 + np.arccosd(np.clip(vel_dot_z, -1.0, 1.0)),
            )

            alpha_gen_eff = (
                alpha_gen
                - (1.0 - AR_3D_eff**0.8)
                * np.sind(2.0 * alpha_gen)
                / 2.0
                * (180.0 / np.pi)
            )

            qc_vec = qc_b - qc_a
            qc_dir = qc_vec / np.linalg.norm(qc_vec)
            qc_dir_list = [qc_dir[0], qc_dir[1], qc_dir[2]]
            vel_dot_qc = np.dot(vel_dir_g, qc_dir_list, manual=True)
            sweep_rad = np.arcsin(np.clip(vel_dot_qc, -0.999, 0.999))

            Re_a = op_point.reynolds(xsec_a.chord)
            Re_b = op_point.reynolds(xsec_b.chord)

            sym_control_surfaces: list[ControlSurface] = []
            for surf in xsec_a.control_surfaces:
                if mirror_across_XZ and not surf.symmetric:
                    surf = surf.copy()
                    surf.deflection = -surf.deflection
                sym_control_surfaces.append(surf)

            kwargs = dict(
                alpha=alpha_gen_eff,
                mach=0.0,
                control_surfaces=sym_control_surfaces,
                model_size=self.model_size,
            )
            aero_a = xsec_a.airfoil.get_aero_from_neuralfoil(Re=Re_a, **kwargs)
            aero_b = xsec_b.airfoil.get_aero_from_neuralfoil(Re=Re_b, **kwargs)

            sect_CL = (
                aero_a["CL"] * a_weight + aero_b["CL"] * b_weight
            ) * AR_3D_eff**0.2
            sect_CDp = aero_a["CD"] * a_weight + aero_b["CD"] * b_weight
            sect_CM = aero_a["CM"] * a_weight + aero_b["CM"] * b_weight

            # Incompressible suction peak corrected for sweep
            sect_Cpmin_2d = aero_a["Cpmin_0"] * a_weight + aero_b["Cpmin_0"] * b_weight
            sect_Cpmin = sect_Cpmin_2d * np.cos(sweep_rad) ** 2

            rho = fluid.density()
            q_local = 0.5 * rho * vel_mag_g**2
            _, p_local, p_vapor = self._local_depth_and_pressure(sect_AC_raw[2])
            sect_sigma = cavitation_number(p_local, p_vapor, q_local)
            sect_cav_margin = sect_sigma + sect_Cpmin
            sect_v_inc = cavitation_inception_speed(
                Cpmin=sect_Cpmin,
                local_pressure=p_local,
                vapor_pressure=p_vapor,
                density=rho,
            )

            if self.include_cavitation_penalty:
                sect_CD_cav = cavitation_drag_increment(sect_Cpmin, sect_sigma)
            else:
                sect_CD_cav = 0.0

            if include_induced_drag:
                sect_CDi = sect_CL**2 / (np.pi * AR_sub * oswalds_eff)
            else:
                sect_CDi = 0.0

            area_sub = areas[sect_id] * f_sub
            sect_L = q_local * area_sub * sect_CL
            sect_D_profile = q_local * area_sub * (sect_CDp + sect_CDi)
            sect_D_cav = q_local * area_sub * sect_CD_cav
            sect_M = q_local * area_sub * sect_CM * mean_chord

            # Ventilation margin
            if surface_z is not None:
                h_over_c_sect = h_sub_mid / np.maximum(mean_chord, 1e-4)
                fr_h_sect = vel_mag_g / np.sqrt(
                    g * np.softmax(h_sub_mid, 1e-3, softness=1e-3)
                )
                vent_foil = ventilation_margin_lifting_foil(h_over_c_sect, sect_CL)
                vent_strut = ventilation_margin_strut(
                    np.sqrt(sect_CL**2 + 1e-6), fr_h_sect
                )
                sect_vent = (
                    horiz_weight * vent_foil + (1.0 - horiz_weight) * vent_strut
                )
            else:
                sect_vent = 10.0

            # Mask out emerged sections above water so they do not trigger false cavitation/ventilation
            mask_offset = (1.0 - f_sub) * 100.0
            cav_margins.append(sect_cav_margin + mask_offset)
            cpmins.append(sect_Cpmin + (1.0 - f_sub) * 10.0)
            sigmas.append(sect_sigma + mask_offset)
            v_incs.append(sect_v_inc + mask_offset)
            vent_margins.append(sect_vent + mask_offset)

            # Force & moment directions
            L_dir_unnorm = [
                zg_local[k] - vel_dot_z * vel_dir_g[k] for k in range(3)
            ]
            L_dir_unnorm = [
                np.where(vel_dot_x > 0, L_dir_unnorm[k], -L_dir_unnorm[k])
                for k in range(3)
            ]
            L_dir_mag = np.sqrt(sum(c**2 for c in L_dir_unnorm) + 1e-30)
            L_dir_g = [L_dir_unnorm[k] / L_dir_mag for k in range(3)]
            D_dir_g = vel_dir_g

            sect_F_base_g = [
                sect_L * L_dir_g[k] + sect_D_profile * D_dir_g[k] for k in range(3)
            ]

            # Record this section's lift vector and submerged quarter-chord segment
            # for the wave-resistance integral.
            y_sign = -1.0 if mirror_across_XZ else 1.0
            t_lo = s_sub - 0.5 * f_sub
            t_hi = s_sub + 0.5 * f_sub
            wave_lift_records.append(
                {
                    "sect_id": sect_id,
                    "mirror": mirror_across_XZ,
                    "start": [
                        ((1.0 - t_lo) * qc_a[k] + t_lo * qc_b[k])
                        * (y_sign if k == 1 else 1.0)
                        for k in range(3)
                    ],
                    "end": [
                        ((1.0 - t_hi) * qc_a[k] + t_hi * qc_b[k])
                        * (y_sign if k == 1 else 1.0)
                        for k in range(3)
                    ],
                    "force_y": sect_L * L_dir_g[1],
                    "force_z": sect_L * L_dir_g[2],
                    "smear": 0.5 * mean_chord,
                    "span": sectional_spans[sect_id] * f_sub,
                }
            )
            sect_F_cav_g = [sect_D_cav * D_dir_g[k] for k in range(3)]

            # Spray drag on surface-piercing section
            if surface_z is not None and self.include_spray_drag:
                c_wl = (1.0 - t_pierce) * xsec_a.chord + t_pierce * xsec_b.chord
                tc_wl = (1.0 - t_pierce) * xsec_tcs[sect_id] + t_pierce * xsec_tcs[
                    sect_id + 1
                ]
                fr_c = vel_mag_g / np.sqrt(g * np.maximum(c_wl, 1e-4))
                d_spray = pierce_w * spray_drag_strut(
                    waterline_chord=c_wl,
                    thickness_to_chord=tc_wl,
                    dynamic_pressure=q_local,
                    froude_chord=fr_c,
                )
                wl_xyz = [
                    (1.0 - t_pierce) * qc_a[k] + t_pierce * qc_b[k] for k in range(3)
                ]
                if mirror_across_XZ:
                    wl_xyz[1] *= -1
                wl_arm = [wl_xyz[k] - self.xyz_ref[k] for k in range(3)]
                sect_F_spray_g = [d_spray * D_dir_g[k] for k in range(3)]
                sect_M_spray_g = np.cross(wl_arm, sect_F_spray_g, manual=True)
            else:
                sect_F_spray_g = [0.0, 0.0, 0.0]
                sect_M_spray_g = [0.0, 0.0, 0.0]

            sect_F_total_g = [
                sect_F_base_g[k] + sect_F_cav_g[k] + sect_F_spray_g[k]
                for k in range(3)
            ]

            M_g_lift_prof = np.cross(
                sect_AC, [sect_F_base_g[k] + sect_F_cav_g[k] for k in range(3)], manual=True
            )
            M_dir_g = np.cross(L_dir_g, D_dir_g, manual=True)
            sect_M_g = [
                M_g_lift_prof[k] + M_dir_g[k] * sect_M + sect_M_spray_g[k]
                for k in range(3)
            ]

            d_prof_w = -op_point.convert_axes(
                sect_F_base_g[0],
                sect_F_base_g[1],
                sect_F_base_g[2],
                from_axes="geometry",
                to_axes="wind",
            )[0]
            d_cav_w = -op_point.convert_axes(
                sect_F_cav_g[0],
                sect_F_cav_g[1],
                sect_F_cav_g[2],
                from_axes="geometry",
                to_axes="wind",
            )[0]
            d_spray_w = -op_point.convert_axes(
                sect_F_spray_g[0],
                sect_F_spray_g[1],
                sect_F_spray_g[2],
                from_axes="geometry",
                to_axes="wind",
            )[0]

            return sect_F_total_g, sect_M_g, d_prof_w, d_cav_w, d_spray_w

        F_g = [0.0, 0.0, 0.0]
        M_g = [0.0, 0.0, 0.0]
        D_profile_total = 0.0
        D_cav_total = 0.0
        D_spray_total = 0.0

        for sect_id in range(n_sects):
            s_F, s_M, dp, dc, ds = compute_section_hydrodynamics(sect_id=sect_id)
            for k in range(3):
                F_g[k] = F_g[k] + s_F[k]
                M_g[k] = M_g[k] + s_M[k]
            D_profile_total = D_profile_total + dp
            D_cav_total = D_cav_total + dc
            D_spray_total = D_spray_total + ds

            if wing.symmetric:
                s_F, s_M, dp, dc, ds = compute_section_hydrodynamics(
                    sect_id=sect_id, mirror_across_XZ=True
                )
                for k in range(3):
                    F_g[k] = F_g[k] + s_F[k]
                    M_g[k] = M_g[k] + s_M[k]
                D_profile_total = D_profile_total + dp
                D_cav_total = D_cav_total + dc
                D_spray_total = D_spray_total + ds

        span_L_effective = np.softmax(
            span_eff_safe * np.abs(np.cosd(wing_dihedral)),
            0.04 * span_eff_safe,
            softness=0.01,
        )
        span_Y_effective = np.softmax(
            span_eff_safe * np.abs(np.sind(wing_dihedral)),
            0.04 * span_eff_safe,
            softness=0.01,
        )

        wave_elements = self._build_wave_lift_elements(
            records=wave_lift_records,
            elliptical=wing.symmetric and not is_strut,
        )

        root_qc = (
            [xsec_quarter_chords[0][k] for k in range(3)]
            if len(xsec_quarter_chords) > 0
            else [0.0, 0.0, 0.0]
        )

        return self.HydroComponentResults(
            s_ref=self.assembly.s_ref,
            c_ref=self.assembly.c_ref,
            b_ref=self.assembly.b_ref,
            op_point=op_point,
            F_g=F_g,
            M_g=M_g,
            span_L_effective=span_L_effective,
            span_Y_effective=span_Y_effective,
            oswalds_efficiency=oswalds_eff,
            D_profile=D_profile_total,
            D_cavitation=D_cav_total,
            D_spray=D_spray_total,
            Cpmin_min=_safe_softmin(cpmins, softness=0.01),
            sigma_min=_safe_softmin(sigmas, softness=0.01),
            cavitation_margin=_safe_softmin(cav_margins, softness=0.01),
            ventilation_margin=_safe_softmin(vent_margins, softness=0.01),
            V_cavitation_inception=_safe_softmin(v_incs, softness=0.05),
            submerged_weight=root_f_sub,
            junction_chord=xsec_chords[0] if xsec_chords else 0.1,
            junction_tc=xsec_tcs[0] if xsec_tcs else 0.12,
            junction_xyz=root_qc,
            wave_elements=wave_elements,
        )

    @staticmethod
    def _build_wave_lift_elements(
        records: list[dict[str, Any]],
        elliptical: bool,
        n_sub: int = 3,
    ) -> dict[str, Any]:
        """
        Turn per-section lift records into line elements for the wave integral.

        The buildup's strip loads carry no spanwise induction, so for a symmetric
        lifting foil each side's total load is redistributed elliptically over its
        submerged span (root at the centreline). Struts and one-sided foils keep
        their per-section loads, uniform along each section.
        """
        elements: dict[str, Any] = {
            "xyz_a": ([], [], []),
            "xyz_b": ([], [], []),
            "force_y": [],
            "force_z": [],
            "smear_length": [],
        }

        def add(start, end, force_y, force_z, smear):
            for k in range(3):
                elements["xyz_a"][k].append(start[k])
                elements["xyz_b"][k].append(end[k])
            elements["force_y"].append(force_y)
            elements["force_z"].append(force_z)
            elements["smear_length"].append(smear)

        if not elliptical:
            for rec in records:
                add(rec["start"], rec["end"], rec["force_y"], rec["force_z"], rec["smear"])
            return elements

        for mirror in (False, True):
            side = [rec for rec in records if rec["mirror"] == mirror]
            if not side:
                continue
            total_fy = sum(rec["force_y"] for rec in side)
            total_fz = sum(rec["force_z"] for rec in side)
            total_span = sum(rec["span"] for rec in side) + 1e-12

            pieces = []
            span_before = 0.0
            for rec in side:
                for j in range(n_sub):
                    eta = (span_before + (j + 0.5) / n_sub * rec["span"]) / total_span
                    d_eta = rec["span"] / n_sub / total_span
                    shape = d_eta * np.sqrt(np.fmax(1.0 - eta**2, 0.0) + 1e-12)
                    t0, t1 = j / n_sub, (j + 1) / n_sub
                    start = [
                        (1.0 - t0) * rec["start"][k] + t0 * rec["end"][k] for k in range(3)
                    ]
                    end = [
                        (1.0 - t1) * rec["start"][k] + t1 * rec["end"][k] for k in range(3)
                    ]
                    pieces.append((start, end, shape, rec["smear"]))
                span_before = span_before + rec["span"]

            shape_sum = sum(piece[2] for piece in pieces)
            for start, end, shape, smear in pieces:
                add(
                    start,
                    end,
                    total_fy * shape / shape_sum,
                    total_fz * shape / shape_sum,
                    smear,
                )
        return elements

    def fuselage_hydrodynamics(
        self,
        fuselage: Fuselage,
        include_induced_drag: bool = True,
    ) -> HydroComponentResults:
        """
        Compute incompressible hydrodynamic forces and moments on a submerged
        fuselage or hydrofoil pod (no compressibility or wave-drag terms).
        """
        op_point = self.op_point
        surface_z = getattr(op_point, "surface_z", None)
        length = fuselage.length()
        Re = op_point.reynolds(reference_length=length)
        q = op_point.dynamic_pressure()
        eta = jorgensen_eta(fuselage.fineness_ratio())

        n_sects = len(fuselage.xsecs) - 1
        if n_sects <= 0:
            return self.HydroComponentResults(
                s_ref=self.assembly.s_ref,
                c_ref=self.assembly.c_ref,
                b_ref=self.assembly.b_ref,
                op_point=op_point,
                F_g=[0.0, 0.0, 0.0],
                M_g=[0.0, 0.0, 0.0],
                span_L_effective=1e-4,
                span_Y_effective=1e-4,
                oswalds_efficiency=0.95,
                submerged_weight=0.0,
            )

        if surface_z is not None:
            sect_f_sub = [
                smooth_submergence_fraction(
                    fuselage.xsecs[i].xyz_c[2],
                    fuselage.xsecs[i + 1].xyz_c[2],
                    surface_z,
                )[0]
                for i in range(n_sects)
            ]
            f_sub_mean = sum(sect_f_sub) / n_sects
        else:
            sect_f_sub = [1.0] * n_sects
            f_sub_mean = 1.0

        span_effective = np.softmax(
            softmax_scalefree(
                [xsec.width for xsec in fuselage.xsecs]
                + [xsec.height for xsec in fuselage.xsecs]
            )
            * f_sub_mean,
            1e-4,
            softness=1e-4,
        )

        F_g = [0.0, 0.0, 0.0]
        M_g = [0.0, 0.0, 0.0]

        xsec_areas = [xsec.xsec_area() for xsec in fuselage.xsecs]
        sect_xyz_a = [xsec.xyz_c for xsec in fuselage.xsecs[:-1]]
        sect_xyz_b = [xsec.xyz_c for xsec in fuselage.xsecs[1:]]
        sect_xyz_mid = [
            [0.5 * (xyz_a[k] + xyz_b[k]) for k in range(3)]
            for xyz_a, xyz_b in zip(sect_xyz_a, sect_xyz_b)
        ]
        sect_lengths = [
            (sum((xyz_a[k] - xyz_b[k]) ** 2 for k in range(3))) ** 0.5
            for xyz_a, xyz_b in zip(sect_xyz_a, sect_xyz_b)
        ]
        sect_dirs = [
            [
                np.where(
                    sect_lengths[i] != 0,
                    (xyz_b[j] - xyz_a[j]) / (sect_lengths[i] + 1e-100),
                    1.0 if j == 0 else 0.0,
                )
                for j in range(3)
            ]
            for i, (xyz_a, xyz_b) in enumerate(zip(sect_xyz_a, sect_xyz_b))
        ]
        sect_areas = [
            (area_a + area_b + (area_a * area_b + 1e-100) ** 0.5) / 3.0
            for area_a, area_b in zip(xsec_areas[:-1], xsec_areas[1:])
        ]

        vel_dir_g = op_point.convert_axes(
            -1.0, 0.0, 0.0, from_axes="wind", to_axes="geometry"
        )
        sin_alpha_force_dir = [
            [np.dot(s, vel_dir_g, manual=True) * vel_dir_g[k] - s[k] for k in range(3)]
            for s in sect_dirs
        ]
        rho_V2 = op_point.atmosphere.density() * op_point.velocity**2
        sin_alpha_moment_dir = [
            np.cross(vel_dir_g, s_dir, manual=True) for s_dir in sect_dirs
        ]

        f_sub_tail = sect_f_sub[-1]
        lift_force_nose = [
            rho_V2 * xsec_areas[-1] * sin_alpha_force_dir[-1][k] * f_sub_tail
            for k in range(3)
        ]
        moment_open_tail = [
            -rho_V2
            * sum(sect_lengths)
            * xsec_areas[-1]
            * sin_alpha_moment_dir[-1][k]
            * f_sub_tail
            for k in range(3)
        ]
        moment_shape = [
            rho_V2
            * sum(
                area * m_dir[k] * slen * f_s
                for area, m_dir, slen, f_s in zip(
                    sect_areas, sin_alpha_moment_dir, sect_lengths, sect_f_sub
                )
            )
            for k in range(3)
        ]
        lift_arm = [fuselage.xsecs[0].xyz_c[k] - self.xyz_ref[k] for k in range(3)]
        moment_lift = np.cross(lift_arm, lift_force_nose, manual=True)

        for k in range(3):
            F_g[k] = F_g[k] + lift_force_nose[k]
            M_g[k] = (
                M_g[k] + moment_open_tail[k] + moment_shape[k] + moment_lift[k]
            )

        # Incompressible profile drag (mach = 0.0, no wave drag)
        base_cd = fuselage_base_drag_coefficient(mach=0.0)
        drag_base = base_cd * fuselage.area_base() * q * f_sub_tail

        form_factor = fuselage_form_factor(
            fineness_ratio=fuselage.fineness_ratio(),
            ratio_of_corner_radius_to_body_width=0.5,
        )
        c_f_ideal = (3.46 * np.log10(np.maximum(Re, 100.0)) - 5.6) ** -2
        c_f = c_f_ideal * form_factor
        drag_skin = c_f * fuselage.area_wetted() * q * f_sub_mean

        drag_profile = drag_base + drag_skin
        drag_profile_g = op_point.convert_axes(
            -drag_profile, 0.0, 0.0, from_axes="wind", to_axes="geometry"
        )
        drag_arm = [
            0.5 * (fuselage.xsecs[0].xyz_c[k] + fuselage.xsecs[-1].xyz_c[k])
            - self.xyz_ref[k]
            for k in range(3)
        ]
        moment_drag_prof = np.cross(drag_arm, drag_profile_g, manual=True)
        for k in range(3):
            F_g[k] = F_g[k] + drag_profile_g[k]
            M_g[k] = M_g[k] + moment_drag_prof[k]

        # Viscous crossflow (mach = 0.0)
        sin_alphas = [
            sum(comp**2 for comp in s) ** 0.5 for s in sin_alpha_force_dir
        ]
        mean_radii = [(area / np.pi + 1e-100) ** 0.5 for area in sect_areas]
        Re_n = [
            s * op_point.reynolds(reference_length=2.0 * r)
            for s, r in zip(sin_alphas, mean_radii)
        ]
        C_d_n = [
            np.where(
                Re_n[i] != 0,
                aerolib.Cd_cylinder(
                    Re_D=Re_n[i], mach=0.0, include_mach_effects=False
                ),
                0.0,
            )
            for i in range(n_sects)
        ]

        vel_dot_x = [
            np.dot(vel_dir_g, sect_dirs[i], manual=True) for i in range(n_sects)
        ]
        norm_dirs_unnorm = [
            [vel_dir_g[j] - vel_dot_x[i] * sect_dirs[i][j] for j in range(3)]
            for i in range(n_sects)
        ]
        for i in range(n_sects):
            norm_dirs_unnorm[i][2] += 1e-100
        norm_mags = [
            (sum(c**2 for c in n) + 1e-100) ** 0.5 for n in norm_dirs_unnorm
        ]
        norm_dirs = [
            [norm_dirs_unnorm[i][j] / norm_mags[i] for j in range(3)]
            for i in range(n_sects)
        ]

        crossflow_forces = [
            [
                sect_lengths[i]
                * rho_V2
                * eta
                * C_d_n[i]
                * norm_dirs[i][j]
                * sum(c**2 for c in sin_alpha_force_dir[i])
                * mean_radii[i]
                * sect_f_sub[i]
                for j in range(3)
            ]
            for i in range(n_sects)
        ]
        crossflow_moments = [
            np.cross(
                [sect_xyz_mid[i][j] - self.xyz_ref[j] for j in range(3)],
                crossflow_forces[i],
                manual=True,
            )
            for i in range(n_sects)
        ]
        for j in range(3):
            F_g[j] = F_g[j] + sum(crossflow_forces[i][j] for i in range(n_sects))
            M_g[j] = M_g[j] + sum(crossflow_moments[i][j] for i in range(n_sects))

        if include_induced_drag:
            _, sideforce, lift = op_point.convert_axes(
                F_g[0], F_g[1], F_g[2], from_axes="geometry", to_axes="wind"
            )
            D_ind = (lift**2 + sideforce**2) / (q * np.pi * span_effective**2)
            D_ind_g = op_point.convert_axes(
                -D_ind, 0.0, 0.0, from_axes="wind", to_axes="geometry"
            )
            for k in range(3):
                F_g[k] = F_g[k] + D_ind_g[k]

        d_prof_w = -op_point.convert_axes(
            F_g[0], F_g[1], F_g[2], from_axes="geometry", to_axes="wind"
        )[0]

        return self.HydroComponentResults(
            s_ref=self.assembly.s_ref,
            c_ref=self.assembly.c_ref,
            b_ref=self.assembly.b_ref,
            op_point=op_point,
            F_g=F_g,
            M_g=M_g,
            span_L_effective=span_effective,
            span_Y_effective=span_effective,
            oswalds_efficiency=0.95,
            D_profile=d_prof_w,
            submerged_weight=f_sub_mean,
        )
