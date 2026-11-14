from typing import Any
import aerosandbox.numpy as np
from aerosandbox.performance import OperatingPoint
from hydrosandbox.hydrodynamics.singularities import (
    calculate_induced_velocity_horseshoe_hydro,
)
from hydrosandbox.library.wave_resistance import (
    fuselage_wave_sources,
    kelvin_wave_resistance,
    merge_wave_elements,
    wing_thickness_wave_sources,
)


def _tall(array):
    return np.reshape(array, (-1, 1))


def _wide(array):
    return np.reshape(array, (1, -1))


def horseshoe_lattice_wave_drag(
    op_point: OperatingPoint,
    surface_z: float,
    image_weight: float,
    left_vortex_vertices: np.ndarray,
    right_vortex_vertices: np.ndarray,
    vortex_strengths: np.ndarray,
    smear_lengths: np.ndarray,
    trailing_vortex_direction: np.ndarray,
    vortex_core_radius: float,
    submerged_assembly: Any = None,
    include_thickness: bool = True,
) -> dict[str, Any]:
    """
    Wave-making drag of a solved horseshoe-vortex lattice (``HydroVLM`` or
    ``HydroLiftingLine``) beneath a free surface, by linear far-field theory.

    For a given circulation distribution the drag due to lift is the
    trailing-vortex drag with a *rigid-lid* image, plus the wave resistance
    ``R_w`` (see ``hydrosandbox.library.wave_resistance``). The lattice was solved
    with an image of weight ``image_weight`` (``kappa``), whose Kutta-Joukowski
    drag already contains ``kappa * D_mutual``; this returns the correction that
    converts it:

        delta_D = R_w - (1 + kappa) * D_mutual

    where ``D_mutual`` is the drag induced on the lattice by its unit-weight image.

    Returns
    -------
    Dict with ``D_wave`` (= ``R_w``), ``D_image_mutual`` (= ``D_mutual``),
    ``delta_D`` [N], and ``xyz_application`` (the point, in geometry axes, at which
    to apply the correction force).
    """
    fluid = op_point.atmosphere
    rho = fluid.density()
    velocity = op_point.velocity
    gravity = getattr(fluid, "gravity", 9.80665)

    vortex_centers = 0.5 * (left_vortex_vertices + right_vortex_vertices)
    bound_legs = right_vortex_vertices - left_vortex_vertices

    ### Mutual drag of the unit-weight image system on the lattice
    def induced_at_centers(weight: float) -> np.ndarray:
        u, v, w = calculate_induced_velocity_horseshoe_hydro(
            x_field=_tall(vortex_centers[:, 0]),
            y_field=_tall(vortex_centers[:, 1]),
            z_field=_tall(vortex_centers[:, 2]),
            x_left=_wide(left_vortex_vertices[:, 0]),
            y_left=_wide(left_vortex_vertices[:, 1]),
            z_left=_wide(left_vortex_vertices[:, 2]),
            x_right=_wide(right_vortex_vertices[:, 0]),
            y_right=_wide(right_vortex_vertices[:, 1]),
            z_right=_wide(right_vortex_vertices[:, 2]),
            trailing_vortex_direction=trailing_vortex_direction,
            gamma=_wide(vortex_strengths),
            vortex_core_radius=vortex_core_radius,
            surface_z=surface_z,
            image_weight=weight,
        )
        return np.stack(
            [np.sum(u, axis=1), np.sum(v, axis=1), np.sum(w, axis=1)], axis=1
        )

    v_image = induced_at_centers(1.0) - induced_at_centers(0.0)
    f_image_g = np.sum(
        rho * np.cross(v_image, bound_legs, axis=1) * _tall(vortex_strengths), axis=0
    )
    # Trefftz-plane sense: the component along the trailing-vortex direction. (When
    # the freestream is pitched relative to the wake/free surface, the wind-axis
    # drag of the image-induced forces would also pick up a share of the image's
    # effect on *lift*, which is not induced drag.)
    wake_direction = np.asarray(trailing_vortex_direction, dtype=float)
    wake_direction = wake_direction / np.linalg.norm(wake_direction)
    D_mutual = float(np.sum(f_image_g * wake_direction))

    ### Far-field wave resistance of the loading (+ thickness sources)
    # Dipole strengths are the linear-theory (freestream Kutta-Joukowski) loads
    # rho * Gamma * (V_inf x l), consistent with the circulation that sets D_mutual.
    freestream_velocity = np.reshape(
        op_point.compute_freestream_velocity_geometry_axes(), (1, 3)
    )
    panel_forces_geometry = (
        rho
        * np.cross(
            np.repeat(freestream_velocity, len(bound_legs), axis=0), bound_legs, axis=1
        )
        * _tall(vortex_strengths)
    )
    lift_elements = {
        "xyz_a": tuple(list(left_vortex_vertices[:, k]) for k in range(3)),
        "xyz_b": tuple(list(right_vortex_vertices[:, k]) for k in range(3)),
        "force_y": list(panel_forces_geometry[:, 1]),
        "force_z": list(panel_forces_geometry[:, 2]),
        "smear_length": list(smear_lengths),
    }
    element_sets = [lift_elements]
    if include_thickness and submerged_assembly is not None:
        for wing in submerged_assembly.wings:
            element_sets.append(
                wing_thickness_wave_sources(wing, velocity=velocity, surface_z=None)
            )
        for fuse in submerged_assembly.fuselages:
            element_sets.append(
                fuselage_wave_sources(fuse, velocity=velocity, surface_z=surface_z)
            )
    elements = merge_wave_elements(*element_sets)
    elements = {
        key: (
            tuple(np.array(v, dtype=float) for v in value)
            if key.startswith("xyz")
            else np.array(value, dtype=float)
        )
        for key, value in elements.items()
    }

    D_wave = float(
        kelvin_wave_resistance(
            velocity=float(velocity),
            density=float(rho),
            surface_z=float(surface_z),
            gravity=gravity,
            **elements,
        )
    )

    force_mag = np.linalg.norm(panel_forces_geometry[:, 1:], axis=1)
    if np.sum(force_mag) > 0:
        xyz_application = np.sum(
            vortex_centers * _tall(force_mag), axis=0
        ) / np.sum(force_mag)
    else:
        xyz_application = np.mean(vortex_centers, axis=0)

    return {
        "D_wave": D_wave,
        "D_image_mutual": D_mutual,
        "delta_D": D_wave - (1.0 + image_weight) * D_mutual,
        "xyz_application": xyz_application,
    }
