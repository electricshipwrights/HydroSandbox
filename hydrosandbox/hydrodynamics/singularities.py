from aerosandbox.aerodynamics.aero_3D.singularities.uniform_strength_horseshoe_singularities import (
    calculate_induced_velocity_horseshoe,
)
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


def calculate_induced_velocity_horseshoe_hydro(
    x_field: Vectorizable,
    y_field: Vectorizable,
    z_field: Vectorizable,
    x_left: Vectorizable,
    y_left: Vectorizable,
    z_left: Vectorizable,
    x_right: Vectorizable,
    y_right: Vectorizable,
    z_right: Vectorizable,
    gamma: Vectorizable = 1.0,
    trailing_vortex_direction: np.ndarray | None = None,
    vortex_core_radius: float = 0.0,
    surface_z: Vectorizable | None = None,
    image_weight: Vectorizable = 0.0,
) -> tuple[Vectorizable, Vectorizable, Vectorizable]:
    """
    Calculate the 3D induced velocity due to a horseshoe vortex (and optionally its
    method-of-images reflection across a flat free surface at ``z = surface_z``).

    With ``z_image = 2 * surface_z - z``, ``u_trail_image = (ux, uy, -uz)``, and
    ``gamma_image = image_weight * gamma``:

    - ``image_weight = +1.0`` enforces ``phi = 0`` (zero tangential perturbation
      velocity ``u = v = 0``) on ``z = surface_z`` (high-Froude free surface).
    - ``image_weight = -1.0`` enforces ``dphi/dz = 0`` (zero normal velocity
      ``w = 0``) on ``z = surface_z`` (rigid-lid / ground-effect wall).
    - ``image_weight = 0.0`` (or ``surface_z is None``) evaluates an unbounded fluid.
    """
    u, v, w = calculate_induced_velocity_horseshoe(
        x_field=x_field,
        y_field=y_field,
        z_field=z_field,
        x_left=x_left,
        y_left=y_left,
        z_left=z_left,
        x_right=x_right,
        y_right=y_right,
        z_right=z_right,
        gamma=gamma,
        trailing_vortex_direction=trailing_vortex_direction,
        vortex_core_radius=vortex_core_radius,
    )

    if surface_z is None or (
        not np.is_casadi_type(image_weight)
        and isinstance(image_weight, (int, float))
        and image_weight == 0.0
    ):
        return u, v, w

    if trailing_vortex_direction is None:
        tvd = np.array([1.0, 0.0, 0.0])
    else:
        tvd = trailing_vortex_direction

    tvd_image = np.array([tvd[0], tvd[1], -tvd[2]])

    u_img, v_img, w_img = calculate_induced_velocity_horseshoe(
        x_field=x_field,
        y_field=y_field,
        z_field=z_field,
        x_left=x_left,
        y_left=y_left,
        z_left=2.0 * surface_z - z_left,
        x_right=x_right,
        y_right=y_right,
        z_right=2.0 * surface_z - z_right,
        gamma=gamma * image_weight,
        trailing_vortex_direction=tvd_image,
        vortex_core_radius=vortex_core_radius,
    )

    return u + u_img, v + v_img, w + w_img
