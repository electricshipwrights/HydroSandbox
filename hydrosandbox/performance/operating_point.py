from aerosandbox.performance import OperatingPoint
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.fluid import Water


class HydroOperatingPoint(OperatingPoint):
    """
    Instantaneous hydrodynamic operating conditions of a hydrofoil assembly.

    Extends ``asb.OperatingPoint`` with a ``Water`` fluid model, optional
    free-surface elevation ``surface_z`` in geometry axes (+z up),
    incompressible stagnation pressure, cavitation number, Froude numbers,
    and Weber number.
    """

    def __init__(
        self,
        water: Water | None = None,
        velocity: Vectorizable = 10.0,
        alpha: Vectorizable = 0.0,
        beta: Vectorizable = 0.0,
        p: Vectorizable = 0.0,
        q: Vectorizable = 0.0,
        r: Vectorizable = 0.0,
        surface_z: Vectorizable | None = None,
    ):
        """
        Initialize a new HydroOperatingPoint.

        Parameters
        ----------
        water : Water | None
            The ``Water`` fluid model. Defaults to ``Water.seawater()`` if None.
        velocity : Vectorizable
            Forward speed relative to the water [m/s].
        alpha : Vectorizable
            Pitch angle of attack [deg].
        beta : Vectorizable
            Sideslip / leeway angle [deg]. Positive beta indicates oncoming flow
            from the starboard (+y) side.
        p : Vectorizable
            Roll rate about the body x_b axis [rad/s].
        q : Vectorizable
            Pitch rate about the body y_b axis [rad/s].
        r : Vectorizable
            Yaw rate about the body z_b axis [rad/s].
        surface_z : Vectorizable | None
            Geometry-axis z coordinate [m] of the undisturbed free water surface
            (+z is upward). Local submergence depth at geometry coordinate z_g is
            ``surface_z - z_g``. If None, models an unbounded (infinite) fluid
            using ``water.depth - z_g`` for local hydrostatic pressure.
        """
        if water is None:
            water = Water.seawater()

        self.water = water
        self.velocity = velocity
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.q = q
        self.r = r
        self.surface_z = surface_z

    @property
    def atmosphere(self) -> Water:
        """Alias ``water`` as ``atmosphere`` for AeroSandbox compatibility."""
        return self.water

    @atmosphere.setter
    def atmosphere(self, value: Water) -> None:
        self.water = value

    @property
    def state(self) -> dict[str, Water | Vectorizable | None]:
        """Return the state variables of this HydroOperatingPoint as a dictionary."""
        return {
            "water": self.water,
            "velocity": self.velocity,
            "alpha": self.alpha,
            "beta": self.beta,
            "p": self.p,
            "q": self.q,
            "r": self.r,
            "surface_z": self.surface_z,
        }

    def local_depth(self, z_g: Vectorizable = 0.0, x_g: Vectorizable = 0.0) -> Vectorizable:
        """
        Return local submergence depth [m] at geometry coordinates ``(x_g, z_g)``.

        Positive values lie below the free surface.
        """
        if self.surface_z is not None:
            if callable(self.surface_z):
                return -self.surface_z(x_g, z_g)
            return self.surface_z - z_g
        return self.water.depth - z_g

    def local_pressure(self, z_g: Vectorizable = 0.0, x_g: Vectorizable = 0.0) -> Vectorizable:
        """Return local absolute hydrostatic pressure [Pa] at geometry ``(x_g, z_g)``."""
        return self.water.pressure(depth=self.local_depth(z_g, x_g=x_g))

    def total_pressure(self, z_g: Vectorizable = 0.0, x_g: Vectorizable = 0.0) -> Vectorizable:
        """
        Return total (stagnation) pressure [Pa] via incompressible Bernoulli:
        ``p_static(x_g, z_g) + 0.5 * rho * V^2``.
        """
        return self.local_pressure(z_g, x_g=x_g) + self.dynamic_pressure()

    def total_temperature(self) -> Vectorizable:
        """Return water temperature [K] (incompressible liquid)."""
        return self.water.temperature()

    def cavitation_number(self, z_g: Vectorizable = 0.0, x_g: Vectorizable = 0.0) -> Vectorizable:
        """
        Return the local cavitation number ``sigma = (p_local - p_v) / q`` [-].
        """
        p_local = self.local_pressure(z_g, x_g=x_g)
        p_vapor = self.water.vapor_pressure()
        q_dyn = np.maximum(self.dynamic_pressure(), 1e-12)
        return (p_local - p_vapor) / q_dyn

    def froude_number(self, reference_length: Vectorizable) -> Vectorizable:
        """
        Return the Froude number ``V / sqrt(g * L)`` [-].
        """
        return self.velocity / np.sqrt(
            self.water.gravity * np.maximum(reference_length, 1e-12)
        )

    def depth_froude_number(self, z_g: Vectorizable = 0.0) -> Vectorizable:
        """
        Return the submergence-depth Froude number ``V / sqrt(g * h)`` [-].
        """
        h_eff = np.softmax(self.local_depth(z_g), 1e-3, softness=1e-3)
        return self.froude_number(h_eff)

    def weber_number(self, reference_length: Vectorizable) -> Vectorizable:
        """
        Return the Weber number ``rho * V^2 * L / surface_tension`` [-].
        """
        return (
            self.water.density()
            * self.velocity**2
            * reference_length
            / self.water.surface_tension()
        )
