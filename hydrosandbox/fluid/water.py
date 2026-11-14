from aerosandbox.common import AeroSandboxObject
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable


class Water(AeroSandboxObject):
    """
    Model liquid water (seawater or freshwater) and hydrostatic properties.

    Implements the ``Atmosphere`` interface expected by AeroSandbox components
    (``pressure``, ``temperature``, ``density``, ``dynamic_viscosity``,
    ``kinematic_viscosity``, ``speed_of_sound``, ``ratio_of_specific_heats``)
    alongside hydrodynamic properties such as ``vapor_pressure`` and
    ``surface_tension``.
    """

    def __init__(
        self,
        depth: Vectorizable = 0.0,
        temperature_C: Vectorizable = 15.0,
        salinity: Vectorizable = 35.0,
        atmospheric_pressure: Vectorizable = 101325.0,
        gravity: float = 9.80665,
        surface_tension_override: Vectorizable | None = None,
    ):
        """
        Initialize a new Water fluid model.

        Parameters
        ----------
        depth : Vectorizable
            Reference submergence depth below the free surface [m]. Positive values
            indicate depth below the water surface.
        temperature_C : Vectorizable
            Water temperature [deg C].
        salinity : Vectorizable
            Absolute/practical salinity [g/kg] (or PSU; ~35 g/kg for standard seawater,
            0 g/kg for freshwater).
        atmospheric_pressure : Vectorizable
            Ambient pressure at the free surface [Pa]. Defaults to 101325 Pa.
        gravity : float
            Gravitational acceleration [m/s^2]. Defaults to 9.80665 m/s^2.
        surface_tension_override : Vectorizable | None
            Optional override for surface tension [N/m]. If None, computed from
            temperature and salinity.
        """
        self.depth = depth
        self.temperature_C = temperature_C
        self.salinity = salinity
        self.atmospheric_pressure = atmospheric_pressure
        self.gravity = gravity
        self.surface_tension_override = surface_tension_override

    @classmethod
    def seawater(
        cls,
        depth: Vectorizable = 0.0,
        temperature_C: Vectorizable = 15.0,
        salinity: Vectorizable = 35.0,
        **kwargs,
    ) -> "Water":
        """Construct a standard seawater model (default salinity 35 g/kg at 15 deg C)."""
        return cls(
            depth=depth,
            temperature_C=temperature_C,
            salinity=salinity,
            **kwargs,
        )

    @classmethod
    def freshwater(
        cls,
        depth: Vectorizable = 0.0,
        temperature_C: Vectorizable = 15.0,
        **kwargs,
    ) -> "Water":
        """Construct a standard freshwater model (salinity 0 g/kg)."""
        kwargs.pop("salinity", None)
        return cls(
            depth=depth,
            temperature_C=temperature_C,
            salinity=0.0,
            **kwargs,
        )

    def __repr__(self) -> str:
        try:
            return (
                f"Water (depth: {self.depth:.2f} m, "
                f"T: {self.temperature_C:.1f} C, "
                f"salinity: {self.salinity:.1f} g/kg)"
            )
        except (ValueError, TypeError):
            return (
                f"Water (depth: {self.depth} m, "
                f"T: {self.temperature_C} C, "
                f"salinity: {self.salinity} g/kg)"
            )

    def __len__(self) -> int:
        length = 1
        fields = [
            self.depth,
            self.temperature_C,
            self.salinity,
            self.atmospheric_pressure,
        ]
        if self.surface_tension_override is not None:
            fields.append(self.surface_tension_override)

        for v in fields:
            lv = np.length(v)
            if lv != 1:
                if length == 1:
                    length = lv
                elif length != lv:
                    raise ValueError(
                        "State variables appear vectorized, but of different lengths!"
                    )
        return length

    def __getitem__(self, index) -> "Water":
        """Index one item from each vectorized attribute of a Water instance."""
        self_length = len(self)

        def get_item_of_attribute(a):
            if hasattr(a, "__len__") and hasattr(a, "__getitem__"):
                if len(a) == 1:
                    return a[0]
                elif len(a) == self_length:
                    return a[index]
                else:
                    try:
                        return a[index]
                    except IndexError:
                        raise IndexError(
                            f"A state variable could not be indexed; it has length {len(a)} "
                            f"while the parent has length {self_length}."
                        )
            return a

        inputs = {
            "depth": self.depth,
            "temperature_C": self.temperature_C,
            "salinity": self.salinity,
            "atmospheric_pressure": self.atmospheric_pressure,
        }
        st_override = (
            get_item_of_attribute(self.surface_tension_override)
            if self.surface_tension_override is not None
            else None
        )

        return self.__class__(
            **{k: get_item_of_attribute(v) for k, v in inputs.items()},
            gravity=self.gravity,
            surface_tension_override=st_override,
        )

    def __array__(self, dtype="O"):
        """Allow NumPy array creation without infinite recursion in __len__ and __getitem__."""
        return np.fromiter([self], dtype=dtype).reshape(())

    def temperature_celsius(self) -> Vectorizable:
        """Return the water temperature in degrees Celsius."""
        return self.temperature_C

    def temperature(self) -> Vectorizable:
        """Return the absolute water temperature in Kelvin."""
        return self.temperature_C + 273.15

    def density(self) -> Vectorizable:
        """
        Return the density of water in kg/m^3.

        Uses the UNESCO EOS-80 (Millero & Poisson, 1981) equation of state
        at atmospheric pressure.
        """
        T = self.temperature_C
        S_plus = np.maximum(self.salinity, 0.0)

        rho_w = (
            999.842594
            + 6.793952e-2 * T
            - 9.095290e-3 * T**2
            + 1.001685e-4 * T**3
            - 1.120083e-6 * T**4
            + 6.536332e-9 * T**5
        )
        A = (
            0.824493
            - 4.0899e-3 * T
            + 7.6438e-5 * T**2
            - 8.2467e-7 * T**3
            + 5.3875e-9 * T**4
        )
        B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T**2
        C = 4.8314e-4

        # Offset S_plus by 1e-12 before **1.5 to keep second derivatives finite at S=0.
        return rho_w + A * S_plus + B * (S_plus + 1e-12) ** 1.5 + C * S_plus**2

    def dynamic_viscosity(self) -> Vectorizable:
        """
        Return the dynamic viscosity (mu) of water in Pa*s (kg/(m*s)).

        Uses the Sharqawy et al. (2010) / ITTC 2011 correlation.
        """
        T = self.temperature_C
        s = np.maximum(self.salinity, 0.0) / 1000.0

        mu_w = 4.2844e-5 + 1.0 / (0.157 * (T + 64.993) ** 2 - 91.296)
        A_mu = 1.541 + 1.998e-2 * T - 9.52e-5 * T**2
        B_mu = 7.974 - 7.561e-2 * T + 4.724e-4 * T**2

        return mu_w * (1.0 + A_mu * s + B_mu * s**2)

    def kinematic_viscosity(self) -> Vectorizable:
        """Return the kinematic viscosity (nu = mu / rho) in m^2/s."""
        return self.dynamic_viscosity() / self.density()

    def vapor_pressure(self) -> Vectorizable:
        """
        Return the saturation vapor pressure of water in Pa.

        Uses Buck's pure-water vapor pressure correlation with the
        Sharqawy et al. (2010) seawater salinity depression factor.
        """
        T = self.temperature_C
        S_plus = np.clip(self.salinity, 0.0, 250.0)

        p_vw = 611.21 * np.exp((18.678 - T / 234.5) * T / (257.14 + T))
        return p_vw / (1.0 + 0.57357 * S_plus / (1000.0 - S_plus))

    def pressure(self, depth: Vectorizable | None = None) -> Vectorizable:
        """
        Return the absolute hydrostatic pressure [Pa] at a given submergence depth.

        Clamped smoothly above the water surface so emerged regions experience
        ambient atmospheric pressure.
        """
        d = self.depth if depth is None else depth
        d_sub = np.softmax(d, 0.0, softness=1e-3)
        return self.atmospheric_pressure + self.density() * self.gravity * d_sub

    def speed_of_sound(self, depth: Vectorizable | None = None) -> Vectorizable:
        """
        Return the speed of sound in water [m/s] using Mackenzie (1981).
        """
        T = self.temperature_C
        S = self.salinity
        d = np.softmax(self.depth if depth is None else depth, 0.0, softness=1e-3)

        return (
            1448.96
            + 4.591 * T
            - 5.304e-2 * T**2
            + 2.374e-4 * T**3
            + 1.340 * (S - 35.0)
            + 1.630e-2 * d
            + 1.675e-7 * d**2
        )

    def ratio_of_specific_heats(self) -> float:
        """Return 1.0 (incompressible liquid)."""
        return 1.0

    def surface_tension(self) -> Vectorizable:
        """Return the air-water surface tension [N/m]."""
        if self.surface_tension_override is not None:
            return self.surface_tension_override

        T = self.temperature_C
        S_plus = np.maximum(self.salinity, 0.0)
        sigma_w = 0.07564 - 1.45e-4 * T
        return sigma_w * (1.0 + 3.766e-4 * S_plus + 2.347e-6 * S_plus * T)
