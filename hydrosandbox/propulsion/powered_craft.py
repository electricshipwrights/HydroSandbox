"""
A hull, its foils and its electric drive as one craft: the drivetrain's component
masses load the hull where they are mounted, and the propeller shaft sets where and
in what direction the thrust acts.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _onp

import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import HydrofoilAssembly
from hydrosandbox.hull.craft import CraftBuildup
from hydrosandbox.hull.hull import Hull
from hydrosandbox.performance import HydroOperatingPoint
from hydrosandbox.propulsion.drivetrain import ElectricPropulsionBuildup

Vectorizable = float | _onp.ndarray | Any


def _scalar(value) -> float:
    return float(_onp.ravel(value)[0])


class PoweredCraftBuildup(asb.ExplicitAnalysis):
    """
    ``CraftBuildup`` (hull + foils, free to sink and trim) coupled to
    ``ElectricPropulsionBuildup`` (wake, propeller, gearbox, motor, battery).

    The two depend on each other:

    - the drive's masses (motor, controller and gearbox at ``xyz_motor``; battery
      at ``xyz_battery``) add to the craft's mass and move its centre of gravity,
      and so its trim, sinkage and drag;
    - the propeller at ``xyz_propeller`` on a shaft inclined ``shaft_angle`` sets the
      thrust's line of action, and so its pitching moment and vertical component;
    - the craft's thrust requirement, and the hull's wake at the propeller at the
      craft's actual attitude, set the power, and through endurance the battery mass.

    ``run()`` evaluates one consistent-in-attitude state for *given* drive masses
    and returns the mismatches as residuals:

    - ``mass_drivetrain_residual``, ``mass_battery_residual`` [kg]: given minus
      what the propulsion model sizes;
    - ``thrust_residual`` [N]: the B-series propeller's thrust at the given RPM
      minus what the craft needs;
    - ``F_z_net``, ``M_y_net``: the attitude balance, as on ``CraftBuildup``.

    In ``asb.Opti``, make the masses, RPM, sinkage and trim variables and constrain
    all five residuals to zero. For a fixed design, ``PoweredCraftBuildup.solve``
    closes them by iteration.
    """

    def __init__(
        self,
        hull: Hull,
        velocity: Vectorizable,
        mass: Vectorizable,
        xyz_cg: Sequence[Vectorizable],
        xyz_propeller: Sequence[Vectorizable],
        xyz_motor: Sequence[Vectorizable],
        xyz_battery: Sequence[Vectorizable],
        mass_drivetrain: Vectorizable = 0.0,
        mass_battery: Vectorizable = 0.0,
        shaft_angle: Vectorizable = 0.0,
        assembly: HydrofoilAssembly | None = None,
        sinkage: Vectorizable = 0.0,
        trim: Vectorizable = 0.0,
        water: Water | None = None,
        propulsion_options: dict[str, Any] | None = None,
        craft_options: dict[str, Any] | None = None,
        thrust_deduction: Vectorizable | None = None,
    ):
        """
        Parameters
        ----------
        mass, xyz_cg : Mass [kg] and centre of gravity [m, geometry axes] of the
            craft *without* its drivetrain and battery (hull, structure, payload).
        xyz_propeller : Propeller disc centre, geometry axes [m].
        xyz_motor : Where the motor, controller and gearbox sit [m].
        xyz_battery : Where the battery sits [m].
        mass_drivetrain, mass_battery : The masses used in the balance [kg].
        shaft_angle : Shaft inclination to the hull's x-axis [deg], positive for a
            shaft running down and aft.
        thrust_deduction : Hull thrust-deduction fraction ``t`` in [0, 1).
            If None (default), taken from ``ElectricPropulsionBuildup``'s hull
            interaction model and fed back into both the shaft thrust sizing and
            aft hull suction drag at the centre of buoyancy.
        propulsion_options : Keyword arguments for ``ElectricPropulsionBuildup``
            (propeller geometry and RPM, gearbox, motor, ``endurance_hours``, ...).
        craft_options : Further keyword arguments for ``CraftBuildup``.
        """
        super().__init__()
        self.hull = hull
        self.velocity = velocity
        self.mass = mass
        self.xyz_cg = xyz_cg
        self.xyz_propeller = xyz_propeller
        self.xyz_motor = xyz_motor
        self.xyz_battery = xyz_battery
        self.mass_drivetrain = mass_drivetrain
        self.mass_battery = mass_battery
        self.shaft_angle = shaft_angle
        self.assembly = assembly
        self.sinkage = sinkage
        self.trim = trim
        self.water = Water.seawater() if water is None else water
        self.propulsion_options = dict(propulsion_options or {})
        self.craft_options = dict(craft_options or {})
        self.thrust_deduction = thrust_deduction

    def _craft_inputs(self, thrust_deduction: Vectorizable = 0.0) -> dict[str, Any]:
        """The ``CraftBuildup`` arguments this design implies, attitude excluded."""
        opts = dict(self.craft_options)
        opts.setdefault("thrust_deduction", thrust_deduction)
        return dict(
            hull=self.hull,
            velocity=self.velocity,
            mass=self.mass,
            xyz_cg=self.xyz_cg,
            assembly=self.assembly,
            water=self.water,
            xyz_thrust=self.xyz_propeller,
            shaft_angle=self.shaft_angle,
            point_masses=[
                (self.mass_drivetrain, self.xyz_motor),
                (self.mass_battery, self.xyz_battery),
            ],
            **opts,
        )

    def _craft(self, thrust_deduction: Vectorizable = 0.0) -> CraftBuildup:
        return CraftBuildup(
            sinkage=self.sinkage,
            trim=self.trim,
            **self._craft_inputs(thrust_deduction=thrust_deduction),
        )

    def _propulsion(self, craft: CraftBuildup, drag_bare: Vectorizable) -> dict[str, Any]:
        """The drive overcoming bare horizontal drag ``drag_bare`` at this attitude."""
        # The propeller where the trimmed, sunk hull actually carries it.
        x_disc, z_disc = craft._earth(self.xyz_propeller[0], self.xyz_propeller[2])
        prop_opts = dict(self.propulsion_options)
        if self.thrust_deduction is not None:
            prop_opts.setdefault("thrust_deduction", self.thrust_deduction)
        return ElectricPropulsionBuildup(
            op_point=HydroOperatingPoint(water=self.water, velocity=self.velocity),
            water=self.water,
            drag_bare_N=drag_bare,
            hull=self.hull,
            disc_xyz=[x_disc, self.xyz_propeller[1], z_disc],
            sinkage=self.sinkage,
            trim=self.trim,
            pivot_xz=(craft.xyz_cg[0], craft.xyz_cg[2]),
            shaft_depth=-z_disc,
            shaft_angle_deg=self.trim + self.shaft_angle,
            **prop_opts,
        ).run()

    def run(self, with_stability: bool = False) -> dict[str, Any]:
        explicit_t = (
            self.thrust_deduction
            if self.thrust_deduction is not None
            else self.propulsion_options.get(
                "thrust_deduction",
                self.craft_options.get("thrust_deduction", None),
            )
        )
        if explicit_t is not None:
            craft = self._craft(thrust_deduction=explicit_t)
            res = craft.run_with_stability() if with_stability else craft.run()
            propulsion = self._propulsion(craft, res["D"])
        else:
            probe_craft = self._craft(thrust_deduction=0.0)
            base = probe_craft.run()
            propulsion = self._propulsion(probe_craft, base["D"])
            t_ded = propulsion["thrust_deduction"]
            if with_stability:
                craft = self._craft(thrust_deduction=t_ded)
                res = craft.run_with_stability()
            else:
                res = base
                cos_shaft = np.cosd(self.trim + self.shaft_angle)
                sin_shaft = np.sind(self.trim + self.shaft_angle)
                thrust_new = res["D"] / np.fmax((1.0 - t_ded) * cos_shaft, 1e-6)
                delta_t = thrust_new - res["thrust"]
                suction_drag = t_ded * thrust_new * cos_shaft
                x_cg, z_cg = probe_craft.xyz_cg[0], probe_craft.xyz_cg[2]
                _, z_cg_earth = probe_craft._earth(x_cg, z_cg)
                x_t, z_t = probe_craft._earth(
                    self.xyz_propeller[0], self.xyz_propeller[2]
                )
                res["thrust"] = thrust_new
                res["F_z_net"] = res["F_z_net"] + delta_t * sin_shaft
                res["M_y_net"] = (
                    res["M_y_net"]
                    + (
                        (z_t - z_cg_earth) * (-delta_t * cos_shaft)
                        - (x_t - x_cg) * (delta_t * sin_shaft)
                    )
                    + (res["z_cb_hull"] - z_cg_earth) * suction_drag
                )

        sized_drivetrain = (
            propulsion["mass_motor_kg"]
            + propulsion["mass_controller_kg"]
            + propulsion["mass_gearbox_kg"]
        )
        res.update(
            propulsion=propulsion,
            P_battery=propulsion["P_battery"],
            eta_overall=propulsion["eta_overall"],
            mass_drivetrain_sized=sized_drivetrain,
            mass_battery_sized=propulsion["mass_battery_kg"],
            mass_drivetrain_residual=self.mass_drivetrain - sized_drivetrain,
            mass_battery_residual=self.mass_battery - propulsion["mass_battery_kg"],
            thrust_residual=propulsion["thrust_residual_N"],
        )
        return res

    @classmethod
    def solve(
        cls,
        *args,
        rpm_bounds: tuple[float, float] = (100.0, 8000.0),
        mass_tolerance: float = 0.02,
        max_iterations: int = 12,
        with_stability: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Close the loop for a fixed (numeric) design: find the attitude, the
        propeller RPM (B-series) and the drive masses at which the craft is in
        balance, the propeller delivers the thrust needed, and the masses in the
        balance are the ones the propulsion model sizes.

        Takes the constructor's arguments; ``mass_drivetrain``, ``mass_battery``,
        ``sinkage`` and ``trim``, if given, are starting guesses. Iterates
        attitude -> RPM -> masses until both masses move by less than
        ``mass_tolerance`` [kg].
        """
        from scipy.optimize import brentq

        kwargs = dict(kwargs)
        mass_drivetrain = float(kwargs.pop("mass_drivetrain", 0.0))
        mass_battery = float(kwargs.pop("mass_battery", 0.0))
        sinkage = float(kwargs.pop("sinkage", 0.0))
        trim = float(kwargs.pop("trim", 0.0))
        propulsion_options = dict(kwargs.pop("propulsion_options", None) or {})
        uses_bseries = propulsion_options.get("propeller_model", "bseries") == "bseries"
        explicit_t = kwargs.get(
            "thrust_deduction",
            propulsion_options.get(
                "thrust_deduction",
                (kwargs.get("craft_options") or {}).get("thrust_deduction", None),
            ),
        )
        t_ded_iter = float(explicit_t) if explicit_t is not None else 0.0

        def build(**state) -> "PoweredCraftBuildup":
            return cls(*args, propulsion_options=propulsion_options, **kwargs, **state)

        for iteration in range(max_iterations):
            masses = dict(mass_drivetrain=mass_drivetrain, mass_battery=mass_battery)
            # 1. Attitude, with the current masses and thrust deduction on board.
            attitude = CraftBuildup.solve_equilibrium(
                sinkage_guess=sinkage,
                trim_guess=trim,
                **dict(
                    build(**masses)._craft_inputs(thrust_deduction=t_ded_iter),
                    include_interference_breakdown=False,
                ),
            )
            sinkage, trim = attitude["sinkage"], attitude["trim"]
            state = dict(masses, sinkage=sinkage, trim=trim)

            # 2. Propeller speed that delivers the thrust (B-series only). The craft
            #    does not change with RPM, so only the drive is re-evaluated.
            if uses_bseries:
                design = build(**state)
                craft = design._craft(thrust_deduction=t_ded_iter)
                drag_bare = _scalar(craft.run()["D"])

                def thrust_gap(rpm: float) -> float:
                    design.propulsion_options["propeller_rpm"] = rpm
                    return _scalar(design._propulsion(craft, drag_bare)["thrust_residual_N"])

                lo, hi = rpm_bounds
                if thrust_gap(lo) > 0.0 or thrust_gap(hi) < 0.0:
                    raise RuntimeError(
                        f"The propeller cannot deliver the required thrust between "
                        f"{lo:.0f} and {hi:.0f} RPM."
                    )
                propulsion_options["propeller_rpm"] = brentq(thrust_gap, lo, hi, xtol=1e-3)

            # 3. Masses (and thrust deduction) the propulsion model sizes at this state.
            res = build(**state).run()
            new_drivetrain = _scalar(res["mass_drivetrain_sized"])
            new_battery = _scalar(res["mass_battery_sized"])
            new_t_ded = _scalar(res["propulsion"]["thrust_deduction"])
            change = max(abs(new_drivetrain - mass_drivetrain), abs(new_battery - mass_battery))
            t_change = abs(new_t_ded - t_ded_iter)
            if verbose:
                print(
                    f"  iter {iteration}: trim {trim:+.3f} deg, sinkage {1000 * sinkage:+.1f} mm, "
                    f"drivetrain {new_drivetrain:.2f} kg, battery {new_battery:.2f} kg"
                )
            mass_drivetrain, mass_battery = new_drivetrain, new_battery
            t_ded_iter = new_t_ded
            if change < mass_tolerance and t_change < 1e-4:
                break
        else:
            raise RuntimeError("Drive masses did not converge.")

        final = build(
            mass_drivetrain=mass_drivetrain, mass_battery=mass_battery,
            sinkage=sinkage, trim=trim,
        )
        out = final.run(with_stability=with_stability)
        out["propeller_rpm"] = propulsion_options.get("propeller_rpm")
        return out


__all__ = ["PoweredCraftBuildup"]
