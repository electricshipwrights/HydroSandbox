"""
The boat model as the design API uses it: named quantities for one scenario.

``quantities()`` evaluates a boat with a design at one state of its loads and one
attitude, and returns the named quantities the objective is written in, plus the
two balance residuals. It works for numbers and for CasADi expressions alike, so
the optimizer and ``evaluate()`` see the same model.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as onp

import aerosandbox.numpy as np
from hydrosandbox.design.boat import Boat
from hydrosandbox.design.design_space import DesignSpace
from hydrosandbox.design.distributions import is_free, mean_of
from hydrosandbox.hull import CraftBuildup

#: The quantities the model reports for every scenario, in order. A design space
#: adds its own after these.
MODEL_QUANTITIES = (
    "drag", "drag_hull", "drag_foils", "drag_wave",
    "sinkage", "trim",
    "pitch_stiffness", "heave_stiffness", "pitch_damping",
    "foil_lift", "foil_lift_fraction", "lift_coefficient_max", "lift_coefficient_min",
    "cavitation_margin", "ventilation_margin",
    "thrust", "mass", "x_cg", "y_cg", "z_cg",
)

#: Margins are infinite when there is nothing to cavitate or ventilate; they are
#: reported capped at this.
MARGIN_CAP = 10.0


@dataclass
class ThrustAt:
    """Propulsion as a thrust, equal to the drag, acting at a point ``[x, y, z]`` along the hull's axis."""

    position: Sequence[float]


def placed_variables(boat: Boat) -> dict[str, tuple[float, float, float]]:
    """The design variables for loads the optimizer places: name -> (low, high, start)."""
    out = {}
    for load in boat.loads:
        for axis, component in zip("xyz", load.position):
            if is_free(component):
                out[f"{load.name}.{axis}"] = (component.low, component.high, component.initial())
    return out


def centre_of_gravity(boat: Boat, values: dict[str, Any], mass, moment, placed_masses: dict[str, Any]):
    """Total mass and centre of gravity, from a scenario and the placed loads' positions."""
    total = mass
    moment = [moment[k] for k in range(3)]
    for load in boat.loads:
        if not load.placed:
            continue
        m = placed_masses[load.name]
        total = total + m
        for k, (axis, component) in enumerate(zip("xyz", load.position)):
            position = values[f"{load.name}.{axis}"] if is_free(component) else mean_of(component)
            moment[k] = moment[k] + m * position
    return total, [moment[k] / total for k in range(3)]


def quantities(
    boat: Boat,
    velocity: float,
    design: DesignSpace,
    values: dict[str, Any],
    mass,
    cg: Sequence[Any],
    sinkage,
    trim,
    propulsion: ThrustAt | None = None,
    tables: dict[str, Any] | None = None,
    need: set[str] | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    """
    The named quantities of one scenario, and the vertical-force and pitching-moment
    residuals (as fractions of the weight, and of weight times hull length).

    ``tables`` holds the hull's attitude tables (``hull_dynamics``,
    ``hull_wave_amplitude``) for the optimizer; without them the hull is integrated
    directly. ``need`` names the quantities wanted; the stiffnesses, which are
    derivatives of the model and its costliest part, are left out if not among them
    (None: everything).
    """
    hull, water = boat.hull, boat.water
    assembly = design.builder(values, boat)
    options = dict(
        wave_length_scales=(1.0, 1.5 * hull.length, 0.01),
        include_interference_breakdown=False,
    )
    options.update(boat.craft_options)
    options.update(tables or {})
    craft = CraftBuildup(
        hull=hull, velocity=velocity, mass=mass, xyz_cg=[cg[0], 0.0, cg[2]], water=water,
        assembly=assembly, sinkage=sinkage, trim=trim, demihull_spacing=boat.demihull_spacing,
        xyz_thrust=None if propulsion is None else list(propulsion.position),
        **options,
    )
    with_stiffness = need is None or bool(need & {"pitch_stiffness", "heave_stiffness"})
    with_damping = need is None or "pitch_damping" in need
    if with_stiffness:
        res = craft.run_with_stability(include_pitch_damping=with_damping)
    else:
        res = craft.run()
        if with_damping and assembly is not None:
            pitching = copy.copy(craft)
            pitching.pitch_rate = 0.01
            res["c_pitch_foils"] = -(pitching.run()["M_y_net"] - res["M_y_net"]) / 0.01
        else:
            res["c_pitch_foils"] = 0.0 if with_damping else None
    weight = mass * water.gravity
    q = 0.5 * water.density() * velocity**2

    if assembly is not None:
        foil = res["foil"]
        coefficients = [
            component.L / (q * wing.area())
            for wing, component in zip(assembly.wings, foil["wing_hydro_components"])
            if not getattr(wing, "is_strut", False)
        ]
        cl_max = cl_min = coefficients[0] if coefficients else 0.0
        for c in coefficients[1:]:
            cl_max, cl_min = np.fmax(cl_max, c), np.fmin(cl_min, c)
        cavitation = np.fmin(foil["cavitation_margin"], MARGIN_CAP)
        ventilation = np.fmin(foil["ventilation_margin"], MARGIN_CAP)
        foil_lift, drag_foils = res["L_foil"], res["D_foil"]
    else:
        cl_max = cl_min = foil_lift = drag_foils = 0.0
        cavitation = ventilation = MARGIN_CAP

    out = {
        "drag": res["D"],
        "drag_hull": res["D_hull_viscous"],
        "drag_foils": drag_foils,
        "drag_wave": res["D_wave"],
        "sinkage": sinkage,
        "trim": trim,
        "pitch_stiffness": res["GM_L"] if with_stiffness else None,
        "heave_stiffness": res["k_heave"] if with_stiffness else None,
        "pitch_damping": res["c_pitch_foils"] if with_damping else None,
        "foil_lift": foil_lift,
        "foil_lift_fraction": foil_lift / weight,
        "lift_coefficient_max": cl_max,
        "lift_coefficient_min": cl_min,
        "cavitation_margin": cavitation,
        "ventilation_margin": ventilation,
        "thrust": res["thrust"],
        "mass": mass,
        "x_cg": cg[0],
        "y_cg": cg[1],
        "z_cg": cg[2],
    }
    for name, function in design.quantities.items():
        if name in out:
            raise ValueError(f"The design space's quantity {name!r} has the name of a model quantity.")
        out[name] = function(values, boat)
    out = {name: value for name, value in out.items() if value is not None}
    return out, res["F_z_net"] / weight, res["M_y_net"] / (weight * hull.length)


def hull_tables(boat: Boat, velocity: float, trim_bounds: tuple[float, float], jobs: int | None = None) -> dict[str, Any]:
    """The hull's dynamic-force and wave-amplitude tables over attitude, for the optimizer."""
    hull, water = boat.hull, boat.water
    scales = boat.craft_options.get("wave_length_scales", (1.0, 1.5 * hull.length, 0.01))
    ranges = dict(
        sinkage_range=(-0.4, 0.2),
        trim_range=(min(float(trim_bounds[0]), -2.0), max(float(trim_bounds[1]), 4.0)),
        n_sinkage=11, n_trim=11, demihull_spacing=boat.demihull_spacing, jobs=jobs,
    )
    return {
        "hull_dynamics": hull.dynamic_force_table(velocity, water.density(), water.gravity, **ranges),
        "hull_wave_amplitude": hull.wave_amplitude_table(velocity, water.gravity, length_scales=scales, **ranges),
    }


def balance(
    boat: Boat, velocity: float, design: DesignSpace, values: dict[str, float], mass: float, cg: Sequence[float],
    propulsion: ThrustAt | None = None, tables: dict[str, Any] | None = None,
    guess: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    """The sinkage and trim at which a numeric design balances in one scenario."""
    options = dict(wave_length_scales=(1.0, 1.5 * boat.hull.length, 0.01))
    options.update(boat.craft_options)
    options.update(tables or {})
    eq = CraftBuildup.solve_equilibrium(
        hull=boat.hull, velocity=velocity, mass=mass, xyz_cg=[cg[0], 0.0, cg[2]], water=boat.water,
        assembly=design.builder(values, boat), demihull_spacing=boat.demihull_spacing,
        xyz_thrust=None if propulsion is None else list(propulsion.position),
        sinkage_guess=guess[0], trim_guess=guess[1], **options,
    )
    return float(eq["sinkage"]), float(eq["trim"])


def as_numbers(named: dict[str, Any]) -> dict[str, float]:
    return {name: float(onp.ravel(value)[0]) for name, value in named.items()}
