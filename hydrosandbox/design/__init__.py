"""
Design optimization of a boat with appendages, for the loads it will carry.

A problem is a ``Boat`` (hull, fixed mass, and ``PointMass`` loads that may be
uncertain or the optimizer's to place), a ``DesignSpace`` (variables and a builder
of geometry), and an ``Objective`` written in the model's named quantities. See
``optimize`` and ``evaluate``.
"""

from hydrosandbox.design.boat import Boat, PointMass
from hydrosandbox.design.design_space import DesignSpace
from hydrosandbox.design.distributions import Discrete, Free, Normal, Uniform
from hydrosandbox.design.layouts import (
    centreline_t_foil,
    foil_section,
    no_appendages,
    standard_penalties,
    twin_t_foils,
    two_strut_foil,
)
from hydrosandbox.design.model import MODEL_QUANTITIES, ThrustAt
from hydrosandbox.design.objective import AtLeast, AtMost, Cost, Objective
from hydrosandbox.design.optimize import Solver, evaluate, optimize
from hydrosandbox.design.scenarios import Scenario, load_scenarios

__all__ = [
    "AtLeast", "AtMost", "Boat", "Cost", "DesignSpace", "Discrete", "Free", "MODEL_QUANTITIES",
    "Normal", "Objective", "PointMass", "Scenario", "Solver", "ThrustAt", "Uniform",
    "centreline_t_foil", "evaluate", "foil_section", "load_scenarios", "no_appendages",
    "optimize", "standard_penalties", "twin_t_foils", "two_strut_foil",
]
