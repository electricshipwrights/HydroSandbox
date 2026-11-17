"""The design variables, and how a set of their values becomes geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DesignSpace:
    """
    Parameters
    ----------
    builder : ``builder(values, boat)`` returns the ``HydrofoilAssembly`` (foils,
        struts, bodies) for a dict of variable values, or None for no appendages.
        The values may be numbers or CasADi expressions.
    variables : Each variable's ``(low, high)`` bounds, by name.
    start : A starting value for each variable.
    quantities : Further named quantities that follow from the design alone, each a
        function ``(values, boat)``; e.g. a bending stress or a clearance. They are
        reported with the model's own quantities and can be used in the objective.
    """

    builder: Callable[[dict[str, Any], Any], Any]
    variables: dict[str, tuple[float, float]]
    start: dict[str, float]
    quantities: dict[str, Callable[[dict[str, Any], Any], Any]] = field(default_factory=dict)

    def __post_init__(self):
        missing = set(self.variables) - set(self.start)
        if missing:
            raise ValueError(f"No starting value for: {sorted(missing)}.")
        for name, (low, high) in self.variables.items():
            if not low <= self.start[name] <= high:
                raise ValueError(f"Start of {name!r} ({self.start[name]}) is outside its bounds ({low}, {high}).")
