"""The boat a design problem is about: hull, water, fixed mass and loads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from hydrosandbox.design.distributions import Free, is_free, is_uncertain, mean_of

#: The name under which the fixed part's position appears among the placed items
#: (its design variables are ``fixed.x``, ``fixed.y``, ``fixed.z``).
FIXED = "fixed"
from hydrosandbox.fluid import Water
from hydrosandbox.hull import Hull


@dataclass
class PointMass:
    """
    One load: a mass at a position ``[x, y, z]`` in the boat's axes (x aft, y to
    starboard, z up from the design waterline) [kg, m].

    ``mass`` is a number or a distribution. Each position component is a number, a
    distribution, or ``Free(low, high)`` for a load the optimizer may place. A load
    with a ``Free`` component is a *placed* load: its other components must be plain
    numbers. ``probability`` is the chance that the load is aboard at all.
    """

    name: str
    mass: Any
    position: Sequence[Any]
    probability: float = 1.0

    def __post_init__(self):
        self.position = list(self.position)
        if len(self.position) != 3:
            raise ValueError(f"Load {self.name!r}: position must be [x, y, z].")
        if is_free(self.mass):
            raise ValueError(f"Load {self.name!r}: a mass cannot be Free; give a number or a distribution.")
        if not 0.0 < self.probability <= 1.0:
            raise ValueError(f"Load {self.name!r}: probability must be in (0, 1].")
        if self.placed and any(is_uncertain(c) for c in self.position):
            raise ValueError(
                f"Load {self.name!r}: a load with a Free position component cannot also have an "
                "uncertain one; give its other components as numbers."
            )

    @property
    def placed(self) -> bool:
        """True if the optimizer chooses any component of the position."""
        return any(is_free(c) for c in self.position)


@dataclass
class Boat:
    """
    What is being designed for.

    Parameters
    ----------
    hull : The hull, or one demihull of a catamaran.
    fixed_mass, fixed_cg : Mass [kg] and centre of gravity ``[x, y, z]`` [m] of
        everything that is not listed in ``loads``. A component of ``fixed_cg`` may
        be ``Free(low, high)``: the designer can still move the boat's own mass
        that far (ballast, battery, fit-out), and the optimizer chooses where.
    loads : Further masses, each possibly uncertain or the optimizer's to place.
    water : The water; sea water if None.
    demihull_spacing : Centreline-to-centreline spacing of a catamaran's demihulls
        [m]; None for a monohull.
    craft_options : Further arguments for ``hydrosandbox.hull.CraftBuildup``
        (e.g. ``wave_length_scales``).
    """

    hull: Hull
    fixed_mass: float
    fixed_cg: Sequence[float]
    loads: list[PointMass] = field(default_factory=list)
    water: Water | None = None
    demihull_spacing: float | None = None
    craft_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.water is None:
            self.water = Water.seawater()
        self.fixed_cg = list(self.fixed_cg)
        if len(self.fixed_cg) != 3:
            raise ValueError("fixed_cg must be [x, y, z].")
        if any(is_uncertain(c) for c in self.fixed_cg):
            raise ValueError("fixed_cg cannot be uncertain; describe what varies as a load.")
        self.fixed_cg = [c if is_free(c) else float(c) for c in self.fixed_cg]
        names = [load.name for load in self.loads]
        if len(set(names)) != len(names):
            raise ValueError("Loads must have distinct names.")
        if FIXED in names:
            raise ValueError(f"{FIXED!r} is reserved for the fixed part; give the load another name.")

    @property
    def fixed_placed(self) -> bool:
        """True if the optimizer chooses where the fixed part's mass sits."""
        return any(is_free(c) for c in self.fixed_cg)

    def placed_items(self) -> list[tuple[str, Any]]:
        """Everything whose position the optimizer chooses, as (name, position)."""
        items = [(FIXED, self.fixed_cg)] if self.fixed_placed else []
        return items + [(load.name, load.position) for load in self.loads if load.placed]

    def placed_names(self) -> list[str]:
        return [name for name, _ in self.placed_items()]

    def nominal_fixed_cg(self) -> list[float]:
        """The fixed part's centre of gravity, a free component at its start."""
        return [mean_of(c) for c in self.fixed_cg]
