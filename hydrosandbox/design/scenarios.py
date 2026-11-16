"""
From uncertain loads to a few weighted scenarios.

The boat model responds to its loads only through a few summaries of them: the
total mass and the centre of gravity (today its fore-aft position and height; its
sideways position once heel is modelled). However many loads are uncertain, the
scenarios are placed on the joint spread of those summaries: one at the centre and
two along each direction in which they actually vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Sequence

import numpy as onp
from scipy.stats import qmc

from hydrosandbox.design.boat import Boat
from hydrosandbox.design.distributions import is_free, is_uncertain, mean_of

#: The components of the centre of gravity the model responds to (0 = fore-aft,
#: 1 = sideways, 2 = height). Sideways joins when heel is modelled.
RESPONDS_TO_CG = (0, 2)


@dataclass
class Scenario:
    """
    One weighted state of the loads, as the model sees it: the mass and the moment
    (mass times position, about the axes' origin) of everything whose position is
    not the optimizer's to choose, and the mass of each placed load.
    """

    weight: float
    mass: float
    moment: onp.ndarray
    placed_masses: dict[str, float] = field(default_factory=dict)
    name: str = ""

    def parameters(self, placed_names: Sequence[str]) -> list[float]:
        """As a flat list: mass, the three moments, then the placed loads' masses."""
        return [self.mass, *self.moment.tolist(), *[self.placed_masses[n] for n in placed_names]]


def _slots(boat: Boat) -> list[tuple[int, str]]:
    """Every uncertain number among the loads, as (load index, field)."""
    slots = []
    for i, load in enumerate(boat.loads):
        if is_uncertain(load.mass):
            slots.append((i, "mass"))
        for k, component in enumerate(load.position):
            if is_uncertain(component):
                slots.append((i, f"position{k}"))
    return slots


def _state(boat: Boat, values: dict[tuple[int, str], float], aboard: Sequence[bool]) -> Scenario:
    """The scenario for given values of the uncertain numbers (others at their own value)."""
    mass = float(boat.fixed_mass)
    moment = mass * onp.asarray(boat.fixed_cg, dtype=float)
    placed = {}
    for i, load in enumerate(boat.loads):
        m = values.get((i, "mass"), mean_of(load.mass)) if aboard[i] else 0.0
        if load.placed:
            placed[load.name] = m
            continue
        position = onp.array([values.get((i, f"position{k}"), mean_of(c)) for k, c in enumerate(load.position)])
        mass += m
        moment = moment + m * position
    return Scenario(weight=1.0, mass=mass, moment=moment, placed_masses=placed)


def _vector(s: Scenario, placed_names: Sequence[str], length: float) -> onp.ndarray:
    """What the model responds to, in comparable units (kg)."""
    return onp.array(
        [s.mass, *[s.moment[k] / length for k in RESPONDS_TO_CG], *[s.placed_masses[n] for n in placed_names]]
    )


def _from_vector(v: onp.ndarray, like: Scenario, placed_names: Sequence[str], length: float) -> Scenario:
    moment = like.moment.copy()
    for j, k in enumerate(RESPONDS_TO_CG):
        moment[k] = v[1 + j] * length
    n = 1 + len(RESPONDS_TO_CG)
    return Scenario(1.0, float(v[0]), moment, {name: float(v[n + j]) for j, name in enumerate(placed_names)})


def _spread_points(boat: Boat, aboard: Sequence[bool], n_draws: int = 2048):
    """
    Points and weights that reproduce the mean and covariance of the model's load
    summaries for one aboard/not-aboard case: the centre, and a pair along each
    direction of variation. Also whether anything varied that the model ignores.
    """
    slots = [s for s in _slots(boat) if aboard[s[0]]]
    placed_names = [load.name for load in boat.loads if load.placed]
    length = boat.hull.length
    centre = _state(boat, {}, aboard)
    if not slots:
        return [centre], [1.0], False
    draws = qmc.Sobol(len(slots), seed=0).random(n_draws)
    draws = onp.clip(draws, 1e-6, 1 - 1e-6)
    states = [
        _state(boat, {slot: getattr_dist(boat, slot).quantile(u) for slot, u in zip(slots, row)}, aboard)
        for row in draws
    ]
    vectors = onp.array([_vector(s, placed_names, length) for s in states])
    mean, cov = vectors.mean(axis=0), onp.cov(vectors.T)
    cov = onp.atleast_2d(cov)
    ignored = [k for k in range(3) if k not in RESPONDS_TO_CG]
    ignored_varies = any(onp.std([s.moment[k] for s in states]) > 1e-9 * max(centre.mass, 1.0) for k in ignored)
    mean_state = Scenario(1.0, 0.0, onp.mean([s.moment for s in states], axis=0), {})
    eigenvalues, eigenvectors = onp.linalg.eigh(cov)
    keep = eigenvalues > 1e-10 * max(eigenvalues.max(), 1e-12)
    keep &= eigenvalues > 1e-12
    r = int(keep.sum())
    if r == 0:
        return [_from_vector(mean, mean_state, placed_names, length)], [1.0], ignored_varies
    kappa = max(3.0 - r, 1.0)
    points, weights = [mean], [kappa / (r + kappa)]
    for value, direction in zip(eigenvalues[keep], eigenvectors[:, keep].T):
        step = onp.sqrt((r + kappa) * value) * direction
        points += [mean + step, mean - step]
        weights += [0.5 / (r + kappa)] * 2
    return [_from_vector(p, mean_state, placed_names, length) for p in points], weights, ignored_varies


def getattr_dist(boat: Boat, slot: tuple[int, str]):
    load = boat.loads[slot[0]]
    return load.mass if slot[1] == "mass" else load.position[int(slot[1][-1])]


def load_scenarios(boat: Boat, scenarios: Any = None) -> tuple[list[Scenario], list[str]]:
    """
    The weighted scenarios for a boat's loads, and notes on anything that had no
    effect.

    ``scenarios`` is None (points placed on the spread of the load summaries the
    model responds to), 1 (every load at its expected value), a larger number (a
    fixed low-discrepancy sample of that many equally weighted draws of the loads),
    or a list of explicit cases, each a dict with a ``weight``, an optional ``name``
    and ``loads``: per load name, a dict with any of ``mass``, ``position`` and
    ``aboard``.
    """
    notes: list[str] = []
    loads = boat.loads
    optional = [i for i, load in enumerate(loads) if load.probability < 1.0]

    if isinstance(scenarios, (list, tuple)):
        out = []
        for n, case in enumerate(scenarios):
            given = case.get("loads", {})
            unknown = set(given) - {load.name for load in loads}
            if unknown:
                raise KeyError(f"Scenario names loads the boat does not have: {sorted(unknown)}.")
            values, aboard = {}, []
            for i, load in enumerate(loads):
                g = given.get(load.name, {})
                aboard.append(bool(g.get("aboard", True)))
                if "mass" in g:
                    values[(i, "mass")] = float(g["mass"])
                for k, component in enumerate(g.get("position", [])):
                    if not is_free(load.position[k]):
                        values[(i, f"position{k}")] = float(component)
            s = _state(boat, values, aboard)
            s.weight, s.name = float(case.get("weight", 1.0)), case.get("name", f"case {n + 1}")
            out.append(s)
        total = sum(s.weight for s in out)
        for s in out:
            s.weight /= total
        return out, notes

    if scenarios == 1:
        # Expected values: an optional load counts for its probability.
        s = _state(boat, {}, [True] * len(loads))
        for i in optional:
            load = loads[i]
            m = mean_of(load.mass) * (1.0 - load.probability)
            if load.placed:
                s.placed_masses[load.name] -= m
            else:
                s.mass -= m
                s.moment = s.moment - m * onp.array([mean_of(c) for c in load.position])
        s.name = "nominal"
        return [s], notes

    if scenarios is not None:
        n = int(scenarios)
        slots = _slots(boat)
        draws = qmc.Sobol(len(slots) + len(optional), seed=0).random(n) if slots or optional else onp.zeros((n, 0))
        draws = onp.clip(draws, 1e-6, 1 - 1e-6)
        out = []
        for j, row in enumerate(draws):
            aboard = [True] * len(loads)
            for i, u in zip(optional, row[len(slots):]):
                aboard[i] = bool(u < loads[i].probability)
            values = {slot: getattr_dist(boat, slot).quantile(u) for slot, u in zip(slots, row)}
            s = _state(boat, values, aboard)
            s.weight, s.name = 1.0 / n, f"draw {j + 1}"
            out.append(s)
        return out, notes

    if len(optional) > 3:
        raise ValueError(
            "More than three optional loads: give `scenarios` as a number of draws, or as explicit cases."
        )
    out, ignored_any = [], False
    for mask in product([True, False], repeat=len(optional)):
        aboard = [True] * len(loads)
        probability = 1.0
        for i, is_aboard in zip(optional, mask):
            aboard[i] = is_aboard
            probability *= loads[i].probability if is_aboard else 1.0 - loads[i].probability
        points, weights, ignored = _spread_points(boat, aboard)
        ignored_any |= ignored
        case = ", ".join(f"{loads[i].name} {'aboard' if a else 'not aboard'}" for i, a in zip(optional, mask))
        for j, (s, w) in enumerate(zip(points, weights)):
            s.weight = probability * w
            s.name = ("; ".join(filter(None, [case, "centre" if j == 0 else f"spread {j}"])))
            out.append(s)
    if ignored_any or any(
        is_uncertain(load.position[k]) for load in loads for k in range(3) if k not in RESPONDS_TO_CG
    ):
        notes.append(
            "Some loads vary sideways. The model has no heel, so this has no effect on the result."
        )
    return out, notes
