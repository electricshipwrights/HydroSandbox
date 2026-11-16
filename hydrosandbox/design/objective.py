"""
One objective for a design: what to maximize, discounted by costs and penalties.

For scenario ``k`` with weight ``w_k`` the utility is

    U_k = S_k * prod_j (ref_j / q_jk) ** p_j * exp(-sum_i lambda_i * v_ik ** 2)

and the optimizer maximizes ``J = sum_k w_k U_k - risk_aversion * sd_k(U_k)``.
``S`` is the quantity to maximize (1 if none), ``q`` are costs with exponents ``p``,
and ``v`` are violations: how far a quantity is past a threshold, in units of a
scale, and zero when it is inside. A zero exponent or weight switches a term off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import aerosandbox.numpy as np

Quantity = str | Callable[[dict[str, Any]], Any]


def _value(quantity: Quantity, quantities: dict[str, Any]):
    if callable(quantity):
        return quantity(quantities)
    if quantity not in quantities:
        raise KeyError(
            f"Unknown quantity {quantity!r}. Available: {', '.join(sorted(quantities))}."
        )
    return quantities[quantity]


def _label(quantity: Quantity) -> str:
    return quantity if isinstance(quantity, str) else getattr(quantity, "__name__", "custom quantity")


@dataclass
class Cost:
    """
    A quantity to keep small: multiplies the utility by ``(reference / q) ** exponent``.
    With ``reference=None`` it is the quantity's value for the starting design in the
    nominal scenario.
    """

    quantity: Quantity
    exponent: float = 1.0
    reference: float | None = None


@dataclass
class _Penalty:
    quantity: Quantity
    threshold: float
    weight: float = 1.0
    scale: float | None = None

    def __post_init__(self):
        if self.scale is None:
            if self.threshold == 0:
                raise ValueError(
                    f"{type(self).__name__}({_label(self.quantity)!r}, 0): give a `scale`, "
                    "the amount past zero that counts as one unit of violation."
                )
            self.scale = 0.1 * abs(self.threshold)
        if self.scale <= 0 or self.weight < 0:
            raise ValueError("A penalty's scale must be positive and its weight not negative.")

    def excess(self, value):
        raise NotImplementedError

    def violation(self, quantities: dict[str, Any]):
        """How far past the threshold, in units of ``scale``; zero when inside."""
        return np.fmax(self.excess(_value(self.quantity, quantities)), 0.0) / self.scale


class AtLeast(_Penalty):
    """
    Penalize a quantity for falling below ``threshold``. One unit of violation is
    ``scale`` below it (a tenth of the threshold by default) and multiplies the
    utility by ``exp(-weight)``; the penalty grows with the square of the violation.
    """

    def excess(self, value):
        return self.threshold - value


class AtMost(_Penalty):
    """Penalize a quantity for rising above ``threshold`` (see ``AtLeast``)."""

    def excess(self, value):
        return value - self.threshold


@dataclass
class Objective:
    """
    Parameters
    ----------
    maximize : The quantity wanted (a name, or a function of the quantities). None:
        the utility is costs and penalties alone, e.g. ``costs=[Cost("drag")]``
        minimizes drag.
    costs : Quantities to keep small, each with an exponent.
    penalties : ``AtLeast`` / ``AtMost`` terms.
    risk_aversion : Standard deviations of the utility across scenarios to subtract
        from its mean.
    """

    maximize: Quantity | None = None
    costs: list[Cost] = field(default_factory=list)
    penalties: list[_Penalty] = field(default_factory=list)
    risk_aversion: float = 0.0

    def log_utility(self, quantities: dict[str, Any], references: dict[int, float], measure_scale: float = 1.0):
        """
        The natural logarithm of one scenario's utility. ``references`` holds each
        cost's reference by its position in ``costs``; ``measure_scale`` is a typical
        size of the maximized quantity, used only to keep it positive.

        The optimizer works with the logarithm: far past a threshold the utility
        itself is flat at zero, while its logarithm keeps a slope to follow.
        """
        if self.maximize is None:
            log_u = 0.0
        else:
            s = _value(self.maximize, quantities)
            # (kept positive, so that a penalty can never turn into a reward)
            log_u = np.log(0.5 * (s + np.sqrt(s**2 + (1e-3 * measure_scale) ** 2)))
        for index, cost in enumerate(self.costs):
            if cost.exponent != 0:
                log_u = log_u + cost.exponent * (
                    np.log(references[index]) - np.log(_value(cost.quantity, quantities))
                )
        for term in self.penalties:
            if term.weight != 0:
                log_u = log_u - term.weight * term.violation(quantities) ** 2
        return log_u

    def utility(self, quantities: dict[str, Any], references: dict[int, float], measure_scale: float = 1.0):
        """The utility of one scenario (see ``log_utility``)."""
        return np.exp(self.log_utility(quantities, references, measure_scale))

    def needed(self) -> set[str] | None:
        """The names of the quantities this objective uses; None if it cannot tell."""
        used = [self.maximize] + [c.quantity for c in self.costs] + [t.quantity for t in self.penalties]
        used = [u for u in used if u is not None]
        if any(callable(u) for u in used):
            return None
        return set(used)

    def violations(self, quantities: dict[str, Any]) -> list[dict[str, Any]]:
        """The penalty terms that are active for numeric ``quantities``."""
        active = []
        for term in self.penalties:
            v = float(term.violation(quantities))
            if v > 1e-6 and term.weight != 0:
                active.append({
                    "quantity": _label(term.quantity),
                    "kind": "at least" if isinstance(term, AtLeast) else "at most",
                    "threshold": term.threshold,
                    "value": float(_value(term.quantity, quantities)),
                    "violation": v,
                    "utility_factor": float(np.exp(-term.weight * v**2)),
                })
        return active
