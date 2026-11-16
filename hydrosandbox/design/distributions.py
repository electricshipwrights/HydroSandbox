"""Numbers that are uncertain, or the optimizer's to choose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as onp
from scipy.stats import norm


class Distribution:
    """An uncertain number. Subclasses give its mean and its quantile function."""

    def mean(self) -> float:
        raise NotImplementedError

    def quantile(self, u: float) -> float:
        """The value below which a fraction ``u`` of the probability lies."""
        raise NotImplementedError


@dataclass(frozen=True)
class Normal(Distribution):
    """Normally distributed, with a mean and a standard deviation."""

    mean_value: float
    sd: float

    def mean(self) -> float:
        return float(self.mean_value)

    def quantile(self, u: float) -> float:
        return float(self.mean_value + self.sd * norm.ppf(u))


@dataclass(frozen=True)
class Uniform(Distribution):
    """Equally likely anywhere between ``low`` and ``high``."""

    low: float
    high: float

    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    def quantile(self, u: float) -> float:
        return float(self.low + u * (self.high - self.low))


class Discrete(Distribution):
    """One of a list of values, with weights (equal if not given)."""

    def __init__(self, values: Sequence[float], weights: Sequence[float] | None = None):
        self.values = onp.asarray(values, dtype=float)
        w = onp.ones(len(self.values)) if weights is None else onp.asarray(weights, dtype=float)
        if len(w) != len(self.values) or onp.any(w < 0) or w.sum() <= 0:
            raise ValueError("Discrete needs one non-negative weight per value.")
        self.weights = w / w.sum()

    def mean(self) -> float:
        return float(self.weights @ self.values)

    def quantile(self, u: float) -> float:
        index = int(onp.searchsorted(onp.cumsum(self.weights), u, side="right"))
        return float(self.values[min(index, len(self.values) - 1)])

    def __repr__(self) -> str:
        return f"Discrete({self.values.tolist()}, {self.weights.tolist()})"


@dataclass(frozen=True)
class Free:
    """A number the optimizer chooses, between ``low`` and ``high``."""

    low: float
    high: float
    start: float | None = None

    def initial(self) -> float:
        return 0.5 * (self.low + self.high) if self.start is None else float(self.start)


def is_uncertain(value) -> bool:
    return isinstance(value, Distribution)


def is_free(value) -> bool:
    return isinstance(value, Free)


def mean_of(value) -> float:
    """The mean of a distribution, the start of a free number, or the number itself."""
    if isinstance(value, Distribution):
        return value.mean()
    if isinstance(value, Free):
        return value.initial()
    return float(value)
