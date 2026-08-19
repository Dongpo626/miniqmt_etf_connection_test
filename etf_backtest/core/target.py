"""Immutable strategy target-portfolio value object."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol


@dataclass(frozen=True, slots=True)
class NoRebalance:
    """Explicit strategy directive to keep holdings without creating a target."""


NO_REBALANCE = NoRebalance()


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """A complete target-weight mapping produced by a strategy.

    Symbols omitted from ``weights`` have an explicit target weight of zero.
    The input mapping is defensively copied, sorted by symbol, and exposed as
    an immutable mapping so callers cannot alter a strategy decision later.
    """

    weights: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        """Validate Decimal weights and freeze a canonical defensive copy."""
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must be a mapping")

        canonical_weights: dict[str, Decimal] = {}
        total = Decimal("0")

        for symbol, weight in self.weights.items():
            if not isinstance(symbol, str):
                raise TypeError("target symbol must be a string")
            canonical_symbol = normalize_symbol(symbol)
            if not isinstance(weight, Decimal):
                raise TypeError("target weight must be Decimal")
            if not weight.is_finite():
                raise ValueError("target weight must be finite")
            if weight < Decimal("0") or weight > Decimal("1"):
                raise ValueError("target weight must be between zero and one")

            if canonical_symbol in canonical_weights:
                raise ValueError("target symbols must be unique after normalization")
            canonical_weights[canonical_symbol] = weight
            total += weight

        if total > Decimal("1"):
            raise ValueError("target weights must not sum to more than one")

        sorted_weights = dict(sorted(canonical_weights.items()))
        object.__setattr__(self, "weights", MappingProxyType(sorted_weights))

    def weight_for(self, symbol: str) -> Decimal:
        """Return a symbol's target weight, or zero when it is omitted."""
        return self.weights.get(normalize_symbol(symbol), Decimal("0"))


StrategyTarget = TargetPortfolio | NoRebalance

__all__ = ["NO_REBALANCE", "NoRebalance", "StrategyTarget", "TargetPortfolio"]
