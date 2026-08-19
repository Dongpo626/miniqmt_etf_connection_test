"""Framework-neutral portfolio policies for daily Model predictions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from types import MappingProxyType
from typing import Literal, Protocol, cast, runtime_checkable

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.strategy.model_contracts import PredictionRecord

PortfolioWeightInput = Decimal | str | int | float
WeightingMode = Literal["equal", "score_proportional", "softmax"]
AllocationFunction = Callable[
    [tuple[PredictionRecord, ...]],
    Mapping[str, PortfolioWeightInput],
]


@runtime_checkable
class ModelPortfolioPolicy(Protocol):
    """Allocate one signal date's predictions into a complete target portfolio."""

    @property
    def max_total_weight(self) -> Decimal:
        """Return the maximum exposure this policy may allocate."""

    def allocate(
        self,
        predictions: Sequence[PredictionRecord],
    ) -> Mapping[str, Decimal]:
        """Return validated target weights keyed by predicted symbol."""

    def resolved_dict(self) -> dict[str, object]:
        """Return JSON-compatible policy provenance."""


def _decimal_value(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must not be boolean")
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, str | int | float):
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"{field_name} must be valid decimal text") from exc
    else:
        raise TypeError(f"{field_name} must be Decimal or decimal-compatible")
    if not parsed.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _exposure(value: object) -> Decimal:
    parsed = _decimal_value(value, "total_weight")
    if not Decimal("0") < parsed <= Decimal("1"):
        raise ValueError("total_weight must be in (0, 1]")
    return parsed


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _freeze_predictions(
    predictions: Sequence[PredictionRecord],
) -> tuple[PredictionRecord, ...]:
    supplied = cast(object, predictions)
    if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
        raise TypeError("predictions must be a sequence")
    frozen = tuple(predictions)
    if any(not isinstance(prediction, PredictionRecord) for prediction in frozen):
        raise TypeError("predictions may contain only PredictionRecord values")
    keys = tuple(prediction.key for prediction in frozen)
    if len(keys) != len(set(keys)):
        raise ValueError("predictions contain duplicate sample keys")
    signal_dates = {prediction.key.signal_date for prediction in frozen}
    if len(signal_dates) > 1:
        raise ValueError("predictions must share the same signal_date")
    return frozen


def _validated_weights(
    value: object,
    *,
    predictions: tuple[PredictionRecord, ...],
    exposure_cap: Decimal,
) -> Mapping[str, Decimal]:
    if not isinstance(value, Mapping):
        raise TypeError("portfolio allocator output must be a mapping")
    predicted_symbols = {prediction.key.symbol for prediction in predictions}
    canonical: dict[str, Decimal] = {}
    total = Decimal("0")
    for supplied_symbol, supplied_weight in value.items():
        if not isinstance(supplied_symbol, str):
            raise TypeError("portfolio symbol must be a string")
        symbol = normalize_symbol(supplied_symbol)
        if symbol not in predicted_symbols:
            raise ValueError("portfolio symbol must occur in the current prediction set")
        if symbol in canonical:
            raise ValueError("portfolio symbols must be unique after normalization")
        weight = _decimal_value(supplied_weight, "portfolio weight")
        if not Decimal("0") <= weight <= Decimal("1"):
            raise ValueError("portfolio weight must be between zero and one")
        canonical[symbol] = weight
        total += weight
    if total > exposure_cap:
        raise ValueError("portfolio weights exceed the declared exposure cap")
    positive = {
        symbol: weight for symbol, weight in sorted(canonical.items()) if weight > Decimal("0")
    }
    return MappingProxyType(positive)


def validate_model_allocation(
    value: object,
    *,
    predictions: Sequence[PredictionRecord],
    exposure_cap: object,
) -> Mapping[str, Decimal]:
    """Apply the non-bypassable validation boundary to any policy output."""

    return _validated_weights(
        value,
        predictions=_freeze_predictions(predictions),
        exposure_cap=_exposure(exposure_cap),
    )


def _normalized_weights(
    ranked: tuple[PredictionRecord, ...],
    raw_weights: tuple[Decimal, ...],
    total_weight: Decimal,
) -> Mapping[str, Decimal]:
    if not ranked:
        return MappingProxyType({})
    raw_total = sum(raw_weights, start=Decimal("0"))
    if raw_total <= Decimal("0"):
        raise ValueError("portfolio normalization weights must be positive")
    allocated = Decimal("0")
    weights: dict[str, Decimal] = {}
    with localcontext() as context:
        context.prec = 50
        for prediction, raw_weight in zip(ranked[:-1], raw_weights[:-1], strict=True):
            weight = total_weight * raw_weight / raw_total
            weights[prediction.key.symbol] = weight
            allocated += weight
        weights[ranked[-1].key.symbol] = total_weight - allocated
    return _validated_weights(
        weights,
        predictions=ranked,
        exposure_cap=total_weight,
    )


@dataclass(frozen=True, slots=True)
class TopKPortfolio:
    """Select the highest qualifying scores and allocate one exposure budget."""

    top_k: int = 1
    total_weight: PortfolioWeightInput = Decimal("0.90")
    min_score: float = 0.0
    weighting: WeightingMode = "score_proportional"
    softmax_temperature: float = 1.0

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        total_weight = _exposure(self.total_weight)
        min_score = _finite_float(self.min_score, "min_score")
        temperature = _finite_float(self.softmax_temperature, "softmax_temperature")
        if self.weighting not in {"equal", "score_proportional", "softmax"}:
            raise ValueError("weighting must be equal, score_proportional, or softmax")
        if temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        object.__setattr__(self, "total_weight", total_weight)
        object.__setattr__(self, "min_score", min_score)
        object.__setattr__(self, "softmax_temperature", temperature)

    @property
    def max_total_weight(self) -> Decimal:
        return cast(Decimal, self.total_weight)

    def allocate(
        self,
        predictions: Sequence[PredictionRecord],
    ) -> Mapping[str, Decimal]:
        frozen = _freeze_predictions(predictions)
        ranked = tuple(
            sorted(
                (prediction for prediction in frozen if prediction.score > self.min_score),
                key=lambda prediction: (prediction.score, prediction.key.symbol),
                reverse=True,
            )[: self.top_k]
        )
        if not ranked:
            return MappingProxyType({})
        if self.weighting == "equal":
            raw_weights = tuple(Decimal("1") for _ in ranked)
        elif self.weighting == "score_proportional":
            threshold = Decimal(str(self.min_score))
            raw_weights = tuple(Decimal(str(item.score)) - threshold for item in ranked)
        else:
            maximum = ranked[0].score
            raw_weights = tuple(
                Decimal(str(math.exp((item.score - maximum) / self.softmax_temperature)))
                for item in ranked
            )
        return _normalized_weights(ranked, raw_weights, self.max_total_weight)

    def resolved_dict(self) -> dict[str, object]:
        return {
            "type": "top_k",
            "top_k": self.top_k,
            "total_weight": str(self.max_total_weight),
            "min_score": self.min_score,
            "weighting": self.weighting,
            "softmax_temperature": self.softmax_temperature,
        }


@dataclass(frozen=True, slots=True)
class CustomPortfolio:
    """Validate a trusted local allocation function defined in ``model.py``."""

    name: str
    total_weight: PortfolioWeightInput
    allocator: AllocationFunction

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must not be blank")
        if not callable(self.allocator):
            raise TypeError("allocator must be callable")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "total_weight", _exposure(self.total_weight))

    @property
    def max_total_weight(self) -> Decimal:
        return cast(Decimal, self.total_weight)

    def allocate(
        self,
        predictions: Sequence[PredictionRecord],
    ) -> Mapping[str, Decimal]:
        frozen = _freeze_predictions(predictions)
        supplied = self.allocator(frozen)
        return validate_model_allocation(
            supplied,
            predictions=frozen,
            exposure_cap=self.max_total_weight,
        )

    def resolved_dict(self) -> dict[str, object]:
        return {
            "type": "custom",
            "name": self.name,
            "total_weight": str(self.max_total_weight),
        }


__all__ = [
    "AllocationFunction",
    "CustomPortfolio",
    "ModelPortfolioPolicy",
    "PortfolioWeightInput",
    "TopKPortfolio",
    "WeightingMode",
    "validate_model_allocation",
]
