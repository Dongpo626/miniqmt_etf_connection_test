"""Framework-neutral daily strategy driven by a predictor bundle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from etf_backtest.core.market import MarketBarView
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView, StrategyContext
from etf_backtest.strategy.model_contracts import (
    FeatureBuilder,
    PredictionRecord,
    PredictorBundle,
    feature_fingerprint,
    feature_records_for_signal,
    validate_feature_builder,
)
from etf_backtest.strategy.portfolio import (
    ModelPortfolioPolicy,
    validate_model_allocation,
)
from etf_backtest.strategy.scheduler import EveryTradingDayScheduler


@dataclass(frozen=True, slots=True)
class ModelAllocationRecord:
    """One prediction's rank and applied target weight on a signal date."""

    signal_date: date
    symbol: str
    score: float
    rank: int
    selected: bool
    target_weight: Decimal


class DailyModelStrategy(BaseStrategy):
    """Turn daily model scores into one validated multi-asset target."""

    __slots__ = (
        "_allocations",
        "_bundle",
        "_feature_builder",
        "_lookback",
        "_portfolio",
        "_predictions",
        "_scheduler",
    )

    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder,
        bundle: PredictorBundle,
        portfolio: ModelPortfolioPolicy,
    ) -> None:
        feature_names, lookback = validate_feature_builder(feature_builder)
        if not isinstance(bundle, PredictorBundle):
            raise TypeError("bundle must satisfy PredictorBundle")
        metadata = bundle.metadata
        if metadata.feature_names != feature_names:
            raise ValueError("bundle feature_names do not match FeatureBuilder")
        if metadata.feature_fingerprint != feature_fingerprint(feature_names, metadata.label_name):
            raise ValueError("bundle feature fingerprint is invalid")
        if not isinstance(portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        self._feature_builder = feature_builder
        self._bundle = bundle
        self._lookback = lookback
        self._portfolio = portfolio
        self._predictions: list[PredictionRecord] = []
        self._allocations: list[ModelAllocationRecord] = []
        self._scheduler = EveryTradingDayScheduler()

    @property
    def predictions(self) -> tuple[PredictionRecord, ...]:
        return tuple(self._predictions)

    @property
    def allocations(self) -> tuple[ModelAllocationRecord, ...]:
        return tuple(self._allocations)

    @property
    def required_history_trading_days(self) -> int:
        return self._lookback

    def should_generate_target(self, frame_index: int) -> bool:
        return self._scheduler.should_decide(frame_index)

    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> TargetPortfolio:
        del account_view, context
        if signal_date <= self._bundle.metadata.trained_through:
            raise ValueError("model signal_date must follow the training interval")
        records = feature_records_for_signal(
            builder=self._feature_builder,
            market_views=market_history,
            signal_date=signal_date,
        )
        if not records:
            return TargetPortfolio(weights={})
        predictions = self._bundle.predict(records)
        expected_keys = tuple(record.key for record in records)
        actual_keys = tuple(prediction.key for prediction in predictions)
        if len(actual_keys) != len(set(actual_keys)) or frozenset(actual_keys) != frozenset(
            expected_keys
        ):
            raise ValueError("bundle predictions must exactly match feature record keys")
        self._predictions.extend(predictions)
        weights = validate_model_allocation(
            self._portfolio.allocate(predictions),
            predictions=predictions,
            exposure_cap=self._portfolio.max_total_weight,
        )
        ranked = sorted(
            predictions,
            key=lambda prediction: (prediction.score, prediction.key.symbol),
            reverse=True,
        )
        self._allocations.extend(
            ModelAllocationRecord(
                signal_date=prediction.key.signal_date,
                symbol=prediction.key.symbol,
                score=prediction.score,
                rank=rank,
                selected=weights.get(prediction.key.symbol, Decimal("0")) > Decimal("0"),
                target_weight=weights.get(prediction.key.symbol, Decimal("0")),
            )
            for rank, prediction in enumerate(ranked, start=1)
        )
        return TargetPortfolio(weights=weights)


__all__ = ["DailyModelStrategy", "ModelAllocationRecord"]
