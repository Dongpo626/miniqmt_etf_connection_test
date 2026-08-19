"""Framework-neutral daily model allocation tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from etf_backtest.core.account import Account
from etf_backtest.core.market import MarketBarView, TurnoverRule
from etf_backtest.core.position import Position
from etf_backtest.strategy.context import AccountView, StrategyContext
from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    MODEL_BUNDLE_SCHEMA_VERSION,
    DateRange,
    FeatureRecord,
    ModelDataIdentity,
    ModelMetadata,
    PredictionRecord,
    canonical_json,
    feature_fingerprint,
)
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.portfolio import TopKPortfolio


class _CloseFeatureBuilder:
    feature_names = ("close",)
    required_history_trading_days = 1

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> tuple[Decimal, ...]:
        del symbol, signal_date
        return (history[-1].close,)


class _ScoreBundle:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        names = ("close",)
        self.metadata = ModelMetadata(
            schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
            model_id="fake",
            model_class_name="Fake",
            model_parameters_json=canonical_json({}),
            training_parameters_json=canonical_json({}),
            feature_names=names,
            feature_fingerprint=feature_fingerprint(names, DAILY_FORWARD_RETURN_LABEL),
            label_name=DAILY_FORWARD_RETURN_LABEL,
            train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
            valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
            test_range=DateRange(date(2024, 1, 1), date(2024, 12, 31)),
            random_seed=1,
            data_identity=ModelDataIdentity(
                dataset_version="test",
                manifest_sha256="a" * 64,
                snapshot_started_at_utc=datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC),
            ),
            trained_through=date(2022, 12, 30),
            framework_name="fake",
            framework_version="1",
        )

    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]:
        return tuple(
            PredictionRecord(key=record.key, score=self._scores[record.key.symbol])
            for record in records
        )


def _view(symbol: str, close: str) -> MarketBarView:
    trade_date = date(2024, 1, 2)
    value = Decimal(close)
    return MarketBarView(
        source_record_key=f"QMT:{symbol.split('.')[1]}:{trade_date.isoformat()}:1:1",
        symbol=symbol,
        trade_date=trade_date,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=1000,
        suspended=False,
    )


def _inputs(
    symbols: tuple[str, ...] = ("SH.510300", "SH.518880"),
) -> tuple[tuple[str, ...], AccountView, StrategyContext]:
    account = Account(
        cash=Decimal("1000000"),
        positions={
            symbol: Position(
                symbol=symbol,
                turnover_rule=(TurnoverRule.T0 if symbol == "SH.518880" else TurnoverRule.T1),
            )
            for symbol in symbols
        },
    )
    account_view = AccountView.from_account(account)
    context = StrategyContext(
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        frame_index=0,
        symbols=symbols,
        account_view=account_view,
        current_weights_by_symbol={symbol: Decimal("0") for symbol in symbols},
    )
    return symbols, account_view, context


@pytest.mark.unit
def test_daily_model_strategy_selects_highest_positive_score() -> None:
    symbols, account_view, context = _inputs()
    strategy = DailyModelStrategy(
        feature_builder=_CloseFeatureBuilder(),
        bundle=_ScoreBundle({symbols[0]: 0.01, symbols[1]: 0.02}),
        portfolio=TopKPortfolio(),
    )

    target = strategy.generate_target(
        signal_date=context.signal_date,
        market_history=(_view(symbols[0], "4"), _view(symbols[1], "6")),
        account_view=account_view,
        context=context,
    )

    assert dict(target.weights) == {symbols[1]: Decimal("0.90")}
    assert {prediction.key.symbol for prediction in strategy.predictions} == set(symbols)


@pytest.mark.unit
def test_daily_model_strategy_holds_cash_without_positive_score() -> None:
    symbols, account_view, context = _inputs()
    strategy = DailyModelStrategy(
        feature_builder=_CloseFeatureBuilder(),
        bundle=_ScoreBundle({symbols[0]: -0.01, symbols[1]: 0.0}),
        portfolio=TopKPortfolio(),
    )

    target = strategy.generate_target(
        signal_date=context.signal_date,
        market_history=(_view(symbols[0], "4"), _view(symbols[1], "6")),
        account_view=account_view,
        context=context,
    )

    assert dict(target.weights) == {}


@pytest.mark.unit
def test_daily_model_strategy_generates_and_records_dynamic_multi_asset_target() -> None:
    symbols = ("SH.510300", "SH.518880", "SH.588000")
    _, account_view, context = _inputs(symbols)
    strategy = DailyModelStrategy(
        feature_builder=_CloseFeatureBuilder(),
        bundle=_ScoreBundle(
            {
                symbols[0]: 0.01,
                symbols[1]: 0.02,
                symbols[2]: -0.01,
            }
        ),
        portfolio=TopKPortfolio(top_k=2, total_weight="0.90"),
    )

    target = strategy.generate_target(
        signal_date=context.signal_date,
        market_history=(
            _view(symbols[0], "4"),
            _view(symbols[1], "6"),
            _view(symbols[2], "1"),
        ),
        account_view=account_view,
        context=context,
    )

    assert dict(target.weights) == {
        symbols[0]: Decimal("0.30"),
        symbols[1]: Decimal("0.60"),
    }
    assert [
        (row.symbol, row.rank, row.selected, row.target_weight) for row in strategy.allocations
    ] == [
        (symbols[1], 1, True, Decimal("0.60")),
        (symbols[0], 2, True, Decimal("0.30")),
        (symbols[2], 3, False, Decimal("0")),
    ]


@pytest.mark.unit
def test_daily_model_strategy_rejects_invalid_custom_allocation_before_target() -> None:
    symbols, account_view, context = _inputs()

    class UnvalidatedPortfolio:
        max_total_weight = Decimal("0.90")

        def allocate(
            self,
            predictions: Sequence[PredictionRecord],
        ) -> dict[str, Decimal]:
            del predictions
            return {"SH.588000": Decimal("0.90")}

        def resolved_dict(self) -> dict[str, object]:
            return {"type": "test-unvalidated"}

    strategy = DailyModelStrategy(
        feature_builder=_CloseFeatureBuilder(),
        bundle=_ScoreBundle({symbols[0]: 0.01, symbols[1]: 0.02}),
        portfolio=UnvalidatedPortfolio(),
    )

    with pytest.raises(ValueError, match="current prediction"):
        strategy.generate_target(
            signal_date=context.signal_date,
            market_history=(_view(symbols[0], "4"), _view(symbols[1], "6")),
            account_view=account_view,
            context=context,
        )
