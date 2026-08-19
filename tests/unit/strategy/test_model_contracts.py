"""Framework-neutral model contract and daily alignment tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    MODEL_BUNDLE_SCHEMA_VERSION,
    DateRange,
    ModelDataIdentity,
    ModelMetadata,
    canonical_json,
    feature_fingerprint,
)
from etf_backtest.strategy.model_training import DailyTorchDatasetBuilder


class _CloseFeatureBuilder:
    feature_names = ("close",)
    required_history_trading_days = 1

    def __init__(self) -> None:
        self.visible_dates: list[tuple[date, tuple[date, ...]]] = []

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> tuple[Decimal, ...]:
        del symbol
        self.visible_dates.append((signal_date, tuple(view.trade_date for view in history)))
        return (history[-1].close,)


def _views(count: int = 11) -> tuple[MarketBarView, ...]:
    start = date(2024, 1, 1)
    rows: list[MarketBarView] = []
    for index in range(count):
        trade_date = start + timedelta(days=index)
        close = Decimal("10") + Decimal(index)
        rows.append(
            MarketBarView(
                source_record_key=f"QMT:510300:{trade_date.isoformat()}:1:1",
                symbol="SH.510300",
                trade_date=trade_date,
                open=close,
                high=close,
                low=close,
                close=close,
                volume=1000 + index,
                suspended=False,
            )
        )
    return tuple(rows)


@pytest.mark.unit
def test_daily_dataset_owns_d_alignment_and_d1_to_d2_label() -> None:
    views = _views()
    builder = _CloseFeatureBuilder()
    dataset = DailyTorchDatasetBuilder(builder).build(
        market_views=views,
        train_range=DateRange(date(2024, 1, 1), date(2024, 1, 3)),
        valid_range=DateRange(date(2024, 1, 4), date(2024, 1, 6)),
        test_range=DateRange(date(2024, 1, 7), date(2024, 1, 9)),
    )

    first = dataset.train[0]
    assert first.key.signal_date == date(2024, 1, 1)
    assert first.features == (Decimal("10"),)
    assert first.label == Decimal("12") / Decimal("11") - Decimal("1")
    assert dataset.label_name == DAILY_FORWARD_RETURN_LABEL
    assert all(max(visible) <= signal for signal, visible in builder.visible_dates)
    assert {row.key.signal_date for row in dataset.test} == {
        date(2024, 1, 7),
        date(2024, 1, 8),
        date(2024, 1, 9),
    }


@pytest.mark.unit
def test_model_metadata_round_trip_preserves_compatibility_identity() -> None:
    feature_names = ("close",)
    identity = ModelDataIdentity(
        dataset_version="snapshot-v1",
        manifest_sha256="a" * 64,
        snapshot_started_at_utc=datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC),
    )
    metadata = ModelMetadata(
        schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
        model_id="tiny_mlp",
        model_class_name="TinyMlp",
        model_parameters_json=canonical_json({"hidden": 4}),
        training_parameters_json=canonical_json({"seed": 7}),
        feature_names=feature_names,
        feature_fingerprint=feature_fingerprint(feature_names, DAILY_FORWARD_RETURN_LABEL),
        label_name=DAILY_FORWARD_RETURN_LABEL,
        train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
        valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
        test_range=DateRange(date(2024, 1, 1), date(2024, 12, 31)),
        random_seed=7,
        data_identity=identity,
        trained_through=date(2022, 12, 30),
        framework_name="torch",
        framework_version="test",
    )

    assert ModelMetadata.from_mapping(metadata.to_dict()) == metadata
