from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from etf_backtest.config.schema import (
    BacktestConfig,
    DatabaseConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
    UniverseConfig,
    etf_code,
    normalize_symbol,
)


def _config(**overrides: object) -> BacktestConfig:
    payload: dict[str, object] = {
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_cash": "1000000",
        "database": {"user": "reader", "password": "secret"},
        "universe": {"symbols": ["510300", "SZ.159915"], "pools": ["gold_etf"]},
        "strategy": {"kind": "rule"},
    }
    payload.update(overrides)
    return BacktestConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [("510300", "SH.510300"), ("510300.SH", "SH.510300"), ("sz.159915", "SZ.159915")],
)
def test_symbol_mapping_is_bidirectional(supplied: str, expected: str) -> None:
    assert normalize_symbol(supplied) == expected
    assert etf_code(expected) == expected[-6:]


def test_symbol_mapping_rejects_conflicting_or_non_etf_codes() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        normalize_symbol("SZ.510300")
    with pytest.raises(ValueError, match="start with"):
        normalize_symbol("000001")


def test_universe_requires_a_source_and_deduplicates_after_normalization() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        UniverseConfig()
    with pytest.raises(ValidationError, match="unique"):
        UniverseConfig(symbols=("510300", "SH.510300"))


def test_rule_config_exposes_daily_strategy_controls() -> None:
    config = _config()
    assert isinstance(config.strategy, RuleStrategyConfig)
    assert config.strategy.lookback_trading_days == 20
    assert config.strategy.target_weight == Decimal("0.90")
    custom = _config(
        strategy={
            "kind": "rule",
            "lookback_trading_days": 60,
            "rebalance_every_trading_days": 5,
        }
    )
    assert custom.strategy.lookback_trading_days == 60
    assert custom.strategy.rebalance_every_trading_days == 5
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(bar_interval="1h")
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(trade_price_mode="OPEN")
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(strategy={"kind": "rule", "strategy_name": "legacy"})


def test_model_splits_are_configurable_but_must_match_backtest() -> None:
    config = _config(strategy={"kind": "model"})
    assert isinstance(config.strategy, ModelStrategyConfig)
    assert config.strategy.train_start == date(2021, 1, 1)
    assert config.strategy.test_end == date(2024, 12, 31)
    custom = _config(
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy={
            "kind": "model",
            "train_start": "2020-01-01",
            "train_end": "2021-12-31",
            "valid_start": "2022-01-01",
            "valid_end": "2022-12-31",
            "test_start": "2023-01-01",
            "test_end": "2023-12-31",
        },
    )
    assert custom.strategy.train_start == date(2020, 1, 1)
    with pytest.raises(ValidationError, match="must equal the configured test interval"):
        _config(
            start_date="2024-02-01",
            strategy={"kind": "model"},
        )
    for removed_field in ("model_name", "ridge_alpha", "load_bundle"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            _config(strategy={"kind": "model", removed_field: "legacy"})


def test_decimal_configuration_rejects_binary_floats() -> None:
    with pytest.raises(TypeError, match="decimal text"):
        _config(initial_cash=1_000_000.0)
    with pytest.raises(TypeError, match="decimal text"):
        _config(slippage={"rate": 0.0005})


def test_database_secret_is_resolved_but_never_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEMP_QMT_PASSWORD", "env-secret")
    database = DatabaseConfig(user="reader", password_env="TEMP_QMT_PASSWORD")
    assert database.resolved_password() == "env-secret"
    config = _config(database={"user": "reader", "password": "literal-secret"})
    serialized = config.to_resolved_json()
    resolved = json.loads(serialized)
    assert "literal-secret" not in serialized
    assert resolved["database"]["password"] == "***"
    assert resolved["database"]["database"] == "qmt_etf_quant"
    assert resolved["fixed_execution_price"] == "CLOSE"
    assert resolved["pit_compliant"] is False
    assert "price_limit_data" not in resolved


def test_only_the_unified_qmt_database_is_supported() -> None:
    assert DatabaseConfig(user="reader", password="secret").database == "qmt_etf_quant"
    with pytest.raises(ValidationError):
        DatabaseConfig(
            database="qmt_etf_quant_trade_status_export",
            user="reader",
            password="secret",
        )


def test_unified_snapshot_defaults_and_removed_fields_are_strict() -> None:
    snapshot = _config().data_snapshot

    assert snapshot.dataset_version == "qmt_etf_quant_20260812"
    assert snapshot.snapshot_date == date(2026, 8, 12)
    assert snapshot.snapshot_started_at_utc == datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC)
    assert snapshot.manifest_sha256 == (
        "39502872c5544cbf4dc1671ea67488479021e39db97f34aecdedf1bc56cc62a7"
    )
    assert snapshot.trade_status_table == "etf_trade_status_daily"
    assert snapshot.data_mode == "RETROSPECTIVE_SNAPSHOT"

    for removed_field in (
        "snapshot_cutoff_utc",
        "revision_strategy",
        "share_manifest_sha256",
        "index_manifest_sha256",
    ):
        with pytest.raises(ValidationError, match="Extra inputs"):
            _config(data_snapshot={removed_field: "legacy"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        _config(price_limit_data={})


def test_snapshot_started_at_is_timezone_aware_and_normalized_to_utc() -> None:
    supplied = datetime(2026, 8, 12, 16, 48, 57, tzinfo=timezone(timedelta(hours=8)))
    snapshot = _config(data_snapshot={"snapshot_started_at_utc": supplied}).data_snapshot
    assert snapshot.snapshot_started_at_utc == datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC)

    with pytest.raises(ValidationError, match="snapshot_started_at_utc must be timezone-aware"):
        _config(data_snapshot={"snapshot_started_at_utc": "2026-08-12T08:48:57"})


def test_rule_fund_data_sources_are_frozen_in_the_data_snapshot() -> None:
    csv_path = Path("D:/readonly/huijin_combined.csv")
    config = _config(
        data_snapshot={
            "share_table": "etf_share_daily",
            "huijin_holders_csv": str(csv_path),
            "huijin_holders_csv_sha256": "b" * 64,
            "index_table": "index_quote_daily",
            "rule_index_codes": ["000001.SH"],
        }
    )

    assert config.data_snapshot.share_table == "etf_share_daily"
    assert config.data_snapshot.huijin_holders_csv == csv_path
    assert config.data_snapshot.huijin_holders_csv_sha256 == "b" * 64
    assert config.data_snapshot.index_table == "index_quote_daily"
    assert config.data_snapshot.rule_index_codes == ("000001.SH",)
    with pytest.raises(ValidationError, match="together"):
        _config(data_snapshot={"huijin_holders_csv": str(csv_path)})
    with pytest.raises(ValidationError, match="together"):
        _config(data_snapshot={"index_table": "index_quote_daily"})
    with pytest.raises(ValidationError, match="together"):
        _config(data_snapshot={"rule_index_codes": ["000001.SH"]})
