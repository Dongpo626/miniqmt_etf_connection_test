"""Minimal Rule runtime assembly and canonical Universe tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.application import runtime_factory
from etf_backtest.application.runtime_factory import (
    build_rule_runtime,
    canonical_universe_identity,
)
from etf_backtest.application.strategy_source import load_rule_strategy_source
from etf_backtest.config.schema import BacktestConfig, RuleStrategyConfig
from etf_backtest.core.effective_rules import EffectiveDatedEtfRuleResolver
from etf_backtest.core.market import EtfInfo, Exchange
from etf_backtest.data.mysql import QmtDailyDataset, QmtDailyRepository
from etf_backtest.data.portal import DailyDataPortal
from etf_backtest.universe.resolver import FrozenUniverse

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_PATH = _PROJECT_ROOT / "private_strategy" / "beginner_example" / "experiment.yaml"
_SYSTEM_PATH = _PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"


@pytest.mark.unit
def test_canonical_universe_is_stable_across_input_order() -> None:
    forward = canonical_universe_identity(("510300.SH", "SH.518880"))
    reversed_order = canonical_universe_identity(("SH.518880", "SH.510300"))

    assert forward == reversed_order
    assert forward[0] == ("SH.510300", "SH.518880")
    assert forward[1] == '["SH.510300","SH.518880"]'
    assert len(forward[2]) == 64


@pytest.mark.unit
def test_build_rule_runtime_wires_shared_components_and_requested_history_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_rule_strategy_source(_EXPERIMENT_PATH, system_path=_SYSTEM_PATH)
    config = source.experiment.build_case(
        source.system,
        strategy=RuleStrategyConfig(
            lookback_trading_days=source.rule.lookback_trading_days,
            rebalance_every_trading_days=source.rule.rebalance_every_trading_days,
            target_weight=source.rule.target_weight,
        ),
    )
    engine = Mock(spec=Engine)
    connection = Mock(spec=Connection)
    repository = Mock(spec=QmtDailyRepository)
    info = EtfInfo(
        symbol="SH.510300",
        exchange=Exchange.SSE,
        name="ETF",
        primary_category="domestic",
        fund_type="stock",
        list_date=date(2012, 1, 1),
        delist_date=None,
        current_status="active",
    )
    universe = Mock(spec=FrozenUniverse)
    universe.symbols = ("SH.510300",)
    universe.etf_infos = (info,)
    dataset = Mock(spec=QmtDailyDataset)
    repository.load_daily_dataset.return_value = dataset
    portal = Mock(spec=DailyDataPortal)
    rules = Mock(spec=EffectiveDatedEtfRuleResolver)
    repository_calls: list[tuple[BacktestConfig, Engine, Connection | None]] = []

    def fake_create_repository(
        config_arg: BacktestConfig,
        engine_arg: Engine,
        *,
        connection: Connection | None = None,
    ) -> QmtDailyRepository:
        repository_calls.append((config_arg, engine_arg, connection))
        return cast(QmtDailyRepository, repository)

    monkeypatch.setattr(runtime_factory, "create_repository", fake_create_repository)
    monkeypatch.setattr(runtime_factory, "resolve_universe", lambda *_args: universe)
    monkeypatch.setattr(runtime_factory, "DailyDataPortal", lambda _dataset: portal)
    monkeypatch.setattr(
        runtime_factory,
        "load_effective_rule_resolver",
        lambda *_args: rules,
    )
    load_start = date(2023, 10, 1)
    load_end = date(2024, 6, 28)

    runtime = build_rule_runtime(
        source=source,
        config=config,
        project_root=_PROJECT_ROOT,
        engine=cast(Engine, engine),
        connection=cast(Connection, connection),
        load_start=load_start,
        load_end=load_end,
    )

    assert repository_calls == [(config, engine, connection)]
    repository.load_daily_dataset.assert_called_once_with(
        ("SH.510300",),
        load_start,
        load_end,
        etf_infos=(info,),
    )
    assert runtime.source is source
    assert runtime.repository is repository
    assert runtime.universe is universe
    assert runtime.dataset is dataset
    assert runtime.portal is portal
    assert runtime.rule_resolver is rules
    assert runtime.universe_json == '["SH.510300"]'
    assert len(runtime.universe_sha256) == 64
