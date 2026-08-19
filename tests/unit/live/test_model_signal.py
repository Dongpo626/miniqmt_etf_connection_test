from __future__ import annotations

from contextlib import nullcontext
from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

import etf_backtest.live.jobs as jobs_module
from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.account_adapter import AdaptedAccountState
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import ModelSignalService
from etf_backtest.strategy.context import AccountView


@pytest.mark.unit
def test_model_signal_uses_one_transaction_same_decision_service_and_data_through_d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_date = date(2026, 8, 18)
    connection = Mock(spec=Connection)
    connection.begin.return_value = nullcontext()
    engine = Mock(spec=Engine)
    engine.connect.return_value = nullcontext(connection)
    repository = Mock()
    repository.load_sse_calendar.return_value = (object(),)
    runtime = SimpleNamespace(repository=repository, portal=Mock())
    build_runtime = Mock(return_value=runtime)
    monkeypatch.setattr(jobs_module, "build_model_signal_runtime", build_runtime)
    monkeypatch.setattr(jobs_module, "SseTradingCalendar", lambda rows: Mock())
    monkeypatch.setattr(jobs_module, "_model_backtest_config", lambda source: Mock())

    expected = DailyDecisionResult(
        signal_date=signal_date,
        execution_date=date(2026, 8, 19),
        schedule_index=8,
        status=DecisionStatus.TARGET_CREATED,
        target_portfolio=TargetPortfolio({"SH.510300": Decimal("0.9")}),
    )
    decision_service = Mock()
    decision_service.evaluate.return_value = expected
    resolver = Mock()
    resolver.next_trading_day.return_value = date(2026, 8, 19)
    resolver.resolve.return_value = 8
    strategy = Mock()
    strategy.required_history_trading_days = 21
    live_config = SimpleNamespace(
        deployment=SimpleNamespace(schedule_anchor_date=date(2026, 1, 1))
    )
    service = ModelSignalService(
        live_config=cast(LiveConfig, live_config),
        source=Mock(),
        strategy=strategy,
        strategy_engine=cast(Engine, engine),
        decision_service=decision_service,
        schedule_resolver=resolver,
    )
    account = SimpleNamespace(
        account_view=AccountView(cash=Decimal("1000000"), positions={}),
        current_weights_by_symbol={"SH.510300": Decimal("0")},
    )

    result = service.evaluate(
        signal_date=signal_date,
        symbols=("SH.510300",),
        account=cast(AdaptedAccountState, account),
    )

    assert result is expected
    call = build_runtime.call_args.kwargs
    assert call["connection"] is connection
    assert call["frozen_symbols"] == ("SH.510300",)
    assert call["load_end"] == signal_date
    decision_service.evaluate.assert_called_once()
    assert decision_service.evaluate.call_args.kwargs["strategy"] is strategy
    assert decision_service.evaluate.call_args.kwargs["portal"] is runtime.portal
    repository.load_sse_calendar.assert_called_once()
    engine.connect.assert_called_once_with()
