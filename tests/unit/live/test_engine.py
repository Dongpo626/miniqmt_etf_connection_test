from datetime import date
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.engine import LiveTradingEngine
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.live.state import DeploymentStatus

ROOT = Path(__file__).parents[3]


def _engine(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LiveTradingEngine, Mock, Mock, Mock, Mock, Mock]:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    config = load_live_config(ROOT / "qmt_example/configs/live/beginner_example_paper.yaml")
    connection = Mock(spec=Connection)
    state_engine = Mock(spec=Engine)
    state_engine.connect.return_value = connection
    repository = Mock(spec=LiveStateRepository)
    broker = Mock(spec=BrokerGateway)
    quotes = Mock(spec=QuoteProvider)
    jobs = Mock(spec=LiveDailyJobs)
    jobs.startup_reconcile.return_value = {
        "status": DeploymentStatus.ACTIVE,
        "universe_json": '["SH.510300"]',
    }
    scheduler = Mock(spec=LiveScheduler)
    engine = LiveTradingEngine(
        config=config,
        state_engine=cast(Engine, state_engine),
        repository=cast(LiveStateRepository, repository),
        broker=cast(BrokerGateway, broker),
        quote_provider=cast(QuoteProvider, quotes),
        jobs=cast(LiveDailyJobs, jobs),
        scheduler=cast(LiveScheduler, scheduler),
    )
    return engine, connection, broker, quotes, jobs, scheduler


def _record[ValueT](events: list[str], label: str, result: ValueT) -> ValueT:
    events.append(label)
    return result


def test_account_lock_failure_does_not_connect_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, connection, broker, _, _, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: False)
    with pytest.raises(RuntimeError, match="account lock"):
        engine.start(date(2026, 8, 19))
    broker.connect.assert_not_called()
    scheduler.start.assert_not_called()
    connection.close.assert_called_once()


def test_successful_lifecycle_order_and_startup_failure_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    engine, connection, broker, quotes, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr(
        "etf_backtest.live.engine.acquire_account_lock",
        lambda *args: _record(events, "lock", True),
    )
    monkeypatch.setattr(
        "etf_backtest.live.engine.release_account_lock",
        lambda *args: events.append("unlock"),
    )
    broker.connect.side_effect = lambda: events.append("connect")
    broker.subscribe_account.side_effect = lambda account: events.append("subscribe_account")
    jobs.startup_reconcile.side_effect = lambda *args, **kwargs: _record(
        events,
        "startup",
        {
            "status": DeploymentStatus.ACTIVE,
            "universe_json": '["SH.510300"]',
        },
    )
    quotes.subscribe.side_effect = lambda symbols: events.append("subscribe_quotes")
    scheduler.start.side_effect = lambda: events.append("scheduler_start")
    scheduler.stop.side_effect = lambda: events.append("scheduler_stop")
    broker.disconnect.side_effect = lambda: events.append("disconnect")

    engine.start(date(2026, 8, 19))
    engine.stop()
    assert events == [
        "lock",
        "connect",
        "subscribe_account",
        "startup",
        "subscribe_quotes",
        "scheduler_start",
        "scheduler_stop",
        "disconnect",
        "unlock",
    ]
    connection.close.assert_called_once()

    failed, connection2, broker2, _, jobs2, scheduler2 = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)
    jobs2.startup_reconcile.side_effect = RuntimeError("unsafe startup")
    with pytest.raises(RuntimeError, match="unsafe startup"):
        failed.start(date(2026, 8, 19))
    scheduler2.start.assert_not_called()
    broker2.disconnect.assert_called_once()
    connection2.close.assert_called_once()


def test_callback_only_sets_event_and_main_loop_performs_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, broker, quotes, jobs, scheduler = _engine(monkeypatch)
    monkeypatch.setattr("etf_backtest.live.engine.acquire_account_lock", lambda *args: True)
    monkeypatch.setattr("etf_backtest.live.engine.release_account_lock", lambda *args: None)

    engine.start(date(2026, 8, 19))
    engine.notify_broker_unhealthy()
    scheduler.stop.assert_not_called()
    broker.disconnect.assert_not_called()

    engine._recover_broker()

    scheduler.stop.assert_called_once()
    assert scheduler.start.call_count == 2
    assert broker.connect.call_count == 2
    assert broker.disconnect.call_count == 1
    assert jobs.startup_reconcile.call_count == 2
    assert quotes.subscribe.call_count == 2
    engine.stop()
