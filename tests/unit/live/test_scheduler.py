from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.live.config import load_live_config
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.scheduler import LiveScheduler, TradingDaySource

ROOT = Path(__file__).parents[3]


class _Jobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []
        self.completed: set[tuple[str, date]] = set()
        self.rebalance_activity = False

    def has_job_completed(self, name: str, trade_date: date) -> bool:
        return (name, trade_date) in self.completed

    def has_rebalance_activity(self, trade_date: date) -> bool:
        del trade_date
        return self.rebalance_activity

    def record_job_skipped(self, name: str, trade_date: date, reason: str) -> None:
        self.calls.append((f"{name}:{reason}", trade_date))

    def __getattr__(self, name: str) -> Any:
        def call(trade_date: date, **kwargs: object) -> None:
            del kwargs
            self.calls.append((name, trade_date))

        return call


class _Calendar:
    def __init__(self, open_day: bool = True) -> None:
        self.open_day = open_day

    def is_trading_day(self, trade_date: date) -> bool:
        del trade_date
        return self.open_day


def _scheduler(monkeypatch: pytest.MonkeyPatch, jobs: _Jobs) -> LiveScheduler:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    config = load_live_config(ROOT / "qmt_example/configs/live/beginner_example_paper.yaml")
    return LiveScheduler(
        jobs=cast(LiveDailyJobs, jobs),
        calendar=cast(TradingDaySource, _Calendar()),
        config=config,
    )


def test_scheduler_has_only_three_configuration_driven_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = _Jobs()
    scheduler = _scheduler(monkeypatch, jobs)
    trade_date = date(2026, 8, 19)
    for hour, minute in ((14, 50), (14, 50), (15, 10), (16, 30)):
        scheduler.tick(datetime(2026, 8, 19, hour, minute, 30, tzinfo=MARKET_TIMEZONE))

    assert [name for name, _ in jobs.calls] == [
        "execute_pending_target",
        "eod",
        "prepare_signal",
    ]
    assert all(day == trade_date for _, day in jobs.calls)


def test_scheduler_catches_up_recovery_but_never_opens_a_late_order_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = _Jobs()
    scheduler = _scheduler(monkeypatch, jobs)
    scheduler.tick(datetime(2026, 8, 19, 15, 0, tzinfo=MARKET_TIMEZONE))
    assert jobs.calls == [("rebalance:MISSED_ORDER_WINDOW", date(2026, 8, 19))]

    recovery_jobs = _Jobs()
    recovery_jobs.rebalance_activity = True
    recovery = _scheduler(monkeypatch, recovery_jobs)
    recovery.tick(datetime(2026, 8, 19, 15, 0, tzinfo=MARKET_TIMEZONE))
    assert recovery_jobs.calls == [("execute_pending_target", date(2026, 8, 19))]


def test_scheduler_skips_closed_days_and_persisted_terminal_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = _Jobs()
    scheduler = _scheduler(monkeypatch, jobs)
    scheduler.calendar = cast(TradingDaySource, _Calendar(False))
    scheduler.tick(datetime(2026, 8, 19, 16, 30, tzinfo=MARKET_TIMEZONE))
    assert not jobs.calls

    jobs.completed.add(("rebalance", date(2026, 8, 19)))
    scheduler.calendar = cast(TradingDaySource, _Calendar())
    scheduler.tick(datetime(2026, 8, 19, 15, 0, tzinfo=MARKET_TIMEZONE))
    assert not jobs.calls
