from datetime import date, datetime
from typing import Any, cast

from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.scheduler import LiveScheduler, TradingDaySource


class _Jobs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, date]] = []

    def __getattr__(self, name: str) -> Any:
        def call(trade_date: date, **kwargs: object) -> None:
            self.calls.append((name, trade_date))

        return call


class _Calendar:
    def __init__(self, open_day: bool = True) -> None:
        self.open_day = open_day

    def is_trading_day(self, trade_date: date) -> bool:
        return self.open_day


def test_scheduler_has_only_five_once_daily_triggers() -> None:
    jobs = _Jobs()
    scheduler = LiveScheduler(
        jobs=cast(LiveDailyJobs, jobs), calendar=cast(TradingDaySource, _Calendar())
    )
    trade_date = date(2026, 8, 19)
    for hour, minute in ((14, 50), (14, 50), (14, 57), (15, 5), (15, 10), (16, 30)):
        scheduler.tick(datetime(2026, 8, 19, hour, minute, 30).astimezone())

    assert [name for name, _ in jobs.calls] == [
        "execute_pending_target",
        "cancel_open_orders",
        "reconcile_eod",
        "snapshot_eod",
        "prepare_signal",
    ]
    assert all(day == trade_date for _, day in jobs.calls)


def test_scheduler_does_not_trigger_closed_day_or_catch_up() -> None:
    closed_jobs = _Jobs()
    closed = LiveScheduler(
        jobs=cast(LiveDailyJobs, closed_jobs),
        calendar=cast(TradingDaySource, _Calendar(False)),
    )
    closed.tick(datetime(2026, 8, 19, 14, 50).astimezone())
    assert not closed_jobs.calls

    late_jobs = _Jobs()
    late = LiveScheduler(
        jobs=cast(LiveDailyJobs, late_jobs), calendar=cast(TradingDaySource, _Calendar())
    )
    late.tick(datetime(2026, 8, 19, 15, 0).astimezone())
    assert not late_jobs.calls
    assert [name for _, name in late.missed_jobs] == [
        "execute_pending_target",
        "cancel_open_orders",
    ]
