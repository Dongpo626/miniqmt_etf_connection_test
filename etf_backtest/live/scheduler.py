"""Five fixed Asia/Shanghai live-job triggers with no retry or catch-up."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from typing import Protocol
from zoneinfo import ZoneInfo

from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.state import JobTriggerSource

SHANGHAI = ZoneInfo("Asia/Shanghai")


class TradingDaySource(Protocol):
    def is_trading_day(self, trade_date: date) -> bool: ...


class LiveScheduler:
    def __init__(
        self,
        *,
        jobs: LiveDailyJobs,
        calendar: TradingDaySource,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time as time_module

        self.jobs = jobs
        self.calendar = calendar
        self.clock = clock or (lambda: datetime.now(SHANGHAI))
        self.sleep = sleep or time_module.sleep
        self._running = False
        self._handled: set[tuple[date, str]] = set()
        self._missed: list[tuple[date, str]] = []

    @property
    def missed_jobs(self) -> tuple[tuple[date, str], ...]:
        return tuple(self._missed)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def tick(self, now: datetime | None = None) -> None:
        current = (now or self.clock()).astimezone(SHANGHAI)
        trade_date = current.date()
        if not self.calendar.is_trading_day(trade_date):
            return
        for job_time, name, call in self._schedule(trade_date):
            key = (trade_date, name)
            if key in self._handled:
                continue
            current_minute = (current.hour, current.minute)
            scheduled_minute = (job_time.hour, job_time.minute)
            if current_minute == scheduled_minute:
                self._handled.add(key)
                call()
            elif current_minute > scheduled_minute:
                self._handled.add(key)
                self._missed.append(key)

    def run_forever(self) -> None:
        self.start()
        while self._running:
            self.tick()
            self.sleep(1.0)

    def _schedule(
        self, trade_date: date
    ) -> tuple[tuple[time, str, Callable[[], object]], ...]:
        scheduled = JobTriggerSource.SCHEDULED
        return (
            (
                time(14, 50),
                "execute_pending_target",
                lambda: self.jobs.execute_pending_target(
                    trade_date, trigger_source=scheduled
                ),
            ),
            (
                time(14, 57),
                "cancel_open_orders",
                lambda: self.jobs.cancel_open_orders(trade_date, trigger_source=scheduled),
            ),
            (
                time(15, 5),
                "reconcile_eod",
                lambda: self.jobs.reconcile_eod(trade_date, trigger_source=scheduled),
            ),
            (
                time(15, 10),
                "snapshot_eod",
                lambda: self.jobs.snapshot_eod(trade_date, trigger_source=scheduled),
            ),
            (
                time(16, 30),
                "prepare_signal",
                lambda: self.jobs.prepare_signal(trade_date, trigger_source=scheduled),
            ),
        )


__all__ = ["LiveScheduler"]
