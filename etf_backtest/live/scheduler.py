"""Configuration-driven scheduler for the three daily PAPER business jobs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Protocol
from zoneinfo import ZoneInfo

from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import JobAlreadySucceeded, JobSkipped, LiveDailyJobs
from etf_backtest.live.state import JobTriggerSource

SHANGHAI = ZoneInfo("Asia/Shanghai")
LOGGER = logging.getLogger(__name__)


class TradingDaySource(Protocol):
    def is_trading_day(self, trade_date: date) -> bool: ...


class LiveScheduler:
    """Run each business job once per process and deduplicate it in MySQL."""

    def __init__(
        self,
        *,
        jobs: LiveDailyJobs,
        calendar: TradingDaySource,
        config: LiveConfig,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time as time_module

        self.jobs = jobs
        self.calendar = calendar
        self.config = config
        self.clock = clock or (lambda: datetime.now(SHANGHAI))
        self.sleep = sleep or time_module.sleep
        self._running = False
        self._handled: set[tuple[date, str]] = set()
        self._missed: list[tuple[date, str]] = []

    @property
    def running(self) -> bool:
        return self._running

    @property
    def missed_jobs(self) -> tuple[tuple[date, str], ...]:
        return tuple(self._missed)

    def start(self) -> None:
        # A reconnect intentionally re-evaluates today's jobs. Persisted job rows,
        # rather than this process-local cache, are the authoritative duplicate guard.
        self._handled.clear()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def tick(self, now: datetime | None = None) -> None:
        current = (now or self.clock()).astimezone(SHANGHAI)
        trade_date = current.date()
        local_time = current.time().replace(tzinfo=None)
        if not self.calendar.is_trading_day(trade_date):
            return
        for job_time, name, call in self._schedule(trade_date):
            key = (trade_date, name)
            if key in self._handled or local_time < job_time:
                continue
            self._handled.add(key)
            try:
                if self.jobs.has_job_completed(name, trade_date):
                    continue
                if (
                    name == "rebalance"
                    and local_time >= self.config.execution.stop_new_orders
                    and not self.jobs.has_rebalance_activity(trade_date)
                ):
                    self._missed.append(key)
                    self.jobs.record_job_skipped("rebalance", trade_date, "MISSED_ORDER_WINDOW")
                    continue
                call()
            except (JobAlreadySucceeded, JobSkipped):
                # Both outcomes are durable and normal during catch-up.
                continue
            except Exception:
                # Avoid retrying every second. Reconnect/restart re-enters through state.
                LOGGER.exception("scheduled live job failed: %s", name)

    def run_forever(self) -> None:
        self.start()
        while self._running:
            self.tick()
            self.sleep(1.0)

    def _schedule(self, trade_date: date) -> tuple[tuple[time, str, Callable[[], object]], ...]:
        scheduled = JobTriggerSource.SCHEDULED
        return (
            (
                self.config.execution.submit_start,
                "rebalance",
                lambda: self.jobs.execute_pending_target(trade_date, trigger_source=scheduled),
            ),
            (
                self.config.eod.run_time,
                "eod",
                lambda: self.jobs.eod(trade_date, trigger_source=scheduled),
            ),
            (
                self.config.signal.run_time,
                "prepare_signal",
                lambda: self.jobs.prepare_signal(trade_date, trigger_source=scheduled),
            ),
        )


__all__ = ["LiveScheduler", "TradingDaySource"]
