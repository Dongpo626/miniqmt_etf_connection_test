"""Main-thread lifecycle owner for account lock, broker recovery and scheduling."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date, datetime
from threading import Event
from typing import Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.engine import Connection, Engine

from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_account_lock,
    release_account_lock,
)
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.live.state import DeploymentStatus

SHANGHAI = ZoneInfo("Asia/Shanghai")
LOGGER = logging.getLogger(__name__)


class EventConsumer(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...


class LiveTradingEngine:
    def __init__(
        self,
        *,
        config: LiveConfig,
        state_engine: Engine,
        repository: LiveStateRepository,
        broker: BrokerGateway,
        quote_provider: QuoteProvider,
        jobs: LiveDailyJobs,
        scheduler: LiveScheduler,
        event_consumer: EventConsumer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.state_engine = state_engine
        self.repository = repository
        self.broker = broker
        self.quote_provider = quote_provider
        self.jobs = jobs
        self.scheduler = scheduler
        self.event_consumer = event_consumer
        self.clock = clock or (lambda: datetime.now(SHANGHAI))
        self._connection: Connection | None = None
        self._lock_acquired = False
        self._broker_connected = False
        self._scheduler_started = False
        self._consumer_started = False
        self._broker_unhealthy = Event()
        self._shutdown = Event()

    def set_event_consumer(self, event_consumer: EventConsumer) -> None:
        if self._connection is not None:
            raise RuntimeError("cannot replace the event consumer while engine is running")
        self.event_consumer = event_consumer

    def notify_broker_unhealthy(self) -> None:
        """Thread-safe callback target; recovery itself stays on the engine thread."""

        self._broker_unhealthy.set()

    def start(self, trade_date: date) -> None:
        if self._connection is not None:
            raise RuntimeError("live engine is already started")
        self._shutdown.clear()
        self._broker_unhealthy.clear()
        account_id = self.config.deployment.account_id()
        self._connection = self.state_engine.connect()
        try:
            if not acquire_account_lock(self._connection, account_id):
                raise RuntimeError("account lock is already held")
            self._lock_acquired = True
            if self.event_consumer is not None:
                self.event_consumer.start()
                self._consumer_started = True
            self._connect_reconcile_and_schedule(trade_date)
        except Exception:
            self.stop()
            raise

    def run_forever(self) -> None:
        """Own scheduler ticks and all reconnect work on this calling thread."""

        self.start(self.clock().astimezone(SHANGHAI).date())
        try:
            while not self._shutdown.wait(1.0):
                if self._broker_unhealthy.is_set():
                    self._recover_broker()
                elif self._scheduler_started:
                    self.scheduler.tick()
        finally:
            self.stop()

    def stop(self) -> None:
        self._shutdown.set()
        connection = self._connection
        try:
            self._stop_scheduler()
        finally:
            try:
                self._disconnect_broker()
            finally:
                try:
                    if self._consumer_started and self.event_consumer is not None:
                        self.event_consumer.stop()
                        self._consumer_started = False
                finally:
                    try:
                        if connection is not None and self._lock_acquired:
                            release_account_lock(connection, self.config.deployment.account_id())
                            self._lock_acquired = False
                    finally:
                        if connection is not None:
                            connection.close()
                            self._connection = None

    def _connect_reconcile_and_schedule(self, trade_date: date) -> None:
        assert self._connection is not None
        account_id = self.config.deployment.account_id()
        self.broker.connect()
        self._broker_connected = True
        self.broker.subscribe_account(account_id)
        deployment = self.jobs.startup_reconcile(trade_date, lock_connection=self._connection)
        status = deployment["status"]
        if status not in {DeploymentStatus.ACTIVE, DeploymentStatus.ACTIVE.value}:
            raise RuntimeError("startup did not produce an ACTIVE deployment")
        raw_symbols = json.loads(str(deployment["universe_json"]))
        if not isinstance(raw_symbols, list):
            raise ValueError("deployment universe_json is invalid")
        self.quote_provider.subscribe(tuple(str(symbol) for symbol in raw_symbols))
        self.scheduler.start()
        self._scheduler_started = True

    def _recover_broker(self) -> None:
        self._stop_scheduler()
        self._disconnect_broker()
        interval = float(self.config.miniqmt.reconnect_interval_seconds)
        while not self._shutdown.is_set():
            self._broker_unhealthy.clear()
            try:
                self._connect_reconcile_and_schedule(self.clock().astimezone(SHANGHAI).date())
                return
            except Exception:
                LOGGER.exception("MiniQMT recovery attempt failed")
                self._stop_scheduler()
                self._disconnect_broker()
                if self._shutdown.wait(interval):
                    return

    def _stop_scheduler(self) -> None:
        if self._scheduler_started:
            self.scheduler.stop()
            self._scheduler_started = False

    def _disconnect_broker(self) -> None:
        if self._broker_connected:
            try:
                self.broker.disconnect()
            finally:
                self._broker_connected = False


__all__ = ["LiveTradingEngine"]
