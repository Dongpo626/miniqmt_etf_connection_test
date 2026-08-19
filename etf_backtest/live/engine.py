"""Lifecycle-only owner of the account lock, broker and scheduler."""

from __future__ import annotations

from datetime import date
from typing import Protocol

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
    ) -> None:
        self.config = config
        self.state_engine = state_engine
        self.repository = repository
        self.broker = broker
        self.quote_provider = quote_provider
        self.jobs = jobs
        self.scheduler = scheduler
        self.event_consumer = event_consumer
        self._connection: Connection | None = None
        self._lock_acquired = False
        self._broker_connected = False
        self._scheduler_started = False
        self._consumer_started = False

    def start(self, trade_date: date) -> None:
        if self._connection is not None:
            raise RuntimeError("live engine is already started")
        account_id = self.config.deployment.account_id()
        self._connection = self.state_engine.connect()
        try:
            if not acquire_account_lock(self._connection, account_id):
                raise RuntimeError("account lock is already held")
            self._lock_acquired = True
            if self.event_consumer is not None:
                self.event_consumer.start()
                self._consumer_started = True
            self.broker.connect()
            self._broker_connected = True
            self.broker.subscribe_account(account_id)
            deployment = self.jobs.startup_reconcile(
                trade_date, lock_connection=self._connection
            )
            status = deployment["status"]
            if status not in {DeploymentStatus.ACTIVE, DeploymentStatus.ACTIVE.value}:
                raise RuntimeError("startup did not produce an ACTIVE deployment")
            import json

            raw_symbols = json.loads(str(deployment["universe_json"]))
            if not isinstance(raw_symbols, list):
                raise ValueError("deployment universe_json is invalid")
            self.quote_provider.subscribe(tuple(str(symbol) for symbol in raw_symbols))
            self.scheduler.start()
            self._scheduler_started = True
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        connection = self._connection
        try:
            if self._scheduler_started:
                self.scheduler.stop()
                self._scheduler_started = False
        finally:
            try:
                if self._broker_connected:
                    self.broker.disconnect()
                    self._broker_connected = False
            finally:
                try:
                    if self._consumer_started and self.event_consumer is not None:
                        self.event_consumer.stop()
                        self._consumer_started = False
                finally:
                    try:
                        if connection is not None and self._lock_acquired:
                            release_account_lock(
                                connection, self.config.deployment.account_id()
                            )
                            self._lock_acquired = False
                    finally:
                        if connection is not None:
                            connection.close()
                            self._connection = None


__all__ = ["LiveTradingEngine"]
