"""Minimal xtquant callback-to-queue bridge and single worker consumer."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from queue import Empty, Queue
from threading import Event, Thread

from etf_backtest.live.broker.mapper import map_order, map_trade
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import BrokerOrderSnapshot, BrokerTradeSnapshot

LOGGER = logging.getLogger(__name__)


class BrokerEventType(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    ACCOUNT_STATUS = "ACCOUNT_STATUS"
    ORDER = "ORDER"
    TRADE = "TRADE"
    ORDER_ERROR = "ORDER_ERROR"
    CANCEL_ERROR = "CANCEL_ERROR"


@dataclass(frozen=True, slots=True)
class BrokerEvent:
    event_type: BrokerEventType
    payload: object | None = None
    account_id: str | None = None
    status: int | None = None
    error: str | None = None


def enqueue_disconnected(events: Queue[BrokerEvent]) -> None:
    events.put(BrokerEvent(BrokerEventType.DISCONNECTED))


def enqueue_account_status(events: Queue[BrokerEvent], status: object) -> None:
    events.put(
        BrokerEvent(
            BrokerEventType.ACCOUNT_STATUS,
            account_id=str(getattr(status, "account_id", "")) or None,
            status=int(getattr(status, "status", -1)),
        )
    )


def enqueue_order(events: Queue[BrokerEvent], order: BrokerOrderSnapshot) -> None:
    events.put(BrokerEvent(BrokerEventType.ORDER, payload=order, account_id=order.account_id))


def enqueue_trade(events: Queue[BrokerEvent], trade: BrokerTradeSnapshot) -> None:
    events.put(BrokerEvent(BrokerEventType.TRADE, payload=trade, account_id=trade.account_id))


def enqueue_error(
    events: Queue[BrokerEvent], event_type: BrokerEventType, error: object
) -> None:
    events.put(
        BrokerEvent(
            event_type,
            account_id=str(getattr(error, "account_id", "")) or None,
            error=(
                f"order_id={getattr(error, 'order_id', '')} "
                f"error_id={getattr(error, 'error_id', '')} "
                f"error_msg={getattr(error, 'error_msg', '')}"
            ).strip(),
        )
    )


def create_xtquant_callback(events: Queue[BrokerEvent]) -> object:
    """Import the SDK only when a production callback is actually constructed."""

    try:
        module = import_module("xtquant.xttrader")
        constants = import_module("xtquant.xtconstant")
    except ImportError as error:
        raise RuntimeError("当前环境未安装 xtquant, 无法创建交易运行时。") from error

    class QueueingCallback(module.XtQuantTraderCallback):  # type: ignore[name-defined,misc]
        def on_disconnected(self) -> None:
            enqueue_disconnected(events)

        def on_account_status(self, status: object) -> None:
            enqueue_account_status(events, status)

        def on_stock_order(self, order: object) -> None:
            enqueue_order(events, map_order(order, constants=constants))

        def on_stock_trade(self, trade: object) -> None:
            enqueue_trade(events, map_trade(trade, constants=constants))

        def on_order_error(self, error: object) -> None:
            enqueue_error(events, BrokerEventType.ORDER_ERROR, error)

        def on_cancel_error(self, error: object) -> None:
            enqueue_error(events, BrokerEventType.CANCEL_ERROR, error)

    return QueueingCallback()


class BrokerEventConsumer:
    """Persist individual callbacks idempotently; active queries remain authoritative."""

    def __init__(
        self,
        *,
        events: Queue[BrokerEvent],
        repository: LiveStateRepository,
        account_id: str,
        on_unhealthy: Callable[[], None],
    ) -> None:
        self._events = events
        self._repository = repository
        self._account_id = account_id
        self._on_unhealthy = on_unhealthy
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="qmt-broker-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def process(self, event: BrokerEvent) -> None:
        if event.event_type is BrokerEventType.DISCONNECTED:
            self._on_unhealthy()
            return
        if event.event_type is BrokerEventType.ACCOUNT_STATUS:
            if event.account_id not in {None, self._account_id} or event.status != 0:
                self._on_unhealthy()
            return
        if event.event_type in {BrokerEventType.ORDER_ERROR, BrokerEventType.CANCEL_ERROR}:
            LOGGER.error("MiniQMT %s: %s", event.event_type, event.error)
            return
        if event.event_type is BrokerEventType.ORDER:
            self._persist_order(event.payload)
        elif event.event_type is BrokerEventType.TRADE:
            self._persist_trade(event.payload)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._events.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.process(event)
            except Exception:
                LOGGER.exception("failed to consume MiniQMT callback event")

    def _persist_order(self, payload: object | None) -> None:
        if not isinstance(payload, BrokerOrderSnapshot):
            raise TypeError("ORDER event requires BrokerOrderSnapshot")
        account_id = payload.account_id or self._account_id
        with self._repository.transaction() as connection:
            saved = self._repository.get_broker_order(
                account_id, payload.broker_order_id, connection=connection
            )
            if saved is not None:
                intent_id = str(saved["intent_id"])
                remark = str(saved["remark_token"])
            elif payload.remark_token:
                intent = self._repository.get_intent_by_remark_token(
                    payload.remark_token, account_id=account_id, connection=connection
                )
                if intent is None:
                    LOGGER.warning("unknown callback order: %s", payload.broker_order_id)
                    return
                intent_id = str(intent["intent_id"])
                remark = payload.remark_token
            else:
                LOGGER.warning("callback order has no local identity: %s", payload.broker_order_id)
                return
            self._repository.bind_broker_order(
                account_id=account_id,
                intent_id=intent_id,
                remark_token=remark,
                order=payload,
                connection=connection,
            )

    def _persist_trade(self, payload: object | None) -> None:
        if not isinstance(payload, BrokerTradeSnapshot):
            raise TypeError("TRADE event requires BrokerTradeSnapshot")
        account_id = payload.account_id or self._account_id
        with self._repository.transaction() as connection:
            order = self._repository.get_broker_order(
                account_id, payload.broker_order_id, connection=connection
            )
            if order is None:
                LOGGER.warning("unknown callback trade: %s", payload.broker_trade_id)
                return
            self._repository.insert_broker_trade_if_absent(
                account_id=account_id,
                intent_id=str(order["intent_id"]),
                trade=payload,
                connection=connection,
            )


__all__ = [
    "BrokerEvent",
    "BrokerEventConsumer",
    "BrokerEventType",
    "create_xtquant_callback",
    "enqueue_account_status",
    "enqueue_disconnected",
    "enqueue_error",
    "enqueue_order",
    "enqueue_trade",
]
