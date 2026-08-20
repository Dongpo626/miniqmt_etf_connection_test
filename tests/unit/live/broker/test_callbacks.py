from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from etf_backtest.core.order import OrderSide
from etf_backtest.live.broker.callbacks import (
    BrokerEvent,
    BrokerEventConsumer,
    BrokerEventType,
    enqueue_account_status,
    enqueue_disconnected,
    enqueue_error,
    enqueue_order,
    enqueue_trade,
)
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerTradeSnapshot,
)

NOW = datetime(2026, 8, 19, 14, 50, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_six_callback_events_only_enter_the_internal_queue() -> None:
    events: Queue[BrokerEvent] = Queue()
    order = BrokerOrderSnapshot(
        broker_order_id="1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("4"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        account_id="paper-1",
    )
    trade = BrokerTradeSnapshot(
        broker_trade_id="t1",
        broker_order_id="1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("4"),
        traded_at=NOW,
        account_id="paper-1",
    )
    enqueue_order(events, order)
    enqueue_trade(events, trade)
    enqueue_disconnected(events)
    enqueue_account_status(events, SimpleNamespace(account_id="paper-1", status=3))
    error = SimpleNamespace(order_id=1, error_id=2, error_msg="failed")
    enqueue_error(events, BrokerEventType.ORDER_ERROR, error)
    enqueue_error(events, BrokerEventType.CANCEL_ERROR, error)

    assert [events.get_nowait().event_type for _ in range(6)] == [
        BrokerEventType.ORDER,
        BrokerEventType.TRADE,
        BrokerEventType.DISCONNECTED,
        BrokerEventType.ACCOUNT_STATUS,
        BrokerEventType.ORDER_ERROR,
        BrokerEventType.CANCEL_ERROR,
    ]


def test_consumer_persists_single_order_and_trade_and_stops_on_unhealthy() -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.transaction.return_value = nullcontext(Mock())
    repository.get_broker_order.side_effect = [
        {"intent_id": "intent-1", "remark_token": "L123"},
        {"intent_id": "intent-1", "remark_token": "L123"},
    ]
    unhealthy = Mock()
    events: Queue[BrokerEvent] = Queue()
    consumer = BrokerEventConsumer(
        events=events,
        repository=repository,
        account_id="paper-1",
        on_unhealthy=unhealthy,
    )
    order = BrokerOrderSnapshot(
        broker_order_id="1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=0,
        limit_price=Decimal("4"),
        status=BrokerOrderStatus.PENDING,
        captured_at=NOW,
        account_id="paper-1",
    )
    trade = BrokerTradeSnapshot(
        broker_trade_id="t1",
        broker_order_id="1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("4"),
        traded_at=NOW,
        account_id="paper-1",
    )

    consumer.process(BrokerEvent(BrokerEventType.ORDER, payload=order))
    consumer.process(BrokerEvent(BrokerEventType.TRADE, payload=trade))
    consumer.process(BrokerEvent(BrokerEventType.DISCONNECTED))
    consumer.process(
        BrokerEvent(
            BrokerEventType.ACCOUNT_STATUS,
            account_id="paper-1",
            status=3,
        )
    )
    consumer.process(BrokerEvent(BrokerEventType.ORDER_ERROR, error="rejected"))
    consumer.process(BrokerEvent(BrokerEventType.CANCEL_ERROR, error="cancel failed"))

    repository.bind_broker_order.assert_called_once()
    repository.insert_broker_trade_if_absent.assert_called_once()
    assert unhealthy.call_count == 4
