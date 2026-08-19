import inspect
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy.engine import Connection

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.order import OrderSide
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerTradeSnapshot,
    OrderIntentStatus,
)

NOW = datetime(2026, 8, 19, 15, 5, tzinfo=MARKET_TIMEZONE)


class _MemoryRepository:
    def __init__(self) -> None:
        self.intents: dict[str, dict[str, Any]] = {}
        self.orders: dict[tuple[str, str], dict[str, Any]] = {}
        self.trades: set[tuple[str, str]] = set()

    @contextmanager
    def transaction(self, connection: Connection | None = None) -> Any:
        yield connection

    def get_broker_order(
        self, account_id: str, broker_order_id: str, *, connection: object = None
    ) -> dict[str, Any] | None:
        return self.orders.get((account_id, broker_order_id))

    def get_intent_by_remark_token(
        self,
        remark_token: str,
        *,
        account_id: str | None = None,
        connection: object = None,
    ) -> dict[str, Any] | None:
        return self.intents.get(remark_token)

    def bind_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: object = None,
    ) -> None:
        self.orders[(account_id, order.broker_order_id)] = {
            "intent_id": intent_id,
            "remark_token": remark_token,
        }
        self.intents[remark_token]["status"] = OrderIntentStatus.SUBMITTED

    def insert_broker_trade_if_absent(
        self,
        *,
        account_id: str,
        intent_id: str,
        trade: BrokerTradeSnapshot,
        connection: object = None,
    ) -> bool:
        key = (account_id, trade.broker_trade_id)
        if key in self.trades:
            return False
        self.trades.add(key)
        return True

    def list_unresolved_intents(
        self, *, account_id: str, connection: object = None
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            intent
            for intent in self.intents.values()
            if intent["status"]
            in {OrderIntentStatus.SUBMITTING, OrderIntentStatus.SUBMIT_UNKNOWN}
        )


def _order(*, token: str | None) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id="order-1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        requested_quantity=100,
        filled_quantity=100,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.FILLED,
        captured_at=NOW,
        remark_token=token,
    )


def _trade(order_id: str = "order-1") -> BrokerTradeSnapshot:
    return BrokerTradeSnapshot(
        broker_trade_id="trade-1",
        broker_order_id=order_id,
        symbol="510300.SH",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("10"),
        traded_at=NOW,
    )


def test_remark_recovers_unknown_submission_and_repeat_is_idempotent() -> None:
    repository = _MemoryRepository()
    token = "L" + "A" * 20
    repository.intents[token] = {
        "intent_id": "intent-1",
        "status": OrderIntentStatus.SUBMIT_UNKNOWN,
    }
    service = ReconciliationService()
    typed = cast(LiveStateRepository, repository)

    first = service.reconcile(
        account_id="account-1",
        broker_orders=[_order(token=token)],
        broker_trades=[_trade()],
        repository=typed,
    )
    second = service.reconcile(
        account_id="account-1",
        broker_orders=[_order(token=token)],
        broker_trades=[_trade()],
        repository=typed,
    )

    assert first.matched_order_count == second.matched_order_count == 1
    assert first.inserted_trade_count == 1 and second.inserted_trade_count == 0
    assert repository.intents[token]["status"] is OrderIntentStatus.SUBMITTED
    assert len(repository.orders) == len(repository.trades) == 1
    assert not first.has_unresolved and not second.has_unresolved


def test_missing_order_keeps_submit_unknown_unresolved() -> None:
    repository = _MemoryRepository()
    token = "L" + "B" * 20
    repository.intents[token] = {
        "intent_id": "intent-2",
        "status": OrderIntentStatus.SUBMIT_UNKNOWN,
    }
    report = ReconciliationService().reconcile(
        account_id="account-1",
        broker_orders=[],
        broker_trades=[],
        repository=cast(LiveStateRepository, repository),
    )

    assert repository.intents[token]["status"] is OrderIntentStatus.SUBMIT_UNKNOWN
    assert report.unresolved_intent_ids == ("intent-2",)
    assert report.has_unresolved


def test_unknown_order_and_trade_are_reported_without_synthetic_state() -> None:
    repository = _MemoryRepository()
    report = ReconciliationService().reconcile(
        account_id="account-1",
        broker_orders=[_order(token="L" + "Z" * 20)],
        broker_trades=[_trade(order_id="unknown-order")],
        repository=cast(LiveStateRepository, repository),
    )

    assert report.unknown_broker_order_ids == ("order-1",)
    assert report.unknown_broker_trade_ids == ("trade-1",)
    assert not repository.intents and not repository.orders and not repository.trades
    source = inspect.getsource(ReconciliationService)
    assert "submit_order" not in source and "cancel_order" not in source
