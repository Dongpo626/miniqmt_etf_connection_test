from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy.engine import Connection

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.account_adapter import adapt_broker_account
from etf_backtest.live.execution.planner import LiveRebalancePlanner
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    OrderIntent,
    OrderIntentStatus,
    SubmitOrderResult,
    SubmitOrderStatus,
)

NOW = datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE)


class _ClosedLoopRepository:
    def __init__(self, intent: OrderIntent) -> None:
        self.intent = {
            "intent_id": "intent-1",
            "remark_token": intent.remark_token,
            "symbol": intent.symbol,
            "side": intent.side,
            "requested_quantity": intent.requested_quantity,
            "limit_price": intent.limit_price,
            "status": OrderIntentStatus.PLANNED,
        }
        self.order: dict[str, Any] | None = None
        self.trade_ids: set[str] = set()

    @contextmanager
    def transaction(self, connection: Connection | None = None) -> Any:
        yield connection

    def get_broker_order(
        self, account_id: str, broker_order_id: str, *, connection: object = None
    ) -> dict[str, Any] | None:
        if self.order is not None and self.order["broker_order_id"] == broker_order_id:
            return self.order
        return None

    def get_intent_by_remark_token(
        self,
        remark_token: str,
        *,
        account_id: str | None = None,
        connection: object = None,
    ) -> dict[str, Any] | None:
        return self.intent if self.intent["remark_token"] == remark_token else None

    def get_intent(self, intent_id: str, *, connection: object = None) -> dict[str, Any] | None:
        del connection
        return self.intent if self.intent["intent_id"] == intent_id else None

    def mark_intent_completed(self, intent_id: str, *, connection: object = None) -> None:
        del connection
        assert self.intent["intent_id"] == intent_id
        self.intent["status"] = OrderIntentStatus.COMPLETED

    def mark_intent_incomplete(
        self, intent_id: str, reason: str, *, connection: object = None
    ) -> None:
        del reason, connection
        assert self.intent["intent_id"] == intent_id
        self.intent["status"] = OrderIntentStatus.INCOMPLETE

    def bind_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: object = None,
    ) -> None:
        self.order = {
            "account_id": account_id,
            "broker_order_id": order.broker_order_id,
            "intent_id": intent_id,
            "remark_token": remark_token,
        }
        self.intent["status"] = OrderIntentStatus.SUBMITTED

    def insert_broker_trade_if_absent(
        self,
        *,
        account_id: str,
        intent_id: str,
        trade: BrokerTradeSnapshot,
        connection: object = None,
    ) -> bool:
        del account_id, intent_id, connection
        if trade.broker_trade_id in self.trade_ids:
            return False
        self.trade_ids.add(trade.broker_trade_id)
        return True

    def list_unresolved_intents(
        self, *, account_id: str, connection: object = None
    ) -> tuple[dict[str, Any], ...]:
        del account_id, connection
        return ()


class _StubBroker:
    def __init__(self, intent: OrderIntent) -> None:
        self.intent = intent
        self.submit_calls = 0
        self.order: BrokerOrderSnapshot | None = None
        self.trade: BrokerTradeSnapshot | None = None

    def submit_once(self) -> SubmitOrderResult:
        if self.submit_calls:
            raise AssertionError("economic order was submitted twice")
        self.submit_calls += 1
        self.order = BrokerOrderSnapshot(
            broker_order_id="broker-1",
            symbol=self.intent.symbol,
            side=self.intent.side,
            requested_quantity=self.intent.requested_quantity,
            filled_quantity=self.intent.requested_quantity,
            limit_price=self.intent.limit_price,
            status=BrokerOrderStatus.FILLED,
            captured_at=NOW,
            remark_token=self.intent.remark_token,
        )
        self.trade = BrokerTradeSnapshot(
            broker_trade_id="trade-1",
            broker_order_id="broker-1",
            symbol=self.intent.symbol,
            side=self.intent.side,
            quantity=self.intent.requested_quantity,
            price=self.intent.limit_price,
            traded_at=NOW,
        )
        return SubmitOrderResult(SubmitOrderStatus.ACCEPTED, broker_order_id="broker-1")


def test_minimal_rule_target_to_trade_reconcile_and_snapshot_closed_loop() -> None:
    intents = LiveRebalancePlanner().plan(
        deployment_id="deployment-1",
        decision_id="decision-1",
        execution_date=date(2026, 8, 19),
        symbols=("510300.SH",),
        target=TargetPortfolio({"510300.SH": Decimal("0.5")}),
        total_asset=Decimal("10000"),
        available_cash=Decimal("10000"),
        positions={},
        active_orders=(),
        valuation_prices={"SH.510300": Decimal("10")},
        limit_prices={"SH.510300": Decimal("10")},
        lot_size=100,
    )
    assert len(intents) == 1
    intent = intents[0]
    repository = _ClosedLoopRepository(intent)
    broker = _StubBroker(intent)

    repository.intent["status"] = OrderIntentStatus.SUBMITTING
    result = broker.submit_once()
    assert result.status is SubmitOrderStatus.ACCEPTED
    assert broker.order is not None and broker.trade is not None
    repository.bind_broker_order(
        account_id="account-1",
        intent_id="intent-1",
        remark_token=intent.remark_token,
        order=broker.order,
    )
    service = ReconciliationService()
    first = service.reconcile(
        account_id="account-1",
        broker_orders=(broker.order,),
        broker_trades=(broker.trade,),
        repository=cast(LiveStateRepository, repository),
    )
    second = service.reconcile(
        account_id="account-1",
        broker_orders=(broker.order,),
        broker_trades=(broker.trade,),
        repository=cast(LiveStateRepository, repository),
    )
    assert first.inserted_trade_count == 1 and second.inserted_trade_count == 0
    assert broker.submit_calls == 1 and len(repository.trade_ids) == 1

    position = BrokerPositionSnapshot(
        symbol="510300.SH",
        total_quantity=500,
        available_quantity=0,
        today_buy_quantity=500,
        market_value=Decimal("5000"),
        turnover_rule=TurnoverRule.T1,
        captured_at=NOW,
    )
    snapshot = adapt_broker_account(
        asset=BrokerAssetSnapshot(
            total_asset=Decimal("10000"),
            available_cash=Decimal("5000"),
            captured_at=NOW,
            account_id="account-1",
        ),
        positions=(position,),
        symbols=("510300.SH",),
    )
    assert snapshot.positions_by_symbol["SH.510300"].total_quantity == 500
    assert repository.intent["status"] is OrderIntentStatus.COMPLETED
