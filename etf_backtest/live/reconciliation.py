"""Reconcile already-normalized broker facts with persisted live state."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy.engine import Connection

from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerTradeSnapshot,
    ReconciliationReport,
)


class ReconciliationService:
    def reconcile(
        self,
        *,
        account_id: str,
        broker_orders: Sequence[BrokerOrderSnapshot],
        broker_trades: Sequence[BrokerTradeSnapshot],
        repository: LiveStateRepository,
        connection: Connection | None = None,
    ) -> ReconciliationReport:
        matched_orders = 0
        inserted_trades = 0
        unknown_orders: list[str] = []
        unknown_trades: list[str] = []
        active_orders: list[str] = []
        incomplete_intents: list[str] = []
        quantity_mismatches: list[str] = []
        identity_mismatches: list[str] = []
        trade_identity_mismatches: list[str] = []
        unknown_statuses: list[str] = []
        matched: dict[str, tuple[BrokerOrderSnapshot, str]] = {}
        accepted_trades: list[BrokerTradeSnapshot] = []

        with repository.transaction(connection) as active:
            for order in broker_orders:
                if order.account_id not in {None, account_id}:
                    identity_mismatches.append(order.broker_order_id)
                    continue
                saved_order = repository.get_broker_order(
                    account_id, order.broker_order_id, connection=active
                )
                intent = None
                if saved_order is not None:
                    intent_id = str(saved_order["intent_id"])
                    remark_token = str(saved_order["remark_token"])
                elif order.remark_token:
                    intent = repository.get_intent_by_remark_token(
                        order.remark_token, account_id=account_id, connection=active
                    )
                    if intent is None:
                        unknown_orders.append(order.broker_order_id)
                        continue
                    intent_id = str(intent["intent_id"])
                    remark_token = order.remark_token
                else:
                    unknown_orders.append(order.broker_order_id)
                    continue
                intent_row = repository.get_intent(intent_id, connection=active)
                if intent_row is None or not self._order_matches_intent(order, intent_row):
                    identity_mismatches.append(order.broker_order_id)
                    continue
                repository.bind_broker_order(
                    account_id=account_id,
                    intent_id=intent_id,
                    remark_token=remark_token,
                    order=order,
                    connection=active,
                )
                matched_orders += 1
                matched[order.broker_order_id] = (order, intent_id)

            for trade in broker_trades:
                if trade.account_id not in {None, account_id}:
                    trade_identity_mismatches.append(trade.broker_trade_id)
                    continue
                saved_order = repository.get_broker_order(
                    account_id, trade.broker_order_id, connection=active
                )
                if saved_order is None:
                    unknown_trades.append(trade.broker_trade_id)
                    continue
                intent_row = repository.get_intent(str(saved_order["intent_id"]), connection=active)
                if intent_row is None or not self._trade_matches_intent(trade, intent_row):
                    trade_identity_mismatches.append(trade.broker_trade_id)
                    continue
                if repository.insert_broker_trade_if_absent(
                    account_id=account_id,
                    intent_id=str(saved_order["intent_id"]),
                    trade=trade,
                    connection=active,
                ):
                    inserted_trades += 1
                accepted_trades.append(trade)

            traded_by_order: dict[str, int] = {}
            for trade in accepted_trades:
                traded_by_order[trade.broker_order_id] = (
                    traded_by_order.get(trade.broker_order_id, 0) + trade.quantity
                )
            for broker_order_id, (order, intent_id) in matched.items():
                traded_quantity = traded_by_order.get(broker_order_id, 0)
                if traded_quantity != order.filled_quantity:
                    quantity_mismatches.append(broker_order_id)
                    continue
                if order.status is BrokerOrderStatus.UNKNOWN:
                    unknown_statuses.append(broker_order_id)
                    continue
                if order.status.is_active:
                    active_orders.append(broker_order_id)
                elif (
                    order.status is BrokerOrderStatus.FILLED
                    and order.filled_quantity == order.requested_quantity
                ):
                    repository.mark_intent_completed(intent_id, connection=active)
                else:
                    repository.mark_intent_incomplete(
                        intent_id,
                        f"BROKER_TERMINAL_{order.status.value}",
                        connection=active,
                    )
                    incomplete_intents.append(intent_id)

            unresolved = repository.list_unresolved_intents(
                account_id=account_id, connection=active
            )

        return ReconciliationReport(
            matched_order_count=matched_orders,
            inserted_trade_count=inserted_trades,
            unresolved_intent_ids=tuple(sorted(str(row["intent_id"]) for row in unresolved)),
            active_broker_order_ids=tuple(sorted(set(active_orders))),
            incomplete_intent_ids=tuple(sorted(set(incomplete_intents))),
            order_trade_mismatch_ids=tuple(sorted(set(quantity_mismatches))),
            order_identity_mismatch_ids=tuple(sorted(set(identity_mismatches))),
            trade_identity_mismatch_ids=tuple(sorted(set(trade_identity_mismatches))),
            unknown_order_status_ids=tuple(sorted(set(unknown_statuses))),
            unknown_broker_order_ids=tuple(sorted(set(unknown_orders))),
            unknown_broker_trade_ids=tuple(sorted(set(unknown_trades))),
        )

    @staticmethod
    def _order_matches_intent(order: BrokerOrderSnapshot, intent: dict[str, object]) -> bool:
        return (
            str(intent["symbol"]) == order.symbol
            and str(intent["side"]) == order.side.value
            and int(str(intent["requested_quantity"])) == order.requested_quantity
            and Decimal(str(intent["limit_price"])) == order.limit_price
            and str(intent["remark_token"]) == str(order.remark_token)
        )

    @staticmethod
    def _trade_matches_intent(trade: BrokerTradeSnapshot, intent: dict[str, object]) -> bool:
        return str(intent["symbol"]) == trade.symbol and str(intent["side"]) == trade.side.value


__all__ = ["ReconciliationService"]
