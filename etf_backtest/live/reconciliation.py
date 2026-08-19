"""Reconcile already-normalized broker facts with persisted live state."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.engine import Connection

from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
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

        with repository.transaction(connection) as active:
            for order in broker_orders:
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
                repository.bind_broker_order(
                    account_id=account_id,
                    intent_id=intent_id,
                    remark_token=remark_token,
                    order=order,
                    connection=active,
                )
                matched_orders += 1

            for trade in broker_trades:
                saved_order = repository.get_broker_order(
                    account_id, trade.broker_order_id, connection=active
                )
                if saved_order is None:
                    unknown_trades.append(trade.broker_trade_id)
                    continue
                if repository.insert_broker_trade_if_absent(
                    account_id=account_id,
                    intent_id=str(saved_order["intent_id"]),
                    trade=trade,
                    connection=active,
                ):
                    inserted_trades += 1

            unresolved = repository.list_unresolved_intents(
                account_id=account_id, connection=active
            )

        return ReconciliationReport(
            matched_order_count=matched_orders,
            inserted_trade_count=inserted_trades,
            unresolved_intent_ids=tuple(sorted(str(row["intent_id"]) for row in unresolved)),
            unknown_broker_order_ids=tuple(sorted(set(unknown_orders))),
            unknown_broker_trade_ids=tuple(sorted(set(unknown_trades))),
        )


__all__ = ["ReconciliationService"]
