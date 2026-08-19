"""Protocol used by live jobs without importing an external broker SDK."""

from __future__ import annotations

from typing import Protocol

from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    OrderIntent,
    QueryResult,
    SubmitOrderResult,
)


class BrokerGateway(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def subscribe_account(self, account_id: str) -> None: ...

    def query_asset(self) -> QueryResult[BrokerAssetSnapshot]: ...

    def query_positions(self) -> QueryResult[BrokerPositionSnapshot]: ...

    def query_orders(self) -> QueryResult[BrokerOrderSnapshot]: ...

    def query_trades(self) -> QueryResult[BrokerTradeSnapshot]: ...

    def submit_order(self, intent: OrderIntent) -> SubmitOrderResult: ...

    def cancel_order(self, broker_order_id: str) -> bool: ...


__all__ = ["BrokerGateway"]
