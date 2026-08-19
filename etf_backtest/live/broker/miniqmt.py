"""Synchronous MiniQMT implementation of the existing BrokerGateway protocol."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from importlib import import_module
from pathlib import Path
from queue import Queue
from types import ModuleType
from typing import Any, TypeVar

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.live.broker.callbacks import BrokerEvent, create_xtquant_callback
from etf_backtest.live.broker.mapper import (
    internal_to_external_symbol,
    map_asset,
    map_order,
    map_position,
    map_submit_result,
    map_trade,
    side_to_xt,
)
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    OrderIntent,
    QueryResult,
    SubmitOrderResult,
    SubmitOrderStatus,
)

RecordT = TypeVar("RecordT")


def _load_xtquant() -> tuple[ModuleType, ModuleType, ModuleType]:
    try:
        return (
            import_module("xtquant.xttrader"),
            import_module("xtquant.xttype"),
            import_module("xtquant.xtconstant"),
        )
    except ImportError as error:
        raise RuntimeError("当前环境未安装 xtquant, 无法创建交易运行时。") from error


class MiniQmtBrokerGateway:
    def __init__(
        self,
        *,
        userdata_path: Path,
        session_id: int,
        account_id: str,
        event_queue: Queue[BrokerEvent],
        account_type: str = "STOCK",
        strategy_name: str = "qmt-etf-live",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        xttrader, xttype, constants = _load_xtquant()
        self._userdata_path = Path(userdata_path)
        self._session_id = session_id
        self._account_id = account_id
        self._account_type = account_type
        self._strategy_name = strategy_name
        self._events = event_queue
        self._clock = clock or (lambda: datetime.now(MARKET_TIMEZONE))
        self._xttrader = xttrader
        self._xttype = xttype
        self._constants = constants
        self._trader: Any | None = None
        self._account: Any | None = None
        self._subscribed = False
        self._trading_enabled = False

    def connect(self) -> None:
        if not self._userdata_path.is_dir():
            raise FileNotFoundError(f"userdata_mini directory is missing: {self._userdata_path}")
        trader = self._xttrader.XtQuantTrader(
            str(self._userdata_path), self._session_id
        )
        trader.register_callback(create_xtquant_callback(self._events))
        trader.start()
        try:
            result = trader.connect()
        except Exception:
            trader.stop()
            raise
        if result != 0:
            trader.stop()
            raise RuntimeError(f"MiniQMT connect failed: {result}")
        try:
            account = self._xttype.StockAccount(self._account_id, self._account_type)
        except Exception:
            trader.stop()
            raise
        self._trader = trader
        self._account = account

    def subscribe_account(self, account_id: str) -> None:
        if account_id != self._account_id:
            raise ValueError("subscribed account does not match configured account_id")
        trader, account = self._connected()
        result = trader.subscribe(account)
        if result != 0:
            raise RuntimeError(f"MiniQMT account subscribe failed: {result}")
        self._subscribed = True
        self._trading_enabled = True

    def disconnect(self) -> None:
        trader, account = self._trader, self._account
        self._trading_enabled = False
        try:
            if trader is not None and account is not None and self._subscribed:
                result = trader.unsubscribe(account)
                if result not in {None, 0}:
                    raise RuntimeError(f"MiniQMT account unsubscribe failed: {result}")
        finally:
            try:
                if trader is not None:
                    trader.stop()
            finally:
                self._subscribed = False
                self._trader = None
                self._account = None

    def disable_trading(self) -> None:
        self._trading_enabled = False

    def query_asset(self) -> QueryResult[BrokerAssetSnapshot]:
        try:
            trader, account = self._connected()
            value = trader.query_stock_asset(account)
            if value is None:
                return QueryResult(success=False, error="MiniQMT asset query returned None")
            return QueryResult(success=True, records=(map_asset(value, captured_at=self._clock()),))
        except Exception as error:
            return QueryResult(success=False, error=str(error))

    def query_positions(self) -> QueryResult[BrokerPositionSnapshot]:
        return self._query_list("positions", "query_stock_positions", map_position)

    def query_orders(self) -> QueryResult[BrokerOrderSnapshot]:
        return self._query_list(
            "orders",
            "query_stock_orders",
            lambda value, **_: map_order(value, constants=self._constants),
        )

    def query_trades(self) -> QueryResult[BrokerTradeSnapshot]:
        return self._query_list(
            "trades",
            "query_stock_trades",
            lambda value, **_: map_trade(value, constants=self._constants),
        )

    def submit_order(self, intent: OrderIntent) -> SubmitOrderResult:
        if not self._trading_enabled:
            return SubmitOrderResult(
                SubmitOrderStatus.UNKNOWN, error="MiniQMT trading is not enabled"
            )
        if len(intent.remark_token.encode("ascii")) > 24:
            return SubmitOrderResult(
                SubmitOrderStatus.REJECTED, error="order_remark exceeds 24 ASCII bytes"
            )
        try:
            trader, account = self._connected()
            order_id = trader.order_stock(
                account,
                internal_to_external_symbol(intent.symbol),
                side_to_xt(intent.side, self._constants),
                int(intent.requested_quantity),
                int(self._constants.FIX_PRICE),
                float(intent.limit_price),
                self._strategy_name,
                intent.remark_token,
            )
        except Exception as error:
            return SubmitOrderResult(SubmitOrderStatus.UNKNOWN, error=str(error))
        return map_submit_result(order_id)

    def cancel_order(self, broker_order_id: str) -> bool:
        if not self._trading_enabled:
            raise RuntimeError("MiniQMT trading is not enabled")
        trader, account = self._connected()
        try:
            result = trader.cancel_order_stock(account, int(broker_order_id))
        except Exception as error:
            raise RuntimeError(f"MiniQMT cancel failed: {error}") from error
        if result == 0:
            return True
        raise RuntimeError(f"MiniQMT cancel failed: {result}")

    def _connected(self) -> tuple[Any, Any]:
        if self._trader is None or self._account is None:
            raise RuntimeError("MiniQMT broker is not connected")
        return self._trader, self._account

    def _query_list(
        self,
        label: str,
        method_name: str,
        mapper: Callable[..., RecordT],
    ) -> QueryResult[RecordT]:
        try:
            trader, account = self._connected()
            values = getattr(trader, method_name)(account)
            if values is None:
                return QueryResult(
                    success=False, error=f"MiniQMT {label} query returned None"
                )
            if not isinstance(values, (list, tuple)):
                return QueryResult(
                    success=False,
                    error=f"MiniQMT {label} query returned an invalid collection",
                )
            captured_at = self._clock()
            return QueryResult(
                success=True,
                records=tuple(mapper(value, captured_at=captured_at) for value in values),
            )
        except Exception as error:
            return QueryResult(success=False, error=str(error))


__all__ = ["MiniQmtBrokerGateway"]
