"""The single translation boundary between xtquant values and live domain values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from types import ModuleType
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    LiveQuote,
    SubmitOrderResult,
    SubmitOrderStatus,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")

# Published xtquant values. Runtime order submission still reads the named
# constants from the installed SDK, so a changed SDK cannot silently submit.
STOCK_BUY = 23
STOCK_SELL = 24
OFFSET_BUY = 48
OFFSET_SELL = 49

_ORDER_STATUSES = {
    48: BrokerOrderStatus.PENDING,  # 未报
    49: BrokerOrderStatus.PENDING,  # 待报
    50: BrokerOrderStatus.PENDING,  # 已报
    51: BrokerOrderStatus.PENDING,  # 已报待撤
    52: BrokerOrderStatus.PARTIALLY_FILLED,  # 部成待撤
    53: BrokerOrderStatus.CANCELED,  # 部撤
    54: BrokerOrderStatus.CANCELED,  # 已撤
    55: BrokerOrderStatus.PARTIALLY_FILLED,  # 部成
    56: BrokerOrderStatus.FILLED,  # 已成
    57: BrokerOrderStatus.REJECTED,  # 废单
    255: BrokerOrderStatus.UNKNOWN,
}


def decimal_value(value: object) -> Decimal:
    """Convert an external numeric value without constructing Decimal from float."""

    return Decimal(str(value))


def integer_value(value: object) -> int:
    return int(cast(Any, value))


def internal_to_external_symbol(symbol: str) -> str:
    exchange, _, code = normalize_symbol(symbol).partition(".")
    return f"{code}.{exchange}"


def external_to_internal_symbol(symbol: str) -> str:
    return normalize_symbol(symbol)


def side_to_xt(side: OrderSide, constants: ModuleType | Any) -> int:
    if side is OrderSide.BUY:
        return int(constants.STOCK_BUY)
    if side is OrderSide.SELL:
        return int(constants.STOCK_SELL)
    raise ValueError(f"unsupported order side: {side}")


def side_from_xt(
    order_type: object | None,
    *,
    offset_flag: object | None = None,
    constants: ModuleType | Any | None = None,
) -> OrderSide:
    buys = {STOCK_BUY, OFFSET_BUY}
    sells = {STOCK_SELL, OFFSET_SELL}
    if constants is not None:
        buys.add(int(constants.STOCK_BUY))
        sells.add(int(constants.STOCK_SELL))
    values = {integer_value(value) for value in (order_type, offset_flag) if value is not None}
    if values & buys:
        return OrderSide.BUY
    if values & sells:
        return OrderSide.SELL
    raise ValueError(f"unsupported stock order side: {values}")


def order_status_from_xt(value: object) -> BrokerOrderStatus:
    try:
        return _ORDER_STATUSES.get(integer_value(value), BrokerOrderStatus.UNKNOWN)
    except (TypeError, ValueError):
        return BrokerOrderStatus.UNKNOWN


def market_datetime(
    value: object,
    *,
    unit: Literal["seconds", "milliseconds"],
) -> datetime:
    """Convert one documented SDK epoch value using an explicit unit.

    XtOrder/XtTrade times are treated as seconds and xtdata tick times as
    milliseconds. The target SDK version must confirm these units.
    """

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=SHANGHAI)
        return value.astimezone(SHANGHAI)
    divisor = Decimal("1000") if unit == "milliseconds" else Decimal("1")
    seconds = decimal_value(value) / divisor
    return datetime.fromtimestamp(float(seconds), tz=SHANGHAI)


def _attribute(source: object, name: str, default: object | None = None) -> object:
    return getattr(source, name, default)


def map_asset(source: object, *, captured_at: datetime) -> BrokerAssetSnapshot:
    return BrokerAssetSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        available_cash=decimal_value(_attribute(source, "cash", 0)),
        frozen_cash=decimal_value(_attribute(source, "frozen_cash", 0)),
        market_value=decimal_value(_attribute(source, "market_value", 0)),
        total_asset=decimal_value(_attribute(source, "total_asset", 0)),
        captured_at=captured_at,
    )


def map_position(source: object, *, captured_at: datetime) -> BrokerPositionSnapshot:
    on_road = integer_value(_attribute(source, "on_road_volume", 0))
    average_cost = _attribute(source, "avg_price", _attribute(source, "open_price"))
    return BrokerPositionSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        total_quantity=integer_value(_attribute(source, "volume", 0)),
        available_quantity=integer_value(_attribute(source, "can_use_volume", 0)),
        # The SDK's explicit on-road field is used; this is deliberately not
        # inferred from total_quantity - available_quantity.
        today_buy_quantity=max(0, on_road),
        market_value=decimal_value(_attribute(source, "market_value", 0)),
        turnover_rule=TurnoverRule.T1,
        captured_at=captured_at,
        frozen_quantity=integer_value(_attribute(source, "frozen_volume", 0)),
        on_road_quantity=on_road,
        yesterday_quantity=integer_value(_attribute(source, "yesterday_volume", 0)),
        average_cost=None if average_cost is None else decimal_value(average_cost),
    )


def map_order(source: object, *, constants: ModuleType | Any | None = None) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        broker_order_id=str(_attribute(source, "order_id")),
        broker_order_sysid=str(_attribute(source, "order_sysid", "")) or None,
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        side=side_from_xt(
            _attribute(source, "order_type"),
            offset_flag=_attribute(source, "offset_flag"),
            constants=constants,
        ),
        requested_quantity=integer_value(_attribute(source, "order_volume", 0)),
        filled_quantity=integer_value(_attribute(source, "traded_volume", 0)),
        limit_price=decimal_value(_attribute(source, "price", 0)),
        traded_price=decimal_value(_attribute(source, "traded_price", 0)),
        status=order_status_from_xt(_attribute(source, "order_status")),
        remark_token=str(_attribute(source, "order_remark", "")) or None,
        captured_at=market_datetime(_attribute(source, "order_time"), unit="seconds"),
    )


def map_trade(source: object, *, constants: ModuleType | Any | None = None) -> BrokerTradeSnapshot:
    return BrokerTradeSnapshot(
        account_id=str(_attribute(source, "account_id", "")),
        broker_trade_id=str(_attribute(source, "traded_id")),
        broker_order_id=str(_attribute(source, "order_id")),
        broker_order_sysid=str(_attribute(source, "order_sysid", "")) or None,
        symbol=external_to_internal_symbol(str(_attribute(source, "stock_code"))),
        side=side_from_xt(
            _attribute(source, "order_type"),
            offset_flag=_attribute(source, "offset_flag"),
            constants=constants,
        ),
        quantity=integer_value(_attribute(source, "traded_volume", 0)),
        price=decimal_value(_attribute(source, "traded_price", 0)),
        remark_token=str(_attribute(source, "order_remark", "")) or None,
        traded_at=market_datetime(_attribute(source, "traded_time"), unit="seconds"),
    )


def map_submit_result(value: object) -> SubmitOrderResult:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return SubmitOrderResult(
            SubmitOrderStatus.ACCEPTED, broker_order_id=str(value)
        )
    if value == -1:
        return SubmitOrderResult(
            SubmitOrderStatus.REJECTED, error="MiniQMT order_stock returned -1"
        )
    return SubmitOrderResult(
        SubmitOrderStatus.UNKNOWN,
        error=f"MiniQMT order_stock returned unexpected result: {value!r}",
    )


def _first_price(values: object) -> Decimal | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        return None
    value = decimal_value(values[0])
    return value if value > 0 else None


def map_quote(
    external_symbol: str,
    tick: Mapping[str, object],
    instrument: Mapping[str, object],
) -> LiveQuote:
    stock_status = integer_value(tick.get("stockStatus", 0))
    is_trading = bool(instrument.get("IsTrading", False))
    instrument_status = str(instrument.get("InstrumentStatus", "")).casefold()
    tradable_tick_status = stock_status in {11, 12, 13, 18, 19, 22}
    suspended = (
        not is_trading
        or not tradable_tick_status
        or "停牌" in instrument_status
        or "suspend" in instrument_status
    )
    return LiveQuote(
        symbol=external_to_internal_symbol(external_symbol),
        last_price=decimal_value(tick.get("lastPrice", 0)),
        bid1=_first_price(tick.get("bidPrice")),
        ask1=_first_price(tick.get("askPrice")),
        lower_limit=_optional_positive(instrument.get("DownStopPrice")),
        upper_limit=_optional_positive(instrument.get("UpStopPrice")),
        price_tick=_optional_positive(instrument.get("PriceTick")) or Decimal("0.001"),
        suspended=suspended,
        quoted_at=market_datetime(tick["time"], unit="milliseconds"),
    )


def _optional_positive(value: object | None) -> Decimal | None:
    if value is None:
        return None
    converted = decimal_value(value)
    return converted if converted > 0 else None


__all__ = [
    "decimal_value",
    "external_to_internal_symbol",
    "internal_to_external_symbol",
    "map_asset",
    "map_order",
    "map_position",
    "map_quote",
    "map_submit_result",
    "map_trade",
    "market_datetime",
    "order_status_from_xt",
    "side_from_xt",
    "side_to_xt",
]
