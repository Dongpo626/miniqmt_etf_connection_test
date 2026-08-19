"""Immutable daily-close order, approval and formal-fill contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum

from etf_backtest.config.schema import MARKET_TIMEZONE, normalize_symbol
from etf_backtest.core.market import PriceLimitSource


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class RuleReasonCode(StrEnum):
    APPROVED = "APPROVED"
    LISTING_OR_WINDOW = "LISTING_OR_WINDOW"
    SUSPENDED = "SUSPENDED"
    QUOTE_UNAVAILABLE = "QUOTE_UNAVAILABLE"
    PRICE_LIMIT = "PRICE_LIMIT"
    LOT_SIZE = "LOT_SIZE"
    TURNOVER_RULE = "TURNOVER_RULE"
    VOLUME_LIMIT = "VOLUME_LIMIT"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"


class FillStatus(StrEnum):
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _quantity(value: object, field_name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _money(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    signed: bool = False,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and not signed and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class Order:
    """A target-derived quantity bound to the next SSE close."""

    order_id: str
    signal_date: date
    execution_date: date
    symbol: str
    side: OrderSide
    requested_quantity: int
    target_value_gap: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _text(self.order_id, "order_id"))
        signal_date = _plain_date(self.signal_date, "signal_date")
        execution_date = _plain_date(self.execution_date, "execution_date")
        if execution_date <= signal_date:
            raise ValueError("execution_date must strictly follow signal_date")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be OrderSide")
        _quantity(self.requested_quantity, "requested_quantity")
        gap = _money(self.target_value_gap, "target_value_gap", signed=True)
        if self.side is OrderSide.BUY and gap < 0:
            raise ValueError("BUY target_value_gap must be non-negative")
        if self.side is OrderSide.SELL and gap > 0:
            raise ValueError("SELL target_value_gap must be non-positive")


@dataclass(frozen=True, slots=True)
class TradePriceQuote:
    """One cached raw close quote and its legal daily price boundaries."""

    source_record_key: str
    symbol: str
    trade_date: date
    base_trade_price: Decimal
    price_limit_down: Decimal
    price_limit_up: Decimal
    price_limit_source: PriceLimitSource = PriceLimitSource.DERIVED_RULE_FALLBACK

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_record_key", _text(self.source_record_key, "source_record_key")
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        base = _money(self.base_trade_price, "base_trade_price", positive=True)
        lower = _money(self.price_limit_down, "price_limit_down", positive=True)
        upper = _money(self.price_limit_up, "price_limit_up", positive=True)
        if not lower <= base <= upper:
            raise ValueError("base close must stay inside the legal price range")
        if not isinstance(self.price_limit_source, PriceLimitSource):
            raise TypeError("price_limit_source must be PriceLimitSource")

    @property
    def trade_time(self) -> datetime:
        return datetime.combine(self.trade_date, time(15, 0), tzinfo=MARKET_TIMEZONE)

    @property
    def price_source(self) -> str:
        return "CLOSE"


@dataclass(frozen=True, slots=True)
class ExecutionEstimate:
    """Direction-adjusted price and fee estimate before quantity approval."""

    order_id: str
    requested_quantity: int
    base_trade_price: Decimal
    fill_price: Decimal
    estimated_trade_amount: Decimal
    estimated_fee: Decimal
    estimated_total_cash_required: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _text(self.order_id, "order_id"))
        quantity = _quantity(self.requested_quantity, "requested_quantity")
        _money(self.base_trade_price, "base_trade_price", positive=True)
        fill_price = _money(self.fill_price, "fill_price", positive=True)
        amount = _money(self.estimated_trade_amount, "estimated_trade_amount")
        fee = _money(self.estimated_fee, "estimated_fee")
        total = _money(self.estimated_total_cash_required, "estimated_total_cash_required")
        if amount != fill_price * quantity:
            raise ValueError("estimated trade amount is inconsistent")
        if total != amount + fee:
            raise ValueError("estimated total cash is inconsistent")


@dataclass(frozen=True, slots=True)
class RuleCheckResult:
    """The sole owner of an order's final approved quantity."""

    order_id: str
    requested_quantity: int
    approved_quantity: int
    passed: bool
    reason_code: RuleReasonCode
    message: str
    base_trade_price: Decimal | None = None
    price_limit_down: Decimal | None = None
    price_limit_up: Decimal | None = None
    price_limit_source: PriceLimitSource | None = None
    price_limit_fallback_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _text(self.order_id, "order_id"))
        requested = _quantity(self.requested_quantity, "requested_quantity")
        approved = _quantity(self.approved_quantity, "approved_quantity")
        if approved > requested:
            raise ValueError("approved quantity exceeds requested quantity")
        if type(self.passed) is not bool:
            raise TypeError("passed must be bool")
        if not isinstance(self.reason_code, RuleReasonCode):
            raise TypeError("reason_code must be RuleReasonCode")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if self.passed != (approved > 0):
            raise ValueError("passed must exactly reflect positive approval")
        if self.passed and self.reason_code is not RuleReasonCode.APPROVED:
            raise ValueError("positive approval must use APPROVED")
        if not self.passed and self.reason_code is RuleReasonCode.APPROVED:
            raise ValueError("zero approval cannot use APPROVED")
        prices = (self.base_trade_price, self.price_limit_down, self.price_limit_up)
        if all(value is None for value in prices):
            if self.price_limit_source is not None or self.price_limit_fallback_reason is not None:
                raise ValueError("price-limit source/reason requires complete quote evidence")
            return
        if any(value is None for value in prices):
            raise ValueError("rule-check quote evidence must be supplied as a complete set")
        assert self.base_trade_price is not None
        assert self.price_limit_down is not None
        assert self.price_limit_up is not None
        base = _money(self.base_trade_price, "base_trade_price", positive=True)
        lower = _money(self.price_limit_down, "price_limit_down", positive=True)
        upper = _money(self.price_limit_up, "price_limit_up", positive=True)
        if not lower <= base <= upper:
            raise ValueError("rule-check base close must stay inside legal price limits")
        if not isinstance(self.price_limit_source, PriceLimitSource):
            raise TypeError("price_limit_source must be PriceLimitSource")
        if self.price_limit_source is PriceLimitSource.DERIVED_RULE_FALLBACK:
            if self.price_limit_fallback_reason != "NO_EXPLICIT_PRICE_LIMIT":
                raise ValueError("derived price limits require the fixed fallback reason")
        elif self.price_limit_fallback_reason is not None:
            raise ValueError("explicit price limits must not carry a fallback reason")


@dataclass(frozen=True, slots=True, init=False)
class FillResult:
    """A positive formal fill; direct public construction is forbidden."""

    order_id: str
    signal_date: date
    execution_date: date
    source_record_key: str
    symbol: str
    side: OrderSide
    requested_quantity: int
    fill_quantity: int
    base_trade_price: Decimal
    fill_price: Decimal
    trade_amount: Decimal
    fee: Decimal
    status: FillStatus

    def __init__(self) -> None:
        raise TypeError("FillResult must be created with from_approved")

    def __post_init__(self) -> None:
        _text(self.order_id, "order_id")
        signal = _plain_date(self.signal_date, "signal_date")
        execution = _plain_date(self.execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")
        _text(self.source_record_key, "source_record_key")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.side, OrderSide):
            raise TypeError("side must be OrderSide")
        requested = _quantity(self.requested_quantity, "requested_quantity", positive=True)
        filled = _quantity(self.fill_quantity, "fill_quantity", positive=True)
        if filled > requested:
            raise ValueError("fill quantity exceeds request")
        _money(self.base_trade_price, "base_trade_price", positive=True)
        price = _money(self.fill_price, "fill_price", positive=True)
        amount = _money(self.trade_amount, "trade_amount", positive=True)
        _money(self.fee, "fee")
        if amount != price * filled:
            raise ValueError("trade amount is inconsistent")
        expected = FillStatus.FILLED if filled == requested else FillStatus.PARTIALLY_FILLED
        if self.status is not expected:
            raise ValueError("fill status is inconsistent")

    @classmethod
    def from_approved(
        cls,
        *,
        order: Order,
        quote: TradePriceQuote,
        estimate: ExecutionEstimate,
        approval: RuleCheckResult,
        trade_amount: Decimal,
        fee: Decimal,
    ) -> FillResult:
        """Materialize a fill only after preserving every upstream identity."""

        if not approval.passed or approval.approved_quantity <= 0:
            raise ValueError("formal fill requires a positive approval")
        if order.order_id != estimate.order_id or order.order_id != approval.order_id:
            raise ValueError("order identity changed in the execution chain")
        if quote.symbol != order.symbol or quote.trade_date != order.execution_date:
            raise ValueError("quote identity changed in the execution chain")
        if estimate.requested_quantity != order.requested_quantity:
            raise ValueError("estimate quantity changed in the execution chain")
        if approval.requested_quantity != order.requested_quantity:
            raise ValueError("approval quantity changed in the execution chain")
        if estimate.base_trade_price != quote.base_trade_price:
            raise ValueError("base price changed in the execution chain")

        instance = object.__new__(cls)
        values: dict[str, object] = {
            "order_id": order.order_id,
            "signal_date": order.signal_date,
            "execution_date": order.execution_date,
            "source_record_key": quote.source_record_key,
            "symbol": order.symbol,
            "side": order.side,
            "requested_quantity": order.requested_quantity,
            "fill_quantity": approval.approved_quantity,
            "base_trade_price": estimate.base_trade_price,
            "fill_price": estimate.fill_price,
            "trade_amount": trade_amount,
            "fee": fee,
            "status": (
                FillStatus.FILLED
                if approval.approved_quantity == order.requested_quantity
                else FillStatus.PARTIALLY_FILLED
            ),
        }
        for field_name, value in values.items():
            object.__setattr__(instance, field_name, value)
        instance.__post_init__()
        return instance

    @property
    def trade_time(self) -> datetime:
        return datetime.combine(self.execution_date, time(15, 0), tzinfo=MARKET_TIMEZONE)

    @property
    def price_source(self) -> str:
        return "CLOSE"


__all__ = [
    "ExecutionEstimate",
    "FillResult",
    "FillStatus",
    "Order",
    "OrderSide",
    "RuleCheckResult",
    "RuleReasonCode",
    "TradePriceQuote",
]
