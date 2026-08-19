"""Select valuation and legal limit prices from one near-close quote."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import LiveQuote


@dataclass(frozen=True, slots=True)
class NearClosePrice:
    valuation_price: Decimal
    limit_price: Decimal


def _is_positive(value: Decimal | None) -> bool:
    return value is not None and value.is_finite() and value > 0


class NearCloseLimitPolicy:
    """Produce a side-independent valuation price and side-specific order price."""

    def calculate(
        self,
        *,
        side: OrderSide,
        quote: LiveQuote,
        tick_size: Decimal,
        price_offset_ticks: int,
        now: datetime,
        quote_stale_seconds: int,
    ) -> NearClosePrice | None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be a timezone-aware datetime")
        if quote_stale_seconds < 0 or price_offset_ticks < 0:
            raise ValueError("staleness and price offset must be non-negative")
        age = (now - quote.quoted_at).total_seconds()
        if age < 0 or age > quote_stale_seconds or quote.suspended:
            return None
        if not _is_positive(quote.last_price) or not _is_positive(tick_size):
            return None
        if not _is_positive(quote.lower_limit) or not _is_positive(quote.upper_limit):
            return None
        assert quote.lower_limit is not None
        assert quote.upper_limit is not None
        if quote.lower_limit > quote.upper_limit:
            return None

        offset = tick_size * price_offset_ticks
        if side is OrderSide.BUY:
            base = quote.ask1 if _is_positive(quote.ask1) else quote.last_price
            assert base is not None
            raw_price = min(base + offset, quote.upper_limit)
            limit_price = (raw_price / tick_size).to_integral_value(
                rounding=ROUND_CEILING
            ) * tick_size
        else:
            base = quote.bid1 if _is_positive(quote.bid1) else quote.last_price
            assert base is not None
            raw_price = max(base - offset, quote.lower_limit)
            limit_price = (raw_price / tick_size).to_integral_value(
                rounding=ROUND_FLOOR
            ) * tick_size
        if not _is_positive(limit_price):
            return None
        if not quote.lower_limit <= limit_price <= quote.upper_limit:
            return None
        return NearClosePrice(valuation_price=quote.last_price, limit_price=limit_price)


__all__ = ["NearCloseLimitPolicy", "NearClosePrice"]
