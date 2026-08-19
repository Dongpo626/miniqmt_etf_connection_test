from datetime import datetime, timedelta
from decimal import Decimal

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.order import OrderSide
from etf_backtest.live.execution.near_close_limit import NearCloseLimitPolicy
from etf_backtest.live.state import LiveQuote


def _quote(*, suspended: bool = False, quoted_at: datetime | None = None) -> LiveQuote:
    return LiveQuote(
        symbol="510300.SH",
        last_price=Decimal("10.000"),
        bid1=Decimal("9.991"),
        ask1=Decimal("10.009"),
        lower_limit=Decimal("9.980"),
        upper_limit=Decimal("10.020"),
        suspended=suspended,
        quoted_at=quoted_at
        or datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE),
    )


def test_prices_use_same_valuation_and_side_quotes_with_tick_and_limit_caps() -> None:
    now = datetime(2026, 8, 19, 14, 50, 1, tzinfo=MARKET_TIMEZONE)
    policy = NearCloseLimitPolicy()
    buy = policy.calculate(
        side=OrderSide.BUY,
        quote=_quote(),
        tick_size=Decimal("0.001"),
        price_offset_ticks=20,
        now=now,
        quote_stale_seconds=5,
    )
    sell = policy.calculate(
        side=OrderSide.SELL,
        quote=_quote(),
        tick_size=Decimal("0.001"),
        price_offset_ticks=20,
        now=now,
        quote_stale_seconds=5,
    )

    assert buy is not None and sell is not None
    assert buy.valuation_price == sell.valuation_price == Decimal("10.000")
    assert buy.limit_price == Decimal("10.020")
    assert sell.limit_price == Decimal("9.980")


def test_stale_or_suspended_quote_has_no_price_result() -> None:
    now = datetime(2026, 8, 19, 14, 51, tzinfo=MARKET_TIMEZONE)
    policy = NearCloseLimitPolicy()
    inputs = (
        _quote(quoted_at=now - timedelta(seconds=6)),
        _quote(suspended=True, quoted_at=now),
    )

    for quote in inputs:
        assert (
            policy.calculate(
                side=OrderSide.BUY,
                quote=quote,
                tick_size=Decimal("0.001"),
                price_offset_ticks=2,
                now=now,
                quote_stale_seconds=5,
            )
            is None
        )
