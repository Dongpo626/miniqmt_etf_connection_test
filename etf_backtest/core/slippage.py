"""Direction-aware proportional close-price slippage."""

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from etf_backtest.config.schema import SlippageConfig
from etf_backtest.core.order import OrderSide


def _positive_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return value


def _is_tick_aligned(value: Decimal, tick_size: Decimal) -> bool:
    return value / tick_size == (value / tick_size).to_integral_value()


class SlippageModel:
    """Move a raw close against the order and stay inside legal daily limits."""

    __slots__ = ("_rate",)

    def __init__(self, config: SlippageConfig) -> None:
        if not isinstance(config, SlippageConfig):
            raise TypeError("config must be SlippageConfig")
        self._rate = config.rate

    @property
    def rate(self) -> Decimal:
        return self._rate

    def apply(
        self,
        *,
        base_trade_price: Decimal,
        side: OrderSide,
        tick_size: Decimal,
        price_limit_down: Decimal,
        price_limit_up: Decimal,
    ) -> Decimal:
        """Return a conservative tick price capped only by the legal limit.

        The raw bar high and low are deliberately absent from this interface.
        They are observations, not legal execution boundaries for proportional
        close-price slippage.
        """

        price = _positive_decimal(base_trade_price, "base_trade_price")
        tick = _positive_decimal(tick_size, "tick_size")
        lower = _positive_decimal(price_limit_down, "price_limit_down")
        upper = _positive_decimal(price_limit_up, "price_limit_up")
        if not isinstance(side, OrderSide):
            raise TypeError("side must be OrderSide")
        if not lower <= price <= upper:
            raise ValueError("base_trade_price must be inside the legal price limits")
        if not _is_tick_aligned(lower, tick) or not _is_tick_aligned(upper, tick):
            raise ValueError("legal price limits must be tick aligned")

        if side is OrderSide.BUY:
            adjusted = price * (Decimal("1") + self._rate)
            rounded = (adjusted / tick).to_integral_value(rounding=ROUND_CEILING) * tick
            return min(rounded, upper)

        adjusted = price * (Decimal("1") - self._rate)
        rounded = (adjusted / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        return max(_positive_decimal(rounded, "fill_price"), lower)


__all__ = ["SlippageModel"]
