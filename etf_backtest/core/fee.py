"""Pure Decimal fee calculation for each individual formal fill."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Final

from etf_backtest.config.schema import FeeConfig
from etf_backtest.core.order import OrderSide

_ZERO: Final = Decimal("0")
_MONEY_QUANTUM: Final = Decimal("0.001")


def _quantize_money(value: Decimal) -> Decimal:
    """Return a fee amount at the shared financial result precision."""

    return value.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _require_non_negative_money(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value < _ZERO:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """Immutable per-fill fee components."""

    commission: Decimal
    stamp_duty: Decimal
    total: Decimal

    def __post_init__(self) -> None:
        commission = _require_non_negative_money(self.commission, "commission")
        stamp_duty = _require_non_negative_money(self.stamp_duty, "stamp_duty")
        total = _require_non_negative_money(self.total, "total")
        if total != commission + stamp_duty:
            raise ValueError("fee total is inconsistent with its components")


class FeeModel:
    """Calculate commission and direction-specific fees per trade."""

    __slots__ = ("_config",)

    def __init__(self, config: FeeConfig) -> None:
        if not isinstance(config, FeeConfig):
            raise TypeError("config must be FeeConfig")
        self._config = config

    def calculate(
        self,
        *,
        trade_amount: Decimal,
        side: OrderSide,
    ) -> FeeBreakdown:
        """Return one independent fee breakdown for a candidate fill."""
        amount = _require_non_negative_money(trade_amount, "trade_amount")
        if not isinstance(side, OrderSide):
            raise TypeError("side must be OrderSide")
        if amount == _ZERO:
            return FeeBreakdown(
                commission=_ZERO,
                stamp_duty=_ZERO,
                total=_ZERO,
            )

        commission = _quantize_money(
            max(
                amount * self._config.commission_rate,
                self._config.minimum_commission,
            )
        )
        stamp_duty = (
            _quantize_money(amount * self._config.stamp_duty_rate)
            if side is OrderSide.SELL
            else _ZERO
        )
        return FeeBreakdown(
            commission=commission,
            stamp_duty=stamp_duty,
            total=commission + stamp_duty,
        )


__all__ = ["FeeBreakdown", "FeeModel"]
