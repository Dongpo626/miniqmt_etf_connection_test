"""Immutable T+0/T+1 ETF holding state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import FillResult, OrderSide


def _quantity(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class Position:
    """One registered ETF position with explicit sellable-share buckets."""

    symbol: str
    turnover_rule: TurnoverRule
    total_quantity: int = 0
    available_quantity: int = 0
    today_buy_quantity: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.turnover_rule, TurnoverRule):
            raise TypeError("turnover_rule must be TurnoverRule")
        total = _quantity(self.total_quantity, "total_quantity")
        available = _quantity(self.available_quantity, "available_quantity")
        today = _quantity(self.today_buy_quantity, "today_buy_quantity")
        if available > total or today > total:
            raise ValueError("position buckets cannot exceed total_quantity")
        if self.turnover_rule is TurnoverRule.T0:
            if available != total or today != 0:
                raise ValueError("T0 requires all shares immediately available")
        elif available + today != total:
            raise ValueError("T1 available and today buckets must sum to total")

    def apply_buy(self, fill: FillResult) -> Position:
        if not isinstance(fill, FillResult):
            raise TypeError("position updates require a formal FillResult")
        if fill.side is not OrderSide.BUY or fill.symbol != self.symbol:
            raise ValueError("BUY fill does not match the position")
        total = self.total_quantity + fill.fill_quantity
        if self.turnover_rule is TurnoverRule.T0:
            return replace(
                self,
                total_quantity=total,
                available_quantity=self.available_quantity + fill.fill_quantity,
            )
        return replace(
            self,
            total_quantity=total,
            today_buy_quantity=self.today_buy_quantity + fill.fill_quantity,
        )

    def apply_sell(self, fill: FillResult) -> Position:
        if not isinstance(fill, FillResult):
            raise TypeError("position updates require a formal FillResult")
        if fill.side is not OrderSide.SELL or fill.symbol != self.symbol:
            raise ValueError("SELL fill does not match the position")
        if fill.fill_quantity > self.available_quantity:
            raise ValueError("sell quantity exceeds available_quantity")
        return replace(
            self,
            total_quantity=self.total_quantity - fill.fill_quantity,
            available_quantity=self.available_quantity - fill.fill_quantity,
        )

    def on_new_trade_date(self) -> Position:
        """Release T+1 purchases exactly once at the next engine date."""

        if self.turnover_rule is TurnoverRule.T0 or self.today_buy_quantity == 0:
            return self
        return replace(
            self,
            available_quantity=self.available_quantity + self.today_buy_quantity,
            today_buy_quantity=0,
        )

    def market_value(self, raw_close: Decimal) -> Decimal:
        if not isinstance(raw_close, Decimal):
            raise TypeError("raw_close must be Decimal")
        if not raw_close.is_finite() or raw_close <= 0:
            raise ValueError("raw_close must be finite and positive")
        return raw_close * self.total_quantity

    @property
    def is_empty(self) -> bool:
        return self.total_quantity == 0


__all__ = ["Position"]
