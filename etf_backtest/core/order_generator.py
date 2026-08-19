"""Convert a complete target portfolio into next-close quantity requests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import AccountSnapshot
from etf_backtest.core.order import Order, OrderSide
from etf_backtest.core.target import TargetPortfolio


class OrderGenerationError(ValueError):
    """Report a missing valuation input at the target-to-order boundary."""


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _order_id(*, signal_date: date, execution_date: date, symbol: str, side: OrderSide) -> str:
    payload = f"{signal_date.isoformat()}|{execution_date.isoformat()}|{symbol}|{side.value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class OrderGenerator:
    """Generate deterministic requests; only the rule engine may resize them."""

    lot_size: int = 100

    def __post_init__(self) -> None:
        if type(self.lot_size) is not int or self.lot_size <= 0:
            raise ValueError("lot_size must be a positive integer")

    def generate(
        self,
        *,
        target_portfolio: TargetPortfolio,
        valuation_snapshot: AccountSnapshot,
        signal_date: date,
        execution_date: date,
    ) -> tuple[Order, ...]:
        if not isinstance(target_portfolio, TargetPortfolio):
            raise TypeError("target_portfolio must be TargetPortfolio")
        if not isinstance(valuation_snapshot, AccountSnapshot):
            raise TypeError("valuation_snapshot must be AccountSnapshot")
        signal = _plain_date(signal_date, "signal_date")
        execution = _plain_date(execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")

        symbols = set(target_portfolio.weights)
        symbols.update(
            symbol
            for symbol, position in valuation_snapshot.positions.items()
            if position.total_quantity > 0
        )
        missing = sorted(symbols - set(valuation_snapshot.mark_close_prices))
        if missing:
            raise OrderGenerationError("valuation snapshot lacks prices for: " + ", ".join(missing))

        orders: list[Order] = []
        for supplied_symbol in sorted(symbols):
            symbol = normalize_symbol(supplied_symbol)
            price = valuation_snapshot.mark_close_prices[symbol]
            current_value = valuation_snapshot.position_values.get(symbol, Decimal("0"))
            target_value = valuation_snapshot.total_asset * target_portfolio.weight_for(symbol)
            gap = target_value - current_value
            shares = int((abs(gap) / price).to_integral_value(rounding=ROUND_DOWN))
            requested = (shares // self.lot_size) * self.lot_size
            if requested == 0:
                continue
            side = OrderSide.BUY if gap > 0 else OrderSide.SELL
            orders.append(
                Order(
                    order_id=_order_id(
                        signal_date=signal,
                        execution_date=execution,
                        symbol=symbol,
                        side=side,
                    ),
                    signal_date=signal,
                    execution_date=execution,
                    symbol=symbol,
                    side=side,
                    requested_quantity=requested,
                    target_value_gap=gap,
                )
            )
        return tuple(sorted(orders, key=lambda item: (item.side is OrderSide.BUY, item.symbol)))


__all__ = ["OrderGenerationError", "OrderGenerator"]
