"""Pure dynamic metrics derived from bt_daily and bt_trade projections.

The evaluator consumes immutable values corresponding to persisted result
columns.  It does not query or write a database, inspect strategies, or use
intraday bars.  Every return calculation is based on the daily total-asset
sequence and explicitly anchors its first observation to ``initial_cash``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Final

_ZERO: Final = Decimal("0")
_ONE: Final = Decimal("1")
_TRADING_DAYS_PER_YEAR: Final = Decimal("252")
_HALF: Final = Decimal("0.5")


class BacktestMetricsError(ValueError):
    """Raised when persisted result projections cannot produce valid metrics."""


def _non_negative_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if value < _ZERO:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class DailyMetricRow:
    """Read-only projection of one bt_daily row used by metrics."""

    trade_date: date
    cash: Decimal
    market_value: Decimal
    total_asset: Decimal

    def __post_init__(self) -> None:
        if isinstance(self.trade_date, datetime) or not isinstance(self.trade_date, date):
            raise TypeError("trade_date must be datetime.date")
        cash = _non_negative_decimal(self.cash, "cash")
        market_value = _non_negative_decimal(self.market_value, "market_value")
        total_asset = _non_negative_decimal(self.total_asset, "total_asset")
        if total_asset != cash + market_value:
            raise ValueError("total_asset must equal cash + market_value")


@dataclass(frozen=True, slots=True)
class TradeMetricRow:
    """Read-only projection of one formal fill used by cost metrics."""

    trade_amount: Decimal
    fee: Decimal
    base_trade_price: Decimal
    fill_price: Decimal
    fill_quantity: int

    def __post_init__(self) -> None:
        amount = _non_negative_decimal(self.trade_amount, "trade_amount")
        _non_negative_decimal(self.fee, "fee")
        base_price = _non_negative_decimal(self.base_trade_price, "base_trade_price")
        fill_price = _non_negative_decimal(self.fill_price, "fill_price")
        if amount <= _ZERO:
            raise ValueError("trade_amount must be positive")
        if base_price <= _ZERO or fill_price <= _ZERO:
            raise ValueError("trade prices must be positive")
        if type(self.fill_quantity) is not int or self.fill_quantity <= 0:
            raise ValueError("fill_quantity must be a positive int")
        if amount != fill_price * self.fill_quantity:
            raise ValueError("trade_amount must equal fill_price * fill_quantity")

    @property
    def slippage_cost(self) -> Decimal:
        """Return adverse price movement relative to the raw close quote."""

        return abs(self.fill_price - self.base_trade_price) * self.fill_quantity


@dataclass(frozen=True, slots=True)
class BacktestMetricResult:
    """Complete dynamically calculated backtest metric set."""

    daily_returns: MappingProxyType[date, Decimal]
    cumulative_return: Decimal
    annualized_return: Decimal
    max_drawdown: Decimal
    sharpe: Decimal
    sharpe_reason: str | None
    total_fee: Decimal
    total_slippage_cost: Decimal
    total_transaction_cost: Decimal
    transaction_cost_rate: Decimal
    trade_count: int
    turnover: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.daily_returns, MappingProxyType):
            raise TypeError("daily_returns must be an immutable mapping")
        if type(self.trade_count) is not int or self.trade_count < 0:
            raise ValueError("trade_count must be a non-negative int")


class BacktestMetrics:
    """Calculate fixed-formula metrics without persistence side effects."""

    @staticmethod
    def calculate(
        *,
        initial_cash: Decimal,
        daily_rows: Sequence[DailyMetricRow],
        trade_rows: Sequence[TradeMetricRow],
    ) -> BacktestMetricResult:
        """Return daily-return, risk, fee, count, and turnover metrics."""

        initial = _non_negative_decimal(initial_cash, "initial_cash")
        if initial <= _ZERO:
            raise ValueError("initial_cash must be positive")
        if not isinstance(daily_rows, Sequence):
            raise TypeError("daily_rows must be a sequence")
        if not isinstance(trade_rows, Sequence):
            raise TypeError("trade_rows must be a sequence")
        if any(not isinstance(row, DailyMetricRow) for row in daily_rows):
            raise TypeError("daily_rows may contain only DailyMetricRow")
        if any(not isinstance(row, TradeMetricRow) for row in trade_rows):
            raise TypeError("trade_rows may contain only TradeMetricRow")

        ordered_daily = tuple(sorted(daily_rows, key=lambda row: row.trade_date))
        if not ordered_daily:
            raise BacktestMetricsError("at least one bt_daily row is required")
        dates = tuple(row.trade_date for row in ordered_daily)
        if len(set(dates)) != len(dates):
            raise BacktestMetricsError("bt_daily trade_date values must be unique")

        previous_asset = initial
        daily_returns: dict[date, Decimal] = {}
        for row in ordered_daily:
            if previous_asset <= _ZERO:
                raise BacktestMetricsError("daily return is undefined after a zero total-asset day")
            daily_returns[row.trade_date] = row.total_asset / previous_asset - _ONE
            previous_asset = row.total_asset

        final_asset = ordered_daily[-1].total_asset
        cumulative_return = final_asset / initial - _ONE
        annualized_return = BacktestMetrics._annualized_return(
            final_asset=final_asset,
            initial_cash=initial,
            day_count=len(ordered_daily),
        )
        max_drawdown = BacktestMetrics._max_drawdown(
            initial_cash=initial,
            daily_rows=ordered_daily,
        )
        sharpe, sharpe_reason = BacktestMetrics._sharpe(tuple(daily_returns.values()))
        total_fee = sum((row.fee for row in trade_rows), start=_ZERO)
        total_slippage_cost = sum((row.slippage_cost for row in trade_rows), start=_ZERO)
        total_transaction_cost = total_fee + total_slippage_cost
        total_trade_amount = sum((row.trade_amount for row in trade_rows), start=_ZERO)
        transaction_cost_rate = (
            total_transaction_cost / total_trade_amount if total_trade_amount > _ZERO else _ZERO
        )
        trade_count = len(trade_rows)
        if trade_count == 0:
            turnover = _ZERO
        else:
            mean_asset = sum(
                (row.total_asset for row in ordered_daily),
                start=_ZERO,
            ) / len(ordered_daily)
            if mean_asset <= _ZERO:
                raise BacktestMetricsError("turnover is undefined when mean total_asset is zero")
            turnover = _HALF * total_trade_amount / mean_asset

        return BacktestMetricResult(
            daily_returns=MappingProxyType(daily_returns),
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe=sharpe,
            sharpe_reason=sharpe_reason,
            total_fee=total_fee,
            total_slippage_cost=total_slippage_cost,
            total_transaction_cost=total_transaction_cost,
            transaction_cost_rate=transaction_cost_rate,
            trade_count=trade_count,
            turnover=turnover,
        )

    @staticmethod
    def _annualized_return(
        *,
        final_asset: Decimal,
        initial_cash: Decimal,
        day_count: int,
    ) -> Decimal:
        ratio = final_asset / initial_cash
        if ratio == _ZERO:
            return -_ONE
        exponent = _TRADING_DAYS_PER_YEAR / day_count
        return (ratio.ln() * exponent).exp() - _ONE

    @staticmethod
    def _max_drawdown(
        *,
        initial_cash: Decimal,
        daily_rows: tuple[DailyMetricRow, ...],
    ) -> Decimal:
        running_max = initial_cash
        maximum = _ZERO
        for row in daily_rows:
            running_max = max(running_max, row.total_asset)
            drawdown = _ONE - row.total_asset / running_max
            maximum = max(maximum, drawdown)
        return maximum

    @staticmethod
    def _sharpe(daily_returns: tuple[Decimal, ...]) -> tuple[Decimal, str | None]:
        if len(daily_returns) < 2:
            return _ZERO, "fewer than two daily returns"
        mean_return = sum(daily_returns, start=_ZERO) / len(daily_returns)
        squared_deviations = sum(
            ((value - mean_return) ** 2 for value in daily_returns),
            start=_ZERO,
        )
        sample_variance = squared_deviations / (len(daily_returns) - 1)
        if sample_variance == _ZERO:
            return _ZERO, "daily return sample standard deviation is zero"
        sample_standard_deviation = sample_variance.sqrt()
        return (
            mean_return / sample_standard_deviation * _TRADING_DAYS_PER_YEAR.sqrt(),
            None,
        )


__all__ = [
    "BacktestMetricResult",
    "BacktestMetrics",
    "BacktestMetricsError",
    "DailyMetricRow",
    "TradeMetricRow",
]
