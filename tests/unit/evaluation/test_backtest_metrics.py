"""Fixed-formula and boundary tests for dynamic backtest metrics."""

from datetime import date
from decimal import Decimal
from types import MappingProxyType

import pytest

from etf_backtest.evaluation.backtest_metrics import (
    BacktestMetrics,
    BacktestMetricsError,
    DailyMetricRow,
    TradeMetricRow,
)


def _daily(day: int, total_asset: str) -> DailyMetricRow:
    total = Decimal(total_asset)
    return DailyMetricRow(
        trade_date=date(2025, 1, day),
        cash=total,
        market_value=Decimal("0"),
        total_asset=total,
    )


@pytest.mark.unit
def test_fixed_return_drawdown_sharpe_fee_and_turnover_formulas() -> None:
    daily_rows = (
        _daily(2, "110"),
        _daily(3, "99"),
        _daily(4, "120"),
    )
    trades = (
        TradeMetricRow(
            trade_amount=Decimal("50"),
            fee=Decimal("1"),
            base_trade_price=Decimal("0.49"),
            fill_price=Decimal("0.50"),
            fill_quantity=100,
        ),
        TradeMetricRow(
            trade_amount=Decimal("50"),
            fee=Decimal("2"),
            base_trade_price=Decimal("0.52"),
            fill_price=Decimal("0.50"),
            fill_quantity=100,
        ),
    )

    result = BacktestMetrics.calculate(
        initial_cash=Decimal("100"),
        daily_rows=daily_rows,
        trade_rows=trades,
    )

    assert isinstance(result.daily_returns, MappingProxyType)
    assert result.daily_returns[date(2025, 1, 2)] == Decimal("0.1")
    assert result.daily_returns[date(2025, 1, 3)] == Decimal("-0.1")
    assert result.daily_returns[date(2025, 1, 4)] == Decimal("120") / Decimal("99") - 1
    assert result.cumulative_return == Decimal("0.2")
    expected_annualized = (Decimal("1.2").ln() * (Decimal("252") / Decimal("3"))).exp() - Decimal(
        "1"
    )
    assert result.annualized_return == expected_annualized
    assert result.max_drawdown == Decimal("0.1")
    assert result.sharpe != Decimal("0")
    assert result.sharpe_reason is None
    assert result.total_fee == Decimal("3")
    assert result.total_slippage_cost == Decimal("3")
    assert result.total_transaction_cost == Decimal("6")
    assert result.transaction_cost_rate == Decimal("0.06")
    assert result.trade_count == 2
    assert abs(result.turnover - Decimal("150") / Decimal("329")) < Decimal("1e-27")


@pytest.mark.unit
def test_zero_daily_rows_fail_explicitly() -> None:
    with pytest.raises(BacktestMetricsError, match="at least one"):
        BacktestMetrics.calculate(
            initial_cash=Decimal("100"),
            daily_rows=(),
            trade_rows=(),
        )


@pytest.mark.unit
def test_one_day_return_is_valid_and_sharpe_is_zero() -> None:
    result = BacktestMetrics.calculate(
        initial_cash=Decimal("100"),
        daily_rows=(_daily(2, "105"),),
        trade_rows=(),
    )

    assert result.daily_returns == {date(2025, 1, 2): Decimal("0.05")}
    assert result.cumulative_return == Decimal("0.05")
    assert result.sharpe == Decimal("0")
    assert result.sharpe_reason == "fewer than two daily returns"


@pytest.mark.unit
def test_zero_volatility_sharpe_is_zero_with_reason() -> None:
    result = BacktestMetrics.calculate(
        initial_cash=Decimal("100"),
        daily_rows=(
            _daily(2, "100"),
            _daily(3, "100"),
            _daily(4, "100"),
        ),
        trade_rows=(),
    )

    assert result.sharpe == Decimal("0")
    assert result.sharpe_reason == "daily return sample standard deviation is zero"


@pytest.mark.unit
def test_no_trades_have_zero_fee_count_and_turnover() -> None:
    result = BacktestMetrics.calculate(
        initial_cash=Decimal("100"),
        daily_rows=(_daily(2, "100"), _daily(3, "101")),
        trade_rows=(),
    )

    assert result.total_fee == Decimal("0")
    assert result.total_slippage_cost == Decimal("0")
    assert result.total_transaction_cost == Decimal("0")
    assert result.transaction_cost_rate == Decimal("0")
    assert result.trade_count == 0
    assert result.turnover == Decimal("0")


@pytest.mark.unit
def test_initial_cash_is_first_drawdown_anchor() -> None:
    result = BacktestMetrics.calculate(
        initial_cash=Decimal("100"),
        daily_rows=(_daily(2, "80"), _daily(3, "90")),
        trade_rows=(),
    )

    assert result.max_drawdown == Decimal("0.2")


@pytest.mark.unit
def test_duplicate_daily_date_is_rejected() -> None:
    with pytest.raises(BacktestMetricsError, match="unique"):
        BacktestMetrics.calculate(
            initial_cash=Decimal("100"),
            daily_rows=(_daily(2, "100"), _daily(2, "101")),
            trade_rows=(),
        )
