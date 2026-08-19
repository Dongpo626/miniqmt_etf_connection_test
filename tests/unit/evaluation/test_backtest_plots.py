"""Basic coverage for final full-period backtest charts."""

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.evaluation.backtest_metrics import DailyMetricRow
from etf_backtest.evaluation.backtest_plots import render_backtest_plots


@pytest.mark.unit
def test_render_backtest_plots_returns_three_png_files() -> None:
    rows = (
        DailyMetricRow(date(2024, 1, 2), Decimal("6000"), Decimal("4000"), Decimal("10000")),
        DailyMetricRow(date(2024, 1, 3), Decimal("5500"), Decimal("4700"), Decimal("10200")),
        DailyMetricRow(date(2024, 1, 4), Decimal("5300"), Decimal("4600"), Decimal("9900")),
    )

    plots = render_backtest_plots(initial_cash=Decimal("10000"), daily_rows=rows)

    assert set(plots) == {"cumulative_return.png", "drawdown.png", "cash.png"}
    assert all(content.startswith(b"\x89PNG\r\n\x1a\n") for content in plots.values())
