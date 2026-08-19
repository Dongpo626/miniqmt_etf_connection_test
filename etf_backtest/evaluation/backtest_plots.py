"""Render full-period charts after a daily backtest has completed."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from io import BytesIO
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from etf_backtest.evaluation.backtest_metrics import DailyMetricRow

_ZERO = Decimal("0")


def _ordered_rows(daily_rows: Sequence[DailyMetricRow]) -> tuple[DailyMetricRow, ...]:
    if not isinstance(daily_rows, Sequence):
        raise TypeError("daily_rows must be a sequence")
    if any(not isinstance(row, DailyMetricRow) for row in daily_rows):
        raise TypeError("daily_rows may contain only DailyMetricRow")
    ordered = tuple(sorted(daily_rows, key=lambda row: row.trade_date))
    if not ordered:
        raise ValueError("at least one daily row is required for plotting")
    dates = tuple(row.trade_date for row in ordered)
    if len(dates) != len(set(dates)):
        raise ValueError("daily row dates must be unique")
    return ordered


def _new_figure(*, title: str, ylabel: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(11, 4.8), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_title(title, fontsize=14, pad=12)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    locator = AutoDateLocator(minticks=3, maxticks=9)  # type: ignore[no-untyped-call]
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(
        ConciseDateFormatter(locator)  # type: ignore[no-untyped-call]
    )
    axis.margins(x=0.01)
    return figure, axis


def _png_bytes(figure: Figure) -> bytes:
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, facecolor="white")
    return output.getvalue()


def render_backtest_plots(
    *,
    initial_cash: Decimal,
    daily_rows: Sequence[DailyMetricRow],
) -> dict[str, bytes]:
    """Return the three final charts for the complete backtest period."""

    if not isinstance(initial_cash, Decimal):
        raise TypeError("initial_cash must be Decimal")
    if not initial_cash.is_finite() or initial_cash <= _ZERO:
        raise ValueError("initial_cash must be finite and positive")
    ordered = _ordered_rows(daily_rows)
    dates = tuple(row.trade_date for row in ordered)
    plot_dates: Any = dates  # Matplotlib accepts dates although its stubs omit them.

    cumulative_returns = tuple(float(row.total_asset / initial_cash - 1) for row in ordered)
    cumulative_figure, cumulative_axis = _new_figure(
        title="Cumulative Return",
        ylabel="Return",
    )
    cumulative_axis.axhline(0, color="#666666", linewidth=0.8)
    cumulative_axis.plot(
        plot_dates,
        cumulative_returns,
        color="#0072B2",
        linewidth=1.8,
    )
    cumulative_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    running_peak = initial_cash
    drawdowns: list[float] = []
    for row in ordered:
        running_peak = max(running_peak, row.total_asset)
        drawdowns.append(float(row.total_asset / running_peak - 1))
    drawdown_figure, drawdown_axis = _new_figure(title="Drawdown", ylabel="Drawdown")
    drawdown_axis.plot(
        plot_dates,
        drawdowns,
        color="#D55E00",
        linewidth=1.6,
    )
    drawdown_axis.fill_between(
        plot_dates,
        drawdowns,
        0,
        color="#D55E00",
        alpha=0.18,
    )
    drawdown_axis.axhline(0, color="#666666", linewidth=0.8)
    drawdown_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    cash_values = tuple(float(row.cash) for row in ordered)
    cash_figure, cash_axis = _new_figure(title="Cash", ylabel="Cash Amount")
    cash_axis.plot(
        plot_dates,
        cash_values,
        color="#009E73",
        linewidth=1.8,
    )
    cash_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    return {
        "cumulative_return.png": _png_bytes(cumulative_figure),
        "drawdown.png": _png_bytes(drawdown_figure),
        "cash.png": _png_bytes(cash_figure),
    }


__all__ = ["render_backtest_plots"]
