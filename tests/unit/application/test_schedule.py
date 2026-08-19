"""Stable SSE trading-day schedule index behavior."""

from __future__ import annotations

from datetime import date

import pytest

from etf_backtest.application.schedule import TradingDayIndexResolver


class StubCalendar:
    def __init__(self, trading_dates: tuple[date, ...]) -> None:
        self._trading_dates = trading_dates

    def is_trading_day(self, trade_date: date) -> bool:
        return trade_date in self._trading_dates

    def require_trading_day(self, trade_date: date) -> date:
        if not self.is_trading_day(trade_date):
            raise ValueError("not a trading day")
        return trade_date

    def trading_dates(self, start_date: date, end_date: date) -> tuple[date, ...]:
        if start_date > end_date:
            raise ValueError("invalid range")
        return tuple(value for value in self._trading_dates if start_date <= value <= end_date)

    def previous_trading_day(self, trade_date: date) -> date:
        return self._trading_dates[self._trading_dates.index(trade_date) - 1]

    def next_trading_day(self, trade_date: date) -> date:
        return self._trading_dates[self._trading_dates.index(trade_date) + 1]


@pytest.mark.unit
def test_fixed_anchor_produces_stable_schedule_index_and_adjacent_days() -> None:
    trading_dates = (
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 5),
    )
    calendar = StubCalendar(trading_dates)
    resolver = TradingDayIndexResolver()

    assert (
        resolver.resolve(
            calendar=calendar,
            anchor_date=trading_dates[0],
            signal_date=trading_dates[2],
        )
        == 2
    )
    assert resolver.is_trading_day(calendar=calendar, trade_date=trading_dates[2])
    assert (
        resolver.previous_trading_day(calendar=calendar, trade_date=trading_dates[2])
        == trading_dates[1]
    )
    assert (
        resolver.next_trading_day(calendar=calendar, trade_date=trading_dates[2])
        == trading_dates[3]
    )
