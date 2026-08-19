"""Stable strategy schedule indexes derived from the SSE trading calendar."""

from __future__ import annotations

from datetime import date
from typing import Protocol


class _SseCalendar(Protocol):
    def is_trading_day(self, trade_date: date) -> bool: ...

    def require_trading_day(self, trade_date: date) -> date: ...

    def trading_dates(self, start_date: date, end_date: date) -> tuple[date, ...]: ...

    def previous_trading_day(self, trade_date: date) -> date: ...

    def next_trading_day(self, trade_date: date) -> date: ...


class TradingDayIndexResolver:
    """Resolve a stable zero-based index from an immutable trading-day anchor."""

    def resolve(
        self,
        *,
        calendar: _SseCalendar,
        anchor_date: date,
        signal_date: date,
    ) -> int:
        anchor = calendar.require_trading_day(anchor_date)
        signal = calendar.require_trading_day(signal_date)
        calendar.next_trading_day(signal)
        return len(calendar.trading_dates(anchor, signal)) - 1

    def is_trading_day(self, *, calendar: _SseCalendar, trade_date: date) -> bool:
        return calendar.is_trading_day(trade_date)

    def previous_trading_day(self, *, calendar: _SseCalendar, trade_date: date) -> date:
        return calendar.previous_trading_day(trade_date)

    def next_trading_day(self, *, calendar: _SseCalendar, trade_date: date) -> date:
        return calendar.next_trading_day(trade_date)


__all__ = ["TradingDayIndexResolver"]
