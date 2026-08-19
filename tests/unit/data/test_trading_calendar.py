"""Frozen natural-date SSE daily calendar contract."""

from __future__ import annotations

from datetime import date

import pytest

from etf_backtest.data.calendar import CalendarCoverageError, SseTradingCalendar
from etf_backtest.data.mysql import QmtSseCalendarDay

THURSDAY = date(2025, 1, 2)
FRIDAY = date(2025, 1, 3)
SATURDAY = date(2025, 1, 4)
SUNDAY = date(2025, 1, 5)
MONDAY = date(2025, 1, 6)
VERSION = "qmt-calendar-v1"


def _day(
    value: date,
    *,
    is_open: bool,
    previous_open: date,
    next_open: date,
    version: str = VERSION,
) -> QmtSseCalendarDay:
    return QmtSseCalendarDay(
        cal_date=value,
        is_open=is_open,
        previous_open_date=previous_open,
        next_open_date=next_open,
        source_system="AKSHARE_SINA",
        calendar_version=version,
    )


def _days() -> tuple[QmtSseCalendarDay, ...]:
    return (
        _day(
            THURSDAY,
            is_open=True,
            previous_open=date(2024, 12, 31),
            next_open=FRIDAY,
        ),
        _day(FRIDAY, is_open=True, previous_open=THURSDAY, next_open=MONDAY),
        _day(SATURDAY, is_open=False, previous_open=FRIDAY, next_open=MONDAY),
        _day(SUNDAY, is_open=False, previous_open=FRIDAY, next_open=MONDAY),
        _day(
            MONDAY,
            is_open=True,
            previous_open=FRIDAY,
            next_open=date(2025, 1, 7),
        ),
    )


@pytest.mark.unit
def test_calendar_uses_loaded_open_flags_and_sse_for_all_policy() -> None:
    calendar = SseTradingCalendar(_days())

    assert calendar.calendar_policy == "SSE_FOR_ALL"
    assert calendar.open_dates == (THURSDAY, FRIDAY, MONDAY)
    assert calendar.next_trading_day(FRIDAY) == MONDAY
    assert calendar.previous_trading_day(MONDAY) == FRIDAY
    assert calendar.trading_dates(THURSDAY, MONDAY) == (THURSDAY, FRIDAY, MONDAY)
    assert calendar.frame_key(MONDAY).close_time.hour == 15


@pytest.mark.unit
def test_closed_and_out_of_coverage_dates_fail_explicitly() -> None:
    calendar = SseTradingCalendar(_days())

    assert not calendar.is_trading_day(SATURDAY)
    with pytest.raises(ValueError, match="not an SSE trading date"):
        calendar.frame_key(SATURDAY)
    with pytest.raises(CalendarCoverageError, match="outside SSE calendar"):
        calendar.day(date(2025, 1, 7))
    with pytest.raises(CalendarCoverageError, match="no next"):
        calendar.next_trading_day(MONDAY)


@pytest.mark.unit
def test_incomplete_natural_dates_and_mixed_versions_are_rejected() -> None:
    incomplete = tuple(day for day in _days() if day.cal_date != SATURDAY)
    with pytest.raises(ValueError, match="incomplete SSE natural-date calendar"):
        SseTradingCalendar(incomplete)

    mixed = (
        *_days()[:-1],
        _day(
            MONDAY,
            is_open=True,
            previous_open=FRIDAY,
            next_open=date(2025, 1, 7),
            version="other",
        ),
    )
    with pytest.raises(ValueError, match="source and version must be uniform"):
        SseTradingCalendar(mixed)
