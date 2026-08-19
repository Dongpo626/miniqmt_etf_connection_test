"""Pure daily SSE calendar and non-leaking DataPortal tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time
from decimal import Decimal

import pytest

from etf_backtest.core.market import EtfInfo, Exchange, IndexBarView
from etf_backtest.data.calendar import CalendarCoverageError, SseTradingCalendar
from etf_backtest.data.mysql import (
    HuijinHolderRatioRecord,
    QmtDailyDataset,
    QmtDailyFrame,
    QmtEtfMasterRecord,
    QmtEtfShareRecord,
    QmtFrontDailyBar,
    QmtRawDailyBar,
    QmtSseCalendarDay,
)
from etf_backtest.data.portal import DailyDataPortal, DataQualityError

FIRST = date(2024, 1, 2)
SECOND = date(2024, 1, 3)
VERSION = "qmt-20260803-sha"


def _calendar_day(trade_date: date) -> QmtSseCalendarDay:
    if trade_date == FIRST:
        previous, following = date(2023, 12, 29), SECOND
    else:
        previous, following = FIRST, date(2024, 1, 4)
    return QmtSseCalendarDay(
        cal_date=trade_date,
        is_open=True,
        previous_open_date=previous,
        next_open_date=following,
        source_system="AKSHARE_SINA",
        calendar_version=VERSION,
    )


def _master(symbol: str = "SZ.159919") -> QmtEtfMasterRecord:
    code = symbol.partition(".")[2]
    return QmtEtfMasterRecord(
        symbol=symbol,
        etf_code=code,
        qmt_symbol=f"{code}.SZ",
        exchange=Exchange.SZSE,
        name="ETF",
        list_date=date(2012, 5, 28),
        delist_date=None,
        current_status="LISTED",
        primary_category="纯境内",
        fund_type="股票型",
        etf_type="纯境内",
        source_system="TUSHARE",
    )


def _info(symbol: str = "SZ.159919") -> EtfInfo:
    return EtfInfo(
        symbol=symbol,
        exchange=Exchange.SZSE,
        name="ETF",
        primary_category="纯境内",
        fund_type="股票型",
        list_date=date(2012, 5, 28),
        delist_date=None,
        current_status="LISTED",
    )


def _frame(trade_date: date, *, symbol: str = "SZ.159919") -> QmtDailyFrame:
    paired_key = f"QMT:{VERSION}:{symbol.partition('.')[2]}:{trade_date.isoformat()}"
    raw = QmtRawDailyBar(
        source_record_key=paired_key,
        symbol=symbol,
        trade_date=trade_date,
        bar_start_time=datetime.combine(trade_date, time(9, 30)).astimezone(),
        bar_end_time=datetime.combine(trade_date, time(15)).astimezone(),
        open=Decimal("4.0"),
        high=Decimal("4.2"),
        low=Decimal("3.8"),
        close=Decimal("4.1"),
        pre_close=Decimal("3.9"),
        volume=1000,
        amount=Decimal("4100"),
        source_system="QMT",
    )
    front = QmtFrontDailyBar(
        source_record_key=paired_key,
        symbol=symbol,
        trade_date=trade_date,
        open=Decimal("4.0"),
        high=Decimal("4.2"),
        low=Decimal("3.8"),
        close=Decimal("4.1"),
        source_system="QMT",
    )
    return QmtDailyFrame(
        trade_date=trade_date,
        bar_start_time=raw.bar_start_time,
        bar_end_time=raw.bar_end_time,
        raw_by_symbol={symbol: raw},
        front_by_symbol={symbol: front},
    )


def _dataset(*, frames: tuple[QmtDailyFrame, ...] | None = None) -> QmtDailyDataset:
    symbol = "SZ.159919"
    return QmtDailyDataset(
        symbols=(symbol,),
        start_date=FIRST,
        end_date=SECOND,
        dataset_version=VERSION,
        calendar_source="AKSHARE_SINA",
        calendar=(_calendar_day(FIRST), _calendar_day(SECOND)),
        etf_master=(_master(),),
        etf_infos=(_info(),),
        frames=frames or (_frame(FIRST), _frame(SECOND)),
    )


@pytest.mark.unit
def test_sse_calendar_drives_both_exchange_symbols_and_strict_next_day() -> None:
    calendar = SseTradingCalendar((_calendar_day(FIRST), _calendar_day(SECOND)))

    assert calendar.calendar_policy == "SSE_FOR_ALL"
    assert calendar.next_trading_day(FIRST) == SECOND
    assert calendar.frame_key(FIRST).close_time.hour == 15
    with pytest.raises(CalendarCoverageError, match="no next"):
        calendar.next_trading_day(SECOND)


@pytest.mark.unit
def test_portal_exposes_raw_frames_but_date_gates_front_history() -> None:
    portal = DailyDataPortal(_dataset())

    first_frame = portal.raw_frame(FIRST)
    assert first_frame is not None
    assert first_frame.bar_for("159919.SZ").source_record_key == (
        "QMT:qmt-20260803-sha:159919:2024-01-02"
    )
    assert [view.trade_date for view in portal.views_through(FIRST)] == [FIRST]
    assert [view.trade_date for view in portal.views_through(SECOND)] == [FIRST, SECOND]
    assert [view.trade_date for view in portal.views_through(SECOND, lookback_trading_days=1)] == [
        SECOND
    ]
    next_frame = portal.next_execution_frame(FIRST)
    assert next_frame is not None and next_frame.trade_date == SECOND


@pytest.mark.unit
def test_portal_rejects_missing_active_daily_coverage() -> None:
    dataset = _dataset(frames=(_frame(FIRST),))

    with pytest.raises(DataQualityError, match="active daily coverage"):
        DailyDataPortal(dataset)


@pytest.mark.unit
def test_history_symbol_filter_rejects_outside_universe() -> None:
    portal = DailyDataPortal(_dataset())

    with pytest.raises(ValueError, match="outside the universe"):
        portal.views_through(FIRST, symbols=["SH.510300"])


@pytest.mark.unit
def test_portal_gates_daily_share_and_selects_latest_ratio_strictly_before_signal() -> None:
    symbol = "SZ.159919"
    dataset = replace(
        _dataset(),
        share_records=(
            QmtEtfShareRecord(
                symbol=symbol,
                asof_date=FIRST,
                total_share=Decimal("100.0000"),
                source_system="TUSHARE",
            ),
            QmtEtfShareRecord(
                symbol=symbol,
                asof_date=SECOND,
                total_share=Decimal("110.0000"),
                source_system="TUSHARE",
            ),
        ),
        huijin_ratio_records=(
            HuijinHolderRatioRecord(
                symbol=symbol,
                end_date=date(2023, 12, 31),
                entity="中央汇金投资有限责任公司",
                ratio=Decimal("0.0900"),
            ),
            HuijinHolderRatioRecord(
                symbol=symbol,
                end_date=date(2023, 12, 31),
                entity="中央汇金资产管理有限责任公司",
                ratio=Decimal("0.0100"),
            ),
            HuijinHolderRatioRecord(
                symbol=symbol,
                end_date=FIRST,
                entity="中央汇金投资有限责任公司",
                ratio=Decimal("0.1000"),
            ),
        ),
    )
    portal = DailyDataPortal(dataset)

    first_shares = portal.share_history_through(FIRST)
    assert dict(first_shares[symbol]) == {FIRST: Decimal("100.0000")}
    assert portal.huijin_ratios_as_of(FIRST)[symbol]["中央汇金投资有限责任公司"] == (
        date(2023, 12, 31),
        Decimal("0.0900"),
    )
    assert portal.huijin_ratios_as_of(SECOND)[symbol]["中央汇金投资有限责任公司"] == (
        FIRST,
        Decimal("0.1000"),
    )
    assert portal.combined_huijin_ratios_as_of(FIRST)[symbol] == (
        date(2023, 12, 31),
        Decimal("0.1000"),
    )
    assert portal.combined_huijin_ratios_as_of(SECOND)[symbol] == (
        FIRST,
        Decimal("0.1000"),
    )


@pytest.mark.unit
def test_portal_date_gates_index_history_independently_from_etf_universe() -> None:
    dataset = replace(
        _dataset(),
        index_records=(
            IndexBarView(
                index_code="000001.SH",
                trade_date=FIRST,
                open=Decimal("3000"),
                high=Decimal("3020"),
                low=Decimal("2980"),
                close=Decimal("3010"),
                pre_close=Decimal("2990"),
                pct_change=Decimal("0.6689"),
                source_system="TUSHARE",
            ),
            IndexBarView(
                index_code="000001.SH",
                trade_date=SECOND,
                open=Decimal("3010"),
                high=Decimal("3030"),
                low=Decimal("2995"),
                close=Decimal("3005"),
                pre_close=Decimal("3010"),
                pct_change=Decimal("-0.1661"),
                source_system="TUSHARE",
            ),
        ),
    )
    portal = DailyDataPortal(dataset)

    assert [bar.trade_date for bar in portal.index_history_through(FIRST)["000001.SH"]] == [FIRST]
    assert [
        bar.trade_date
        for bar in portal.index_history_through(SECOND, lookback_trading_days=1)["000001.SH"]
    ] == [SECOND]
