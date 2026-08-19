from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.market import (
    EtfCategory,
    EtfInfo,
    EtfTradingRule,
    Exchange,
    FrameKey,
    MarketBar,
    MarketBarView,
    MarketFrame,
    PriceLimitSource,
    TurnoverRule,
)


def _key(symbol: str) -> str:
    return f"daily-bar:{symbol}:2024-01-02"


def _bar(symbol: str = "SH.510300") -> MarketBar:
    return MarketBar(
        source_record_key=_key(symbol),
        symbol=symbol,
        trade_date=date(2024, 1, 2),
        open=Decimal("3.8"),
        high=Decimal("4.1"),
        low=Decimal("3.7"),
        close=Decimal("4"),
        pre_close=Decimal("3.9"),
        volume=10000,
        amount=Decimal("40000"),
        suspended=False,
    )


def test_raw_bar_accepts_an_opaque_source_key_and_allows_zero_volume() -> None:
    bar = _bar()
    assert bar.source_record_key == "daily-bar:SH.510300:2024-01-02"
    zero = replace(bar, volume=0, amount=Decimal("0"), suspended=True)
    assert zero.volume == 0
    assert not hasattr(bar, "raw_revision")


def test_source_key_is_non_blank_but_has_no_embedded_revision_contract() -> None:
    arbitrary = replace(_bar(), source_record_key="database-primary-key-42")
    assert arbitrary.source_record_key == "database-primary-key-42"
    with pytest.raises(ValueError, match="must not be blank"):
        replace(_bar(), source_record_key="  ")


def test_front_view_is_independent_and_contains_only_strategy_market_fields() -> None:
    view = MarketBarView(
        source_record_key=_key("SH.510300"),
        symbol="510300",
        trade_date=date(2024, 1, 2),
        open=Decimal("2"),
        high=Decimal("2.1"),
        low=Decimal("1.9"),
        close=Decimal("2.05"),
        volume=10000,
        suspended=False,
    )
    assert view.close == Decimal("2.05")
    assert not hasattr(view, "raw_bar")
    assert not hasattr(view, "front_revision")
    assert not hasattr(view, "adjustment_asof_date")
    assert not hasattr(view, "anchor_trade_date")
    assert not hasattr(view, "front_adjustment_ratio")
    assert view.signal_time.hour == 15


@pytest.mark.unit
def test_market_bar_requires_an_atomic_valid_explicit_price_limit_pair() -> None:
    valid = replace(
        _bar(),
        price_limit_down=Decimal("3.500"),
        price_limit_up=Decimal("4.500"),
        price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
    )
    assert valid.price_limit_source is PriceLimitSource.TUSHARE_EXPLICIT

    with pytest.raises(ValueError, match="complete pair"):
        replace(_bar(), price_limit_down=Decimal("3.500"))
    with pytest.raises(ValueError, match="inside explicit"):
        replace(
            _bar(),
            price_limit_down=Decimal("4.200"),
            price_limit_up=Decimal("4.500"),
            price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
        )


def test_frame_is_daily_and_defensively_sorted() -> None:
    second = _bar("SZ.159915")
    frame = MarketFrame.from_bars(FrameKey(date(2024, 1, 2), "calendar-v1"), (second, _bar()))
    assert frame.canonical_symbols == ("SH.510300", "SZ.159915")
    assert frame.frame_key.close_time.hour == 15


def test_frame_rejects_duplicate_opaque_source_record_keys() -> None:
    first = _bar()
    second = replace(_bar("SZ.159915"), source_record_key=first.source_record_key)
    with pytest.raises(ValueError, match="duplicate symbol or source record"):
        MarketFrame.from_bars(FrameKey(date(2024, 1, 2), "calendar-v1"), (first, second))


def test_etf_metadata_lifecycle_and_rules_do_not_guess_20pct_from_prefix() -> None:
    info = EtfInfo(
        symbol="588000",
        exchange=Exchange.SSE,
        name="科创50ETF",
        primary_category="纯境内",
        fund_type="股票型",
        list_date=date(2020, 9, 28),
        delist_date=None,
        current_status="上市",
    )
    assert info.is_active(date(2024, 1, 2))
    rule = EtfTradingRule(
        symbol=info.symbol,
        etf_category=EtfCategory.DOMESTIC_STOCK_ETF,
        turnover_rule=TurnoverRule.T1,
        price_limit_ratio=Decimal("0.10"),
    )
    assert rule.price_limit_ratio == Decimal("0.10")
    with pytest.raises(ValueError, match="turnover"):
        EtfTradingRule(
            symbol="518880",
            etf_category=EtfCategory.GOLD_ETF,
            turnover_rule=TurnoverRule.T1,
            price_limit_ratio=Decimal("0.10"),
        )
