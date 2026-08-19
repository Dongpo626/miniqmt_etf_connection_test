"""D signal to D+1 close execution regression."""

import inspect
from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.market import MarketFrame, PriceLimitSource
from tests.unit.core.conftest import DATES, SYMBOL


@pytest.mark.unit
def test_engine_run_is_a_pure_in_memory_boundary(engine_components) -> None:
    engine, *_ = engine_components

    assert tuple(inspect.signature(engine.run).parameters) == ("start_date", "end_date")


@pytest.mark.unit
def test_engine_executes_prior_target_at_next_raw_close_in_fixed_event_order(
    engine_components,
) -> None:
    engine, _portal, resolver, rule_engine, _strategy = engine_components

    result = engine.run(start_date=DATES[0], end_date=DATES[-1])

    assert [(fill.signal_date, fill.execution_date) for fill in result.fills] == [
        (DATES[0], DATES[1]),
        (DATES[1], DATES[2]),
    ]
    assert [fill.fill_price for fill in result.fills] == [Decimal("10"), Decimal("20")]
    assert [daily.total_asset for daily in result.daily_snapshots] == [
        Decimal("10000.000"),
        Decimal("10000.000"),
        Decimal("20000.000"),
    ]
    assert [daily.cash for daily in result.daily_snapshots] == [
        Decimal("10000.000"),
        Decimal("0.000"),
        Decimal("20000.000"),
    ]
    assert rule_engine.sell_available == [1000]
    assert resolver.calls == [(SYMBOL, trade_date) for trade_date in DATES]


@pytest.mark.unit
def test_strategy_sees_only_front_views_through_signal_and_last_frame_has_no_target(
    engine_components,
) -> None:
    engine, _portal, _resolver, _rule_engine, strategy = engine_components

    result = engine.run(start_date=DATES[0], end_date=DATES[-1])

    assert strategy.signal_dates == [DATES[0], DATES[1]]
    assert strategy.visible_dates == [(DATES[0],), (DATES[0], DATES[1])]
    assert [decision.signal_date for decision in result.decisions] == [DATES[0], DATES[1]]
    assert all(decision.signal_date != DATES[-1] for decision in result.decisions)
    assert dict(strategy.contexts[0].share_history_by_symbol[SYMBOL]) == {DATES[0]: Decimal("100")}
    assert strategy.contexts[1].huijin_ratios_by_symbol[SYMBOL]["中央汇金投资有限责任公司"] == (
        date(2023, 12, 31),
        Decimal("0.0971"),
    )
    assert [bar.trade_date for bar in strategy.contexts[1].index_history_by_code["000001.SH"]] == [
        DATES[0],
        DATES[1],
    ]
    assert strategy.contexts[1].combined_huijin_ratio_by_symbol[SYMBOL] == (
        date(2023, 12, 31),
        Decimal("0.1096"),
    )


@pytest.mark.unit
def test_no_rebalance_creates_no_pending_target_order_or_fill(engine_components) -> None:
    engine, _portal, _resolver, _rule_engine, strategy = engine_components
    strategy.no_rebalance_frames = {0, 1}

    result = engine.run(start_date=DATES[0], end_date=DATES[-1])

    assert result.decisions == ()
    assert result.orders == ()
    assert result.fills == ()


@pytest.mark.unit
def test_engine_freezes_explicit_price_limit_evidence_on_rule_check(engine_components) -> None:
    engine, portal, _resolver, _rule_engine, _strategy = engine_components
    execution_frame = portal._frames[1]
    explicit_bar = replace(
        execution_frame.bar_for(SYMBOL),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("11.000"),
        price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
    )
    portal._frames = (
        portal._frames[0],
        MarketFrame.from_bars(execution_frame.frame_key, (explicit_bar,)),
        portal._frames[2],
    )

    result = engine.run(start_date=DATES[0], end_date=DATES[-1])

    first = result.approvals[0]
    assert first.price_limit_source is PriceLimitSource.TUSHARE_EXPLICIT
    assert first.price_limit_down == Decimal("9.000")
    assert first.price_limit_up == Decimal("11.000")
    assert first.price_limit_fallback_reason is None
