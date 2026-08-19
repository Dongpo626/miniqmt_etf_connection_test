"""Raw-close price and effective legal-limit tests."""

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.market import (
    EtfCategory,
    EtfTradingRule,
    FrameKey,
    MarketBar,
    MarketFrame,
    PriceLimitSource,
    TurnoverRule,
)
from etf_backtest.core.pricing import CloseTradePriceModel, TradePriceQuoteCache


def _bar() -> MarketBar:
    return MarketBar(
        source_record_key="raw:SH.510300:2024-01-03",
        symbol="SH.510300",
        trade_date=date(2024, 1, 3),
        open=Decimal("10.000"),
        high=Decimal("10.200"),
        low=Decimal("9.900"),
        close=Decimal("10.100"),
        pre_close=Decimal("10.005"),
        volume=100000,
        amount=Decimal("1000000"),
        suspended=False,
    )


def _rule() -> EtfTradingRule:
    return EtfTradingRule(
        symbol="SH.510300",
        etf_category=EtfCategory.DOMESTIC_STOCK_ETF,
        turnover_rule=TurnoverRule.T1,
        price_limit_ratio=Decimal("0.10"),
    )


@pytest.mark.unit
def test_close_model_uses_raw_close_and_tick_rounded_legal_limits() -> None:
    quote = CloseTradePriceModel().resolve(execution_bar=_bar(), trading_rule=_rule())

    assert quote.base_trade_price == Decimal("10.100")
    assert quote.price_limit_down == Decimal("9.005")
    assert quote.price_limit_up == Decimal("11.006")
    assert quote.price_source == "CLOSE"
    assert quote.price_limit_source is PriceLimitSource.DERIVED_RULE_FALLBACK
    assert quote.source_record_key == _bar().source_record_key


@pytest.mark.unit
def test_close_model_prefers_explicit_daily_price_limits() -> None:
    bar = replace(
        _bar(),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("10.500"),
        price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
    )

    quote = CloseTradePriceModel().resolve(execution_bar=bar, trading_rule=_rule())

    assert quote.price_limit_down == Decimal("9.000")
    assert quote.price_limit_up == Decimal("10.500")
    assert quote.price_limit_source is PriceLimitSource.TUSHARE_EXPLICIT


class _SpyCloseModel(CloseTradePriceModel):
    def __init__(self) -> None:
        self.calls = 0

    def resolve(self, *, execution_bar: MarketBar, trading_rule: EtfTradingRule):
        self.calls += 1
        return super().resolve(execution_bar=execution_bar, trading_rule=trading_rule)


@pytest.mark.unit
def test_quote_cache_resolves_a_frame_symbol_once() -> None:
    bar = _bar()
    frame = MarketFrame.from_bars(
        FrameKey(trade_date=bar.trade_date, calendar_version="qmt-v1"),
        (bar,),
    )
    model = _SpyCloseModel()
    cache = TradePriceQuoteCache(
        frame=frame,
        trading_rules={bar.symbol: _rule()},
        model=model,
    )

    first = cache.quote_for("510300")
    second = cache.quote_for("SH.510300")

    assert first is second
    assert model.calls == 1


@pytest.mark.unit
def test_rule_identity_must_match_bar() -> None:
    wrong = EtfTradingRule(
        symbol="SH.518880",
        etf_category=EtfCategory.GOLD_ETF,
        turnover_rule=TurnoverRule.T0,
        price_limit_ratio=Decimal("0.10"),
    )
    with pytest.raises(ValueError, match="symbol"):
        CloseTradePriceModel().resolve(execution_bar=_bar(), trading_rule=wrong)
