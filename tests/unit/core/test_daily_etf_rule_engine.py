"""Daily-close ETF quantity approval tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.config.schema import FeeConfig, SlippageConfig
from etf_backtest.core.account import Account
from etf_backtest.core.etf_rules import EtfRuleEngine
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.fill import FillModel
from etf_backtest.core.market import (
    EtfCategory,
    EtfInfo,
    EtfTradingRule,
    Exchange,
    FrameKey,
    MarketBar,
    MarketFrame,
    PriceLimitSource,
    TurnoverRule,
)
from etf_backtest.core.order import Order, OrderSide, RuleCheckResult, RuleReasonCode
from etf_backtest.core.position import Position
from etf_backtest.core.pricing import CloseTradePriceModel
from etf_backtest.core.slippage import SlippageModel

TRADE_DATE = date(2025, 1, 3)
SIGNAL_DATE = date(2025, 1, 2)
STOCK = "SH.510300"
GOLD = "SH.518880"


def _bar(
    symbol: str,
    *,
    close: Decimal = Decimal("10"),
    pre_close: Decimal = Decimal("10"),
    volume: int = 10_000,
    suspended: bool = False,
) -> MarketBar:
    code = symbol.partition(".")[2]
    return MarketBar(
        source_record_key=f"raw:{code}:{TRADE_DATE.isoformat()}",
        symbol=symbol,
        trade_date=TRADE_DATE,
        open=close,
        high=close,
        low=close,
        close=close,
        pre_close=pre_close,
        volume=volume,
        amount=close * volume,
        suspended=suspended,
    )


def _frame(*bars: MarketBar) -> MarketFrame:
    return MarketFrame.from_bars(
        FrameKey(trade_date=TRADE_DATE, calendar_version="test-calendar-v1"),
        bars,
    )


def _rule(symbol: str) -> EtfTradingRule:
    gold = symbol == GOLD
    return EtfTradingRule(
        symbol=symbol,
        etf_category=EtfCategory.GOLD_ETF if gold else EtfCategory.DOMESTIC_STOCK_ETF,
        turnover_rule=TurnoverRule.T0 if gold else TurnoverRule.T1,
        price_limit_ratio=Decimal("0.10"),
    )


def _info(symbol: str, *, list_date: date = date(2020, 1, 1)) -> EtfInfo:
    return EtfInfo(
        symbol=symbol,
        exchange=Exchange.SSE,
        name=symbol,
        primary_category="ETF",
        fund_type="gold" if symbol == GOLD else "stock",
        list_date=list_date,
        delist_date=None,
        current_status="LISTED",
    )


def _position(
    symbol: str,
    *,
    total: int = 0,
    available: int = 0,
    today: int = 0,
) -> Position:
    turnover = TurnoverRule.T0 if symbol == GOLD else TurnoverRule.T1
    return Position(
        symbol=symbol,
        turnover_rule=turnover,
        total_quantity=total,
        available_quantity=available,
        today_buy_quantity=today,
    )


def _account(
    *,
    cash: Decimal,
    stock: Position | None = None,
    gold: Position | None = None,
) -> Account:
    return Account(
        cash=cash,
        positions={
            STOCK: _position(STOCK) if stock is None else stock,
            GOLD: _position(GOLD) if gold is None else gold,
        },
    )


def _order(
    order_id: str,
    symbol: str,
    side: OrderSide,
    quantity: int,
    *,
    gap: Decimal | None = None,
    execution_date: date = TRADE_DATE,
) -> Order:
    target_gap = Decimal(quantity * 10) if gap is None else gap
    if side is OrderSide.SELL:
        target_gap = -abs(target_gap)
    return Order(
        order_id=order_id,
        signal_date=SIGNAL_DATE,
        execution_date=execution_date,
        symbol=symbol,
        side=side,
        requested_quantity=quantity,
        target_value_gap=target_gap,
    )


def _fill_model() -> FillModel:
    return FillModel(
        fee_model=FeeModel(
            FeeConfig(
                commission_rate=Decimal("0"),
                minimum_commission=Decimal("0"),
            )
        ),
        slippage_model=SlippageModel(SlippageConfig(rate=Decimal("0"))),
    )


def _approve(
    *,
    frame: MarketFrame,
    orders: list[Order],
    account: Account,
    infos: dict[str, EtfInfo] | None = None,
    volume_rate: Decimal = Decimal("0.20"),
) -> list[RuleCheckResult]:
    rules = {symbol: _rule(symbol) for symbol in frame.canonical_symbols}
    price_model = CloseTradePriceModel()
    fill_model = _fill_model()
    quotes = {
        symbol: price_model.resolve(
            execution_bar=frame.bar_for(symbol),
            trading_rule=rules[symbol],
        )
        for symbol in frame.canonical_symbols
    }
    estimates = {
        order.order_id: fill_model.create_estimate(
            order=order,
            quote=quotes[order.symbol],
            tick_size=rules[order.symbol].tick_size,
        )
        for order in orders
        if order.execution_date == frame.trade_date
    }
    return EtfRuleEngine(
        fill_model=fill_model,
        volume_participation_rate=volume_rate,
    ).approve_batch(
        frame=frame,
        orders=orders,
        quotes=quotes,
        estimates=estimates,
        account=account,
        trading_rules=rules,
        etf_infos=(
            {symbol: _info(symbol) for symbol in frame.canonical_symbols}
            if infos is None
            else infos
        ),
    )


@pytest.mark.unit
def test_t1_sell_uses_available_bucket_while_gold_t0_can_sell_all() -> None:
    frame = _frame(_bar(STOCK), _bar(GOLD))
    orders = [
        _order("stock-sell", STOCK, OrderSide.SELL, 300),
        _order("gold-sell", GOLD, OrderSide.SELL, 300),
    ]
    account = _account(
        cash=Decimal("0"),
        stock=_position(STOCK, total=300, available=100, today=200),
        gold=_position(GOLD, total=300, available=300),
    )

    before_positions = dict(account.positions)
    results = _approve(frame=frame, orders=orders, account=account)
    by_id = {result.order_id: result for result in results}

    assert by_id["stock-sell"].approved_quantity == 100
    assert by_id["stock-sell"].reason_code is RuleReasonCode.APPROVED
    assert "sellable-holdings" in by_id["stock-sell"].message
    assert by_id["gold-sell"].approved_quantity == 300
    assert account.cash == Decimal("0")
    assert dict(account.positions) == before_positions


@pytest.mark.unit
def test_positive_volume_and_cash_reductions_still_use_approved_reason() -> None:
    volume_frame = _frame(_bar(STOCK, volume=1_000))
    volume_order = _order("volume", STOCK, OrderSide.BUY, 300)
    volume_result = _approve(
        frame=volume_frame,
        orders=[volume_order],
        account=_account(cash=Decimal("10_000")),
    )[0]

    cash_frame = _frame(_bar(STOCK))
    cash_order = _order("cash", STOCK, OrderSide.BUY, 200)
    cash_result = _approve(
        frame=cash_frame,
        orders=[cash_order],
        account=_account(cash=Decimal("1_500")),
    )[0]

    assert (volume_result.approved_quantity, volume_result.reason_code) == (
        200,
        RuleReasonCode.APPROVED,
    )
    assert "daily-volume" in volume_result.message
    assert (cash_result.approved_quantity, cash_result.reason_code) == (
        100,
        RuleReasonCode.APPROVED,
    )
    assert "available-cash" in cash_result.message


@pytest.mark.unit
def test_daily_volume_is_cumulative_across_orders_for_one_symbol() -> None:
    frame = _frame(_bar(STOCK, volume=1_000))
    orders = [
        _order("b", STOCK, OrderSide.SELL, 200),
        _order("a", STOCK, OrderSide.SELL, 200),
    ]
    account = _account(
        cash=Decimal("0"),
        stock=_position(STOCK, total=400, available=400),
    )

    results = _approve(frame=frame, orders=orders, account=account)

    assert [(result.order_id, result.approved_quantity) for result in results] == [
        ("a", 200),
        ("b", 0),
    ]
    assert results[1].reason_code is RuleReasonCode.VOLUME_LIMIT


@pytest.mark.unit
def test_sell_net_proceeds_fund_later_buy_without_mutating_account() -> None:
    frame = _frame(_bar(STOCK), _bar(GOLD))
    orders = [
        _order("buy", GOLD, OrderSide.BUY, 100, gap=Decimal("1_000")),
        _order("sell", STOCK, OrderSide.SELL, 100),
    ]
    account = _account(
        cash=Decimal("0"),
        stock=_position(STOCK, total=100, available=100),
    )

    results = _approve(frame=frame, orders=orders, account=account)

    assert [(result.order_id, result.approved_quantity) for result in results] == [
        ("sell", 100),
        ("buy", 100),
    ]
    assert account.cash == Decimal("0")
    assert account.position_for(STOCK).total_quantity == 100
    assert account.position_for(GOLD).total_quantity == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("side", "close"),
    [
        (OrderSide.BUY, Decimal("11")),
        (OrderSide.SELL, Decimal("9")),
    ],
)
def test_daily_price_limits_are_directional(side: OrderSide, close: Decimal) -> None:
    frame = _frame(_bar(STOCK, close=close))
    order = _order("limit", STOCK, side, 100)
    account = _account(
        cash=Decimal("10_000"),
        stock=_position(STOCK, total=100, available=100),
    )

    result = _approve(frame=frame, orders=[order], account=account)[0]

    assert result.reason_code is RuleReasonCode.PRICE_LIMIT
    assert not result.passed


@pytest.mark.unit
def test_explicit_limit_price_drives_directional_block_and_quote_validation() -> None:
    # The explicit upper limit intentionally differs from the 10% fallback (11.000).
    explicit_bar = replace(
        _bar(STOCK, close=Decimal("10.500")),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("10.500"),
        price_limit_source=PriceLimitSource.TUSHARE_EXPLICIT,
    )

    result = _approve(
        frame=_frame(explicit_bar),
        orders=[_order("explicit-limit", STOCK, OrderSide.BUY, 100)],
        account=_account(cash=Decimal("10000")),
    )[0]

    assert result.reason_code is RuleReasonCode.PRICE_LIMIT
    assert not result.passed


@pytest.mark.unit
def test_listing_suspension_and_lot_checks_fail_closed_in_priority_order() -> None:
    suspended_frame = _frame(_bar(STOCK, volume=0, suspended=True))
    bad_lot = _order("bad-lot", STOCK, OrderSide.BUY, 50)
    future_info = _info(STOCK, list_date=date(2025, 1, 6))

    listing = _approve(
        frame=suspended_frame,
        orders=[bad_lot],
        account=_account(cash=Decimal("10_000")),
        infos={STOCK: future_info},
    )[0]
    suspended = _approve(
        frame=suspended_frame,
        orders=[bad_lot],
        account=_account(cash=Decimal("10_000")),
    )[0]
    lot = _approve(
        frame=_frame(_bar(STOCK)),
        orders=[bad_lot],
        account=_account(cash=Decimal("10_000")),
    )[0]

    assert listing.reason_code is RuleReasonCode.LISTING_OR_WINDOW
    assert suspended.reason_code is RuleReasonCode.SUSPENDED
    assert lot.reason_code is RuleReasonCode.LOT_SIZE


@pytest.mark.unit
def test_execution_date_and_quote_chain_mismatches_are_rejected() -> None:
    frame = _frame(_bar(STOCK))
    account = _account(cash=Decimal("10_000"))
    wrong_date_order = _order(
        "wrong-date",
        STOCK,
        OrderSide.BUY,
        100,
        execution_date=date(2025, 1, 6),
    )
    wrong_date = _approve(
        frame=frame,
        orders=[wrong_date_order],
        account=account,
    )[0]

    order = _order("bad-chain", STOCK, OrderSide.BUY, 100)
    rule = _rule(STOCK)
    quote = CloseTradePriceModel().resolve(
        execution_bar=frame.bar_for(STOCK),
        trading_rule=rule,
    )
    bad_quote = replace(
        quote,
        source_record_key=f"QMT:510300:{TRADE_DATE.isoformat()}:9:9",
    )
    fill_model = _fill_model()
    estimate = fill_model.create_estimate(order=order, quote=quote, tick_size=rule.tick_size)
    bad_chain = EtfRuleEngine(fill_model=fill_model).approve_batch(
        frame=frame,
        orders=[order],
        quotes={STOCK: bad_quote},
        estimates={order.order_id: estimate},
        account=account,
        trading_rules={STOCK: rule},
        etf_infos={STOCK: _info(STOCK)},
    )[0]

    assert wrong_date.reason_code is RuleReasonCode.LISTING_OR_WINDOW
    assert bad_chain.reason_code is RuleReasonCode.QUOTE_UNAVAILABLE


@pytest.mark.unit
def test_constructor_and_batch_boundaries_reject_invalid_inputs() -> None:
    fill_model = _fill_model()
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        EtfRuleEngine(
            fill_model=fill_model,
            volume_participation_rate=Decimal("0"),
        )

    frame = _frame(_bar(STOCK))
    order = _order("duplicate", STOCK, OrderSide.BUY, 100)
    rule = _rule(STOCK)
    quote = CloseTradePriceModel().resolve(
        execution_bar=frame.bar_for(STOCK),
        trading_rule=rule,
    )
    estimate = fill_model.create_estimate(order=order, quote=quote, tick_size=rule.tick_size)
    with pytest.raises(ValueError, match="unique"):
        EtfRuleEngine(fill_model=fill_model).approve_batch(
            frame=frame,
            orders=[order, order],
            quotes={STOCK: quote},
            estimates={order.order_id: estimate},
            account=_account(cash=Decimal("10_000")),
            trading_rules={STOCK: rule},
            etf_infos={STOCK: _info(STOCK)},
        )
