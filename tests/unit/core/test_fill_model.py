"""Estimate and formal-fill tests for the daily-close chain."""

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.config.schema import FeeConfig, SlippageConfig
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.fill import FillModel
from etf_backtest.core.order import (
    FillStatus,
    Order,
    OrderSide,
    RuleCheckResult,
    RuleReasonCode,
    TradePriceQuote,
)
from etf_backtest.core.slippage import SlippageModel


def _model() -> FillModel:
    return FillModel(
        fee_model=FeeModel(FeeConfig()),
        slippage_model=SlippageModel(SlippageConfig(rate=Decimal("0.001"))),
    )


def _order(*, side: OrderSide = OrderSide.BUY, quantity: int = 300) -> Order:
    return Order(
        order_id=f"order-{side.value.lower()}",
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        symbol="SH.510300",
        side=side,
        requested_quantity=quantity,
        target_value_gap=Decimal("3000") if side is OrderSide.BUY else Decimal("-3000"),
    )


def _quote() -> TradePriceQuote:
    return TradePriceQuote(
        source_record_key="QMT:510300:2024-01-03:7:1",
        symbol="SH.510300",
        trade_date=date(2024, 1, 3),
        base_trade_price=Decimal("10"),
        price_limit_down=Decimal("9"),
        price_limit_up=Decimal("11"),
    )


@pytest.mark.unit
def test_estimate_applies_slippage_once_and_cost_reuses_fill_price() -> None:
    order = _order()
    estimate = _model().create_estimate(
        order=order,
        quote=_quote(),
        tick_size=Decimal("0.001"),
    )
    cost = _model().estimate_cost(order=order, estimate=estimate, quantity=100)

    assert estimate.fill_price == Decimal("10.010")
    assert estimate.estimated_trade_amount == Decimal("3003.000")
    assert estimate.estimated_fee == Decimal("5.000")
    assert cost.trade_amount == Decimal("1001.000")
    assert cost.fee == Decimal("5.000")


@pytest.mark.unit
def test_affordability_includes_fee_and_whole_lots() -> None:
    order = _order(quantity=300)
    model = _model()
    estimate = model.create_estimate(order=order, quote=_quote(), tick_size=Decimal("0.001"))

    assert (
        model.max_affordable_buy_quantity(
            order=order,
            estimate=estimate,
            available_cash=Decimal("1005.999"),
            lot_size=100,
        )
        == 0
    )
    assert (
        model.max_affordable_buy_quantity(
            order=order,
            estimate=estimate,
            available_cash=Decimal("2010"),
            lot_size=100,
        )
        == 200
    )


@pytest.mark.unit
def test_positive_approval_materializes_the_only_formal_fill() -> None:
    order = _order(quantity=300)
    model = _model()
    quote = _quote()
    estimate = model.create_estimate(order=order, quote=quote, tick_size=Decimal("0.001"))
    approval = RuleCheckResult(
        order_id=order.order_id,
        requested_quantity=300,
        approved_quantity=100,
        passed=True,
        reason_code=RuleReasonCode.APPROVED,
        message="cash-limited partial approval",
    )

    fill = model.create_fill(
        order=order,
        quote=quote,
        estimate=estimate,
        approval=approval,
    )

    assert fill is not None
    assert fill.fill_quantity == 100
    assert fill.status is FillStatus.PARTIALLY_FILLED
    assert fill.trade_amount == Decimal("1001.000")


@pytest.mark.unit
def test_rejection_materializes_no_fill() -> None:
    order = _order()
    model = _model()
    quote = _quote()
    estimate = model.create_estimate(order=order, quote=quote, tick_size=Decimal("0.001"))
    rejection = RuleCheckResult(
        order_id=order.order_id,
        requested_quantity=order.requested_quantity,
        approved_quantity=0,
        passed=False,
        reason_code=RuleReasonCode.INSUFFICIENT_CASH,
        message="rejected",
    )

    assert (
        model.create_fill(
            order=order,
            quote=quote,
            estimate=estimate,
            approval=rejection,
        )
        is None
    )
