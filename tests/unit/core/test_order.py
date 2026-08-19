from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.order import (
    ExecutionEstimate,
    FillResult,
    Order,
    OrderSide,
    RuleCheckResult,
    RuleReasonCode,
    TradePriceQuote,
)


def _chain() -> tuple[Order, TradePriceQuote, ExecutionEstimate, RuleCheckResult]:
    order = Order(
        order_id="order-1",
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        symbol="510300",
        side=OrderSide.BUY,
        requested_quantity=100,
        target_value_gap=Decimal("400"),
    )
    quote = TradePriceQuote(
        source_record_key="QMT:510300:2024-01-03:1:1",
        symbol="510300",
        trade_date=date(2024, 1, 3),
        base_trade_price=Decimal("4"),
        price_limit_down=Decimal("3.6"),
        price_limit_up=Decimal("4.4"),
    )
    estimate = ExecutionEstimate(
        order_id="order-1",
        requested_quantity=100,
        base_trade_price=Decimal("4"),
        fill_price=Decimal("4.002"),
        estimated_trade_amount=Decimal("400.2"),
        estimated_fee=Decimal("5"),
        estimated_total_cash_required=Decimal("405.2"),
    )
    approval = RuleCheckResult(
        order_id="order-1",
        requested_quantity=100,
        approved_quantity=100,
        passed=True,
        reason_code=RuleReasonCode.APPROVED,
        message="approved",
    )
    return order, quote, estimate, approval


def test_orders_are_strictly_next_date_and_close_only() -> None:
    order, quote, _, _ = _chain()
    assert order.execution_date > order.signal_date
    assert quote.price_source == "CLOSE"
    assert quote.trade_time.hour == 15


def test_fill_requires_positive_rule_approval() -> None:
    order, quote, estimate, approval = _chain()
    fill = FillResult.from_approved(
        order=order,
        quote=quote,
        estimate=estimate,
        approval=approval,
        trade_amount=Decimal("400.2"),
        fee=Decimal("5"),
    )
    assert fill.fill_quantity == 100
    with pytest.raises(TypeError, match="from_approved"):
        FillResult()


def test_fill_rejects_changed_execution_chain_identity() -> None:
    order, quote, estimate, _approval = _chain()
    bad = RuleCheckResult(
        order_id="another-order",
        requested_quantity=100,
        approved_quantity=100,
        passed=True,
        reason_code=RuleReasonCode.APPROVED,
        message="approved",
    )
    with pytest.raises(ValueError, match="identity"):
        FillResult.from_approved(
            order=order,
            quote=quote,
            estimate=estimate,
            approval=bad,
            trade_amount=Decimal("400.2"),
            fee=Decimal("5"),
        )
