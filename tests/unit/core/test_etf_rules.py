"""Public pure-daily ETF rule engine regression tests."""

from __future__ import annotations

from decimal import Decimal

import pytest

from etf_backtest.core.order import OrderSide, RuleReasonCode
from tests.unit.core.test_daily_etf_rule_engine import (
    GOLD,
    STOCK,
    _account,
    _approve,
    _bar,
    _frame,
    _order,
    _position,
)


@pytest.mark.unit
def test_batch_has_one_final_quantity_per_order_in_sell_then_buy_order() -> None:
    frame = _frame(_bar(STOCK), _bar(GOLD))
    orders = [
        _order("buy", GOLD, OrderSide.BUY, 100, gap=Decimal("1000")),
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
    assert all(result.reason_code is RuleReasonCode.APPROVED for result in results)
    assert account.cash == Decimal("0")
    assert account.position_for(STOCK).total_quantity == 100
    assert account.position_for(GOLD).total_quantity == 0


@pytest.mark.unit
def test_zero_approval_keeps_the_primary_rejection_reason() -> None:
    frame = _frame(_bar(STOCK, volume=0, suspended=True))
    order = _order("suspended", STOCK, OrderSide.BUY, 100)

    result = _approve(
        frame=frame,
        orders=[order],
        account=_account(cash=Decimal("10_000")),
    )[0]

    assert result.approved_quantity == 0
    assert not result.passed
    assert result.reason_code is RuleReasonCode.SUSPENDED
