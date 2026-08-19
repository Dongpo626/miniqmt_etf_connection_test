from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.account import Account, DailySnapshot
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import (
    ExecutionEstimate,
    FillResult,
    Order,
    OrderSide,
    RuleCheckResult,
    RuleReasonCode,
    TradePriceQuote,
)
from etf_backtest.core.position import Position


def _fill(side: OrderSide, *, symbol: str = "510300", quantity: int = 100) -> FillResult:
    order = Order(
        order_id=f"{side.value}-1",
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        symbol=symbol,
        side=side,
        requested_quantity=quantity,
        target_value_gap=Decimal("400") * (1 if side is OrderSide.BUY else -1),
    )
    quote = TradePriceQuote(
        source_record_key=f"QMT:{symbol[-6:]}:2024-01-03:1:1",
        symbol=symbol,
        trade_date=date(2024, 1, 3),
        base_trade_price=Decimal("4"),
        price_limit_down=Decimal("3.6"),
        price_limit_up=Decimal("4.4"),
    )
    estimate = ExecutionEstimate(
        order_id=order.order_id,
        requested_quantity=quantity,
        base_trade_price=Decimal("4"),
        fill_price=Decimal("4"),
        estimated_trade_amount=Decimal("4") * quantity,
        estimated_fee=Decimal("5"),
        estimated_total_cash_required=Decimal("4") * quantity + Decimal("5"),
    )
    approval = RuleCheckResult(
        order_id=order.order_id,
        requested_quantity=quantity,
        approved_quantity=quantity,
        passed=True,
        reason_code=RuleReasonCode.APPROVED,
        message="approved",
    )
    return FillResult.from_approved(
        order=order,
        quote=quote,
        estimate=estimate,
        approval=approval,
        trade_amount=Decimal("4") * quantity,
        fee=Decimal("5"),
    )


def test_t1_buy_releases_only_on_new_trade_date() -> None:
    account = Account(
        cash=Decimal("1000"),
        positions={"510300": Position("510300", TurnoverRule.T1)},
    )
    account.apply_fill(_fill(OrderSide.BUY))
    assert account.position_for("510300").available_quantity == 0
    assert account.cash == Decimal("595")
    account.on_new_trade_date()
    assert account.position_for("510300").available_quantity == 100


def test_t0_buy_is_immediately_sellable() -> None:
    account = Account(
        cash=Decimal("1000"),
        positions={"518880": Position("518880", TurnoverRule.T0)},
    )
    account.apply_fill(_fill(OrderSide.BUY, symbol="518880"))
    assert account.position_for("518880").available_quantity == 100


def test_only_formal_fill_mutates_account_and_snapshot_uses_raw_close() -> None:
    account = Account(
        cash=Decimal("1000"),
        positions={"510300": Position("510300", TurnoverRule.T1)},
    )
    with pytest.raises(TypeError, match="formal"):
        account.apply_fill(object())  # type: ignore[arg-type]
    snapshot = account.snapshot({"510300": Decimal("4")})
    daily = DailySnapshot(date(2024, 1, 2), snapshot)
    assert daily.cash == Decimal("1000.000")
    assert daily.total_asset == Decimal("1000.000")
