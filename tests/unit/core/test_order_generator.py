from __future__ import annotations

from datetime import date
from decimal import Decimal

from etf_backtest.core.account import Account
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.position import Position
from etf_backtest.core.target import TargetPortfolio


def test_order_generator_is_deterministic_and_sorts_sells_before_buys() -> None:
    account = Account(
        cash=Decimal("1000"),
        positions={
            "510300": Position(
                "510300", TurnoverRule.T1, total_quantity=100, available_quantity=100
            ),
            "159915": Position("159915", TurnoverRule.T1),
        },
    )
    snapshot = account.snapshot({"510300": Decimal("4"), "159915": Decimal("2")})
    orders = OrderGenerator().generate(
        target_portfolio=TargetPortfolio({"159915": Decimal("0.90")}),
        valuation_snapshot=snapshot,
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
    )
    assert [order.side for order in orders] == [OrderSide.SELL, OrderSide.BUY]
    assert orders[0].requested_quantity == 100
    assert orders[1].requested_quantity == 600
    assert orders == OrderGenerator().generate(
        target_portfolio=TargetPortfolio({"159915": Decimal("0.90")}),
        valuation_snapshot=snapshot,
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
    )
