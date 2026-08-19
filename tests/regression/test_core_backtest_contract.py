"""Frozen calculation contract for the core daily backtest loop."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

import pytest

from etf_backtest.core.engine import BacktestResult
from tests.unit.core.conftest import DATES


def _normalize_result(result: BacktestResult) -> Mapping[str, object]:
    approvals = {approval.order_id: approval for approval in result.approvals}
    return {
        "daily": tuple(
            (
                daily.trade_date,
                daily.cash,
                daily.market_value,
                daily.total_asset,
                tuple(
                    (
                        symbol,
                        position.total_quantity,
                        position.available_quantity,
                        position.today_buy_quantity,
                    )
                    for symbol, position in daily.account_snapshot.positions.items()
                ),
            )
            for daily in result.daily_snapshots
        ),
        "orders": tuple(
            (
                order.signal_date,
                order.execution_date,
                order.symbol,
                order.side.value,
                order.requested_quantity,
                order.target_value_gap,
                approvals[order.order_id].approved_quantity,
                approvals[order.order_id].reason_code.value,
            )
            for order in result.orders
        ),
        "fills": tuple(
            (
                fill.signal_date,
                fill.execution_date,
                fill.symbol,
                fill.side.value,
                fill.fill_quantity,
                fill.base_trade_price,
                fill.fill_price,
                fill.trade_amount,
                fill.fee,
            )
            for fill in result.fills
        ),
    }


@pytest.mark.unit
def test_core_contract_keeps_d_plus_one_orders_fills_and_nav(engine_components) -> None:
    engine, *_ = engine_components

    result = engine.run(start_date=DATES[0], end_date=DATES[-1])

    assert _normalize_result(result) == {
        "daily": (
            (
                DATES[0],
                Decimal("10000.000"),
                Decimal("0"),
                Decimal("10000.000"),
                (("SH.510300", 0, 0, 0),),
            ),
            (
                DATES[1],
                Decimal("0.000"),
                Decimal("10000.000"),
                Decimal("10000.000"),
                (("SH.510300", 1000, 0, 1000),),
            ),
            (
                DATES[2],
                Decimal("20000.000"),
                Decimal("0"),
                Decimal("20000.000"),
                (("SH.510300", 0, 0, 0),),
            ),
        ),
        "orders": (
            (DATES[0], DATES[1], "SH.510300", "BUY", 1000, Decimal("10000.000"), 1000, "APPROVED"),
            (
                DATES[1],
                DATES[2],
                "SH.510300",
                "SELL",
                1000,
                Decimal("-20000.000"),
                1000,
                "APPROVED",
            ),
        ),
        "fills": (
            (
                DATES[0],
                DATES[1],
                "SH.510300",
                "BUY",
                1000,
                Decimal("10"),
                Decimal("10"),
                Decimal("10000"),
                Decimal("0.000"),
            ),
            (
                DATES[1],
                DATES[2],
                "SH.510300",
                "SELL",
                1000,
                Decimal("20"),
                Decimal("20"),
                Decimal("20000"),
                Decimal("0.000"),
            ),
        ),
    }
