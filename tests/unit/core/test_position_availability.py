"""Daily T+0/T+1 availability lives in the Position state."""

from __future__ import annotations

import pytest

from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.position import Position


@pytest.mark.unit
def test_t1_today_bucket_releases_exactly_on_new_engine_trade_date() -> None:
    position = Position(
        symbol="SH.510300",
        turnover_rule=TurnoverRule.T1,
        total_quantity=300,
        available_quantity=100,
        today_buy_quantity=200,
    )

    released = position.on_new_trade_date()

    assert released.total_quantity == 300
    assert released.available_quantity == 300
    assert released.today_buy_quantity == 0
    assert position.available_quantity == 100


@pytest.mark.unit
def test_t0_position_requires_every_share_to_be_immediately_available() -> None:
    position = Position(
        symbol="SH.518880",
        turnover_rule=TurnoverRule.T0,
        total_quantity=300,
        available_quantity=300,
    )

    assert position.on_new_trade_date() is position
    with pytest.raises(ValueError, match="T0 requires all shares"):
        Position(
            symbol="SH.518880",
            turnover_rule=TurnoverRule.T0,
            total_quantity=300,
            available_quantity=100,
            today_buy_quantity=200,
        )


@pytest.mark.unit
def test_t1_buckets_must_partition_total_quantity() -> None:
    with pytest.raises(ValueError, match="must sum to total"):
        Position(
            symbol="SH.510300",
            turnover_rule=TurnoverRule.T1,
            total_quantity=300,
            available_quantity=100,
            today_buy_quantity=100,
        )
