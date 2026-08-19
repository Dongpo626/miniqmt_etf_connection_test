"""Trading-frame schedule tests."""

import pytest

from etf_backtest.strategy.scheduler import (
    EveryTradingDayScheduler,
    PeriodicDecisionScheduler,
)


@pytest.mark.unit
def test_periodic_scheduler_rebalances_every_twenty_sse_frames() -> None:
    scheduler = PeriodicDecisionScheduler(every_trading_days=20)

    assert scheduler.should_decide(0)
    assert not scheduler.should_decide(19)
    assert scheduler.should_decide(20)
    assert scheduler.should_decide(40)


@pytest.mark.unit
def test_model_scheduler_decides_daily() -> None:
    scheduler = EveryTradingDayScheduler()
    assert all(scheduler.should_decide(index) for index in range(5))
