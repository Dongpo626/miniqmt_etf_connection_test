"""Three-state behavior of the shared daily decision service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.application.daily_decision import DailyDecisionService
from etf_backtest.core.account import Account
from etf_backtest.core.market import IndexBarView, MarketBarView, TurnoverRule
from etf_backtest.core.position import Position
from etf_backtest.core.target import NO_REBALANCE, StrategyTarget, TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView, StrategyContext

DATES = (date(2024, 1, 2), date(2024, 1, 3))
SYMBOL = "SH.510300"


class StubPortal:
    def views_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
        lookback_trading_days: int | None = None,
    ) -> tuple[MarketBarView, ...]:
        del as_of_date, symbols, lookback_trading_days
        return ()

    def share_history_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[date, Decimal]]:
        del as_of_date, symbols
        return {}

    def huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]:
        del as_of_date, symbols
        return {}

    def index_history_through(
        self,
        as_of_date: date,
        *,
        lookback_trading_days: int | None = None,
    ) -> Mapping[str, tuple[IndexBarView, ...]]:
        del as_of_date, lookback_trading_days
        return {}

    def combined_huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, tuple[date, Decimal]]:
        del as_of_date, symbols
        return {}


class StubStrategy(BaseStrategy):
    def __init__(self, *, scheduled: bool, target: StrategyTarget) -> None:
        self.scheduled = scheduled
        self.target = target
        self.schedule_indexes: list[int] = []
        self.generate_calls = 0

    @property
    def required_history_trading_days(self) -> int:
        return 2

    def should_generate_target(self, frame_index: int) -> bool:
        self.schedule_indexes.append(frame_index)
        return self.scheduled

    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> StrategyTarget:
        del signal_date, market_history, account_view, context
        self.generate_calls += 1
        return self.target


def evaluate(strategy: StubStrategy) -> DailyDecisionResult:
    account = Account(
        cash=Decimal("10000"),
        positions={SYMBOL: Position(symbol=SYMBOL, turnover_rule=TurnoverRule.T1)},
    )
    return DailyDecisionService().evaluate(
        strategy=strategy,
        portal=StubPortal(),
        signal_date=DATES[0],
        execution_date=DATES[1],
        schedule_index=7,
        symbols=(SYMBOL,),
        account_view=AccountView.from_account(account),
        current_weights_by_symbol={SYMBOL: Decimal("0")},
    )


@pytest.mark.unit
def test_not_scheduled_does_not_generate_a_target() -> None:
    strategy = StubStrategy(
        scheduled=False,
        target=TargetPortfolio(weights={SYMBOL: Decimal("1")}),
    )

    result = evaluate(strategy)

    assert result.status is DecisionStatus.NOT_SCHEDULED
    assert result.target_portfolio is None
    assert strategy.schedule_indexes == [7]
    assert strategy.generate_calls == 0


@pytest.mark.unit
def test_no_rebalance_is_an_explicit_decision_status() -> None:
    strategy = StubStrategy(scheduled=True, target=NO_REBALANCE)

    result = evaluate(strategy)

    assert result.status is DecisionStatus.NO_REBALANCE
    assert result.target_portfolio is None
    assert strategy.generate_calls == 1


@pytest.mark.unit
def test_target_created_returns_the_strategy_portfolio() -> None:
    target = TargetPortfolio(weights={SYMBOL: Decimal("1")})

    result = evaluate(StubStrategy(scheduled=True, target=target))

    assert result.status is DecisionStatus.TARGET_CREATED
    assert result.target_portfolio is target


@pytest.mark.unit
def test_empty_target_portfolio_is_not_no_rebalance() -> None:
    target = TargetPortfolio(weights={})

    result = evaluate(StubStrategy(scheduled=True, target=target))

    assert result.status is DecisionStatus.TARGET_CREATED
    assert result.target_portfolio is target
    assert dict(result.target_portfolio.weights) == {}
