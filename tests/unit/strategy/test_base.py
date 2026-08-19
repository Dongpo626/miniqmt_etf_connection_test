"""The common strategy boundary rejects future adjusted views."""

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.account import Account
from etf_backtest.core.market import MarketBarView, TurnoverRule
from etf_backtest.core.position import Position
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView, StrategyContext


def _view(trade_date: date) -> MarketBarView:
    return MarketBarView(
        source_record_key=f"QMT:510300:{trade_date.isoformat()}:1:1",
        symbol="SH.510300",
        trade_date=trade_date,
        open=Decimal("10"),
        high=Decimal("10"),
        low=Decimal("10"),
        close=Decimal("10"),
        volume=100,
        suspended=False,
    )


class _ProbeStrategy(BaseStrategy):
    @property
    def required_history_trading_days(self) -> int:
        return 1

    def should_generate_target(self, frame_index: int) -> bool:
        return True

    def _generate_target(self, **_kwargs: object) -> TargetPortfolio:
        return TargetPortfolio(weights={})


@pytest.mark.unit
def test_base_strategy_rejects_a_view_after_signal_date() -> None:
    account = Account(
        cash=Decimal("1000"),
        positions={"SH.510300": Position(symbol="SH.510300", turnover_rule=TurnoverRule.T1)},
    )
    account_view = AccountView.from_account(account)
    context = StrategyContext(
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        frame_index=0,
        symbols=("SH.510300",),
        account_view=account_view,
        current_weights_by_symbol={"SH.510300": Decimal("0")},
    )

    with pytest.raises(ValueError, match="future"):
        _ProbeStrategy().generate_target(
            signal_date=context.signal_date,
            market_history=(_view(date(2024, 1, 3)),),
            account_view=account_view,
            context=context,
        )
