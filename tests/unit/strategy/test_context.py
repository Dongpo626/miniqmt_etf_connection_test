"""Price-free strategy context tests."""

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.account import Account
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.position import Position
from etf_backtest.strategy.context import AccountView, StrategyContext


def _account() -> Account:
    return Account(
        cash=Decimal("1000"),
        positions={
            "SH.510300": Position(
                symbol="SH.510300",
                turnover_rule=TurnoverRule.T1,
                total_quantity=100,
                available_quantity=100,
            )
        },
    )


@pytest.mark.unit
def test_account_view_exposes_cash_and_quantities_but_no_prices() -> None:
    view = AccountView.from_account(_account())

    assert view.cash == Decimal("1000")
    assert view.positions["SH.510300"].total_quantity == 100
    assert not hasattr(view, "mark_close_prices")
    assert not hasattr(view.positions["SH.510300"], "market_value")


@pytest.mark.unit
def test_strategy_context_binds_signal_to_strictly_later_execution_date() -> None:
    account_view = AccountView.from_account(_account())
    context = StrategyContext(
        signal_date=date(2024, 1, 2),
        execution_date=date(2024, 1, 3),
        frame_index=0,
        symbols=("SH.510300",),
        account_view=account_view,
        current_weights_by_symbol={"SH.510300": Decimal("0.25")},
    )

    assert context.execution_date > context.signal_date
    assert context.current_weights_by_symbol == {"SH.510300": Decimal("0.25")}
    with pytest.raises(TypeError):
        context.current_weights_by_symbol["SH.510300"] = Decimal("0")  # type: ignore[index]
    with pytest.raises(ValueError, match="follow"):
        StrategyContext(
            signal_date=date(2024, 1, 2),
            execution_date=date(2024, 1, 2),
            frame_index=0,
            symbols=("SH.510300",),
            account_view=account_view,
            current_weights_by_symbol={"SH.510300": Decimal("0.25")},
        )


@pytest.mark.unit
def test_strategy_context_rejects_incomplete_current_weights() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        StrategyContext(
            signal_date=date(2024, 1, 2),
            execution_date=date(2024, 1, 3),
            frame_index=0,
            symbols=("SH.510300",),
            account_view=AccountView.from_account(_account()),
            current_weights_by_symbol={},
        )
