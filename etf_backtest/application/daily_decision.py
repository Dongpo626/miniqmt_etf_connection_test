"""Single shared path for evaluating a daily strategy target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING

from etf_backtest.application.contracts import (
    DailyDecisionResult,
    DecisionStatus,
    StrategyDataPortal,
)

if TYPE_CHECKING:
    from etf_backtest.strategy.base import BaseStrategy
    from etf_backtest.strategy.context import AccountView


class DailyDecisionService:
    """Prepare strategy inputs and classify one daily decision."""

    def evaluate(
        self,
        *,
        strategy: BaseStrategy,
        portal: StrategyDataPortal,
        signal_date: date,
        execution_date: date,
        schedule_index: int,
        symbols: Sequence[str],
        account_view: AccountView,
        current_weights_by_symbol: Mapping[str, Decimal],
    ) -> DailyDecisionResult:
        from etf_backtest.core.target import NoRebalance
        from etf_backtest.strategy.context import StrategyContext

        if not strategy.should_generate_target(schedule_index):
            return DailyDecisionResult(
                signal_date=signal_date,
                execution_date=execution_date,
                schedule_index=schedule_index,
                status=DecisionStatus.NOT_SCHEDULED,
                target_portfolio=None,
            )

        selected_symbols = tuple(symbols)
        lookback = strategy.required_history_trading_days
        context = StrategyContext(
            signal_date=signal_date,
            execution_date=execution_date,
            frame_index=schedule_index,
            symbols=selected_symbols,
            account_view=account_view,
            current_weights_by_symbol=current_weights_by_symbol,
            share_history_by_symbol=portal.share_history_through(
                signal_date,
                symbols=selected_symbols,
            ),
            huijin_ratios_by_symbol=portal.huijin_ratios_as_of(
                signal_date,
                symbols=selected_symbols,
            ),
            index_history_by_code=portal.index_history_through(
                signal_date,
                lookback_trading_days=lookback,
            ),
            combined_huijin_ratio_by_symbol=portal.combined_huijin_ratios_as_of(
                signal_date,
                symbols=selected_symbols,
            ),
        )
        history = portal.views_through(
            signal_date,
            symbols=selected_symbols,
            lookback_trading_days=lookback,
        )
        target = strategy.generate_target(
            signal_date=signal_date,
            market_history=history,
            account_view=account_view,
            context=context,
        )
        if isinstance(target, NoRebalance):
            return DailyDecisionResult(
                signal_date=signal_date,
                execution_date=execution_date,
                schedule_index=schedule_index,
                status=DecisionStatus.NO_REBALANCE,
                target_portfolio=None,
            )
        return DailyDecisionResult(
            signal_date=signal_date,
            execution_date=execution_date,
            schedule_index=schedule_index,
            status=DecisionStatus.TARGET_CREATED,
            target_portfolio=target,
        )


__all__ = ["DailyDecisionService"]
