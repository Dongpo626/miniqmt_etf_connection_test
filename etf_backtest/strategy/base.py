"""Target-only daily strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, datetime
from typing import final

from etf_backtest.core.market import MarketBarView
from etf_backtest.core.target import NoRebalance, StrategyTarget, TargetPortfolio
from etf_backtest.strategy.context import AccountView, StrategyContext


class BaseStrategy(ABC):
    """Validate that strategies see only completed front-adjusted views."""

    @abstractmethod
    def should_generate_target(self, frame_index: int) -> bool:
        """Return whether this signal frame is on the strategy schedule."""

    @property
    @abstractmethod
    def required_history_trading_days(self) -> int:
        """Return the preferred front-view lookback supplied by the engine."""

    @final
    def generate_target(
        self,
        *,
        signal_date: date,
        market_history: Sequence[MarketBarView],
        account_view: AccountView,
        context: StrategyContext,
    ) -> StrategyTarget:
        if isinstance(signal_date, datetime) or not isinstance(signal_date, date):
            raise TypeError("signal_date must be datetime.date")
        if not isinstance(context, StrategyContext):
            raise TypeError("context must be StrategyContext")
        if signal_date != context.signal_date:
            raise ValueError("signal_date must match StrategyContext")
        if not isinstance(account_view, AccountView):
            raise TypeError("account_view must be AccountView")
        if account_view is not context.account_view:
            raise ValueError("account_view must be the view stored in StrategyContext")
        if not isinstance(market_history, Sequence):
            raise TypeError("market_history must be a sequence")

        frozen = tuple(market_history)
        for view in frozen:
            if not isinstance(view, MarketBarView):
                raise TypeError("market_history may contain only MarketBarView")
            if view.trade_date > signal_date:
                raise ValueError("market_history contains a future adjusted view")
            if view.symbol not in context.symbols:
                raise ValueError("market_history contains a symbol outside the universe")
        target = self._generate_target(
            signal_date=signal_date,
            market_history=frozen,
            account_view=account_view,
            context=context,
        )
        if not isinstance(target, TargetPortfolio | NoRebalance):
            raise TypeError("strategy must return TargetPortfolio or NoRebalance")
        return target

    @abstractmethod
    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> StrategyTarget:
        raise NotImplementedError


__all__ = ["BaseStrategy"]
