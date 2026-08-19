"""Shared protocols and values for one daily strategy decision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from etf_backtest.core.market import IndexBarView, MarketBarView
    from etf_backtest.core.target import TargetPortfolio


class DecisionStatus(StrEnum):
    """The three possible outcomes of evaluating one strategy date."""

    NOT_SCHEDULED = "NOT_SCHEDULED"
    NO_REBALANCE = "NO_REBALANCE"
    TARGET_CREATED = "TARGET_CREATED"


@dataclass(frozen=True, slots=True)
class DailyDecisionResult:
    """The scheduling and target outcome for one signal date."""

    signal_date: date
    execution_date: date
    schedule_index: int
    status: DecisionStatus
    target_portfolio: TargetPortfolio | None

    def __post_init__(self) -> None:
        if type(self.schedule_index) is not int or self.schedule_index < 0:
            raise ValueError("schedule_index must be a non-negative integer")
        if not isinstance(self.status, DecisionStatus):
            raise TypeError("status must be DecisionStatus")
        if self.status is DecisionStatus.TARGET_CREATED:
            from etf_backtest.core.target import TargetPortfolio

            if not isinstance(self.target_portfolio, TargetPortfolio):
                raise TypeError("TARGET_CREATED requires a TargetPortfolio")
        elif self.target_portfolio is not None:
            raise ValueError("only TARGET_CREATED may contain a target portfolio")


class StrategyDataPortal(Protocol):
    """Read-only strategy data required to evaluate one daily target."""

    def views_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
        lookback_trading_days: int | None = None,
    ) -> tuple[MarketBarView, ...]: ...

    def share_history_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[date, Decimal]]: ...

    def huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]: ...

    def index_history_through(
        self,
        as_of_date: date,
        *,
        lookback_trading_days: int | None = None,
    ) -> Mapping[str, tuple[IndexBarView, ...]]: ...

    def combined_huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, tuple[date, Decimal]]: ...


__all__ = ["DailyDecisionResult", "DecisionStatus", "StrategyDataPortal"]
