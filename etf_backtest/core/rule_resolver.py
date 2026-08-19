"""Narrow effective-dated rule boundary required by the daily engine."""

from datetime import date
from typing import Protocol, runtime_checkable

from etf_backtest.core.market import EtfTradingRule


@runtime_checkable
class RuleResolver(Protocol):
    """Resolve the rule effective for exactly one symbol and trade date."""

    def resolve(self, symbol: str, trade_date: date) -> EtfTradingRule: ...


__all__ = ["RuleResolver"]
