"""Protocol used to obtain already-normalized live quotes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from etf_backtest.live.state import LiveQuote, QueryResult


class QuoteProvider(Protocol):
    def subscribe(self, symbols: Sequence[str]) -> None: ...

    def latest_quotes(self, symbols: Sequence[str]) -> QueryResult[LiveQuote]: ...


__all__ = ["QuoteProvider"]
