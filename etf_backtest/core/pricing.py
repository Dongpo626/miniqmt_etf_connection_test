"""The single production execution-price model: unadjusted daily close."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import (
    EtfTradingRule,
    MarketBar,
    MarketFrame,
    resolve_legal_price_limits,
)
from etf_backtest.core.order import TradePriceQuote


class CloseTradePriceModel:
    """Resolve raw close and effective-rule price limits without direction."""

    price_source = "CLOSE"

    def resolve(
        self,
        *,
        execution_bar: MarketBar,
        trading_rule: EtfTradingRule,
    ) -> TradePriceQuote:
        if not isinstance(execution_bar, MarketBar):
            raise TypeError("execution_bar must be MarketBar")
        if not isinstance(trading_rule, EtfTradingRule):
            raise TypeError("trading_rule must be EtfTradingRule")
        if trading_rule.symbol != execution_bar.symbol:
            raise ValueError("trading rule symbol does not match the execution bar")

        lower, upper, source = resolve_legal_price_limits(
            execution_bar=execution_bar,
            trading_rule=trading_rule,
        )
        return TradePriceQuote(
            source_record_key=execution_bar.source_record_key,
            symbol=execution_bar.symbol,
            trade_date=execution_bar.trade_date,
            base_trade_price=execution_bar.close,
            price_limit_down=lower,
            price_limit_up=upper,
            price_limit_source=source,
        )


class TradePriceQuoteCache:
    """Cache one close quote for every symbol in a complete daily frame."""

    __slots__ = ("_frame", "_model", "_quotes", "_rules")

    def __init__(
        self,
        *,
        frame: MarketFrame,
        trading_rules: Mapping[str, EtfTradingRule],
        model: CloseTradePriceModel | None = None,
    ) -> None:
        if not isinstance(frame, MarketFrame):
            raise TypeError("frame must be MarketFrame")
        if not isinstance(trading_rules, Mapping):
            raise TypeError("trading_rules must be a mapping")
        resolved: dict[str, EtfTradingRule] = {}
        for supplied_symbol, rule in trading_rules.items():
            symbol = normalize_symbol(supplied_symbol)
            if not isinstance(rule, EtfTradingRule):
                raise TypeError("trading_rules may contain only EtfTradingRule")
            if symbol != rule.symbol or symbol in resolved:
                raise ValueError("trading rule key mismatch or duplicate")
            resolved[symbol] = rule
        self._frame = frame
        self._rules = MappingProxyType(dict(sorted(resolved.items())))
        self._model = CloseTradePriceModel() if model is None else model
        if not isinstance(self._model, CloseTradePriceModel):
            raise TypeError("model must be CloseTradePriceModel")
        self._quotes: dict[str, TradePriceQuote] = {}

    def quote_for(self, symbol: str) -> TradePriceQuote:
        canonical = normalize_symbol(symbol)
        cached = self._quotes.get(canonical)
        if cached is not None:
            return cached
        try:
            rule = self._rules[canonical]
        except KeyError:
            raise KeyError(f"no effective trading rule for {canonical}") from None
        quote = self._model.resolve(
            execution_bar=self._frame.bar_for(canonical),
            trading_rule=rule,
        )
        self._quotes[canonical] = quote
        return quote

    def resolve_all(self, symbols: Iterable[str]) -> Mapping[str, TradePriceQuote]:
        canonical = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
        return MappingProxyType({symbol: self.quote_for(symbol) for symbol in canonical})


__all__ = ["CloseTradePriceModel", "TradePriceQuoteCache"]
