"""Small, typed boundary for user-authored daily Rule strategies."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import ClassVar, cast

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import IndexBarView, MarketBarView
from etf_backtest.core.target import (
    NO_REBALANCE,
    NoRebalance,
    StrategyTarget,
    TargetPortfolio,
)
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountPositionView, AccountView, StrategyContext
from etf_backtest.strategy.scheduler import PeriodicDecisionScheduler

WeightInput = Decimal | str | int | float
RuleOutput = Mapping[str, WeightInput] | NoRebalance

_SYSTEM_PARAMETER_KEYS = frozenset(
    {
        "connection",
        "credentials",
        "database",
        "dsn",
        "host",
        "password",
        "password_env",
        "port",
        "snapshot",
        "user",
        "username",
    }
)


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _strict_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _decimal_weight(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise TypeError("user weights must not be boolean")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("user weight float must be finite")
        return Decimal(str(value))
    if type(value) is int:
        return Decimal(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("user weight text must not be blank")
        try:
            return Decimal(normalized)
        except InvalidOperation as exc:
            raise ValueError("user weight text must be a valid decimal") from exc
    raise TypeError("user weights must be Decimal, integer, float, or decimal text")


def _freeze_parameter(value: object, field_name: str) -> object:
    """Freeze a code-owned Rule parameter without accepting system settings."""

    if value is None or isinstance(value, str | bool):
        return value
    if type(value) is int:
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip():
                raise ValueError(f"{field_name} keys must be nonblank strings")
            if key.casefold() in _SYSTEM_PARAMETER_KEYS:
                raise ValueError(f"{field_name} must not contain system setting {key!r}")
            frozen[key] = _freeze_parameter(item, f"{field_name}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze_parameter(item, f"{field_name}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{field_name} contains unsupported value {type(value).__qualname__}")


def _resolved_parameter(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _resolved_parameter(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_resolved_parameter(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RuleSettings:
    """All Rule-specific controls, declared once inside the user's ``rule.py``.

    Users may change the schedule, target allocation and arbitrary strategy
    constants here.  Database, calendar, fees and execution rules deliberately
    cannot be configured through this object.
    """

    lookback_trading_days: int = 20
    rebalance_every_trading_days: int = 20
    target_weight: WeightInput = "0.90"
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lookback = _strict_positive_int(self.lookback_trading_days, "lookback_trading_days")
        if lookback < 2:
            raise ValueError("lookback_trading_days must be at least two")
        rebalance = _strict_positive_int(
            self.rebalance_every_trading_days,
            "rebalance_every_trading_days",
        )
        weight = _decimal_weight(self.target_weight)
        if not weight.is_finite() or not Decimal("0") < weight <= Decimal("1"):
            raise ValueError("target_weight must be in (0, 1]")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        parameters = _freeze_parameter(self.parameters, "parameters")
        if not isinstance(parameters, Mapping):  # pragma: no cover - checked above
            raise AssertionError("Rule parameter freezing failed")
        object.__setattr__(self, "lookback_trading_days", lookback)
        object.__setattr__(self, "rebalance_every_trading_days", rebalance)
        object.__setattr__(self, "target_weight", weight)
        object.__setattr__(self, "parameters", parameters)

    def resolved_dict(self) -> dict[str, object]:
        """Return stable, JSON-ready settings for experiment provenance."""

        return {
            "lookback_trading_days": self.lookback_trading_days,
            "rebalance_every_trading_days": self.rebalance_every_trading_days,
            "target_weight": str(self.target_weight),
            "parameters": _resolved_parameter(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class RuleMarketData:
    """Immutable, date-gated inputs presented to a :class:`UserRule`.

    Prices are the table-provided front-adjusted daily values.  ``volume`` and
    ``suspended`` retain the validated execution-source semantics.  No raw
    execution price or mutable account object crosses this boundary.
    """

    signal_date: date
    execution_date: date
    frame_index: int
    symbols: tuple[str, ...]
    cash: Decimal
    positions: Mapping[str, AccountPositionView]
    _bars_by_symbol: Mapping[str, tuple[MarketBarView, ...]] = field(repr=False)
    _current_weights_by_symbol: Mapping[str, Decimal] = field(repr=False)
    _share_history_by_symbol: Mapping[str, Mapping[date, Decimal]] = field(
        default_factory=dict,
        repr=False,
    )
    _huijin_ratios_by_symbol: Mapping[str, Mapping[str, tuple[date, Decimal]]] = field(
        default_factory=dict,
        repr=False,
    )
    _index_history_by_code: Mapping[str, tuple[IndexBarView, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    _combined_huijin_ratio_by_symbol: Mapping[str, tuple[date, Decimal]] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        signal_date = _plain_date(self.signal_date, "signal_date")
        execution_date = _plain_date(self.execution_date, "execution_date")
        if execution_date <= signal_date:
            raise ValueError("execution_date must follow signal_date")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")

        supplied_symbols = cast(object, self.symbols)
        if isinstance(supplied_symbols, (str, bytes)) or not isinstance(supplied_symbols, Sequence):
            raise TypeError("symbols must be a sequence")
        symbols = tuple(
            sorted(normalize_symbol(symbol) for symbol in cast(Sequence[str], supplied_symbols))
        )
        if not symbols or len(symbols) != len(set(symbols)):
            raise ValueError("symbols must be non-empty and unique")

        if not isinstance(self.cash, Decimal):
            raise TypeError("cash must be Decimal")
        if not self.cash.is_finite() or self.cash < 0:
            raise ValueError("cash must be finite and non-negative")

        if not isinstance(self.positions, Mapping):
            raise TypeError("positions must be a mapping")
        positions: dict[str, AccountPositionView] = {}
        for supplied_symbol, position in self.positions.items():
            symbol = normalize_symbol(supplied_symbol)
            if not isinstance(position, AccountPositionView):
                raise TypeError("positions may contain only AccountPositionView")
            if position.symbol != symbol or symbol in positions:
                raise ValueError("position key mismatch or duplicate")
            positions[symbol] = position
        if set(positions) != set(symbols):
            raise ValueError("positions must exactly cover symbols")

        if not isinstance(self._bars_by_symbol, Mapping):
            raise TypeError("bars_by_symbol must be a mapping")
        histories: dict[str, tuple[MarketBarView, ...]] = {symbol: () for symbol in symbols}
        seen_history_symbols: set[str] = set()
        for supplied_symbol, supplied_bars in self._bars_by_symbol.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in histories:
                raise ValueError("history contains a symbol outside the universe")
            if symbol in seen_history_symbols:
                raise ValueError("history symbols must be unique after normalization")
            seen_history_symbols.add(symbol)
            bars_value = cast(object, supplied_bars)
            if isinstance(bars_value, (str, bytes)) or not isinstance(bars_value, Sequence):
                raise TypeError("each symbol history must be a sequence")
            bars = tuple(cast(Sequence[MarketBarView], bars_value))
            previous_date: date | None = None
            for bar in bars:
                if not isinstance(bar, MarketBarView):
                    raise TypeError("history may contain only MarketBarView")
                if bar.symbol != symbol:
                    raise ValueError("history key and MarketBarView symbol disagree")
                if bar.trade_date > signal_date:
                    raise ValueError("history contains a future adjusted view")
                if previous_date is not None and bar.trade_date <= previous_date:
                    raise ValueError("symbol history must be strictly chronological")
                previous_date = bar.trade_date
            histories[symbol] = bars

        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "positions", MappingProxyType(dict(sorted(positions.items()))))
        object.__setattr__(
            self,
            "_bars_by_symbol",
            MappingProxyType(dict(sorted(histories.items()))),
        )
        context_view = StrategyContext(
            signal_date=signal_date,
            execution_date=execution_date,
            frame_index=self.frame_index,
            symbols=symbols,
            account_view=AccountView(cash=self.cash, positions=positions),
            current_weights_by_symbol=self._current_weights_by_symbol,
            share_history_by_symbol=self._share_history_by_symbol,
            huijin_ratios_by_symbol=self._huijin_ratios_by_symbol,
            index_history_by_code=self._index_history_by_code,
            combined_huijin_ratio_by_symbol=self._combined_huijin_ratio_by_symbol,
        )
        object.__setattr__(
            self,
            "_share_history_by_symbol",
            context_view.share_history_by_symbol,
        )
        object.__setattr__(
            self,
            "_current_weights_by_symbol",
            context_view.current_weights_by_symbol,
        )
        object.__setattr__(
            self,
            "_huijin_ratios_by_symbol",
            context_view.huijin_ratios_by_symbol,
        )
        object.__setattr__(self, "_index_history_by_code", context_view.index_history_by_code)
        object.__setattr__(
            self,
            "_combined_huijin_ratio_by_symbol",
            context_view.combined_huijin_ratio_by_symbol,
        )

    @classmethod
    def from_strategy_inputs(
        cls,
        *,
        market_history: Sequence[MarketBarView],
        account_view: AccountView,
        context: StrategyContext,
    ) -> RuleMarketData:
        """Build the friendly view from the engine's already-gated inputs."""

        if not isinstance(account_view, AccountView):
            raise TypeError("account_view must be AccountView")
        if not isinstance(context, StrategyContext):
            raise TypeError("context must be StrategyContext")
        if account_view is not context.account_view:
            raise ValueError("account_view must be the one stored in context")
        history_value = cast(object, market_history)
        if isinstance(history_value, (str, bytes)) or not isinstance(history_value, Sequence):
            raise TypeError("market_history must be a sequence")

        grouped: defaultdict[str, list[MarketBarView]] = defaultdict(list)
        for bar in cast(Sequence[MarketBarView], history_value):
            if not isinstance(bar, MarketBarView):
                raise TypeError("market_history may contain only MarketBarView")
            if bar.symbol not in context.symbols:
                raise ValueError("market_history contains a symbol outside the universe")
            grouped[bar.symbol].append(bar)
        ordered = {
            symbol: tuple(sorted(grouped.get(symbol, ()), key=lambda bar: bar.trade_date))
            for symbol in context.symbols
        }
        return cls(
            signal_date=context.signal_date,
            execution_date=context.execution_date,
            frame_index=context.frame_index,
            symbols=context.symbols,
            cash=account_view.cash,
            positions=account_view.positions,
            _bars_by_symbol=ordered,
            _current_weights_by_symbol=context.current_weights_by_symbol,
            _share_history_by_symbol=context.share_history_by_symbol,
            _huijin_ratios_by_symbol=context.huijin_ratios_by_symbol,
            _index_history_by_code=context.index_history_by_code,
            _combined_huijin_ratio_by_symbol=context.combined_huijin_ratio_by_symbol,
        )

    def bars(self, symbol: str) -> tuple[MarketBarView, ...]:
        """Return chronological front-adjusted bars, or an empty tuple."""

        return self._bars_by_symbol[self._known_symbol(symbol)]

    def closes(self, symbol: str) -> tuple[Decimal, ...]:
        return tuple(bar.close for bar in self.bars(symbol))

    def volumes(self, symbol: str) -> tuple[int, ...]:
        return tuple(bar.volume for bar in self.bars(symbol))

    def latest(self, symbol: str) -> MarketBarView | None:
        bars = self.bars(symbol)
        return bars[-1] if bars else None

    def current_weight(self, symbol: str) -> Decimal:
        """Return one universe symbol's signal-date raw-close portfolio weight."""

        return self._current_weights_by_symbol[self._known_symbol(symbol)]

    def share_on(self, symbol: str, asof_date: date) -> Decimal | None:
        """Return the exact daily ETF share observation, without forward filling."""

        value = _plain_date(asof_date, "asof_date")
        if value > self.signal_date:
            raise ValueError("cannot query a future share date")
        return self._share_history_by_symbol[self._known_symbol(symbol)].get(value)

    def share_history(self, symbol: str) -> tuple[tuple[date, Decimal], ...]:
        """Return chronological daily ETF shares visible through signal D."""

        rows = self._share_history_by_symbol[self._known_symbol(symbol)]
        return tuple(rows.items())

    def latest_huijin_ratio(self, symbol: str, company: str) -> Decimal | None:
        """Return the latest pre-D HolderOfListing ratio for one Huijin entity."""

        if not isinstance(company, str) or not company.strip():
            raise ValueError("company must be a nonblank string")
        value = self._huijin_ratios_by_symbol[self._known_symbol(symbol)].get(company.strip())
        return None if value is None else value[1]

    def index_bars(self, index_code: str) -> tuple[IndexBarView, ...]:
        """Return chronological configured index PRICE bars visible through signal D."""

        if not isinstance(index_code, str):
            raise TypeError("index_code must be a string")
        normalized = index_code.strip().upper()
        try:
            return self._index_history_by_code[normalized]
        except KeyError:
            raise ValueError(f"index code is not configured: {normalized}") from None

    def latest_combined_huijin_ratio(self, symbol: str) -> tuple[date, Decimal] | None:
        """Return the latest pre-D same-period ratio sum and its EndDate."""

        return self._combined_huijin_ratio_by_symbol.get(self._known_symbol(symbol))

    def has_history(self, symbol: str, observations: int) -> bool:
        required = _strict_positive_int(observations, "observations")
        return len(self.bars(symbol)) >= required

    def close_return(self, symbol: str, periods: int) -> Decimal | None:
        """Return ``close[D] / close[D-periods] - 1`` when available."""

        distance = _strict_positive_int(periods, "periods")
        closes = self.closes(symbol)
        if len(closes) <= distance:
            return None
        return closes[-1] / closes[-distance - 1] - Decimal("1")

    def position_quantity(self, symbol: str) -> int:
        return self.positions[self._known_symbol(symbol)].total_quantity

    def available_quantity(self, symbol: str) -> int:
        return self.positions[self._known_symbol(symbol)].available_quantity

    def _known_symbol(self, symbol: str) -> str:
        canonical = normalize_symbol(symbol)
        if canonical not in self._bars_by_symbol:
            raise ValueError(f"symbol is outside the configured universe: {canonical}")
        return canonical


class UserRule(ABC):
    """Implement one Rule and keep its settings beside the strategy code."""

    settings: ClassVar[RuleSettings] = RuleSettings()

    @property
    def target_weight(self) -> Decimal:
        """Return the code-owned default target allocation."""

        return cast(Decimal, self.settings.target_weight)

    @property
    def parameters(self) -> Mapping[str, object]:
        """Return immutable strategy parameters declared in ``rule.py``."""

        return self.settings.parameters

    @property
    def lookback_trading_days(self) -> int:
        return self.settings.lookback_trading_days

    @property
    def rebalance_every_trading_days(self) -> int:
        return self.settings.rebalance_every_trading_days

    @abstractmethod
    def generate_weights(self, data: RuleMarketData) -> RuleOutput:
        """Return target weights, or ``NO_REBALANCE`` to create no D+1 target."""


class SimpleRuleStrategy(BaseStrategy):
    """Adapt one :class:`UserRule` to the validated daily strategy engine."""

    __slots__ = ("_lookback_trading_days", "_rule", "_scheduler")

    def __init__(self, *, rule: UserRule) -> None:
        if not isinstance(rule, UserRule):
            raise TypeError("rule must be UserRule")
        if not isinstance(rule.settings, RuleSettings):
            raise TypeError("UserRule.settings must be RuleSettings")
        self._lookback_trading_days = rule.lookback_trading_days
        self._rule = rule
        self._scheduler = PeriodicDecisionScheduler(
            every_trading_days=rule.rebalance_every_trading_days
        )

    @property
    def rule(self) -> UserRule:
        return self._rule

    @property
    def required_history_trading_days(self) -> int:
        return self._lookback_trading_days

    def should_generate_target(self, frame_index: int) -> bool:
        if type(frame_index) is not int:
            raise TypeError("frame_index must be an integer")
        return self._scheduler.should_decide(frame_index)

    def _generate_target(
        self,
        *,
        signal_date: date,
        market_history: tuple[MarketBarView, ...],
        account_view: AccountView,
        context: StrategyContext,
    ) -> StrategyTarget:
        del signal_date
        data = RuleMarketData.from_strategy_inputs(
            market_history=market_history,
            account_view=account_view,
            context=context,
        )
        supplied_weights = self._rule.generate_weights(data)
        if isinstance(supplied_weights, NoRebalance):
            return NO_REBALANCE
        return self._target_from_user_weights(supplied_weights, symbols=data.symbols)

    @staticmethod
    def _target_from_user_weights(
        supplied_weights: Mapping[str, WeightInput],
        *,
        symbols: tuple[str, ...],
    ) -> TargetPortfolio:
        if not isinstance(supplied_weights, Mapping):
            raise TypeError("UserRule.generate_weights must return a mapping")
        allowed = frozenset(symbols)
        converted: dict[str, Decimal] = {}
        for supplied_symbol, supplied_weight in supplied_weights.items():
            if not isinstance(supplied_symbol, str):
                raise TypeError("user target symbols must be strings")
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError(f"user target symbol is outside the universe: {symbol}")
            if symbol in converted:
                raise ValueError("user target symbols must be unique after normalization")
            converted[symbol] = _decimal_weight(supplied_weight)
        return TargetPortfolio(weights=converted)


__all__ = [
    "NO_REBALANCE",
    "NoRebalance",
    "RuleMarketData",
    "RuleOutput",
    "RuleSettings",
    "SimpleRuleStrategy",
    "UserRule",
    "WeightInput",
]
