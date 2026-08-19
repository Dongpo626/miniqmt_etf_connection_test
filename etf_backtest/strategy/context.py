"""Small price-free strategy context for the daily engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import cast

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import Account
from etf_backtest.core.market import IndexBarView, TurnoverRule
from etf_backtest.core.position import Position


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _quantity(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class AccountPositionView:
    """Immutable share buckets with no price or account mutator."""

    symbol: str
    turnover_rule: TurnoverRule
    total_quantity: int
    available_quantity: int
    today_buy_quantity: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.turnover_rule, TurnoverRule):
            raise TypeError("turnover_rule must be TurnoverRule")
        total = _quantity(self.total_quantity, "total_quantity")
        available = _quantity(self.available_quantity, "available_quantity")
        today = _quantity(self.today_buy_quantity, "today_buy_quantity")
        if available > total or today > total:
            raise ValueError("position view buckets cannot exceed total quantity")
        if self.turnover_rule is TurnoverRule.T0:
            if available != total or today != 0:
                raise ValueError("T0 position view requires immediate availability")
        elif available + today != total:
            raise ValueError("T1 position view buckets must sum to total")

    @classmethod
    def from_position(cls, position: Position) -> AccountPositionView:
        if not isinstance(position, Position):
            raise TypeError("position must be Position")
        return cls(
            symbol=position.symbol,
            turnover_rule=position.turnover_rule,
            total_quantity=position.total_quantity,
            available_quantity=position.available_quantity,
            today_buy_quantity=position.today_buy_quantity,
        )


@dataclass(frozen=True, slots=True)
class AccountView:
    """Immutable cash/quantity state that intentionally contains no prices."""

    cash: Decimal
    positions: Mapping[str, AccountPositionView]

    def __post_init__(self) -> None:
        if not isinstance(self.cash, Decimal):
            raise TypeError("cash must be Decimal")
        if not self.cash.is_finite() or self.cash < 0:
            raise ValueError("cash must be finite and non-negative")
        if not isinstance(self.positions, Mapping):
            raise TypeError("positions must be a mapping")
        canonical: dict[str, AccountPositionView] = {}
        for supplied_symbol, value in self.positions.items():
            symbol = normalize_symbol(supplied_symbol)
            if not isinstance(value, AccountPositionView):
                raise TypeError("positions may contain only AccountPositionView")
            if value.symbol != symbol or symbol in canonical:
                raise ValueError("position view key mismatch or duplicate")
            canonical[symbol] = value
        object.__setattr__(self, "positions", MappingProxyType(dict(sorted(canonical.items()))))

    @classmethod
    def from_account(cls, account: Account) -> AccountView:
        if not isinstance(account, Account):
            raise TypeError("account must be Account")
        return cls(
            cash=account.cash,
            positions={
                symbol: AccountPositionView.from_position(position)
                for symbol, position in account.positions.items()
            },
        )


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Known D/D+1 identity plus price-free account and position-weight views."""

    signal_date: date
    execution_date: date
    frame_index: int
    symbols: tuple[str, ...]
    account_view: AccountView
    current_weights_by_symbol: Mapping[str, Decimal]
    share_history_by_symbol: Mapping[str, Mapping[date, Decimal]] = field(default_factory=dict)
    huijin_ratios_by_symbol: Mapping[str, Mapping[str, tuple[date, Decimal]]] = field(
        default_factory=dict
    )
    index_history_by_code: Mapping[str, tuple[IndexBarView, ...]] = field(default_factory=dict)
    combined_huijin_ratio_by_symbol: Mapping[str, tuple[date, Decimal]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        signal = _plain_date(self.signal_date, "signal_date")
        execution = _plain_date(self.execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        symbols = cast(object, self.symbols)
        if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
            raise TypeError("symbols must be a sequence")
        canonical = tuple(
            sorted(normalize_symbol(symbol) for symbol in cast(Sequence[str], symbols))
        )
        if not canonical or len(canonical) != len(set(canonical)):
            raise ValueError("symbols must be non-empty and unique")
        if not isinstance(self.account_view, AccountView):
            raise TypeError("account_view must be AccountView")
        if set(canonical) != set(self.account_view.positions):
            raise ValueError("symbols must exactly cover registered account positions")
        object.__setattr__(self, "symbols", canonical)
        object.__setattr__(
            self,
            "current_weights_by_symbol",
            self._freeze_current_weights(
                self.current_weights_by_symbol,
                symbols=canonical,
            ),
        )
        object.__setattr__(
            self,
            "share_history_by_symbol",
            self._freeze_share_history(
                self.share_history_by_symbol,
                symbols=canonical,
                signal_date=signal,
            ),
        )
        object.__setattr__(
            self,
            "huijin_ratios_by_symbol",
            self._freeze_huijin_ratios(
                self.huijin_ratios_by_symbol,
                symbols=canonical,
                signal_date=signal,
            ),
        )
        object.__setattr__(
            self,
            "index_history_by_code",
            self._freeze_index_history(self.index_history_by_code, signal_date=signal),
        )
        object.__setattr__(
            self,
            "combined_huijin_ratio_by_symbol",
            self._freeze_combined_huijin_ratios(
                self.combined_huijin_ratio_by_symbol,
                symbols=canonical,
                signal_date=signal,
            ),
        )

    @staticmethod
    def _freeze_current_weights(
        supplied: Mapping[str, Decimal],
        *,
        symbols: tuple[str, ...],
    ) -> Mapping[str, Decimal]:
        if not isinstance(supplied, Mapping):
            raise TypeError("current_weights_by_symbol must be a mapping")
        allowed = frozenset(symbols)
        result: dict[str, Decimal] = {}
        total = Decimal("0")
        for supplied_symbol, weight in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError("current weights contain a symbol outside the universe")
            if symbol in result:
                raise ValueError("current weight symbols must be unique after normalization")
            if not isinstance(weight, Decimal):
                raise TypeError("current weights must be Decimal")
            if not weight.is_finite() or not Decimal("0") <= weight <= Decimal("1"):
                raise ValueError("current weights must be finite and in [0, 1]")
            result[symbol] = weight
            total += weight
        if set(result) != allowed:
            raise ValueError("current weights must exactly cover symbols")
        if total > Decimal("1"):
            raise ValueError("current weights must not sum to more than one")
        return MappingProxyType(dict(sorted(result.items())))

    @staticmethod
    def _freeze_combined_huijin_ratios(
        supplied: Mapping[str, tuple[date, Decimal]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, tuple[date, Decimal]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("combined_huijin_ratio_by_symbol must be a mapping")
        allowed = frozenset(symbols)
        result: dict[str, tuple[date, Decimal]] = {}
        for supplied_symbol, supplied_value in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in allowed:
                raise ValueError("combined Huijin ratio contains a symbol outside the universe")
            if symbol in result:
                raise ValueError("combined Huijin ratio symbols must be unique")
            if not isinstance(supplied_value, tuple) or len(supplied_value) != 2:
                raise TypeError("combined Huijin ratio must be an (end_date, Decimal) tuple")
            end_date = _plain_date(supplied_value[0], "combined Huijin end_date")
            ratio = supplied_value[1]
            if end_date >= signal_date:
                raise ValueError("combined Huijin ratio must be strictly earlier than signal_date")
            if not isinstance(ratio, Decimal):
                raise TypeError("combined Huijin ratio must be Decimal")
            if not ratio.is_finite() or not Decimal("0") <= ratio <= Decimal("1"):
                raise ValueError("combined Huijin ratio must be finite and in [0, 1]")
            result[symbol] = (end_date, ratio)
        return MappingProxyType(dict(sorted(result.items())))

    @staticmethod
    def _freeze_index_history(
        supplied: Mapping[str, tuple[IndexBarView, ...]],
        *,
        signal_date: date,
    ) -> Mapping[str, tuple[IndexBarView, ...]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("index_history_by_code must be a mapping")
        histories: dict[str, tuple[IndexBarView, ...]] = {}
        for index_code, supplied_bars in supplied.items():
            supplied_bars_value = cast(object, supplied_bars)
            if isinstance(supplied_bars_value, (str, bytes)) or not isinstance(
                supplied_bars_value, Sequence
            ):
                raise TypeError("each index history must be a sequence")
            bars = tuple(cast(Sequence[IndexBarView], supplied_bars_value))
            previous_date: date | None = None
            for bar in bars:
                if not isinstance(bar, IndexBarView):
                    raise TypeError("index history may contain only IndexBarView")
                if bar.index_code != index_code:
                    raise ValueError("index history key and IndexBarView code disagree")
                if bar.trade_date > signal_date:
                    raise ValueError("index history contains a future view")
                if previous_date is not None and bar.trade_date <= previous_date:
                    raise ValueError("index history must be strictly chronological")
                previous_date = bar.trade_date
            histories[index_code] = bars
        return MappingProxyType(dict(sorted(histories.items())))

    @staticmethod
    def _freeze_share_history(
        supplied: Mapping[str, Mapping[date, Decimal]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, Mapping[date, Decimal]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("share_history_by_symbol must be a mapping")
        histories: dict[str, Mapping[date, Decimal]] = {
            symbol: MappingProxyType({}) for symbol in symbols
        }
        seen: set[str] = set()
        for supplied_symbol, supplied_rows in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in histories:
                raise ValueError("share history contains a symbol outside the universe")
            if symbol in seen:
                raise ValueError("share history symbols must be unique after normalization")
            seen.add(symbol)
            if not isinstance(supplied_rows, Mapping):
                raise TypeError("each share history must be a mapping")
            rows: dict[date, Decimal] = {}
            for supplied_date, total_share in supplied_rows.items():
                asof_date = _plain_date(supplied_date, "share asof_date")
                if asof_date > signal_date:
                    raise ValueError("share history contains a future business date")
                if not isinstance(total_share, Decimal):
                    raise TypeError("total_share must be Decimal")
                if not total_share.is_finite() or total_share < 0:
                    raise ValueError("total_share must be finite and non-negative")
                rows[asof_date] = total_share
            histories[symbol] = MappingProxyType(dict(sorted(rows.items())))
        return MappingProxyType(histories)

    @staticmethod
    def _freeze_huijin_ratios(
        supplied: Mapping[str, Mapping[str, tuple[date, Decimal]]],
        *,
        symbols: tuple[str, ...],
        signal_date: date,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]:
        if not isinstance(supplied, Mapping):
            raise TypeError("huijin_ratios_by_symbol must be a mapping")
        snapshots: dict[str, Mapping[str, tuple[date, Decimal]]] = {
            symbol: MappingProxyType({}) for symbol in symbols
        }
        seen: set[str] = set()
        for supplied_symbol, supplied_entities in supplied.items():
            symbol = normalize_symbol(supplied_symbol)
            if symbol not in snapshots:
                raise ValueError("Huijin ratios contain a symbol outside the universe")
            if symbol in seen:
                raise ValueError("Huijin ratio symbols must be unique after normalization")
            seen.add(symbol)
            if not isinstance(supplied_entities, Mapping):
                raise TypeError("each Huijin ratio snapshot must be a mapping")
            entities: dict[str, tuple[date, Decimal]] = {}
            for supplied_entity, supplied_value in supplied_entities.items():
                if not isinstance(supplied_entity, str) or not supplied_entity.strip():
                    raise ValueError("Huijin entity must be a nonblank string")
                entity = supplied_entity.strip()
                if not isinstance(supplied_value, tuple) or len(supplied_value) != 2:
                    raise TypeError("Huijin ratio must be an (end_date, Decimal) tuple")
                end_date = _plain_date(supplied_value[0], "Huijin end_date")
                ratio = supplied_value[1]
                if end_date >= signal_date:
                    raise ValueError("Huijin ratio must be strictly earlier than signal_date")
                if not isinstance(ratio, Decimal):
                    raise TypeError("Huijin ratio must be Decimal")
                if not ratio.is_finite() or not Decimal("0") <= ratio <= Decimal("1"):
                    raise ValueError("Huijin ratio must be finite and in [0, 1]")
                if entity in entities:
                    raise ValueError("Huijin entities must be unique")
                entities[entity] = (end_date, ratio)
            snapshots[symbol] = MappingProxyType(dict(sorted(entities.items())))
        return MappingProxyType(snapshots)


__all__ = ["AccountPositionView", "AccountView", "StrategyContext"]
