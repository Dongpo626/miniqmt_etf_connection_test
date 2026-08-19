"""Daily-only market-domain values.

Raw QMT bars are used exclusively by execution and account valuation.  The
independent front-adjusted views are the only price objects exposed to a
strategy or model.  No object in this module selects data or opens MySQL.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from types import MappingProxyType

from etf_backtest.config.schema import (
    CALENDAR_POLICY,
    MARKET_TIMEZONE,
    normalize_index_code,
    normalize_symbol,
)


class Exchange(StrEnum):
    """Mainland exchanges represented by the supported symbols."""

    SSE = "SSE"
    SZSE = "SZSE"


class EtfCategory(StrEnum):
    """The only two ETF categories admitted to a run."""

    DOMESTIC_STOCK_ETF = "DOMESTIC_STOCK_ETF"
    GOLD_ETF = "GOLD_ETF"


class TurnoverRule(StrEnum):
    """Same-day sell availability."""

    T0 = "T0"
    T1 = "T1"


class PriceLimitSource(StrEnum):
    """Origin of the legal daily price boundaries used for execution."""

    TUSHARE_EXPLICIT = "TUSHARE_EXPLICIT"
    DERIVED_RULE_FALLBACK = "DERIVED_RULE_FALLBACK"


def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _strict_int(value: object, field_name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _ohlc(
    open_price: Decimal,
    high_price: Decimal,
    low_price: Decimal,
    close_price: Decimal,
) -> None:
    for field_name, value in (
        ("open", open_price),
        ("high", high_price),
        ("low", low_price),
        ("close", close_price),
    ):
        _decimal(value, field_name, positive=True)
    if high_price < max(open_price, close_price, low_price):
        raise ValueError("high must be at least open, close and low")
    if low_price > min(open_price, close_price, high_price):
        raise ValueError("low must be at most open, close and high")


@dataclass(frozen=True, slots=True)
class FrameKey:
    """Identity of one SSE-calendar daily close frame."""

    trade_date: date
    calendar_version: str
    calendar_policy: str = CALENDAR_POLICY

    def __post_init__(self) -> None:
        _plain_date(self.trade_date, "trade_date")
        object.__setattr__(
            self, "calendar_version", _non_blank(self.calendar_version, "calendar_version")
        )
        if self.calendar_policy != CALENDAR_POLICY:
            raise ValueError(f"calendar_policy must be {CALENDAR_POLICY}")

    @property
    def close_time(self) -> datetime:
        """Return the fixed observable/executable daily close timestamp."""

        return datetime.combine(self.trade_date, time(15, 0), tzinfo=MARKET_TIMEZONE)


@dataclass(frozen=True, slots=True)
class MarketBar:
    """One unadjusted QMT daily bar used for execution and valuation."""

    source_record_key: str
    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    pre_close: Decimal
    volume: int
    amount: Decimal
    suspended: bool
    price_limit_down: Decimal | None = None
    price_limit_up: Decimal | None = None
    price_limit_source: PriceLimitSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_record_key",
            _non_blank(self.source_record_key, "source_record_key"),
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        _decimal(self.pre_close, "pre_close", positive=True)
        _strict_int(self.volume, "volume")
        _decimal(self.amount, "amount", non_negative=True)
        if not isinstance(self.suspended, bool):
            raise TypeError("suspended must be bool")
        lower = self.price_limit_down
        upper = self.price_limit_up
        if (lower is None) != (upper is None):
            raise ValueError("explicit price limits must be supplied as a complete pair")
        if lower is None:
            if self.price_limit_source is not None:
                raise ValueError("price_limit_source requires explicit price limits")
            return
        if self.price_limit_source is not PriceLimitSource.TUSHARE_EXPLICIT:
            raise ValueError("explicit price limits must use TUSHARE_EXPLICIT source")
        lower = _decimal(lower, "price_limit_down", positive=True)
        upper = _decimal(upper, "price_limit_up", positive=True)
        if not lower <= self.close <= upper:
            raise ValueError("raw close must stay inside explicit legal price limits")


@dataclass(frozen=True, slots=True)
class MarketBarView:
    """Independent front-adjusted daily strategy view."""

    source_record_key: str
    symbol: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    suspended: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_record_key",
            _non_blank(self.source_record_key, "source_record_key"),
        )
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        _strict_int(self.volume, "volume")
        if not isinstance(self.suspended, bool):
            raise TypeError("suspended must be bool")

    @property
    def signal_time(self) -> datetime:
        """Return the timestamp at which this view first enters strategy history."""

        return datetime.combine(self.trade_date, time(15, 0), tzinfo=MARKET_TIMEZONE)


@dataclass(frozen=True, slots=True)
class IndexBarView:
    """One source-native PRICE index bar visible to Rule strategies only."""

    index_code: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    pre_close: Decimal | None
    pct_change: Decimal | None
    source_system: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index_code", normalize_index_code(self.index_code))
        _plain_date(self.trade_date, "trade_date")
        _ohlc(self.open, self.high, self.low, self.close)
        if self.pre_close is not None:
            _decimal(self.pre_close, "pre_close", positive=True)
        if self.pct_change is not None:
            _decimal(self.pct_change, "pct_change")
        if _non_blank(self.source_system, "source_system") != "TUSHARE":
            raise ValueError("index source_system must be TUSHARE")


@dataclass(frozen=True, slots=True)
class MarketFrame:
    """Complete raw daily bars for one SSE trading date."""

    frame_key: FrameKey
    bars_by_symbol: Mapping[str, MarketBar]

    def __post_init__(self) -> None:
        if not isinstance(self.frame_key, FrameKey):
            raise TypeError("frame_key must be FrameKey")
        if not isinstance(self.bars_by_symbol, Mapping) or not self.bars_by_symbol:
            raise ValueError("bars_by_symbol must be a non-empty mapping")
        canonical: dict[str, MarketBar] = {}
        source_keys: set[str] = set()
        for supplied_symbol, bar in self.bars_by_symbol.items():
            if not isinstance(bar, MarketBar):
                raise TypeError("MarketFrame may contain only MarketBar")
            symbol = normalize_symbol(supplied_symbol)
            if bar.symbol != symbol:
                raise ValueError("bar mapping key must equal MarketBar.symbol")
            if bar.trade_date != self.frame_key.trade_date:
                raise ValueError("all bars must match the frame trade_date")
            if symbol in canonical or bar.source_record_key in source_keys:
                raise ValueError("duplicate symbol or source record in MarketFrame")
            canonical[symbol] = bar
            source_keys.add(bar.source_record_key)
        object.__setattr__(
            self, "bars_by_symbol", MappingProxyType(dict(sorted(canonical.items())))
        )

    @classmethod
    def from_bars(cls, frame_key: FrameKey, bars: Iterable[MarketBar]) -> MarketFrame:
        """Build a deterministic frame from raw bars."""

        indexed: dict[str, MarketBar] = {}
        for bar in bars:
            if not isinstance(bar, MarketBar):
                raise TypeError("bars must contain MarketBar")
            if bar.symbol in indexed:
                raise ValueError(f"duplicate symbol in frame: {bar.symbol}")
            indexed[bar.symbol] = bar
        return cls(frame_key=frame_key, bars_by_symbol=indexed)

    @property
    def trade_date(self) -> date:
        return self.frame_key.trade_date

    @property
    def canonical_symbols(self) -> tuple[str, ...]:
        return tuple(self.bars_by_symbol)

    def bar_for(self, symbol: str) -> MarketBar:
        return self.bars_by_symbol[normalize_symbol(symbol)]


@dataclass(frozen=True, slots=True)
class EtfInfo:
    """Frozen current ``dim_etf`` metadata used to resolve a run universe."""

    symbol: str
    exchange: Exchange
    name: str
    primary_category: str
    fund_type: str
    list_date: date
    delist_date: date | None
    current_status: str
    delist_date_approximated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.exchange, Exchange):
            raise TypeError("exchange must be Exchange")
        object.__setattr__(self, "name", _non_blank(self.name, "name"))
        object.__setattr__(
            self,
            "primary_category",
            _non_blank(self.primary_category, "primary_category"),
        )
        object.__setattr__(self, "fund_type", _non_blank(self.fund_type, "fund_type"))
        list_date = _plain_date(self.list_date, "list_date")
        if self.delist_date is not None:
            delist_date = _plain_date(self.delist_date, "delist_date")
            if delist_date < list_date:
                raise ValueError("delist_date must not precede list_date")
        object.__setattr__(
            self, "current_status", _non_blank(self.current_status, "current_status")
        )
        if not isinstance(self.delist_date_approximated, bool):
            raise TypeError("delist_date_approximated must be bool")
        expected_exchange = Exchange.SSE if self.symbol.startswith("SH.") else Exchange.SZSE
        if self.exchange is not expected_exchange:
            raise ValueError("exchange does not match normalized symbol")

    def is_active(self, trade_date: date) -> bool:
        """Return whether the frozen lifecycle interval includes a date."""

        value = _plain_date(trade_date, "trade_date")
        return self.list_date <= value and (self.delist_date is None or value <= self.delist_date)


@dataclass(frozen=True, slots=True)
class EtfTradingRule:
    """An effective-dated rule result; category and limit are independent."""

    symbol: str
    etf_category: EtfCategory
    turnover_rule: TurnoverRule
    price_limit_ratio: Decimal
    lot_size: int = 100
    tick_size: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        if not isinstance(self.etf_category, EtfCategory):
            raise TypeError("etf_category must be EtfCategory")
        if not isinstance(self.turnover_rule, TurnoverRule):
            raise TypeError("turnover_rule must be TurnoverRule")
        ratio = _decimal(self.price_limit_ratio, "price_limit_ratio", positive=True)
        if ratio >= 1:
            raise ValueError("price_limit_ratio must be less than one")
        _strict_int(self.lot_size, "lot_size", positive=True)
        _decimal(self.tick_size, "tick_size", positive=True)
        expected_turnover = (
            TurnoverRule.T0 if self.etf_category is EtfCategory.GOLD_ETF else TurnoverRule.T1
        )
        if self.turnover_rule is not expected_turnover:
            raise ValueError("turnover rule conflicts with the supported ETF category")


def resolve_legal_price_limits(
    *,
    execution_bar: MarketBar,
    trading_rule: EtfTradingRule,
) -> tuple[Decimal, Decimal, PriceLimitSource]:
    """Resolve explicit legal prices first, then the effective-ratio fallback."""

    if not isinstance(execution_bar, MarketBar):
        raise TypeError("execution_bar must be MarketBar")
    if not isinstance(trading_rule, EtfTradingRule):
        raise TypeError("trading_rule must be EtfTradingRule")
    if trading_rule.symbol != execution_bar.symbol:
        raise ValueError("trading rule symbol does not match the execution bar")

    lower = execution_bar.price_limit_down
    upper = execution_bar.price_limit_up
    if lower is not None and upper is not None:
        for value in (lower, upper):
            ticks = value / trading_rule.tick_size
            if ticks != ticks.to_integral_value():
                raise ValueError("explicit legal price limit is not aligned to the ETF tick")
        return lower, upper, PriceLimitSource.TUSHARE_EXPLICIT

    def round_to_tick(value: Decimal) -> Decimal:
        ticks = (value / trading_rule.tick_size).to_integral_value(rounding=ROUND_HALF_UP)
        rounded = ticks * trading_rule.tick_size
        if not rounded.is_finite() or rounded <= 0:
            raise ValueError("calculated legal price limit must be positive")
        return rounded

    ratio = trading_rule.price_limit_ratio
    return (
        round_to_tick(execution_bar.pre_close * (Decimal("1") - ratio)),
        round_to_tick(execution_bar.pre_close * (Decimal("1") + ratio)),
        PriceLimitSource.DERIVED_RULE_FALLBACK,
    )


__all__ = [
    "EtfCategory",
    "EtfInfo",
    "EtfTradingRule",
    "Exchange",
    "FrameKey",
    "IndexBarView",
    "MarketBar",
    "MarketBarView",
    "MarketFrame",
    "PriceLimitSource",
    "TurnoverRule",
    "resolve_legal_price_limits",
]
