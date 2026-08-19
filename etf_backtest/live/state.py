"""Small immutable values shared by the internal live-trading boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")


class OrderIntentStatus(StrEnum):
    PLANNED = "PLANNED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    REJECTED = "REJECTED"


class BrokerOrderStatus(StrEnum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_active(self) -> bool:
        return self in {BrokerOrderStatus.PENDING, BrokerOrderStatus.PARTIALLY_FILLED}


class DeploymentStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


class JobStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class JobTriggerSource(StrEnum):
    SCHEDULED = "SCHEDULED"
    MANUAL = "MANUAL"
    RECOVERY = "RECOVERY"


class SnapshotType(StrEnum):
    EOD = "EOD"


class SubmitOrderStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class BrokerAssetSnapshot:
    total_asset: Decimal
    available_cash: Decimal
    captured_at: datetime
    account_id: str | None = None
    frozen_cash: Decimal = Decimal("0")
    market_value: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        _aware(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class BrokerPositionSnapshot:
    symbol: str
    total_quantity: int
    available_quantity: int
    today_buy_quantity: int
    market_value: Decimal
    turnover_rule: TurnoverRule
    captured_at: datetime
    account_id: str | None = None
    frozen_quantity: int = 0
    on_road_quantity: int = 0
    yesterday_quantity: int = 0
    average_cost: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class BrokerOrderSnapshot:
    broker_order_id: str
    symbol: str
    side: OrderSide
    requested_quantity: int
    filled_quantity: int
    limit_price: Decimal
    status: BrokerOrderStatus
    captured_at: datetime
    remark_token: str | None = None
    account_id: str | None = None
    broker_order_sysid: str | None = None
    traded_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.captured_at, "captured_at")

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.requested_quantity - self.filled_quantity)


@dataclass(frozen=True, slots=True)
class BrokerTradeSnapshot:
    broker_trade_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    traded_at: datetime
    account_id: str | None = None
    broker_order_sysid: str | None = None
    remark_token: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.traded_at, "traded_at")


@dataclass(frozen=True, slots=True)
class LiveQuote:
    symbol: str
    last_price: Decimal
    bid1: Decimal | None
    ask1: Decimal | None
    lower_limit: Decimal | None
    upper_limit: Decimal | None
    suspended: bool
    quoted_at: datetime
    price_tick: Decimal = Decimal("0.001")

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        _aware(self.quoted_at, "quoted_at")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_key: str
    remark_token: str
    deployment_id: str
    decision_id: str
    execution_date: date
    symbol: str
    side: OrderSide
    requested_quantity: int
    target_weight: Decimal
    valuation_price: Decimal
    limit_price: Decimal
    status: OrderIntentStatus = OrderIntentStatus.PLANNED

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))


@dataclass(frozen=True, slots=True)
class SubmitOrderResult:
    status: SubmitOrderStatus
    broker_order_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status is SubmitOrderStatus.ACCEPTED and not self.broker_order_id:
            raise ValueError("accepted submission requires broker_order_id")
        if self.status is not SubmitOrderStatus.ACCEPTED and self.broker_order_id is not None:
            raise ValueError("only accepted submission may contain broker_order_id")


@dataclass(frozen=True, slots=True)
class QueryResult[RecordT]:
    success: bool
    records: tuple[RecordT, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("successful query cannot contain an error")
        if not self.success and (not self.error or self.records):
            raise ValueError("failed query requires an error and no records")


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    matched_order_count: int
    inserted_trade_count: int
    unresolved_intent_ids: tuple[str, ...] = ()
    unknown_broker_order_ids: tuple[str, ...] = ()
    unknown_broker_trade_ids: tuple[str, ...] = ()

    @property
    def has_unresolved(self) -> bool:
        return bool(
            self.unresolved_intent_ids
            or self.unknown_broker_order_ids
            or self.unknown_broker_trade_ids
        )


__all__ = [
    "BrokerAssetSnapshot",
    "BrokerOrderSnapshot",
    "BrokerOrderStatus",
    "BrokerPositionSnapshot",
    "BrokerTradeSnapshot",
    "DeploymentStatus",
    "JobStatus",
    "JobTriggerSource",
    "LiveQuote",
    "OrderIntent",
    "OrderIntentStatus",
    "QueryResult",
    "ReconciliationReport",
    "SnapshotType",
    "SubmitOrderResult",
    "SubmitOrderStatus",
]
