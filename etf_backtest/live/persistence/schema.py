"""The nine SQLAlchemy Core tables owned by the live state database."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    UniqueConstraint,
)

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import (
    BrokerOrderStatus,
    DeploymentStatus,
    JobStatus,
    JobTriggerSource,
    OrderIntentStatus,
    SnapshotType,
)

metadata = MetaData()

live_deployment = Table(
    "live_deployment",
    metadata,
    Column("deployment_id", String(64), primary_key=True),
    Column("bound_account_id", String(64), nullable=False),
    Column("mode", String(16), nullable=False),
    Column("experiment_path", Text, nullable=False),
    Column("experiment_sha256", String(64), nullable=False),
    Column("strategy_source_sha256", String(64), nullable=False),
    Column("model_bundle_path", Text),
    Column("model_bundle_sha256", String(64)),
    Column("model_id", String(128)),
    Column("schedule_anchor_date", Date, nullable=False),
    Column("universe_json", Text, nullable=False),
    Column("universe_hash", String(64), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("status", Enum(DeploymentStatus), nullable=False),
    Column("pause_reason", Text),
    Column("paused_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("activated_at", DateTime(timezone=True), nullable=False),
    Index("ix_live_deployment_bound_account_id", "bound_account_id"),
    Index("ix_live_deployment_status", "status"),
)

live_job_run = Table(
    "live_job_run",
    metadata,
    Column("job_run_id", String(64), primary_key=True),
    Column("deployment_id", String(64), nullable=False),
    Column("job_type", String(64), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("trigger_source", Enum(JobTriggerSource), nullable=False),
    Column("status", Enum(JobStatus), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("error_type", String(128)),
    Column("error_message", Text),
    Index("ix_live_job_run_deployment_job_date", "deployment_id", "job_type", "trade_date"),
)

live_decision = Table(
    "live_decision",
    metadata,
    Column("decision_id", String(64), primary_key=True),
    Column("deployment_id", ForeignKey("live_deployment.deployment_id"), nullable=False),
    Column("signal_date", Date, nullable=False),
    Column("execution_date", Date, nullable=False),
    Column("schedule_index", Integer, nullable=False),
    Column("status", Enum(DecisionStatus), nullable=False),
    Column("data_as_of", Date, nullable=False),
    Column("strategy_source_sha256", String(64), nullable=False),
    Column("model_id", String(128)),
    Column("config_hash", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("deployment_id", "signal_date", name="uq_live_decision_deployment_signal"),
)

live_target_position = Table(
    "live_target_position",
    metadata,
    Column("decision_id", ForeignKey("live_decision.decision_id"), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("target_weight", Numeric(18, 10), nullable=False),
    UniqueConstraint("decision_id", "symbol", name="uq_live_target_decision_symbol"),
)

live_order_intent = Table(
    "live_order_intent",
    metadata,
    Column("intent_id", String(64), primary_key=True),
    Column("decision_id", ForeignKey("live_decision.decision_id"), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("side", Enum(OrderSide), nullable=False),
    Column("requested_quantity", BigInteger, nullable=False),
    Column("valuation_price", Numeric(24, 8), nullable=False),
    Column("limit_price", Numeric(24, 8), nullable=False),
    Column("intent_key", String(64), nullable=False),
    Column("remark_token", String(24), nullable=False),
    Column("status", Enum(OrderIntentStatus), nullable=False),
    Column("reject_reason", Text),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("intent_key", name="uq_live_order_intent_key"),
    UniqueConstraint("remark_token", name="uq_live_order_remark_token"),
)

live_broker_order = Table(
    "live_broker_order",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("broker_order_id", String(128), nullable=False),
    Column("order_sysid", String(128)),
    Column("intent_id", ForeignKey("live_order_intent.intent_id"), nullable=False),
    Column("requested_quantity", BigInteger, nullable=False),
    Column("filled_quantity", BigInteger, nullable=False),
    Column("average_fill_price", Numeric(24, 8)),
    Column("status", Enum(BrokerOrderStatus), nullable=False),
    Column("remark_token", String(24), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("account_id", "broker_order_id", name="uq_live_broker_order_account_id"),
)

live_broker_trade = Table(
    "live_broker_trade",
    metadata,
    Column("account_id", String(64), nullable=False),
    Column("broker_trade_id", String(128), nullable=False),
    Column("broker_order_id", String(128), nullable=False),
    Column("intent_id", ForeignKey("live_order_intent.intent_id"), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("side", Enum(OrderSide), nullable=False),
    Column("quantity", BigInteger, nullable=False),
    Column("price", Numeric(24, 8), nullable=False),
    Column("trade_time", DateTime(timezone=True), nullable=False),
    UniqueConstraint("account_id", "broker_trade_id", name="uq_live_broker_trade_account_id"),
)

live_account_snapshot = Table(
    "live_account_snapshot",
    metadata,
    Column("deployment_id", ForeignKey("live_deployment.deployment_id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("snapshot_type", Enum(SnapshotType), nullable=False),
    Column("captured_at", DateTime(timezone=True), nullable=False),
    Column("cash", Numeric(24, 8), nullable=False),
    Column("available_cash", Numeric(24, 8), nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("total_asset", Numeric(24, 8), nullable=False),
    Column("frozen_cash", Numeric(24, 8), nullable=False),
    UniqueConstraint(
        "deployment_id", "trade_date", "snapshot_type", name="uq_live_account_snapshot_key"
    ),
)

live_position_snapshot = Table(
    "live_position_snapshot",
    metadata,
    Column("deployment_id", ForeignKey("live_deployment.deployment_id"), nullable=False),
    Column("trade_date", Date, nullable=False),
    Column("snapshot_type", Enum(SnapshotType), nullable=False),
    Column("symbol", String(16), nullable=False),
    Column("total_quantity", BigInteger, nullable=False),
    Column("available_quantity", BigInteger, nullable=False),
    Column("frozen_quantity", BigInteger, nullable=False),
    Column("market_value", Numeric(24, 8), nullable=False),
    Column("last_price", Numeric(24, 8), nullable=False),
    UniqueConstraint(
        "deployment_id",
        "trade_date",
        "snapshot_type",
        "symbol",
        name="uq_live_position_snapshot_key",
    ),
)

LIVE_TABLES = (
    live_deployment,
    live_job_run,
    live_decision,
    live_target_position,
    live_order_intent,
    live_broker_order,
    live_broker_trade,
    live_account_snapshot,
    live_position_snapshot,
)

__all__ = ["LIVE_TABLES", "metadata"]
