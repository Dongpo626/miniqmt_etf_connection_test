"""Transactional SQLAlchemy Core repository for live state."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, delete, insert, or_, select, text, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.base import Executable

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE, normalize_symbol
from etf_backtest.live.persistence.schema import (
    live_account_snapshot,
    live_broker_order,
    live_broker_trade,
    live_decision,
    live_deployment,
    live_job_run,
    live_order_intent,
    live_position_snapshot,
    live_target_position,
)
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerTradeSnapshot,
    DeploymentStatus,
    JobStatus,
    JobTriggerSource,
    OrderIntent,
    OrderIntentStatus,
    SnapshotType,
)

StateRow = dict[str, Any]


def _now() -> datetime:
    return datetime.now(MARKET_TIMEZONE)


def _one(connection: Connection, statement: Executable) -> StateRow | None:
    row = connection.execute(statement).mappings().first()
    return None if row is None else dict(row)


def _many(connection: Connection, statement: Executable) -> tuple[StateRow, ...]:
    rows = connection.execute(statement).mappings().all()
    return tuple(dict(row) for row in rows)


def _lock_name(kind: str, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
    return f"qmt:{kind}:{digest}"


def _acquire_lock(connection: Connection, lock_name: str) -> bool:
    result = connection.execute(
        text("SELECT GET_LOCK(:lock_name, 0)"), {"lock_name": lock_name}
    ).scalar_one_or_none()
    return result == 1


def _release_lock(connection: Connection, lock_name: str) -> None:
    connection.execute(
        text("SELECT RELEASE_LOCK(:lock_name)"), {"lock_name": lock_name}
    ).scalar_one_or_none()


def acquire_account_lock(connection: Connection, account_id: str) -> bool:
    return _acquire_lock(connection, _lock_name("account", account_id))


def release_account_lock(connection: Connection, account_id: str) -> None:
    _release_lock(connection, _lock_name("account", account_id))


def acquire_job_lock(
    connection: Connection, deployment_id: str, job_type: str, trade_date: date
) -> bool:
    identity = f"{deployment_id}|{job_type}|{trade_date.isoformat()}"
    return _acquire_lock(connection, _lock_name("job", identity))


def release_job_lock(
    connection: Connection, deployment_id: str, job_type: str, trade_date: date
) -> None:
    identity = f"{deployment_id}|{job_type}|{trade_date.isoformat()}"
    _release_lock(connection, _lock_name("job", identity))


class LiveStateRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @contextmanager
    def _connection(self, connection: Connection | None, *, write: bool) -> Iterator[Connection]:
        if connection is not None:
            yield connection
            return
        manager = self._engine.begin() if write else self._engine.connect()
        with manager as owned:
            yield owned

    @contextmanager
    def transaction(self, connection: Connection | None = None) -> Iterator[Connection]:
        with self._connection(connection, write=True) as active:
            yield active

    def start_job_run(
        self,
        *,
        deployment_id: str,
        job_type: str,
        trade_date: date,
        trigger_source: JobTriggerSource,
        connection: Connection | None = None,
        job_run_id: str | None = None,
    ) -> str:
        run_id = job_run_id or uuid.uuid4().hex
        with self._connection(connection, write=True) as active:
            active.execute(
                insert(live_job_run).values(
                    job_run_id=run_id,
                    deployment_id=deployment_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                    status=JobStatus.RUNNING,
                    started_at=_now(),
                )
            )
        return run_id

    def finish_job_run(self, job_run_id: str, *, connection: Connection | None = None) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_job_run)
                .where(live_job_run.c.job_run_id == job_run_id)
                .values(status=JobStatus.SUCCEEDED, finished_at=_now())
            )

    def fail_job_run(
        self,
        job_run_id: str,
        *,
        error_type: str,
        error_message: str,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_job_run)
                .where(live_job_run.c.job_run_id == job_run_id)
                .values(
                    status=JobStatus.FAILED,
                    finished_at=_now(),
                    error_type=error_type,
                    error_message=error_message,
                )
            )

    def skip_job_run(
        self,
        job_run_id: str,
        *,
        reason: str,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_job_run)
                .where(live_job_run.c.job_run_id == job_run_id)
                .values(
                    status=JobStatus.SKIPPED,
                    finished_at=_now(),
                    error_type="SKIP_REASON",
                    error_message=reason,
                )
            )

    def has_succeeded_job(
        self,
        deployment_id: str,
        job_type: str,
        trade_date: date,
        *,
        connection: Connection | None = None,
    ) -> bool:
        with self._connection(connection, write=False) as active:
            row = _one(
                active,
                select(live_job_run.c.job_run_id)
                .where(
                    live_job_run.c.deployment_id == deployment_id,
                    live_job_run.c.job_type == job_type,
                    live_job_run.c.trade_date == trade_date,
                    live_job_run.c.status == JobStatus.SUCCEEDED,
                )
                .limit(1),
            )
        return row is not None

    def has_terminal_job(
        self,
        deployment_id: str,
        job_type: str,
        trade_date: date,
        *,
        connection: Connection | None = None,
    ) -> bool:
        with self._connection(connection, write=False) as active:
            row = _one(
                active,
                select(live_job_run.c.job_run_id)
                .where(
                    live_job_run.c.deployment_id == deployment_id,
                    live_job_run.c.job_type == job_type,
                    live_job_run.c.trade_date == trade_date,
                    or_(
                        live_job_run.c.status == JobStatus.SUCCEEDED,
                        and_(
                            live_job_run.c.status == JobStatus.SKIPPED,
                            ~live_job_run.c.error_message.like("FUTURE_%"),
                            live_job_run.c.error_message.not_in(
                                (
                                    "SIGNAL_TIME_NOT_REACHED",
                                    "REBALANCE_TIME_NOT_REACHED",
                                    "CANCEL_TIME_NOT_REACHED",
                                    "EOD_TIME_NOT_REACHED",
                                )
                            ),
                        ),
                    ),
                )
                .limit(1),
            )
        return row is not None

    def get_deployment(
        self, deployment_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_deployment).where(live_deployment.c.deployment_id == deployment_id),
            )

    def get_active_deployment_for_account(
        self, account_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_deployment).where(
                    live_deployment.c.bound_account_id == account_id,
                    live_deployment.c.status == DeploymentStatus.ACTIVE,
                ),
            )

    def ensure_deployment(
        self,
        *,
        deployment_id: str,
        bound_account_id: str,
        mode: str,
        experiment_path: str,
        experiment_sha256: str,
        strategy_source_sha256: str,
        schedule_anchor_date: date,
        universe_json: str,
        universe_hash: str,
        config_hash: str,
        model_bundle_path: str | None = None,
        model_bundle_sha256: str | None = None,
        model_id: str | None = None,
        connection: Connection | None = None,
    ) -> StateRow:
        immutable = {
            "bound_account_id": bound_account_id,
            "experiment_path": experiment_path,
            "experiment_sha256": experiment_sha256,
            "strategy_source_sha256": strategy_source_sha256,
            "schedule_anchor_date": schedule_anchor_date,
            "universe_json": universe_json,
            "universe_hash": universe_hash,
            "config_hash": config_hash,
            "model_bundle_path": model_bundle_path,
            "model_bundle_sha256": model_bundle_sha256,
            "model_id": model_id,
        }
        with self._connection(connection, write=True) as active:
            existing = _one(
                active,
                select(live_deployment)
                .where(live_deployment.c.deployment_id == deployment_id)
                .with_for_update(),
            )
            if existing is not None:
                mismatched = [
                    name for name, value in immutable.items() if existing.get(name) != value
                ]
                if mismatched:
                    raise ValueError("deployment immutable fields differ: " + ", ".join(mismatched))
                if existing["status"] == DeploymentStatus.RETIRED:
                    raise ValueError("RETIRED deployment cannot be reactivated")
                if existing["status"] == DeploymentStatus.PAUSED:
                    return existing
            active_for_account = _one(
                active,
                select(live_deployment)
                .where(
                    live_deployment.c.bound_account_id == bound_account_id,
                    live_deployment.c.status == DeploymentStatus.ACTIVE,
                )
                .with_for_update(),
            )
            now = _now()
            if (
                active_for_account is not None
                and active_for_account["deployment_id"] != deployment_id
            ):
                active.execute(
                    update(live_deployment)
                    .where(live_deployment.c.deployment_id == active_for_account["deployment_id"])
                    .values(status=DeploymentStatus.RETIRED)
                )
            if existing is None:
                active.execute(
                    insert(live_deployment).values(
                        deployment_id=deployment_id,
                        mode=mode,
                        status=DeploymentStatus.ACTIVE,
                        created_at=now,
                        activated_at=now,
                        **immutable,
                    )
                )
            row = self.get_deployment(deployment_id, connection=active)
            assert row is not None
            return row

    def pause_deployment(
        self,
        deployment_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_deployment)
                .where(live_deployment.c.deployment_id == deployment_id)
                .values(status=DeploymentStatus.PAUSED, pause_reason=reason, paused_at=_now())
            )

    def resume_deployment(
        self, deployment_id: str, *, connection: Connection | None = None
    ) -> None:
        with self._connection(connection, write=True) as active:
            result = active.execute(
                update(live_deployment)
                .where(
                    live_deployment.c.deployment_id == deployment_id,
                    live_deployment.c.status == DeploymentStatus.PAUSED,
                )
                .values(
                    status=DeploymentStatus.ACTIVE,
                    pause_reason=None,
                    paused_at=None,
                    activated_at=_now(),
                )
            )
            if result.rowcount == 0:
                raise ValueError("only a PAUSED deployment can be resumed")

    def current_unresolved(
        self, deployment_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        return self.list_unresolved_intents(deployment_id=deployment_id, connection=connection)

    def create_or_get_decision(
        self,
        *,
        decision_id: str,
        deployment_id: str,
        signal_date: date,
        execution_date: date,
        schedule_index: int,
        status: DecisionStatus,
        data_as_of: date,
        strategy_source_sha256: str,
        config_hash: str,
        model_id: str | None = None,
        connection: Connection | None = None,
    ) -> StateRow:
        with self._connection(connection, write=True) as active:
            existing = _one(
                active,
                select(live_decision).where(
                    live_decision.c.deployment_id == deployment_id,
                    live_decision.c.signal_date == signal_date,
                ),
            )
            if existing is not None:
                return existing
            active.execute(
                insert(live_decision).values(
                    decision_id=decision_id,
                    deployment_id=deployment_id,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    schedule_index=schedule_index,
                    status=status,
                    data_as_of=data_as_of,
                    strategy_source_sha256=strategy_source_sha256,
                    model_id=model_id,
                    config_hash=config_hash,
                    created_at=_now(),
                )
            )
            row = _one(
                active,
                select(live_decision).where(live_decision.c.decision_id == decision_id),
            )
            assert row is not None
            return row

    def save_target_positions(
        self,
        decision_id: str,
        target_weights: Mapping[str, Decimal],
        *,
        connection: Connection | None = None,
    ) -> None:
        canonical = {normalize_symbol(symbol): weight for symbol, weight in target_weights.items()}
        with self._connection(connection, write=True) as active:
            rows = _many(
                active,
                select(live_target_position).where(
                    live_target_position.c.decision_id == decision_id
                ),
            )
            existing = {str(row["symbol"]): row["target_weight"] for row in rows}
            if existing:
                if existing != canonical:
                    raise ValueError("saved target positions differ from the requested target")
                return
            if canonical:
                active.execute(
                    insert(live_target_position),
                    [
                        {"decision_id": decision_id, "symbol": symbol, "target_weight": weight}
                        for symbol, weight in sorted(canonical.items())
                    ],
                )

    def pending_decision_for_date(
        self,
        deployment_id: str,
        execution_date: date,
        *,
        connection: Connection | None = None,
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_decision).where(
                    live_decision.c.deployment_id == deployment_id,
                    live_decision.c.execution_date == execution_date,
                    live_decision.c.status == DecisionStatus.TARGET_CREATED,
                ),
            )

    def load_target_positions(
        self, decision_id: str, *, connection: Connection | None = None
    ) -> dict[str, Decimal]:
        with self._connection(connection, write=False) as active:
            rows = _many(
                active,
                select(live_target_position).where(
                    live_target_position.c.decision_id == decision_id
                ),
            )
        return {str(row["symbol"]): row["target_weight"] for row in rows}

    def list_order_intents_for_decision(
        self, decision_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_order_intent).where(live_order_intent.c.decision_id == decision_id),
            )

    def create_order_intent(
        self,
        intent: OrderIntent,
        *,
        connection: Connection | None = None,
        intent_id: str | None = None,
    ) -> StateRow:
        with self._connection(connection, write=True) as active:
            existing = self.get_intent_by_key(intent.intent_key, connection=active)
            expected = {
                "symbol": intent.symbol,
                "side": intent.side,
                "requested_quantity": intent.requested_quantity,
                "valuation_price": intent.valuation_price,
                "limit_price": intent.limit_price,
                "remark_token": intent.remark_token,
            }
            if existing is not None:
                if any(existing[name] != value for name, value in expected.items()):
                    raise ValueError("existing intent_key has different economic content")
                return existing
            token_owner = self.get_intent_by_remark_token(intent.remark_token, connection=active)
            if token_owner is not None:
                raise ValueError("remark_token is already bound to another intent")
            identifier = intent_id or uuid.uuid4().hex
            now = _now()
            active.execute(
                insert(live_order_intent).values(
                    intent_id=identifier,
                    decision_id=intent.decision_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    requested_quantity=intent.requested_quantity,
                    valuation_price=intent.valuation_price,
                    limit_price=intent.limit_price,
                    intent_key=intent.intent_key,
                    remark_token=intent.remark_token,
                    status=intent.status,
                    created_at=now,
                    updated_at=now,
                )
            )
            row = self.get_intent_by_key(intent.intent_key, connection=active)
            assert row is not None
            return row

    def get_intent_by_key(
        self, intent_key: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_order_intent).where(live_order_intent.c.intent_key == intent_key),
            )

    def get_intent_by_remark_token(
        self,
        remark_token: str,
        *,
        account_id: str | None = None,
        connection: Connection | None = None,
    ) -> StateRow | None:
        statement = select(live_order_intent).where(
            live_order_intent.c.remark_token == remark_token
        )
        if account_id is not None:
            statement = (
                select(live_order_intent, live_decision.c.deployment_id)
                .join(live_decision)
                .join(live_deployment)
                .where(
                    live_order_intent.c.remark_token == remark_token,
                    live_deployment.c.bound_account_id == account_id,
                )
            )
        with self._connection(connection, write=False) as active:
            return _one(active, statement)

    def mark_intent_submitting(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> None:
        self._set_intent_status(intent_id, OrderIntentStatus.SUBMITTING, connection=connection)

    def mark_intent_rejected(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id, OrderIntentStatus.REJECTED, connection=connection, reject_reason=reason
        )

    def mark_intent_submit_unknown(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id,
            OrderIntentStatus.SUBMIT_UNKNOWN,
            connection=connection,
            reject_reason=reason,
        )

    def mark_intent_completed(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> None:
        self._set_intent_status(intent_id, OrderIntentStatus.COMPLETED, connection=connection)

    def mark_intent_incomplete(
        self,
        intent_id: str,
        reason: str,
        *,
        connection: Connection | None = None,
    ) -> None:
        self._set_intent_status(
            intent_id,
            OrderIntentStatus.INCOMPLETE,
            connection=connection,
            reject_reason=reason,
        )

    def abandon_planned_intents(
        self,
        deployment_id: str,
        *,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=True) as active:
            rows = _many(
                active,
                select(live_order_intent)
                .join(live_decision)
                .where(
                    live_decision.c.deployment_id == deployment_id,
                    live_order_intent.c.status == OrderIntentStatus.PLANNED,
                ),
            )
            if rows:
                intent_ids = tuple(str(row["intent_id"]) for row in rows)
                active.execute(
                    update(live_order_intent)
                    .where(live_order_intent.c.intent_id.in_(intent_ids))
                    .values(
                        status=OrderIntentStatus.ABANDONED,
                        reject_reason="ABANDONED_STALE_INTENT",
                        updated_at=_now(),
                    )
                )
            return rows

    def decision_has_intent_status(
        self,
        decision_id: str,
        status: OrderIntentStatus,
        *,
        connection: Connection | None = None,
    ) -> bool:
        with self._connection(connection, write=False) as active:
            row = _one(
                active,
                select(live_order_intent.c.intent_id)
                .where(
                    live_order_intent.c.decision_id == decision_id,
                    live_order_intent.c.status == status,
                )
                .limit(1),
            )
        return row is not None

    def list_order_intents(
        self,
        *,
        deployment_id: str | None = None,
        statuses: Sequence[OrderIntentStatus] | None = None,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        statement = select(live_order_intent, live_decision.c.deployment_id).join(live_decision)
        if deployment_id is not None:
            statement = statement.where(live_decision.c.deployment_id == deployment_id)
        if statuses:
            statement = statement.where(live_order_intent.c.status.in_(tuple(statuses)))
        statement = statement.order_by(live_order_intent.c.updated_at.desc())
        with self._connection(connection, write=False) as active:
            return _many(active, statement)

    def _set_intent_status(
        self,
        intent_id: str,
        status: OrderIntentStatus,
        *,
        connection: Connection | None,
        reject_reason: str | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                update(live_order_intent)
                .where(live_order_intent.c.intent_id == intent_id)
                .values(status=status, reject_reason=reject_reason, updated_at=_now())
            )

    def list_unresolved_intents(
        self,
        *,
        deployment_id: str | None = None,
        account_id: str | None = None,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        statement = (
            select(live_order_intent)
            .join(live_decision)
            .join(live_deployment)
            .where(
                live_order_intent.c.status.in_(
                    (OrderIntentStatus.SUBMITTING, OrderIntentStatus.SUBMIT_UNKNOWN)
                )
            )
        )
        if deployment_id is not None:
            statement = statement.where(live_decision.c.deployment_id == deployment_id)
        if account_id is not None:
            statement = statement.where(live_deployment.c.bound_account_id == account_id)
        with self._connection(connection, write=False) as active:
            return _many(active, statement)

    def bind_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            self.upsert_broker_order(
                account_id=account_id,
                intent_id=intent_id,
                remark_token=remark_token,
                order=order,
                connection=active,
            )
            self._set_intent_status(intent_id, OrderIntentStatus.SUBMITTED, connection=active)

    def upsert_broker_order(
        self,
        *,
        account_id: str,
        intent_id: str,
        remark_token: str,
        order: BrokerOrderSnapshot,
        connection: Connection | None = None,
        order_sysid: str | None = None,
        average_fill_price: Decimal | None = None,
    ) -> None:
        values = {
            "account_id": account_id,
            "broker_order_id": order.broker_order_id,
            "order_sysid": order_sysid,
            "intent_id": intent_id,
            "requested_quantity": order.requested_quantity,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": average_fill_price,
            "status": order.status,
            "remark_token": remark_token,
            "updated_at": order.captured_at,
        }
        statement = mysql_insert(live_broker_order).values(**values)
        statement = statement.on_duplicate_key_update(
            order_sysid=statement.inserted.order_sysid,
            requested_quantity=statement.inserted.requested_quantity,
            filled_quantity=statement.inserted.filled_quantity,
            average_fill_price=statement.inserted.average_fill_price,
            status=statement.inserted.status,
            remark_token=statement.inserted.remark_token,
            updated_at=statement.inserted.updated_at,
        )
        with self._connection(connection, write=True) as active:
            active.execute(statement)

    def get_broker_order(
        self,
        account_id: str,
        broker_order_id: str,
        *,
        connection: Connection | None = None,
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_broker_order).where(
                    live_broker_order.c.account_id == account_id,
                    live_broker_order.c.broker_order_id == broker_order_id,
                ),
            )

    def insert_broker_trade_if_absent(
        self,
        *,
        account_id: str,
        intent_id: str,
        trade: BrokerTradeSnapshot,
        connection: Connection | None = None,
    ) -> bool:
        with self._connection(connection, write=True) as active:
            existing = _one(
                active,
                select(live_broker_trade).where(
                    live_broker_trade.c.account_id == account_id,
                    live_broker_trade.c.broker_trade_id == trade.broker_trade_id,
                ),
            )
            if existing is not None:
                return False
            active.execute(
                insert(live_broker_trade).values(
                    account_id=account_id,
                    broker_trade_id=trade.broker_trade_id,
                    broker_order_id=trade.broker_order_id,
                    intent_id=intent_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    quantity=trade.quantity,
                    price=trade.price,
                    trade_time=trade.traded_at,
                )
            )
            return True

    def list_broker_orders_for_intent(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_broker_order).where(live_broker_order.c.intent_id == intent_id),
            )

    def get_intent(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        statement = (
            select(live_order_intent, live_decision.c.deployment_id)
            .join(live_decision)
            .where(live_order_intent.c.intent_id == intent_id)
        )
        with self._connection(connection, write=False) as active:
            return _one(active, statement)

    def list_broker_trades_for_intent(
        self, intent_id: str, *, connection: Connection | None = None
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_broker_trade).where(live_broker_trade.c.intent_id == intent_id),
            )

    def save_account_snapshot(
        self,
        *,
        deployment_id: str,
        trade_date: date,
        snapshot_type: SnapshotType,
        captured_at: datetime,
        cash: Decimal,
        available_cash: Decimal,
        market_value: Decimal,
        total_asset: Decimal,
        frozen_cash: Decimal,
        connection: Connection | None = None,
    ) -> None:
        statement = mysql_insert(live_account_snapshot).values(
            deployment_id=deployment_id,
            trade_date=trade_date,
            snapshot_type=snapshot_type,
            captured_at=captured_at,
            cash=cash,
            available_cash=available_cash,
            market_value=market_value,
            total_asset=total_asset,
            frozen_cash=frozen_cash,
        )
        statement = statement.on_duplicate_key_update(
            captured_at=statement.inserted.captured_at,
            cash=statement.inserted.cash,
            available_cash=statement.inserted.available_cash,
            market_value=statement.inserted.market_value,
            total_asset=statement.inserted.total_asset,
            frozen_cash=statement.inserted.frozen_cash,
        )
        with self._connection(connection, write=True) as active:
            active.execute(statement)

    def save_position_snapshots(
        self,
        *,
        deployment_id: str,
        trade_date: date,
        snapshot_type: SnapshotType,
        positions: Sequence[Mapping[str, object]],
        connection: Connection | None = None,
    ) -> None:
        with self._connection(connection, write=True) as active:
            active.execute(
                delete(live_position_snapshot).where(
                    live_position_snapshot.c.deployment_id == deployment_id,
                    live_position_snapshot.c.trade_date == trade_date,
                    live_position_snapshot.c.snapshot_type == snapshot_type,
                )
            )
            if positions:
                active.execute(
                    insert(live_position_snapshot),
                    [
                        {
                            "deployment_id": deployment_id,
                            "trade_date": trade_date,
                            "snapshot_type": snapshot_type,
                            **position,
                        }
                        for position in positions
                    ],
                )

    def list_job_runs(
        self,
        deployment_id: str,
        *,
        trade_date: date | None = None,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        statement = select(live_job_run).where(live_job_run.c.deployment_id == deployment_id)
        if trade_date is not None:
            statement = statement.where(live_job_run.c.trade_date == trade_date)
        statement = statement.order_by(live_job_run.c.started_at.desc())
        with self._connection(connection, write=False) as active:
            return _many(active, statement)

    def latest_decision_for_deployment(
        self, deployment_id: str, *, connection: Connection | None = None
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_decision)
                .where(live_decision.c.deployment_id == deployment_id)
                .order_by(live_decision.c.signal_date.desc())
                .limit(1),
            )

    def latest_account_snapshot(
        self,
        deployment_id: str,
        snapshot_type: SnapshotType,
        *,
        connection: Connection | None = None,
    ) -> StateRow | None:
        with self._connection(connection, write=False) as active:
            return _one(
                active,
                select(live_account_snapshot)
                .where(
                    live_account_snapshot.c.deployment_id == deployment_id,
                    live_account_snapshot.c.snapshot_type == snapshot_type,
                )
                .order_by(live_account_snapshot.c.captured_at.desc())
                .limit(1),
            )

    def load_position_snapshots(
        self,
        deployment_id: str,
        trade_date: date,
        snapshot_type: SnapshotType,
        *,
        connection: Connection | None = None,
    ) -> tuple[StateRow, ...]:
        with self._connection(connection, write=False) as active:
            return _many(
                active,
                select(live_position_snapshot).where(
                    live_position_snapshot.c.deployment_id == deployment_id,
                    live_position_snapshot.c.trade_date == trade_date,
                    live_position_snapshot.c.snapshot_type == snapshot_type,
                ),
            )


__all__ = [
    "LiveStateRepository",
    "acquire_account_lock",
    "acquire_job_lock",
    "release_account_lock",
    "release_job_lock",
]
