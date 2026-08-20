from contextlib import nullcontext
from datetime import date, datetime
from decimal import Decimal
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.application.contracts import DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.order import OrderSide
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_account_lock,
    acquire_job_lock,
    release_account_lock,
    release_job_lock,
)
from etf_backtest.live.state import (
    BrokerTradeSnapshot,
    DeploymentStatus,
    JobTriggerSource,
    OrderIntent,
    SnapshotType,
)

NOW = datetime(2026, 8, 19, 15, 5, tzinfo=MARKET_TIMEZONE)


def _result(
    *, first: dict[str, object] | None = None, rows: list[dict[str, object]] | None = None
) -> Mock:
    result = Mock()
    mapping = result.mappings.return_value
    mapping.first.return_value = first
    mapping.all.return_value = [] if rows is None else rows
    return result


def _repository(connection: Mock) -> tuple[LiveStateRepository, Mock]:
    engine = Mock(spec=Engine)
    engine.begin.return_value = nullcontext(connection)
    engine.connect.return_value = nullcontext(connection)
    return LiveStateRepository(cast(Engine, engine)), engine


def _intent(quantity: int = 100, token: str = "L" + "A" * 20) -> OrderIntent:
    return OrderIntent(
        intent_key="a" * 64,
        remark_token=token,
        deployment_id="deployment-1",
        decision_id="decision-1",
        execution_date=date(2026, 8, 19),
        symbol="510300.SH",
        side=OrderSide.BUY,
        requested_quantity=quantity,
        target_weight=Decimal("0.5"),
        valuation_price=Decimal("10"),
        limit_price=Decimal("10.01"),
    )


def test_external_connection_is_reused_without_lifecycle_calls() -> None:
    connection = Mock(spec=Connection)
    repository, engine = _repository(connection)
    repository.start_job_run(
        deployment_id="deployment-1",
        job_type="snapshot_eod",
        trade_date=date(2026, 8, 19),
        trigger_source=JobTriggerSource.MANUAL,
        connection=cast(Connection, connection),
        job_run_id="run-1",
    )

    connection.execute.assert_called_once()
    connection.close.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()
    engine.begin.assert_not_called()
    engine.connect.assert_not_called()


def test_owned_reads_use_connect_and_writes_use_begin() -> None:
    connection = Mock(spec=Connection)
    connection.execute.return_value = _result(first=None)
    repository, engine = _repository(connection)

    repository.get_deployment("missing")
    repository.finish_job_run("run-1")

    engine.connect.assert_called_once_with()
    engine.begin.assert_called_once_with()


def test_duplicate_intent_is_returned_only_when_economic_content_matches() -> None:
    intent = _intent()
    existing = {
        "intent_id": "intent-1",
        "symbol": intent.symbol,
        "side": intent.side,
        "requested_quantity": intent.requested_quantity,
        "valuation_price": intent.valuation_price,
        "limit_price": intent.limit_price,
        "remark_token": intent.remark_token,
    }
    connection = Mock(spec=Connection)
    connection.execute.return_value = _result(first=existing)
    repository, _ = _repository(connection)

    assert (
        repository.create_order_intent(intent, connection=cast(Connection, connection)) == existing
    )
    with pytest.raises(ValueError, match="different economic content"):
        repository.create_order_intent(
            _intent(quantity=200), connection=cast(Connection, connection)
        )


def test_remark_collision_fails_without_retry_or_insert() -> None:
    connection = Mock(spec=Connection)
    connection.execute.side_effect = [
        _result(first=None),
        _result(first={"intent_id": "other"}),
    ]
    repository, _ = _repository(connection)

    with pytest.raises(ValueError, match="remark_token"):
        repository.create_order_intent(_intent(), connection=cast(Connection, connection))
    assert connection.execute.call_count == 2


def test_duplicate_trade_is_not_inserted_again() -> None:
    connection = Mock(spec=Connection)
    connection.execute.return_value = _result(first={"broker_trade_id": "trade-1"})
    repository, _ = _repository(connection)
    trade = BrokerTradeSnapshot(
        broker_trade_id="trade-1",
        broker_order_id="order-1",
        symbol="510300.SH",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("10"),
        traded_at=NOW,
    )

    assert not repository.insert_broker_trade_if_absent(
        account_id="account-1",
        intent_id="intent-1",
        trade=trade,
        connection=cast(Connection, connection),
    )
    connection.execute.assert_called_once()


def test_decision_rerun_reuses_existing_id_and_target_mismatch_fails() -> None:
    connection = Mock(spec=Connection)
    decision: dict[str, object] = {"decision_id": "existing-decision"}
    connection.execute.side_effect = [
        _result(first=decision),
        _result(rows=[{"symbol": "SH.510300", "target_weight": Decimal("0.5")}]),
    ]
    repository, _ = _repository(connection)
    row = repository.create_or_get_decision(
        decision_id="new-decision",
        deployment_id="deployment-1",
        signal_date=date(2026, 8, 18),
        execution_date=date(2026, 8, 19),
        schedule_index=1,
        status=DecisionStatus.TARGET_CREATED,
        data_as_of=date(2026, 8, 18),
        strategy_source_sha256="a" * 64,
        config_hash="b" * 64,
        connection=cast(Connection, connection),
    )
    assert row["decision_id"] == "existing-decision"
    with pytest.raises(ValueError, match="differ"):
        repository.save_target_positions(
            "existing-decision",
            {"510300.SH": Decimal("0.4")},
            connection=cast(Connection, connection),
        )


def test_deployment_switch_uses_one_owned_transaction() -> None:
    connection = Mock(spec=Connection)
    new_row: dict[str, object] = {
        "deployment_id": "new",
        "status": DeploymentStatus.ACTIVE,
    }
    connection.execute.side_effect = [
        _result(first=None),
        _result(first={"deployment_id": "old"}),
        _result(),
        _result(),
        _result(first=new_row),
    ]
    repository, engine = _repository(connection)
    row = repository.ensure_deployment(
        deployment_id="new",
        bound_account_id="account-1",
        mode="PAPER",
        experiment_path="experiment.yaml",
        experiment_sha256="a" * 64,
        strategy_source_sha256="b" * 64,
        schedule_anchor_date=date(2026, 1, 1),
        universe_json='["SH.510300"]',
        universe_hash="c" * 64,
        config_hash="d" * 64,
    )

    assert row == new_row
    engine.begin.assert_called_once_with()
    assert connection.execute.call_count == 5


def test_retired_is_terminal_and_paused_is_not_implicitly_reactivated() -> None:
    immutable: dict[str, object] = {
        "bound_account_id": "account-1",
        "experiment_path": "experiment.yaml",
        "experiment_sha256": "a" * 64,
        "strategy_source_sha256": "b" * 64,
        "schedule_anchor_date": date(2026, 1, 1),
        "universe_json": '["SH.510300"]',
        "universe_hash": "c" * 64,
        "config_hash": "d" * 64,
    }
    retired = {"deployment_id": "deployment-1", "status": DeploymentStatus.RETIRED, **immutable}
    connection = Mock(spec=Connection)
    connection.execute.return_value = _result(first=retired)
    repository, _ = _repository(connection)
    with pytest.raises(ValueError, match="RETIRED"):
        repository.ensure_deployment(
            deployment_id="deployment-1",
            mode="PAPER",
            **immutable,  # type: ignore[arg-type]
        )

    paused = {"deployment_id": "deployment-1", "status": DeploymentStatus.PAUSED, **immutable}
    connection.reset_mock()
    connection.execute.side_effect = [_result(first=paused)]
    ensured = repository.ensure_deployment(
        deployment_id="deployment-1",
        mode="PAPER",
        **immutable,  # type: ignore[arg-type]
    )
    assert ensured["status"] is DeploymentStatus.PAUSED
    assert connection.execute.call_count == 1


def test_pause_and_resume_only_issue_minimal_updates() -> None:
    connection = Mock(spec=Connection)
    repository, _ = _repository(connection)
    repository.pause_deployment(
        "deployment-1", "unresolved", connection=cast(Connection, connection)
    )
    repository.resume_deployment("deployment-1", connection=cast(Connection, connection))

    pause_params = connection.execute.call_args_list[0].args[0].compile().params
    resume_params = connection.execute.call_args_list[1].args[0].compile().params
    assert pause_params["status"] is DeploymentStatus.PAUSED
    assert pause_params["pause_reason"] == "unresolved"
    assert resume_params["status"] is DeploymentStatus.ACTIVE
    assert resume_params["pause_reason"] is None


@pytest.mark.parametrize(("database_value", "expected"), [(1, True), (0, False), (None, False)])
def test_advisory_lock_return_semantics(database_value: int | None, expected: bool) -> None:
    connection = Mock(spec=Connection)
    connection.execute.return_value.scalar_one_or_none.return_value = database_value
    assert acquire_account_lock(cast(Connection, connection), "account-1") is expected


def test_release_and_job_lock_use_their_matching_lock_names() -> None:
    connection = Mock(spec=Connection)
    connection.execute.return_value.scalar_one_or_none.return_value = 1
    typed = cast(Connection, connection)
    acquire_account_lock(typed, "account-1")
    release_account_lock(typed, "account-1")
    acquire_job_lock(typed, "deployment-1", "reconcile_eod", date(2026, 8, 19))
    release_job_lock(typed, "deployment-1", "reconcile_eod", date(2026, 8, 19))

    params = [call.args[1]["lock_name"] for call in connection.execute.call_args_list]
    assert params[0] == params[1]
    assert params[2] == params[3]
    assert params[0] != params[2]


def test_latest_current_and_eod_snapshots_are_queried_separately_by_capture_time() -> None:
    connection = Mock(spec=Connection)
    connection.execute.side_effect = [
        _result(first={"snapshot_type": SnapshotType.CURRENT}),
        _result(first={"snapshot_type": SnapshotType.EOD}),
    ]
    repository, _ = _repository(connection)

    current = repository.latest_account_snapshot("deployment-1", SnapshotType.CURRENT)
    eod = repository.latest_account_snapshot("deployment-1", SnapshotType.EOD)

    assert current == {"snapshot_type": SnapshotType.CURRENT}
    assert eod == {"snapshot_type": SnapshotType.EOD}
    statements = [call.args[0] for call in connection.execute.call_args_list]
    assert SnapshotType.CURRENT in statements[0].compile().params.values()
    assert SnapshotType.EOD in statements[1].compile().params.values()
    assert all("captured_at DESC" in str(statement) for statement in statements)
