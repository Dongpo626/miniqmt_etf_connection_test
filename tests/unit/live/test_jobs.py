from contextlib import nullcontext
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.order import OrderSide
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import LiveConfig, load_live_config
from etf_backtest.live.jobs import (
    DeploymentSpec,
    DeploymentSpecFactory,
    LiveDailyJobs,
    SignalEvaluator,
)
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    DeploymentStatus,
    JobTriggerSource,
    LiveQuote,
    OrderIntent,
    OrderIntentStatus,
    QueryResult,
    ReconciliationReport,
    SubmitOrderResult,
    SubmitOrderStatus,
)

ROOT = Path(__file__).parents[3]
NOW = datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE)


def _config(monkeypatch: pytest.MonkeyPatch) -> LiveConfig:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    return load_live_config(ROOT / "qmt_example/configs/live/beginner_example_paper.yaml")


def _record[ValueT](events: list[str], label: str, result: ValueT) -> ValueT:
    events.append(label)
    return result


def _deployment(status: DeploymentStatus = DeploymentStatus.ACTIVE) -> dict[str, object]:
    return {
        "deployment_id": "beginner-example-paper-v1",
        "bound_account_id": "account-1",
        "status": status,
        "universe_json": '["SH.510300","SH.518880"]',
        "strategy_source_sha256": "a" * 64,
        "config_hash": "b" * 64,
    }


def _asset() -> BrokerAssetSnapshot:
    return BrokerAssetSnapshot(
        total_asset=Decimal("10000"),
        available_cash=Decimal("10000"),
        captured_at=NOW,
        account_id="account-1",
    )


def _broker() -> Mock:
    broker = Mock(spec=BrokerGateway)
    broker.query_asset.return_value = QueryResult(success=True, records=(_asset(),))
    broker.query_positions.return_value = QueryResult[BrokerPositionSnapshot](success=True)
    broker.query_orders.return_value = QueryResult[BrokerOrderSnapshot](success=True)
    broker.query_trades.return_value = QueryResult(success=True)
    return broker


def _jobs(
    monkeypatch: pytest.MonkeyPatch,
    repository: Mock,
    broker: Mock,
    *,
    evaluator: object | None = None,
    factory: object | None = None,
    reconciliation: Mock | None = None,
) -> LiveDailyJobs:
    state_engine = Mock(spec=Engine)
    return LiveDailyJobs(
        config=_config(monkeypatch),
        broker=cast(BrokerGateway, broker),
        quote_provider=cast(QuoteProvider, Mock(spec=QuoteProvider)),
        state_repository=cast(LiveStateRepository, repository),
        state_engine=cast(Engine, state_engine),
        signal_evaluator=cast(SignalEvaluator, evaluator or Mock()),
        deployment_spec_factory=cast(DeploymentSpecFactory, factory or Mock()),
        reconciliation_service=cast(ReconciliationService, reconciliation or Mock()),
        clock=lambda: NOW,
    )


def test_startup_queries_and_reconciles_before_ensure_and_query_failure_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.side_effect = lambda *args, **kwargs: _record(
        events, "read", None
    )
    repository.get_active_deployment_for_account.return_value = None
    repository.ensure_deployment.side_effect = lambda **kwargs: _record(
        events, "ensure", _deployment()
    )
    broker = _broker()
    for name in ("query_asset", "query_positions", "query_orders", "query_trades"):
        method = getattr(broker, name)
        original = method.return_value
        method.side_effect = lambda result=original, label=name: _record(events, label, result)
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.side_effect = (
        lambda **kwargs: _record(events, "reconcile", ReconciliationReport(0, 0))
    )
    spec = DeploymentSpec(
        deployment_id="beginner-example-paper-v1",
        bound_account_id="account-1",
        mode="PAPER",
        experiment_path="experiment.yaml",
        experiment_sha256="a" * 64,
        strategy_source_sha256="b" * 64,
        schedule_anchor_date=date(2021, 1, 4),
        symbols=("SH.510300", "SH.518880"),
        universe_json='["SH.510300","SH.518880"]',
        universe_hash="c" * 64,
        config_hash="d" * 64,
    )
    factory = Mock(return_value=spec)
    jobs = _jobs(
        monkeypatch,
        repository,
        broker,
        factory=factory,
        reconciliation=reconciliation,
    )
    jobs._startup_reconcile(date(2026, 8, 19))
    assert events.index("reconcile") < events.index("ensure")
    factory.assert_called_once_with(None)

    repository.reset_mock()
    broker.query_asset.return_value = QueryResult(success=False, error="offline")
    broker.query_asset.side_effect = None
    with pytest.raises(RuntimeError, match="asset query failed"):
        jobs._startup_reconcile(date(2026, 8, 19))
    repository.ensure_deployment.assert_not_called()


def test_startup_resumes_paused_only_after_safe_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    paused = _deployment(DeploymentStatus.PAUSED)
    repository.get_deployment.side_effect = [paused, _deployment()]
    repository.get_active_deployment_for_account.return_value = None
    repository.ensure_deployment.return_value = paused
    broker = _broker()
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    spec = DeploymentSpec(
        deployment_id="beginner-example-paper-v1",
        bound_account_id="account-1",
        mode="PAPER",
        experiment_path="experiment.yaml",
        experiment_sha256="a" * 64,
        strategy_source_sha256="b" * 64,
        schedule_anchor_date=date(2021, 1, 4),
        symbols=("SH.510300", "SH.518880"),
        universe_json='["SH.510300","SH.518880"]',
        universe_hash="c" * 64,
        config_hash="d" * 64,
    )
    factory = Mock(return_value=spec)
    jobs = _jobs(
        monkeypatch,
        repository,
        broker,
        factory=factory,
        reconciliation=reconciliation,
    )

    jobs._startup_reconcile(date(2026, 8, 19))

    reconciliation.reconcile.assert_called_once()
    factory.assert_called_once_with(("SH.510300", "SH.518880"))
    repository.ensure_deployment.assert_called_once()
    repository.resume_deployment.assert_called_once_with("beginner-example-paper-v1")


def test_prepare_signal_saves_complete_zero_target_after_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.return_value = _deployment()
    repository.current_unresolved.return_value = ()
    connection = Mock(spec=Connection)
    repository.transaction.return_value = nullcontext(connection)
    repository.create_or_get_decision.return_value = {"decision_id": "decision-1"}
    broker = _broker()
    evaluator = Mock(spec=SignalEvaluator)
    evaluator.evaluate.return_value = DailyDecisionResult(
        signal_date=date(2026, 8, 19),
        execution_date=date(2026, 8, 20),
        schedule_index=3,
        status=DecisionStatus.TARGET_CREATED,
        target_portfolio=TargetPortfolio({}),
    )
    jobs = _jobs(monkeypatch, repository, broker, evaluator=evaluator)

    jobs._prepare_signal(date(2026, 8, 19))

    evaluator.evaluate.assert_called_once()
    assert evaluator.evaluate.call_args.kwargs["signal_date"] == date(2026, 8, 19)
    repository.save_target_positions.assert_called_once_with(
        "decision-1",
        {"SH.510300": Decimal("0"), "SH.518880": Decimal("0")},
        connection=connection,
    )


def test_execute_submits_once_and_maps_three_submit_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.return_value = _deployment()
    repository.current_unresolved.return_value = ()
    repository.pending_decision_for_date.return_value = {"decision_id": "decision-1"}
    repository.load_target_positions.return_value = {
        "SH.510300": Decimal("0.5"),
        "SH.518880": Decimal("0"),
    }
    planned = {"intent_id": "intent-1", "status": OrderIntentStatus.PLANNED}
    submitted = {"intent_id": "intent-1", "status": OrderIntentStatus.SUBMITTED}
    repository.create_order_intent.side_effect = [planned, submitted]
    repository.get_intent.return_value = submitted
    broker = _broker()
    broker.submit_order.return_value = SubmitOrderResult(
        SubmitOrderStatus.ACCEPTED, broker_order_id="broker-1"
    )
    quote_provider = Mock(spec=QuoteProvider)
    quote_provider.latest_quotes.return_value = QueryResult(
        success=True,
        records=tuple(
            LiveQuote(
                symbol=symbol,
                last_price=Decimal("10"),
                bid1=Decimal("9.999"),
                ask1=Decimal("10.001"),
                lower_limit=Decimal("9"),
                upper_limit=Decimal("11"),
                suspended=False,
                quoted_at=NOW,
            )
            for symbol in ("510300.SH", "518880.SH")
        ),
    )
    reconciliation = Mock(spec=ReconciliationService)
    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    jobs = _jobs(monkeypatch, repository, broker, reconciliation=reconciliation)
    jobs.quote_provider = cast(QuoteProvider, quote_provider)

    jobs._execute_pending_target(date(2026, 8, 19))
    jobs._execute_pending_target(date(2026, 8, 19))
    broker.submit_order.assert_called_once()
    repository.mark_intent_submitting.assert_called_once_with("intent-1")
    repository.bind_broker_order.assert_called_once()
    intent = cast(OrderIntent, broker.submit_order.call_args.args[0])

    repository.create_order_intent.side_effect = None
    for existing_status in (
        OrderIntentStatus.SUBMITTING,
        OrderIntentStatus.SUBMIT_UNKNOWN,
        OrderIntentStatus.SUBMITTED,
        OrderIntentStatus.REJECTED,
    ):
        repository.create_order_intent.return_value = {
            "intent_id": "intent-1",
            "status": existing_status,
        }
        broker.submit_order.reset_mock()
        jobs._execute_pending_target(date(2026, 8, 19))
        broker.submit_order.assert_not_called()

    for status, method in (
        (SubmitOrderStatus.REJECTED, repository.mark_intent_rejected),
        (SubmitOrderStatus.UNKNOWN, repository.mark_intent_submit_unknown),
    ):
        broker.submit_order.reset_mock()
        method.reset_mock()
        broker.submit_order.return_value = SubmitOrderResult(status, error="result")
        jobs._submit_intent(intent, "another-intent", "account-1")
        method.assert_called_once()


def test_unknown_cancel_pauses_without_cancel_and_snapshot_keeps_zero_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.return_value = _deployment()
    repository.transaction.return_value = nullcontext(Mock(spec=Connection))
    repository.get_broker_order.return_value = None
    repository.get_intent_by_remark_token.return_value = None
    broker = _broker()
    broker.query_orders.return_value = QueryResult(
        success=True,
        records=(
            BrokerOrderSnapshot(
                broker_order_id="unknown",
                symbol="510300.SH",
                side=OrderSide.BUY,
                requested_quantity=100,
                filled_quantity=0,
                limit_price=Decimal("10"),
                status=BrokerOrderStatus.PENDING,
                captured_at=NOW,
                remark_token="L" + "Z" * 20,
            ),
        ),
    )
    jobs = _jobs(monkeypatch, repository, broker)
    with pytest.raises(RuntimeError, match="unknown active"):
        jobs._cancel_open_orders()
    broker.cancel_order.assert_not_called()
    repository.pause_deployment.assert_called_once()

    broker.query_orders.return_value = QueryResult[BrokerOrderSnapshot](success=True)
    jobs._snapshot_eod(date(2026, 8, 19))
    rows = repository.save_position_snapshots.call_args.kwargs["positions"]
    assert [row["symbol"] for row in rows] == ["SH.510300", "SH.518880"]
    assert all(row["total_quantity"] == 0 for row in rows)


def test_reconcile_eod_pauses_for_query_failure_or_current_unresolved_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.return_value = _deployment()
    broker = _broker()
    reconciliation = Mock(spec=ReconciliationService)
    jobs = _jobs(monkeypatch, repository, broker, reconciliation=reconciliation)

    broker.query_orders.return_value = QueryResult(success=False, error="offline")
    with pytest.raises(RuntimeError, match="orders query failed"):
        jobs._reconcile_eod()
    repository.pause_deployment.assert_called_once_with(
        "beginner-example-paper-v1", "EOD_QUERY_FAILED"
    )

    broker.query_orders.return_value = QueryResult[BrokerOrderSnapshot](success=True)
    reconciliation.reconcile.return_value = ReconciliationReport(
        0, 0, unresolved_intent_ids=("intent-1",)
    )
    with pytest.raises(RuntimeError, match="unresolved"):
        jobs._reconcile_eod()
    repository.pause_deployment.assert_called_with(
        "beginner-example-paper-v1", "EOD_UNRESOLVED"
    )

    reconciliation.reconcile.return_value = ReconciliationReport(0, 0)
    pause_count = repository.pause_deployment.call_count
    assert jobs._reconcile_eod().has_unresolved is False
    assert repository.pause_deployment.call_count == pause_count


def test_job_lock_failure_marks_run_failed_without_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Mock(spec=LiveStateRepository)
    repository.start_job_run.return_value = "run-1"
    broker = _broker()
    jobs = _jobs(monkeypatch, repository, broker)
    connection = Mock(spec=Connection)
    jobs.state_engine.connect.return_value = nullcontext(connection)  # type: ignore[attr-defined]
    monkeypatch.setattr("etf_backtest.live.jobs.acquire_job_lock", lambda *args: False)
    body = Mock()

    with pytest.raises(RuntimeError, match="job lock"):
        jobs._run_job(
            "prepare_signal",
            date(2026, 8, 19),
            JobTriggerSource.MANUAL,
            body,
            None,
        )
    body.assert_not_called()
    repository.fail_job_run.assert_called_once()
