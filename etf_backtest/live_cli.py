"""Thin command boundary for Rule/Model PAPER jobs and lifecycle operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from queue import Queue

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

from etf_backtest.application.runtime_factory import create_database_engine, create_repository
from etf_backtest.application.strategy_source import (
    ModelStrategySource,
    RuleStrategySource,
    load_model_strategy_source,
    load_rule_strategy_source,
)
from etf_backtest.config.schema import ModelStrategyConfig, RuleStrategyConfig
from etf_backtest.data.mysql import QmtDailyRepository
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.broker.callbacks import (
    BrokerEvent,
    BrokerEventConsumer,
)
from etf_backtest.live.broker.miniqmt import MiniQmtBrokerGateway
from etf_backtest.live.config import LiveConfig, load_live_config
from etf_backtest.live.engine import LiveTradingEngine
from etf_backtest.live.execution.near_close_limit import NearCloseLimitPolicy
from etf_backtest.live.execution.planner import LiveRebalancePlanner
from etf_backtest.live.jobs import (
    DeploymentSpecFactory,
    LiveDailyJobs,
    ModelDeploymentSpecFactory,
    ModelSignalService,
    RuleDeploymentSpecFactory,
    RuleSignalService,
    SignalEvaluator,
)
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.market.xtdata import XtDataQuoteProvider
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_account_lock,
    release_account_lock,
)
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.risk import LiveRiskManager
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.live.state import SnapshotType
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_training import load_daily_torch_bundle_for_inference

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    Path(__file__).resolve().parent / "live/persistence/migrations/001_create_live_tables.sql"
)


@dataclass(frozen=True, slots=True)
class LiveCommandRuntime:
    state_engine: Engine
    repository: LiveStateRepository
    broker: BrokerGateway
    quote_provider: QuoteProvider
    jobs: LiveDailyJobs
    scheduler: LiveScheduler
    engine: LiveTradingEngine
    event_consumer: BrokerEventConsumer | None = None


RuntimeBuilder = Callable[[LiveConfig], LiveCommandRuntime]


def create_state_engine(config: LiveConfig) -> Engine:
    database = config.state_database
    url = URL.create(
        "mysql+pymysql",
        username=database.user,
        password=database.resolved_password(),
        host=database.host,
        port=database.port,
        database=database.database,
        query={"charset": "utf8mb4"},
    )
    return create_engine(url, pool_pre_ping=True)


class _StrategyTradingDaySource:
    def __init__(self, repository: QmtDailyRepository) -> None:
        self._repository = repository
        self._cache: dict[date, bool] = {}

    def is_trading_day(self, trade_date: date) -> bool:
        if trade_date not in self._cache:
            days = self._repository.load_sse_calendar(trade_date, trade_date)
            self._cache[trade_date] = bool(days[0].is_open)
        return self._cache[trade_date]


def build_production_runtime(
    config: LiveConfig,
    *,
    broker_factory: Callable[..., BrokerGateway] = MiniQmtBrokerGateway,
    quote_factory: Callable[[], QuoteProvider] = XtDataQuoteProvider,
) -> LiveCommandRuntime:
    experiment = config.project_path(config.deployment.experiment_path, PROJECT_ROOT)
    system = config.project_path(config.deployment.system_path, PROJECT_ROOT)
    account_id = config.deployment.account_id()
    events: Queue[BrokerEvent] = Queue()
    broker = broker_factory(
        userdata_path=config.miniqmt.userdata_path,
        session_id=config.miniqmt.session_id,
        account_id=account_id,
        account_type=config.deployment.account_type,
        event_queue=events,
    )
    quote_provider = quote_factory()
    state_engine = create_state_engine(config)
    repository = LiveStateRepository(state_engine)
    source: RuleStrategySource | ModelStrategySource
    deployment_factory: DeploymentSpecFactory
    signal_service: SignalEvaluator
    if config.deployment.case == "rule":
        source = load_rule_strategy_source(experiment, system_path=system)
        strategy_engine = create_database_engine(source.system.database)
        deployment_factory = RuleDeploymentSpecFactory(
            live_config=config,
            source=source,
            strategy_engine=strategy_engine,
        )
        signal_service = RuleSignalService(
            live_config=config,
            source=source,
            strategy_engine=strategy_engine,
        )
        strategy_config: RuleStrategyConfig | ModelStrategyConfig = RuleStrategyConfig(
            lookback_trading_days=source.rule.lookback_trading_days,
            rebalance_every_trading_days=source.rule.rebalance_every_trading_days,
            target_weight=source.rule.target_weight,
        )
    else:
        model_source = load_model_strategy_source(experiment, system_path=system)
        strategy_engine = create_database_engine(model_source.system.database)
        assert config.model is not None
        bundle_path = config.project_path(config.model.bundle_path, PROJECT_ROOT)
        loaded = load_daily_torch_bundle_for_inference(
            bundle_path,
            feature_builder=model_source.components.feature_builder,
            model_factory=model_source.components.model_factory,
            portfolio=model_source.components.settings.portfolio,
            signal_date=model_source.experiment.start_date,
        )
        model_strategy = DailyModelStrategy(
            feature_builder=model_source.components.feature_builder,
            bundle=loaded.bundle,
            portfolio=model_source.components.settings.portfolio,
        )
        deployment_factory = ModelDeploymentSpecFactory(
            live_config=config,
            source=model_source,
            strategy_engine=strategy_engine,
            bundle_path=bundle_path,
            loaded_bundle=loaded,
        )
        signal_service = ModelSignalService(
            live_config=config,
            source=model_source,
            strategy=model_strategy,
            strategy_engine=strategy_engine,
        )
        settings = model_source.components.settings
        strategy_config = ModelStrategyConfig(
            max_total_weight=settings.portfolio.max_total_weight,
            train_start=settings.train_range.start_date,
            train_end=settings.train_range.end_date,
            valid_start=settings.valid_range.start_date,
            valid_end=settings.valid_range.end_date,
            test_start=model_source.experiment.start_date,
            test_end=model_source.experiment.end_date,
        )
        source = model_source
    reconciliation = ReconciliationService()
    jobs = LiveDailyJobs(
        config=config,
        broker=broker,
        quote_provider=quote_provider,
        state_repository=repository,
        state_engine=state_engine,
        signal_evaluator=signal_service,
        deployment_spec_factory=deployment_factory,
        reconciliation_service=reconciliation,
        planner=LiveRebalancePlanner(),
        risk_manager=LiveRiskManager(),
        price_policy=NearCloseLimitPolicy(),
    )
    backtest_config = source.experiment.build_case(source.system, strategy=strategy_config)
    calendar_repository = create_repository(backtest_config, strategy_engine)
    scheduler = LiveScheduler(
        jobs=jobs,
        calendar=_StrategyTradingDaySource(calendar_repository),
        config=config,
    )
    engine = LiveTradingEngine(
        config=config,
        state_engine=state_engine,
        repository=repository,
        broker=broker,
        quote_provider=quote_provider,
        jobs=jobs,
        scheduler=scheduler,
    )
    consumer = BrokerEventConsumer(
        events=events,
        repository=repository,
        account_id=account_id,
        on_unhealthy=engine.notify_broker_unhealthy,
    )
    engine.set_event_consumer(consumer)
    return LiveCommandRuntime(
        state_engine=state_engine,
        repository=repository,
        broker=broker,
        quote_provider=quote_provider,
        jobs=jobs,
        scheduler=scheduler,
        engine=engine,
        event_consumer=consumer,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qmt-etf-live")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate",
        "db",
        "service",
        "signal",
        "execute",
        "cancel",
        "snapshot",
        "jobs",
        "status",
    ):
        command = subcommands.add_parser(name)
        if name == "db":
            command.add_argument("action", choices=("init",))
        if name in {"signal", "execute", "cancel", "snapshot", "jobs"}:
            command.add_argument("--date", required=True)
        command.add_argument("--config", required=True)
    reconcile = subcommands.add_parser("reconcile")
    reconcile.add_argument("--phase", choices=("startup", "eod"), required=True)
    reconcile.add_argument("--date")
    reconcile.add_argument("--config", required=True)
    return parser


def _date(value: str | None) -> date:
    return datetime.now().date() if value is None else date.fromisoformat(value)


def _validate(config: LiveConfig) -> None:
    experiment = config.project_path(config.deployment.experiment_path, PROJECT_ROOT)
    system = config.project_path(config.deployment.system_path, PROJECT_ROOT)
    if config.deployment.case == "rule":
        load_rule_strategy_source(experiment, system_path=system)
        return
    source = load_model_strategy_source(experiment, system_path=system)
    assert config.model is not None
    bundle_path = config.project_path(config.model.bundle_path, PROJECT_ROOT)
    load_daily_torch_bundle_for_inference(
        bundle_path,
        feature_builder=source.components.feature_builder,
        model_factory=source.components.model_factory,
        portfolio=source.components.settings.portfolio,
        signal_date=source.experiment.start_date,
    )


def _db_init(config: LiveConfig) -> None:
    engine = create_state_engine(config)
    statements = [part.strip() for part in MIGRATION.read_text(encoding="utf-8").split(";")]
    try:
        with engine.begin() as connection:
            for statement in statements:
                if statement:
                    connection.exec_driver_sql(statement)
    finally:
        engine.dispose()


def _read_repository(config: LiveConfig) -> tuple[Engine, LiveStateRepository]:
    engine = create_state_engine(config)
    return engine, LiveStateRepository(engine)


def _print_jobs(config: LiveConfig, trade_date: date) -> None:
    engine, repository = _read_repository(config)
    try:
        rows = repository.list_job_runs(config.deployment.deployment_id, trade_date=trade_date)
        print(json.dumps(rows, ensure_ascii=False, default=str))
    finally:
        engine.dispose()


def _print_status(config: LiveConfig) -> None:
    engine, repository = _read_repository(config)
    deployment_id = config.deployment.deployment_id
    try:
        deployment = repository.get_deployment(deployment_id)
        decision = repository.latest_decision_for_deployment(deployment_id)
        target: object = None
        intents: tuple[dict[str, object], ...] = ()
        orders: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []
        if decision is not None:
            decision_id = str(decision["decision_id"])
            target = repository.load_target_positions(decision_id)
            intents = repository.list_order_intents_for_decision(decision_id)
            for intent in intents:
                intent_id = str(intent["intent_id"])
                orders.extend(repository.list_broker_orders_for_intent(intent_id))
                trades.extend(repository.list_broker_trades_for_intent(intent_id))
        snapshots: dict[str, object] = {}
        for snapshot_type in (SnapshotType.CURRENT, SnapshotType.EOD):
            account = repository.latest_account_snapshot(deployment_id, snapshot_type)
            positions: tuple[dict[str, object], ...] = ()
            if account is not None:
                positions = repository.load_position_snapshots(
                    deployment_id,
                    account["trade_date"],
                    snapshot_type,
                )
            snapshots[snapshot_type.value] = {
                "account": account,
                "positions": positions,
            }
        print(
            json.dumps(
                {
                    "deployment": deployment,
                    "decision": decision,
                    "target": target,
                    "intents": intents,
                    "broker_orders": orders,
                    "broker_trades": trades,
                    "snapshots": snapshots,
                },
                ensure_ascii=False,
                default=str,
            )
        )
    finally:
        engine.dispose()


def _manual_job(runtime: LiveCommandRuntime, config: LiveConfig, args: argparse.Namespace) -> None:
    account_id = config.deployment.account_id()
    connection = runtime.state_engine.connect()
    connected = False
    locked = False
    consumer_started = False
    try:
        if not acquire_account_lock(connection, account_id):
            raise RuntimeError("account lock is already held")
        locked = True
        if runtime.event_consumer is not None:
            runtime.event_consumer.start()
            consumer_started = True
        runtime.broker.connect()
        connected = True
        runtime.broker.subscribe_account(account_id)
        trade_date = _date(getattr(args, "date", None))
        if args.command == "execute":
            # This is the same recovery boundary as service startup. In particular,
            # stale PLANNED intents are abandoned before manual execution is considered.
            deployment = runtime.jobs.startup_reconcile(trade_date, lock_connection=connection)
            symbols = json.loads(str(deployment["universe_json"]))
            if not isinstance(symbols, list):
                raise ValueError("deployment universe_json is invalid")
            runtime.quote_provider.subscribe(tuple(str(symbol) for symbol in symbols))
        if args.command == "signal":
            runtime.jobs.prepare_signal(trade_date, lock_connection=connection)
        elif args.command == "execute":
            runtime.jobs.execute_pending_target(trade_date, lock_connection=connection)
        elif args.command == "cancel":
            runtime.jobs.cancel_open_orders(trade_date, lock_connection=connection)
        elif args.command == "snapshot":
            runtime.jobs.snapshot_eod(trade_date, lock_connection=connection)
        elif args.command == "reconcile" and args.phase == "startup":
            runtime.jobs.startup_reconcile(trade_date, lock_connection=connection)
        elif args.command == "reconcile":
            runtime.jobs.reconcile_eod(trade_date, lock_connection=connection)
        else:  # pragma: no cover - guarded by parser and caller
            raise ValueError("unsupported manual command")
    finally:
        try:
            if connected:
                runtime.broker.disconnect()
        finally:
            try:
                if consumer_started and runtime.event_consumer is not None:
                    runtime.event_consumer.stop()
            finally:
                try:
                    if locked:
                        release_account_lock(connection, account_id)
                finally:
                    connection.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_builder: RuntimeBuilder | None = None,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_live_config(Path(args.config))
        if args.command == "reconcile" and args.phase == "eod" and args.date is None:
            raise ValueError("reconcile --phase eod requires --date")
        if args.command == "validate":
            _validate(config)
        elif args.command == "db":
            _db_init(config)
        elif args.command == "jobs":
            _print_jobs(config, _date(args.date))
        elif args.command == "status":
            _print_status(config)
        else:
            runtime = (runtime_builder or build_production_runtime)(config)
            if args.command == "service":
                runtime.engine.run_forever()
            else:
                _manual_job(runtime, config, args)
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveCommandRuntime",
    "build_production_runtime",
    "create_state_engine",
    "main",
]
