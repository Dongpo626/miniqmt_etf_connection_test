"""Minimal Rule PAPER daily jobs composed from the existing live services."""

from __future__ import annotations

import json
import time as time_module
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeVar, cast

from sqlalchemy.engine import Connection, Engine

from etf_backtest.application.contracts import DailyDecisionResult, DecisionStatus
from etf_backtest.application.daily_decision import DailyDecisionService
from etf_backtest.application.runtime_factory import (
    build_model_signal_runtime,
    build_rule_signal_runtime,
    canonical_universe_identity,
    create_repository,
    resolve_universe,
)
from etf_backtest.application.schedule import TradingDayIndexResolver
from etf_backtest.application.strategy_source import (
    ModelStrategySource,
    RuleStrategySource,
    sha256_file,
)
from etf_backtest.config.schema import (
    MARKET_TIMEZONE,
    BacktestConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
)
from etf_backtest.core.order import OrderSide
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.data.calendar import SseTradingCalendar
from etf_backtest.live.account_adapter import AdaptedAccountState, adapt_broker_account
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.execution.near_close_limit import NearCloseLimitPolicy
from etf_backtest.live.execution.planner import LiveRebalancePlanner
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import (
    LiveStateRepository,
    acquire_job_lock,
    release_job_lock,
)
from etf_backtest.live.reconciliation import ReconciliationService
from etf_backtest.live.risk import LiveRiskManager
from etf_backtest.live.state import (
    BrokerAssetSnapshot,
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
    BrokerTradeSnapshot,
    DeploymentStatus,
    JobTriggerSource,
    OrderIntent,
    OrderIntentStatus,
    QueryResult,
    ReconciliationReport,
    SnapshotType,
    SubmitOrderStatus,
)
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_training import LoadedInferenceBundle, bundle_sha256

ResultT = TypeVar("ResultT")


class JobAlreadySucceeded(RuntimeError):
    """Signal that one persisted daily job is already terminal."""


class JobSkipped(RuntimeError):
    """Signal a deliberate safe skip that must not be retried as a failure."""


@dataclass(frozen=True, slots=True)
class DeploymentSpec:
    deployment_id: str
    bound_account_id: str
    mode: str
    experiment_path: str
    experiment_sha256: str
    strategy_source_sha256: str
    schedule_anchor_date: date
    symbols: tuple[str, ...]
    universe_json: str
    universe_hash: str
    config_hash: str
    model_bundle_path: str | None = None
    model_bundle_sha256: str | None = None
    model_id: str | None = None

    def repository_values(self) -> dict[str, object]:
        return {
            "deployment_id": self.deployment_id,
            "bound_account_id": self.bound_account_id,
            "mode": self.mode,
            "experiment_path": self.experiment_path,
            "experiment_sha256": self.experiment_sha256,
            "strategy_source_sha256": self.strategy_source_sha256,
            "schedule_anchor_date": self.schedule_anchor_date,
            "universe_json": self.universe_json,
            "universe_hash": self.universe_hash,
            "config_hash": self.config_hash,
            "model_bundle_path": self.model_bundle_path,
            "model_bundle_sha256": self.model_bundle_sha256,
            "model_id": self.model_id,
        }


class DeploymentSpecFactory(Protocol):
    def __call__(self, frozen_symbols: tuple[str, ...] | None) -> DeploymentSpec: ...


class SignalEvaluator(Protocol):
    def evaluate(
        self,
        *,
        signal_date: date,
        symbols: tuple[str, ...],
        account: AdaptedAccountState,
    ) -> DailyDecisionResult: ...


def _rule_backtest_config(source: RuleStrategySource) -> BacktestConfig:
    rule = source.rule
    strategy = RuleStrategyConfig(
        lookback_trading_days=rule.lookback_trading_days,
        rebalance_every_trading_days=rule.rebalance_every_trading_days,
        target_weight=rule.target_weight,
    )
    return source.experiment.build_case(source.system, strategy=strategy)


def _model_backtest_config(source: ModelStrategySource) -> BacktestConfig:
    settings = source.components.settings
    strategy = ModelStrategyConfig(
        max_total_weight=settings.portfolio.max_total_weight,
        train_start=settings.train_range.start_date,
        train_end=settings.train_range.end_date,
        valid_start=settings.valid_range.start_date,
        valid_end=settings.valid_range.end_date,
        test_start=source.experiment.start_date,
        test_end=source.experiment.end_date,
    )
    return source.experiment.build_case(source.system, strategy=strategy)


class RuleDeploymentSpecFactory:
    """Resolve Universe only for first activation; otherwise preserve frozen symbols."""

    def __init__(
        self,
        *,
        live_config: LiveConfig,
        source: RuleStrategySource,
        strategy_engine: Engine,
    ) -> None:
        self._live_config = live_config
        self._source = source
        self._strategy_engine = strategy_engine
        self._backtest_config = _rule_backtest_config(source)

    def __call__(self, frozen_symbols: tuple[str, ...] | None) -> DeploymentSpec:
        if frozen_symbols is None:
            with self._strategy_engine.connect() as connection:
                repository = create_repository(
                    self._backtest_config, self._strategy_engine, connection=connection
                )
                symbols = resolve_universe(self._backtest_config, repository).symbols
        else:
            symbols = frozen_symbols
        ordered, universe_json, universe_hash = canonical_universe_identity(symbols)
        deployment = self._live_config.deployment
        return DeploymentSpec(
            deployment_id=deployment.deployment_id,
            bound_account_id=deployment.account_id(),
            mode=deployment.mode,
            experiment_path=str(self._source.experiment_path),
            experiment_sha256=sha256_file(self._source.experiment_path),
            strategy_source_sha256=self._source.strategy_source_sha256,
            schedule_anchor_date=deployment.schedule_anchor_date,
            symbols=ordered,
            universe_json=universe_json,
            universe_hash=universe_hash,
            config_hash=self._live_config.config_hash,
        )


class ModelDeploymentSpecFactory:
    """Bind one verified bundle to one immutable Model deployment."""

    def __init__(
        self,
        *,
        live_config: LiveConfig,
        source: ModelStrategySource,
        strategy_engine: Engine,
        bundle_path: Path,
        loaded_bundle: LoadedInferenceBundle,
    ) -> None:
        self._live_config = live_config
        self._source = source
        self._strategy_engine = strategy_engine
        self._bundle_path = bundle_path
        self._loaded_bundle = loaded_bundle
        self._backtest_config = _model_backtest_config(source)

    def __call__(self, frozen_symbols: tuple[str, ...] | None) -> DeploymentSpec:
        if bundle_sha256(self._bundle_path) != self._loaded_bundle.file_sha256:
            raise ValueError("model bundle changed after runtime construction")
        if frozen_symbols is None:
            with self._strategy_engine.connect() as connection:
                repository = create_repository(
                    self._backtest_config, self._strategy_engine, connection=connection
                )
                symbols = resolve_universe(self._backtest_config, repository).symbols
        else:
            symbols = frozen_symbols
        ordered, universe_json, universe_hash = canonical_universe_identity(symbols)
        deployment = self._live_config.deployment
        return DeploymentSpec(
            deployment_id=deployment.deployment_id,
            bound_account_id=deployment.account_id(),
            mode=deployment.mode,
            experiment_path=str(self._source.experiment_path),
            experiment_sha256=sha256_file(self._source.experiment_path),
            strategy_source_sha256=self._source.strategy_source_sha256,
            schedule_anchor_date=deployment.schedule_anchor_date,
            symbols=ordered,
            universe_json=universe_json,
            universe_hash=universe_hash,
            config_hash=self._live_config.config_hash,
            model_bundle_path=str(self._bundle_path),
            model_bundle_sha256=self._loaded_bundle.file_sha256,
            model_id=self._loaded_bundle.bundle.metadata.model_id,
        )


class RuleSignalService:
    """Evaluate one Rule signal inside one caller-owned strategy connection."""

    def __init__(
        self,
        *,
        live_config: LiveConfig,
        source: RuleStrategySource,
        strategy_engine: Engine,
        decision_service: DailyDecisionService | None = None,
        schedule_resolver: TradingDayIndexResolver | None = None,
    ) -> None:
        self._live_config = live_config
        self._source = source
        self._strategy_engine = strategy_engine
        self._decision_service = decision_service or DailyDecisionService()
        self._schedule_resolver = schedule_resolver or TradingDayIndexResolver()
        self._backtest_config = _rule_backtest_config(source)

    def evaluate(
        self,
        *,
        signal_date: date,
        symbols: tuple[str, ...],
        account: AdaptedAccountState,
    ) -> DailyDecisionResult:
        load_start = min(
            self._source.experiment.start_date,
            self._live_config.deployment.schedule_anchor_date,
            signal_date,
        )
        load_end = signal_date + timedelta(days=10)
        with self._strategy_engine.connect() as connection, connection.begin():
            runtime = build_rule_signal_runtime(
                source=self._source,
                config=self._backtest_config,
                engine=self._strategy_engine,
                frozen_symbols=symbols,
                load_start=load_start,
                load_end=load_end,
                connection=connection,
            )
            calendar = runtime.portal.trading_calendar
            self._schedule_resolver.previous_trading_day(calendar=calendar, trade_date=signal_date)
            execution_date = self._schedule_resolver.next_trading_day(
                calendar=calendar, trade_date=signal_date
            )
            schedule_index = self._schedule_resolver.resolve(
                calendar=calendar,
                anchor_date=self._live_config.deployment.schedule_anchor_date,
                signal_date=signal_date,
            )
            return self._decision_service.evaluate(
                strategy=self._source.strategy,
                portal=runtime.portal,
                signal_date=signal_date,
                execution_date=execution_date,
                schedule_index=schedule_index,
                symbols=symbols,
                account_view=account.account_view,
                current_weights_by_symbol=account.current_weights_by_symbol,
            )


class ModelSignalService:
    """Run fixed-bundle inference in one caller-owned strategy transaction."""

    def __init__(
        self,
        *,
        live_config: LiveConfig,
        source: ModelStrategySource,
        strategy: DailyModelStrategy,
        strategy_engine: Engine,
        decision_service: DailyDecisionService | None = None,
        schedule_resolver: TradingDayIndexResolver | None = None,
    ) -> None:
        self._live_config = live_config
        self._source = source
        self._strategy = strategy
        self._strategy_engine = strategy_engine
        self._decision_service = decision_service or DailyDecisionService()
        self._schedule_resolver = schedule_resolver or TradingDayIndexResolver()
        self._backtest_config = _model_backtest_config(source)

    def evaluate(
        self,
        *,
        signal_date: date,
        symbols: tuple[str, ...],
        account: AdaptedAccountState,
    ) -> DailyDecisionResult:
        lookback = self._strategy.required_history_trading_days
        load_start = signal_date - timedelta(days=max(90, lookback * 3))
        with self._strategy_engine.connect() as connection, connection.begin():
            runtime = build_model_signal_runtime(
                source=self._source,
                config=self._backtest_config,
                engine=self._strategy_engine,
                frozen_symbols=symbols,
                load_start=load_start,
                load_end=signal_date,
                connection=connection,
            )
            calendar_days = runtime.repository.load_sse_calendar(
                min(load_start, self._live_config.deployment.schedule_anchor_date),
                signal_date + timedelta(days=10),
            )
            calendar = SseTradingCalendar(calendar_days)
            execution_date = self._schedule_resolver.next_trading_day(
                calendar=calendar, trade_date=signal_date
            )
            schedule_index = self._schedule_resolver.resolve(
                calendar=calendar,
                anchor_date=self._live_config.deployment.schedule_anchor_date,
                signal_date=signal_date,
            )
            return self._decision_service.evaluate(
                strategy=self._strategy,
                portal=runtime.portal,
                signal_date=signal_date,
                execution_date=execution_date,
                schedule_index=schedule_index,
                symbols=symbols,
                account_view=account.account_view,
                current_weights_by_symbol=account.current_weights_by_symbol,
            )


def _symbols(deployment: Mapping[str, object]) -> tuple[str, ...]:
    value = json.loads(str(deployment["universe_json"]))
    if not isinstance(value, list) or not all(isinstance(symbol, str) for symbol in value):
        raise ValueError("deployment universe_json is invalid")
    return canonical_universe_identity(value)[0]


def _status[EnumT: Enum](value: object, enum_type: type[EnumT]) -> EnumT:
    return value if isinstance(value, enum_type) else enum_type(cast(str, value))


class LiveDailyJobs:
    def __init__(
        self,
        *,
        config: LiveConfig,
        broker: BrokerGateway,
        quote_provider: QuoteProvider,
        state_repository: LiveStateRepository,
        state_engine: Engine,
        signal_evaluator: SignalEvaluator,
        deployment_spec_factory: DeploymentSpecFactory,
        planner: LiveRebalancePlanner | None = None,
        risk_manager: LiveRiskManager | None = None,
        price_policy: NearCloseLimitPolicy | None = None,
        reconciliation_service: ReconciliationService | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.quote_provider = quote_provider
        self.repository = state_repository
        self.state_engine = state_engine
        self.signal_evaluator = signal_evaluator
        self.deployment_spec_factory = deployment_spec_factory
        self.planner = planner or LiveRebalancePlanner()
        self.risk = risk_manager or LiveRiskManager()
        self.price_policy = price_policy or NearCloseLimitPolicy()
        self.reconciliation = reconciliation_service or ReconciliationService()
        self.clock = clock or (lambda: datetime.now(MARKET_TIMEZONE))
        self.sleep = sleep or time_module.sleep

    def _run_job(
        self,
        job_type: str,
        trade_date: date,
        trigger_source: JobTriggerSource,
        body: Callable[[], ResultT],
        lock_connection: Connection | None,
        *,
        deduplicate: bool = True,
    ) -> ResultT:
        manager = (
            self.state_engine.connect()
            if lock_connection is None
            else _BorrowedConnection(lock_connection)
        )
        with manager as lock:
            acquired = acquire_job_lock(
                lock, self.config.deployment.deployment_id, job_type, trade_date
            )
            if not acquired:
                error = RuntimeError(f"job lock is already held: {job_type}")
                run_id = self.repository.start_job_run(
                    deployment_id=self.config.deployment.deployment_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                )
                self.repository.fail_job_run(
                    run_id, error_type=type(error).__name__, error_message=str(error)
                )
                raise error
            run_id = ""
            try:
                if deduplicate and self.repository.has_terminal_job(
                    self.config.deployment.deployment_id,
                    job_type,
                    trade_date,
                ):
                    raise JobAlreadySucceeded(
                        f"job already completed: {job_type} {trade_date.isoformat()}"
                    )
                run_id = self.repository.start_job_run(
                    deployment_id=self.config.deployment.deployment_id,
                    job_type=job_type,
                    trade_date=trade_date,
                    trigger_source=trigger_source,
                )
                result = body()
            except JobAlreadySucceeded:
                raise
            except JobSkipped as error:
                self.repository.skip_job_run(run_id, reason=str(error))
                raise
            except Exception as error:
                self.repository.fail_job_run(
                    run_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                raise
            else:
                self.repository.finish_job_run(run_id)
                return result
            finally:
                release_job_lock(lock, self.config.deployment.deployment_id, job_type, trade_date)

    def has_job_completed(self, job_type: str, trade_date: date) -> bool:
        return self.repository.has_terminal_job(
            self.config.deployment.deployment_id, job_type, trade_date
        )

    def record_job_skipped(self, job_type: str, trade_date: date, reason: str) -> None:
        def skip() -> None:
            raise JobSkipped(reason)

        try:
            self._run_job(
                job_type,
                trade_date,
                JobTriggerSource.RECOVERY,
                skip,
                None,
            )
        except (JobAlreadySucceeded, JobSkipped):
            return

    def has_rebalance_activity(self, execution_date: date) -> bool:
        """Return whether recovery must enter reconciliation instead of replanning."""

        deployment = self.repository.get_deployment(self.config.deployment.deployment_id)
        if deployment is None:
            return False
        decision = self.repository.pending_decision_for_date(
            str(deployment["deployment_id"]), execution_date
        )
        if decision is None:
            return False
        return bool(self.repository.list_order_intents_for_decision(str(decision["decision_id"])))

    def startup_reconcile(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.RECOVERY,
        lock_connection: Connection | None = None,
    ) -> Mapping[str, object]:
        return self._run_job(
            "startup_reconcile",
            trade_date,
            trigger_source,
            lambda: self._startup_reconcile(trade_date),
            lock_connection,
            deduplicate=False,
        )

    def _startup_reconcile(self, trade_date: date) -> Mapping[str, object]:
        self._require_current_trade_date(trade_date)
        account_id = self.config.deployment.account_id()
        configured = self.repository.get_deployment(self.config.deployment.deployment_id)
        asset, positions, orders, trades = self._account_facts(require_trades=True)
        if asset.account_id != account_id:
            if configured is not None:
                self.repository.pause_deployment(
                    str(configured["deployment_id"]), "ACCOUNT_ID_MISMATCH"
                )
            raise ValueError("broker account_id does not match configured account")
        report = self.reconciliation.reconcile(
            account_id=account_id,
            broker_orders=orders,
            broker_trades=trades,
            repository=self.repository,
        )
        if report.has_unresolved:
            if configured is not None:
                self.repository.pause_deployment(
                    str(configured["deployment_id"]), "STARTUP_RECONCILIATION_UNRESOLVED"
                )
            raise RuntimeError("startup reconciliation has unresolved broker facts")
        if (
            configured is not None
            and _status(configured["status"], DeploymentStatus) is DeploymentStatus.RETIRED
        ):
            raise RuntimeError("RETIRED deployment requires a new deployment_id")
        frozen = None if configured is None else _symbols(configured)
        if configured is not None:
            # PLANNED rows found here belong to a previous Engine lifecycle. They are never
            # submitted after recovery; the next scheduled signal must create a fresh decision.
            self.repository.abandon_planned_intents(str(configured["deployment_id"]))
        spec = self.deployment_spec_factory(frozen)
        outside = [
            position.symbol
            for position in positions
            if position.total_quantity != 0 and position.symbol not in spec.symbols
        ]
        if outside:
            if configured is not None:
                self.repository.pause_deployment(
                    str(configured["deployment_id"]), "POSITION_OUTSIDE_FROZEN_UNIVERSE"
                )
            raise RuntimeError(f"non-zero positions exist outside frozen Universe: {outside}")
        ensured = self.repository.ensure_deployment(
            deployment_id=spec.deployment_id,
            bound_account_id=spec.bound_account_id,
            mode=spec.mode,
            experiment_path=spec.experiment_path,
            experiment_sha256=spec.experiment_sha256,
            strategy_source_sha256=spec.strategy_source_sha256,
            schedule_anchor_date=spec.schedule_anchor_date,
            universe_json=spec.universe_json,
            universe_hash=spec.universe_hash,
            config_hash=spec.config_hash,
            model_bundle_path=spec.model_bundle_path,
            model_bundle_sha256=spec.model_bundle_sha256,
            model_id=spec.model_id,
        )
        if (
            configured is not None
            and _status(configured["status"], DeploymentStatus) is DeploymentStatus.PAUSED
        ):
            self.repository.resume_deployment(spec.deployment_id)
            ensured = self.repository.get_deployment(spec.deployment_id) or ensured
        self._save_account_facts(
            deployment=ensured,
            trade_date=trade_date,
            snapshot_types=(SnapshotType.CURRENT,),
            asset=asset,
            positions=positions,
        )
        return ensured

    def prepare_signal(
        self,
        signal_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> DailyDecisionResult:
        return self._run_job(
            "prepare_signal",
            signal_date,
            trigger_source,
            lambda: self._prepare_signal(signal_date),
            lock_connection,
        )

    def _prepare_signal(self, signal_date: date) -> DailyDecisionResult:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        local_time = now.time().replace(tzinfo=None)
        if signal_date < now.date():
            raise JobSkipped("STALE_SIGNAL_DATE")
        if signal_date > now.date() or (
            signal_date == now.date() and local_time < self.config.signal.run_time
        ):
            raise JobSkipped("SIGNAL_TIME_NOT_REACHED")
        deployment = self._active_deployment()
        self._require_no_unresolved()
        asset = self._single(self.broker.query_asset(), "asset")
        positions = self._records(self.broker.query_positions(), "positions")
        self._validate_asset_positions(asset, positions)
        symbols = _symbols(deployment)
        account = adapt_broker_account(asset=asset, positions=positions, symbols=symbols)
        result = self.signal_evaluator.evaluate(
            signal_date=signal_date, symbols=symbols, account=account
        )
        decision_id = self._decision_id(signal_date)
        with self.repository.transaction() as connection:
            saved = self.repository.create_or_get_decision(
                decision_id=decision_id,
                deployment_id=str(deployment["deployment_id"]),
                signal_date=result.signal_date,
                execution_date=result.execution_date,
                schedule_index=result.schedule_index,
                status=result.status,
                data_as_of=signal_date,
                strategy_source_sha256=str(deployment["strategy_source_sha256"]),
                model_id=(
                    None if deployment.get("model_id") is None else str(deployment["model_id"])
                ),
                config_hash=str(deployment["config_hash"]),
                connection=connection,
            )
            if result.status is DecisionStatus.TARGET_CREATED:
                assert result.target_portfolio is not None
                complete = {
                    symbol: result.target_portfolio.weight_for(symbol) for symbol in symbols
                }
                self.repository.save_target_positions(
                    str(saved["decision_id"]), complete, connection=connection
                )
        return result

    def execute_pending_target(
        self,
        execution_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        return self._run_job(
            "rebalance",
            execution_date,
            trigger_source,
            lambda: self._execute_pending_target(execution_date),
            lock_connection,
        )

    def _execute_pending_target(self, execution_date: date) -> tuple[Mapping[str, object], ...]:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        local_time = now.time().replace(tzinfo=None)
        if execution_date < now.date():
            raise JobSkipped("STALE_EXECUTION_DATE")
        if execution_date > now.date():
            raise JobSkipped("FUTURE_EXECUTION_DATE")
        deployment = self._active_deployment()
        decision = self.repository.pending_decision_for_date(
            str(deployment["deployment_id"]), execution_date
        )
        if decision is None:
            return ()
        decision_id = str(decision["decision_id"])
        if self.repository.decision_has_intent_status(decision_id, OrderIntentStatus.ABANDONED):
            raise JobSkipped("ABANDONED_STALE_INTENT")
        existing_intents = self.repository.list_order_intents_for_decision(decision_id)
        if existing_intents:
            # Any persisted execution activity means this decision has already crossed the
            # planning boundary. Recovery may query/cancel/reconcile it, but never plans a
            # second batch and never submits a second intent for the same economic action.
            return self._finalize_rebalance(execution_date, deployment, tuple(existing_intents))
        if local_time < self.config.execution.submit_start:
            raise JobSkipped("REBALANCE_TIME_NOT_REACHED")
        if local_time >= self.config.execution.stop_new_orders:
            raise JobSkipped("MISSED_ORDER_WINDOW")
        weights = self.repository.load_target_positions(str(decision["decision_id"]))
        symbols = _symbols(deployment)
        asset, positions, orders, trades = self._account_facts(require_trades=True)
        self._reconcile_or_pause(deployment, orders, trades)
        quotes = {
            quote.symbol: quote
            for quote in self._records(self.quote_provider.latest_quotes(symbols), "quotes")
        }
        valuation: dict[str, Decimal] = {}
        buy_limits: dict[str, Decimal] = {}
        sell_limits: dict[str, Decimal] = {}
        now = self.clock()
        for symbol in symbols:
            quote = quotes.get(symbol)
            if quote is None:
                raise RuntimeError(f"quote is missing: {symbol}")
            buy = self.price_policy.calculate(
                side=OrderSide.BUY,
                quote=quote,
                tick_size=Decimal("0.001"),
                price_offset_ticks=self.config.execution.price_offset_ticks,
                now=now,
                quote_stale_seconds=self.config.execution.quote_stale_seconds,
            )
            sell = self.price_policy.calculate(
                side=OrderSide.SELL,
                quote=quote,
                tick_size=Decimal("0.001"),
                price_offset_ticks=self.config.execution.price_offset_ticks,
                now=now,
                quote_stale_seconds=self.config.execution.quote_stale_seconds,
            )
            if buy is None or sell is None:
                raise RuntimeError(f"valid near-close price is unavailable: {symbol}")
            valuation[symbol] = buy.valuation_price
            buy_limits[symbol] = buy.limit_price
            sell_limits[symbol] = sell.limit_price
        target = TargetPortfolio(weights)
        position_map = {position.symbol: position for position in positions}
        preview = self.planner.plan(
            deployment_id=str(deployment["deployment_id"]),
            decision_id=decision_id,
            execution_date=execution_date,
            symbols=symbols,
            target=target,
            total_asset=asset.total_asset,
            available_cash=asset.available_cash,
            positions=position_map,
            active_orders=orders,
            valuation_prices=valuation,
            limit_prices=buy_limits,
            lot_size=self.config.execution.lot_size,
        )
        sides = {intent.symbol: intent.side for intent in preview}
        limits = {
            symbol: sell_limits[symbol]
            if sides.get(symbol) is OrderSide.SELL
            else buy_limits[symbol]
            for symbol in symbols
        }
        intents = self.planner.plan(
            deployment_id=str(deployment["deployment_id"]),
            decision_id=decision_id,
            execution_date=execution_date,
            symbols=symbols,
            target=target,
            total_asset=asset.total_asset,
            available_cash=asset.available_cash,
            positions=position_map,
            active_orders=orders,
            valuation_prices=valuation,
            limit_prices=limits,
            lot_size=self.config.execution.lot_size,
        )
        saved_rows: list[Mapping[str, object]] = []
        daily_notional = Decimal("0")
        for intent in intents:
            available = position_map.get(intent.symbol)
            risk = self.risk.check(
                intent,
                symbols=symbols,
                target_weights=weights,
                max_total_target_weight=self.config.risk.max_total_target_weight,
                available_cash=asset.available_cash,
                available_quantity=0 if available is None else available.available_quantity,
                lot_size=self.config.execution.lot_size,
                max_single_order_notional=self.config.risk.max_single_order_notional,
                max_daily_order_notional=self.config.risk.max_daily_order_notional,
                daily_planned_notional=daily_notional,
                min_order_notional=self.config.risk.min_order_notional,
                quote_valid=True,
            )
            saved = self.repository.create_order_intent(intent)
            if not risk.approved:
                self.repository.mark_intent_rejected(
                    str(saved["intent_id"]), risk.reason or "RISK_REJECTED"
                )
                saved_rows.append(saved)
                continue
            daily_notional += intent.requested_quantity * intent.limit_price
            status = _status(saved["status"], OrderIntentStatus)
            if status is OrderIntentStatus.PLANNED:
                try:
                    self._submit_intent(
                        intent,
                        str(saved["intent_id"]),
                        self.config.deployment.account_id(),
                    )
                except Exception:
                    self.repository.pause_deployment(
                        str(deployment["deployment_id"]), "SUBMIT_RESULT_UNKNOWN"
                    )
                    raise
            saved_rows.append(self.repository.get_intent(str(saved["intent_id"])) or saved)
        return self._finalize_rebalance(execution_date, deployment, tuple(saved_rows))

    def _finalize_rebalance(
        self,
        execution_date: date,
        deployment: Mapping[str, object],
        intent_rows: tuple[Mapping[str, object], ...],
    ) -> tuple[Mapping[str, object], ...]:
        now = self.clock().astimezone(MARKET_TIMEZONE)
        wait_seconds = max(
            0.0,
            (
                datetime.combine(
                    execution_date,
                    self.config.execution.cancel_open_orders,
                    MARKET_TIMEZONE,
                )
                - now
            ).total_seconds(),
        )
        deadline = time_module.monotonic() + wait_seconds
        poll_seconds = float(self.config.miniqmt.reconnect_interval_seconds)
        poll_limit = max(1, int(wait_seconds // poll_seconds) + 2)

        for poll_number in range(poll_limit):
            orders = self._records(self.broker.query_orders(), "orders")
            trades = self._records(self.broker.query_trades(), "trades")
            report = self._reconcile_or_pause(deployment, orders, trades)
            if not report.active_broker_order_ids:
                break
            remaining = deadline - time_module.monotonic()
            if remaining <= 0 or poll_number + 1 >= poll_limit:
                self._cancel_open_orders()
                break
            self.sleep(min(poll_seconds, remaining))

        final_asset, final_positions, final_orders, final_trades = self._account_facts(
            require_trades=True
        )
        final_report = self._reconcile_or_pause(deployment, final_orders, final_trades)
        if final_report.active_broker_order_ids:
            self.repository.pause_deployment(
                str(deployment["deployment_id"]), "ACTIVE_ORDER_AFTER_CANCEL"
            )
            raise RuntimeError("rebalance ended with active broker orders")
        if final_report.has_incomplete:
            self.repository.pause_deployment(
                str(deployment["deployment_id"]), "INCOMPLETE_REBALANCE"
            )
        self._save_account_facts(
            deployment=deployment,
            trade_date=execution_date,
            snapshot_types=(SnapshotType.CURRENT,),
            asset=final_asset,
            positions=final_positions,
        )
        return intent_rows

    def _submit_intent(self, intent: OrderIntent, intent_id: str, account_id: str) -> None:
        self.repository.mark_intent_submitting(intent_id)
        try:
            result = self.broker.submit_order(intent)
        except Exception as error:
            self.repository.mark_intent_submit_unknown(intent_id, str(error))
            raise
        if result.status is SubmitOrderStatus.ACCEPTED:
            assert result.broker_order_id is not None
            self.repository.bind_broker_order(
                account_id=account_id,
                intent_id=intent_id,
                remark_token=intent.remark_token,
                order=BrokerOrderSnapshot(
                    broker_order_id=result.broker_order_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    requested_quantity=intent.requested_quantity,
                    filled_quantity=0,
                    limit_price=intent.limit_price,
                    status=BrokerOrderStatus.PENDING,
                    captured_at=self.clock(),
                    remark_token=intent.remark_token,
                ),
            )
        elif result.status is SubmitOrderStatus.REJECTED:
            self.repository.mark_intent_rejected(intent_id, result.error or "BROKER_REJECTED")
        else:
            self.repository.mark_intent_submit_unknown(
                intent_id, result.error or "SUBMIT_RESULT_UNKNOWN"
            )

    def cancel_open_orders(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> None:
        self._run_job(
            "cancel_open_orders",
            trade_date,
            trigger_source,
            lambda: self._cancel_open_orders_for_date(trade_date),
            lock_connection,
        )

    def _cancel_open_orders_for_date(self, trade_date: date) -> None:
        self._require_current_trade_date(trade_date)
        self._require_time_at_or_after(
            self.config.execution.cancel_open_orders, "CANCEL_TIME_NOT_REACHED"
        )
        self._cancel_open_orders()

    def _cancel_open_orders(self) -> None:
        deployment = self._active_deployment()
        account_id = self.config.deployment.account_id()
        orders = self._records(self.broker.query_orders(), "orders")
        associated: list[BrokerOrderSnapshot] = []
        unknown: list[str] = []
        for order in orders:
            if not order.status.is_active:
                continue
            saved_order = self.repository.get_broker_order(account_id, order.broker_order_id)
            intent = None
            if saved_order is not None:
                intent = self.repository.get_intent(str(saved_order["intent_id"]))
            elif order.remark_token:
                intent = self.repository.get_intent_by_remark_token(
                    order.remark_token, account_id=account_id
                )
            if intent is None or str(intent["deployment_id"]) != str(deployment["deployment_id"]):
                unknown.append(order.broker_order_id)
            else:
                associated.append(order)
        if unknown:
            self.repository.pause_deployment(
                str(deployment["deployment_id"]), "UNKNOWN_ACTIVE_ORDER"
            )
            raise RuntimeError(f"unknown active broker orders: {unknown}")
        for order in associated:
            self.broker.cancel_order(order.broker_order_id)
        orders_after = self._records(self.broker.query_orders(), "orders")
        trades_after = self._records(self.broker.query_trades(), "trades")
        self._reconcile_or_pause(deployment, orders_after, trades_after)

    def reconcile_eod(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> ReconciliationReport:
        return self._run_job(
            "reconcile_eod",
            trade_date,
            trigger_source,
            lambda: self._reconcile_eod(trade_date),
            lock_connection,
        )

    def eod(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> ReconciliationReport:
        return self._run_job(
            "eod",
            trade_date,
            trigger_source,
            lambda: self._eod(trade_date),
            lock_connection,
        )

    def _eod(self, trade_date: date) -> ReconciliationReport:
        self._require_current_trade_date(trade_date)
        self._require_time_at_or_after(self.config.eod.run_time, "EOD_TIME_NOT_REACHED")
        deployment = self._active_deployment(allow_paused=True)
        try:
            asset, positions, orders, trades = self._account_facts(require_trades=True)
        except Exception:
            self.repository.pause_deployment(str(deployment["deployment_id"]), "EOD_QUERY_FAILED")
            raise
        report = self.reconciliation.reconcile(
            account_id=self.config.deployment.account_id(),
            broker_orders=orders,
            broker_trades=trades,
            repository=self.repository,
        )
        if report.has_unresolved or report.active_broker_order_ids:
            self.repository.pause_deployment(str(deployment["deployment_id"]), "EOD_UNRESOLVED")
            raise RuntimeError("end-of-day reconciliation is unresolved")
        if report.has_incomplete:
            self.repository.pause_deployment(
                str(deployment["deployment_id"]), "INCOMPLETE_REBALANCE"
            )
        self._save_account_facts(
            deployment=deployment,
            trade_date=trade_date,
            snapshot_types=(SnapshotType.CURRENT, SnapshotType.EOD),
            asset=asset,
            positions=positions,
        )
        return report

    def _reconcile_eod(self, trade_date: date | None = None) -> ReconciliationReport:
        self._require_current_trade_date(trade_date or self.clock().date())
        self._require_time_at_or_after(self.config.eod.run_time, "EOD_TIME_NOT_REACHED")
        deployment = self._active_deployment(allow_paused=True)
        try:
            orders = self._records(self.broker.query_orders(), "orders")
            trades = self._records(self.broker.query_trades(), "trades")
        except Exception:
            self.repository.pause_deployment(str(deployment["deployment_id"]), "EOD_QUERY_FAILED")
            raise
        report = self.reconciliation.reconcile(
            account_id=self.config.deployment.account_id(),
            broker_orders=orders,
            broker_trades=trades,
            repository=self.repository,
        )
        if report.has_unresolved or report.active_broker_order_ids:
            self.repository.pause_deployment(str(deployment["deployment_id"]), "EOD_UNRESOLVED")
            raise RuntimeError("end-of-day reconciliation is unresolved")
        return report

    def snapshot_eod(
        self,
        trade_date: date,
        *,
        trigger_source: JobTriggerSource = JobTriggerSource.MANUAL,
        lock_connection: Connection | None = None,
    ) -> None:
        self._run_job(
            "snapshot_eod",
            trade_date,
            trigger_source,
            lambda: self._snapshot_eod(trade_date),
            lock_connection,
        )

    def _snapshot_eod(self, trade_date: date) -> None:
        # Keep the historical CLI command, but route it through the same authoritative
        # query/reconcile boundary as the scheduled EOD business job.
        self._eod(trade_date)

    def _save_account_facts(
        self,
        *,
        deployment: Mapping[str, object],
        trade_date: date,
        snapshot_types: tuple[SnapshotType, ...],
        asset: BrokerAssetSnapshot,
        positions: Sequence[BrokerPositionSnapshot],
    ) -> None:
        adapted = adapt_broker_account(
            asset=asset, positions=positions, symbols=_symbols(deployment)
        )
        market_value = sum(
            (position.market_value for position in adapted.positions_by_symbol.values()),
            Decimal("0"),
        )
        frozen_cash = max(Decimal("0"), asset.total_asset - asset.available_cash - market_value)
        with self.repository.transaction() as connection:
            rows = []
            for symbol in _symbols(deployment):
                position = adapted.positions_by_symbol[symbol]
                last_price = (
                    Decimal("0")
                    if position.total_quantity == 0
                    else position.market_value / position.total_quantity
                )
                rows.append(
                    {
                        "symbol": symbol,
                        "total_quantity": position.total_quantity,
                        "available_quantity": position.available_quantity,
                        "frozen_quantity": position.total_quantity - position.available_quantity,
                        "market_value": position.market_value,
                        "last_price": last_price,
                    }
                )
            for snapshot_type in snapshot_types:
                self.repository.save_account_snapshot(
                    deployment_id=str(deployment["deployment_id"]),
                    trade_date=trade_date,
                    snapshot_type=snapshot_type,
                    captured_at=asset.captured_at,
                    cash=asset.available_cash,
                    available_cash=asset.available_cash,
                    market_value=market_value,
                    total_asset=asset.total_asset,
                    frozen_cash=frozen_cash,
                    connection=connection,
                )
                self.repository.save_position_snapshots(
                    deployment_id=str(deployment["deployment_id"]),
                    trade_date=trade_date,
                    snapshot_type=snapshot_type,
                    positions=rows,
                    connection=connection,
                )

    def _active_deployment(self, *, allow_paused: bool = False) -> Mapping[str, object]:
        deployment = self.repository.get_deployment(self.config.deployment.deployment_id)
        if deployment is None:
            raise RuntimeError("deployment does not exist")
        status = _status(deployment["status"], DeploymentStatus)
        if status is not DeploymentStatus.ACTIVE and not (
            allow_paused and status is DeploymentStatus.PAUSED
        ):
            raise RuntimeError("deployment is not ACTIVE")
        return deployment

    def _require_no_unresolved(self) -> None:
        if self.repository.current_unresolved(self.config.deployment.deployment_id):
            raise RuntimeError("deployment has unresolved intents")

    def _reconcile_or_pause(
        self,
        deployment: Mapping[str, object],
        orders: Sequence[BrokerOrderSnapshot],
        trades: Sequence[BrokerTradeSnapshot],
    ) -> ReconciliationReport:
        report = self.reconciliation.reconcile(
            account_id=self.config.deployment.account_id(),
            broker_orders=orders,
            broker_trades=trades,
            repository=self.repository,
        )
        if report.has_unresolved:
            self.repository.pause_deployment(
                str(deployment["deployment_id"]), "RECONCILIATION_UNRESOLVED"
            )
            raise RuntimeError("broker reconciliation is unresolved")
        return report

    def _account_facts(
        self, *, require_trades: bool
    ) -> tuple[
        BrokerAssetSnapshot,
        tuple[BrokerPositionSnapshot, ...],
        tuple[BrokerOrderSnapshot, ...],
        tuple[BrokerTradeSnapshot, ...],
    ]:
        asset = self._single(self.broker.query_asset(), "asset")
        positions = self._records(self.broker.query_positions(), "positions")
        orders = self._records(self.broker.query_orders(), "orders")
        trades = self._records(self.broker.query_trades(), "trades") if require_trades else ()
        self._validate_asset_positions(asset, positions)
        return asset, positions, orders, trades

    def _validate_asset_positions(
        self,
        asset: BrokerAssetSnapshot,
        positions: Sequence[BrokerPositionSnapshot],
    ) -> None:
        account_id = self.config.deployment.account_id()
        if asset.account_id not in {None, account_id}:
            raise RuntimeError("broker asset belongs to a different account")
        if any(position.account_id not in {None, account_id} for position in positions):
            raise RuntimeError("broker positions include a different account")

    def _require_current_trade_date(self, trade_date: date) -> None:
        current = self.clock().astimezone(MARKET_TIMEZONE).date()
        if trade_date < current:
            raise JobSkipped("STALE_TRADE_DATE")
        if trade_date > current:
            raise JobSkipped("FUTURE_TRADE_DATE")

    def _require_time_at_or_after(self, run_time: time, reason: str) -> None:
        current = self.clock().astimezone(MARKET_TIMEZONE).time().replace(tzinfo=None)
        if current < run_time:
            raise JobSkipped(reason)

    @staticmethod
    def _records(result: QueryResult[ResultT], label: str) -> tuple[ResultT, ...]:
        if not result.success:
            raise RuntimeError(f"broker {label} query failed: {result.error}")
        return result.records

    @classmethod
    def _single(cls, result: QueryResult[ResultT], label: str) -> ResultT:
        records = cls._records(result, label)
        if len(records) != 1:
            raise RuntimeError(f"broker {label} query must return exactly one record")
        return records[0]

    def _decision_id(self, signal_date: date) -> str:
        import hashlib

        value = f"{self.config.deployment.deployment_id}|{signal_date.isoformat()}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _BorrowedConnection:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def __enter__(self) -> Connection:
        return self.connection

    def __exit__(self, *args: object) -> None:
        return None


__all__ = [
    "DeploymentSpec",
    "JobAlreadySucceeded",
    "JobSkipped",
    "LiveDailyJobs",
    "ModelDeploymentSpecFactory",
    "ModelSignalService",
    "RuleDeploymentSpecFactory",
    "RuleSignalService",
]
