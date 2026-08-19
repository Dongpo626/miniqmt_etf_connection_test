"""Prepare, validate and run one trusted local Rule or Model experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from etf_backtest.application.runtime_factory import (
    build_rule_runtime,
    create_database_engine,
    create_repository,
    required_project_resource,
    resolve_universe,
)
from etf_backtest.application.strategy_source import (
    RuleStrategySource,
    load_rule_strategy_source,
    sha256_file,
)
from etf_backtest.config.schema import (
    BacktestConfig,
    DatabaseConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
    etf_code,
)
from etf_backtest.core.account import Account
from etf_backtest.core.effective_rules import (
    EffectiveDatedEtfRuleResolver,
    load_effective_rule_resolver,
)
from etf_backtest.core.engine import BacktestEngine, BacktestResult
from etf_backtest.core.etf_rules import EtfRuleEngine
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.fill import FillModel
from etf_backtest.core.market import MarketBarView
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.position import Position
from etf_backtest.core.slippage import SlippageModel
from etf_backtest.data.mysql import QmtDailyDataset, QmtDailyRepository
from etf_backtest.data.portal import DailyDataPortal
from etf_backtest.evaluation.backtest_metrics import (
    BacktestMetricResult,
    BacktestMetrics,
    DailyMetricRow,
    TradeMetricRow,
)
from etf_backtest.evaluation.backtest_plots import render_backtest_plots
from etf_backtest.experiments.config import (
    SystemSettings,
    UserExperimentConfig,
    load_system_settings,
    load_user_experiment_config,
)
from etf_backtest.output.writer import BacktestOutputWriter, ModelArtifacts
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.model import LoadedModelComponents, load_user_model_components
from etf_backtest.strategy.model_contracts import DateRange, ModelDataIdentity, prediction_rows
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_training import (
    DailyTorchDatasetBuilder,
    DailyTorchWorkflow,
    DailyTorchWorkflowResult,
    require_torch,
)
from etf_backtest.strategy.rule import UserRule
from etf_backtest.universe.resolver import FrozenUniverse

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SYSTEM_CONFIG = _PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"


@dataclass(frozen=True, slots=True)
class PreparedExperiment:
    """The one configured strategy plus shared system settings."""

    source_path: Path
    experiment: UserExperimentConfig
    system: SystemSettings
    rule: UserRule | None
    rule_source_sha256: str | None
    model: LoadedModelComponents | None
    rule_source: RuleStrategySource | None = None


@dataclass(slots=True)
class _ModelRuntime:
    strategy: DailyModelStrategy
    workflow: DailyTorchWorkflow
    result: DailyTorchWorkflowResult
    sample_counts: tuple[int, int, int]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _database_engine(database: DatabaseConfig) -> Engine:
    return create_database_engine(database)


def _required_resource(path: Path, root: Path, label: str) -> Path:
    return required_project_resource(path, root, label)


def prepare_experiment(
    experiment_path: Path,
    *,
    system_path: Path = _DEFAULT_SYSTEM_CONFIG,
    project_root: Path = _PROJECT_ROOT,
    require_password: bool = True,
) -> PreparedExperiment:
    """Load configuration and exactly one fixed-name trusted strategy file."""

    root = Path(project_root).resolve()
    source = Path(experiment_path).resolve(strict=True)
    if not source.is_file() or source.suffix.casefold() not in {".yaml", ".yml"}:
        raise ValueError("experiment must be one existing YAML file")
    experiment = load_user_experiment_config(source)
    rule_source: RuleStrategySource | None = None
    if experiment.case == "rule":
        rule_source = load_rule_strategy_source(source, system_path=system_path)
        experiment = rule_source.experiment
        system = rule_source.system
    else:
        system = load_system_settings(Path(system_path).resolve(strict=True))
    _required_resource(system.limit_rules_csv, root, "limit rule CSV")
    _required_resource(system.limit_rules_manifest, root, "limit rule manifest")
    if require_password:
        system.database.resolved_password()

    rule: UserRule | None = None
    rule_hash: str | None = None
    model: LoadedModelComponents | None = None
    if experiment.case == "rule":
        assert rule_source is not None
        rule = rule_source.rule
        rule_hash = rule_source.strategy_source_sha256
    else:
        require_torch()
        model = load_user_model_components(
            source.parent / "model.py",
            allowed_root=source.parent,
        )
    prepared = PreparedExperiment(
        source_path=source,
        experiment=experiment,
        system=system,
        rule=rule,
        rule_source_sha256=rule_hash,
        model=model,
        rule_source=rule_source,
    )
    _strategy_config(prepared)
    return prepared


def _strategy_config(prepared: PreparedExperiment) -> RuleStrategyConfig | ModelStrategyConfig:
    if prepared.experiment.case == "rule":
        assert prepared.rule is not None
        return RuleStrategyConfig(
            lookback_trading_days=prepared.rule.lookback_trading_days,
            rebalance_every_trading_days=prepared.rule.rebalance_every_trading_days,
            target_weight=prepared.rule.target_weight,
        )
    assert prepared.model is not None
    settings = prepared.model.settings
    if settings.valid_range.end_date >= prepared.experiment.start_date:
        raise ValueError("model validation range must end before the backtest starts")
    return ModelStrategyConfig(
        max_total_weight=settings.portfolio.max_total_weight,
        train_start=settings.train_range.start_date,
        train_end=settings.train_range.end_date,
        valid_start=settings.valid_range.start_date,
        valid_end=settings.valid_range.end_date,
        test_start=prepared.experiment.start_date,
        test_end=prepared.experiment.end_date,
    )


def _load_start(prepared: PreparedExperiment) -> date:
    if prepared.rule is not None:
        return prepared.experiment.start_date - timedelta(
            days=max(90, prepared.rule.lookback_trading_days * 3)
        )
    assert prepared.model is not None
    lookback = prepared.model.feature_builder.required_history_trading_days
    return prepared.model.settings.train_range.start_date - timedelta(days=max(90, lookback * 3))


def _table_checks(prepared: PreparedExperiment) -> tuple[tuple[str, str], ...]:
    snapshot = prepared.system.data_snapshot
    checks = [
        ("database", "SELECT 1"),
        (
            "dim_trading_calendar",
            "SELECT cal_date FROM dim_trading_calendar WHERE exchange = 'SSE' LIMIT 1",
        ),
        (snapshot.raw_table, f"SELECT trade_date FROM `{snapshot.raw_table}` LIMIT 1"),
        (snapshot.front_table, f"SELECT trade_date FROM `{snapshot.front_table}` LIMIT 1"),
        ("dim_etf", "SELECT etf_code FROM dim_etf LIMIT 1"),
        (
            snapshot.trade_status_table,
            f"SELECT trade_date FROM `{snapshot.trade_status_table}` LIMIT 1",
        ),
    ]
    if snapshot.share_table is not None:
        checks.append(
            (snapshot.share_table, f"SELECT asof_date FROM `{snapshot.share_table}` LIMIT 1")
        )
    if snapshot.index_table is not None:
        checks.append(
            (snapshot.index_table, f"SELECT trade_date FROM `{snapshot.index_table}` LIMIT 1")
        )
    return tuple(checks)


def validate_experiment(
    experiment_path: Path,
    *,
    system_path: Path = _DEFAULT_SYSTEM_CONFIG,
    project_root: Path = _PROJECT_ROOT,
) -> dict[str, object]:
    """Check local inputs, MySQL access, required tables and explicit symbols."""

    prepared = prepare_experiment(
        experiment_path,
        system_path=system_path,
        project_root=project_root,
        require_password=True,
    )
    sql_engine = _database_engine(prepared.system.database)
    checked: list[str] = []
    try:
        with sql_engine.connect() as connection:
            for label, statement in _table_checks(prepared):
                if connection.execute(text(statement)).first() is None:
                    raise ValueError(f"required MySQL source contains no readable rows: {label}")
                checked.append(label)
            for symbol in prepared.experiment.universe.symbols:
                row = connection.execute(
                    text("SELECT etf_code FROM dim_etf WHERE etf_code = :code LIMIT 1"),
                    {"code": etf_code(symbol)},
                ).first()
                if row is None:
                    raise ValueError(f"configured ETF is missing from dim_etf: {symbol}")
    finally:
        sql_engine.dispose()
    return {
        "status": "valid",
        "name": prepared.experiment.name,
        "case": prepared.experiment.case,
        "experiment_path": str(prepared.source_path),
        "checked_mysql_sources": tuple(checked),
        "checked_symbols": prepared.experiment.universe.symbols,
    }


def _repository(config: BacktestConfig, engine: Engine) -> QmtDailyRepository:
    return create_repository(config, engine)


def _resolve_universe(config: BacktestConfig, repository: QmtDailyRepository) -> FrozenUniverse:
    return resolve_universe(config, repository)


def _model_identity(config: BacktestConfig) -> ModelDataIdentity:
    return ModelDataIdentity(
        dataset_version=config.data_snapshot.dataset_version,
        manifest_sha256=config.data_snapshot.manifest_sha256,
        snapshot_started_at_utc=config.data_snapshot.snapshot_started_at_utc,
    )


def _build_model(
    config: BacktestConfig,
    components: LoadedModelComponents,
    market_views: Sequence[MarketBarView],
) -> _ModelRuntime:
    assert isinstance(config.strategy, ModelStrategyConfig)
    ranges = (
        DateRange(config.strategy.train_start, config.strategy.train_end),
        DateRange(config.strategy.valid_start, config.strategy.valid_end),
        DateRange(config.strategy.test_start, config.strategy.test_end),
    )
    dataset = DailyTorchDatasetBuilder(components.feature_builder).build(
        market_views=market_views,
        train_range=ranges[0],
        valid_range=ranges[1],
        test_range=ranges[2],
    )
    workflow = DailyTorchWorkflow(
        feature_builder=components.feature_builder,
        model_factory=components.model_factory,
        data_identity=_model_identity(config),
        training_config=components.settings.training,
        portfolio=components.settings.portfolio,
    )
    result = workflow.fit(dataset)
    metadata = result.bundle.metadata
    if (metadata.train_range, metadata.valid_range, metadata.test_range) != ranges:
        raise RuntimeError("fitted model split metadata does not match configured ranges")
    if metadata.data_identity != _model_identity(config):
        raise RuntimeError("fitted model data identity does not match the backtest snapshot")
    if metadata.trained_through >= config.start_date:
        raise ValueError("fitted model must be trained before the first backtest signal")
    return _ModelRuntime(
        strategy=DailyModelStrategy(
            feature_builder=components.feature_builder,
            bundle=result.bundle,
            portfolio=components.settings.portfolio,
        ),
        workflow=workflow,
        result=result,
        sample_counts=(len(dataset.train), len(dataset.valid), len(dataset.test)),
    )


def _metrics(initial_cash: Decimal, result: BacktestResult) -> BacktestMetricResult:
    return BacktestMetrics.calculate(
        initial_cash=initial_cash,
        daily_rows=tuple(
            DailyMetricRow(
                trade_date=row.trade_date,
                cash=row.cash,
                market_value=row.market_value,
                total_asset=row.total_asset,
            )
            for row in result.daily_snapshots
        ),
        trade_rows=tuple(
            TradeMetricRow(
                trade_amount=fill.trade_amount,
                fee=fill.fee,
                base_trade_price=fill.base_trade_price,
                fill_price=fill.fill_price,
                fill_quantity=fill.fill_quantity,
            )
            for fill in result.fills
        ),
    )


def _provenance(
    config: BacktestConfig,
    dataset: QmtDailyDataset,
    portal: DailyDataPortal,
    universe: FrozenUniverse,
    rule_resolver: EffectiveDatedEtfRuleResolver,
) -> Mapping[str, object]:
    resource_identity = rule_resolver.resource_identity
    if resource_identity is None:
        raise RuntimeError("effective rule resource identity is missing")
    return {
        "data_mode": config.data_snapshot.data_mode,
        "pit_compliant": False,
        "dataset_version": portal.dataset_version,
        "snapshot_started_at_utc": config.data_snapshot.snapshot_started_at_utc,
        "calendar_source": portal.trading_calendar.calendar_source,
        "calendar_version": portal.trading_calendar.calendar_version,
        "calendar_policy": portal.trading_calendar.calendar_policy,
        "input_manifest_sha256": config.data_snapshot.manifest_sha256,
        "rule_resource": asdict(resource_identity),
        "universe": universe.csv_rows(),
        "universe_approximation_flags": universe.approximation_flags,
        "loaded_share_record_count": len(dataset.share_records),
        "loaded_huijin_ratio_record_count": len(dataset.huijin_ratio_records),
        "loaded_index_record_count": len(dataset.index_records),
        "loaded_explicit_price_limit_count": dataset.explicit_price_limit_count,
        "loaded_derived_price_limit_count": dataset.derived_price_limit_fallback_count,
        "status_only_suspension_carry_keys": dataset.suspension_carry_keys,
    }


def _model_metadata(
    runtime: _ModelRuntime, components: LoadedModelComponents
) -> Mapping[str, object]:
    return {
        "bundle": runtime.result.bundle.metadata.to_dict(),
        "source_path": components.source_path,
        "source_sha256": components.source_sha256,
        "settings": components.settings.resolved_dict(),
        "train_sample_count": runtime.sample_counts[0],
        "validation_sample_count": runtime.sample_counts[1],
        "test_sample_count": runtime.sample_counts[2],
        "best_epoch": runtime.result.best_epoch,
        "epochs_trained": runtime.result.epochs_trained,
        "best_validation_loss": runtime.result.best_validation_loss,
        "validation_metrics": asdict(runtime.result.validation_metrics),
        "test_metrics": asdict(runtime.result.test_metrics),
        "scaler_fit_scope": "TRAIN_ONLY",
    }


def _run_id(case: str) -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{case}-{uuid4().hex[:8]}"


def run_experiment(
    experiment_path: Path,
    *,
    system_path: Path = _DEFAULT_SYSTEM_CONFIG,
    project_root: Path = _PROJECT_ROOT,
) -> dict[str, object]:
    """Run exactly one strategy and atomically publish its fixed result set."""

    root = Path(project_root).resolve()
    prepared = prepare_experiment(
        experiment_path,
        system_path=system_path,
        project_root=root,
        require_password=True,
    )
    config = prepared.experiment.build_case(
        prepared.system,
        strategy=_strategy_config(prepared),
    )
    runs_dir = (root / config.runs_dir).resolve()
    if not runs_dir.is_relative_to(root):
        raise ValueError("runs_dir must stay inside the project")
    run_id = _run_id(prepared.experiment.case)
    writer = BacktestOutputWriter(runs_dir)
    metadata: dict[str, object] = {
        "name": prepared.experiment.name,
        "case": prepared.experiment.case,
        "experiment_path": prepared.source_path,
        "experiment_sha256": _sha256(prepared.source_path),
        "system_path": Path(system_path).resolve(strict=True),
        "resolved_config": config.resolved_dict(),
        "rule_source_sha256": prepared.rule_source_sha256,
    }
    sql_engine: Engine | None = None
    try:
        sql_engine = _database_engine(config.database)
        model_runtime: _ModelRuntime | None = None
        strategy: BaseStrategy
        if prepared.rule_source is not None:
            rule_runtime = build_rule_runtime(
                source=prepared.rule_source,
                config=config,
                project_root=root,
                engine=sql_engine,
                load_start=_load_start(prepared),
                load_end=config.end_date,
            )
            repository = rule_runtime.repository
            universe = rule_runtime.universe
            dataset = rule_runtime.dataset
            portal = rule_runtime.portal
            rule_resolver = rule_runtime.rule_resolver
            strategy = rule_runtime.source.strategy
        else:
            repository = _repository(config, sql_engine)
            universe = _resolve_universe(config, repository)
            dataset = repository.load_daily_dataset(
                universe.symbols,
                _load_start(prepared),
                config.end_date,
                etf_infos=universe.etf_infos,
            )
            portal = DailyDataPortal(dataset)
            rule_resolver = load_effective_rule_resolver(
                universe.etf_infos,
                _required_resource(config.limit_rules_csv, root, "limit rule CSV"),
                _required_resource(config.limit_rules_manifest, root, "limit rule manifest"),
            )
            assert prepared.model is not None
            model_runtime = _build_model(
                config,
                prepared.model,
                dataset.front_market_bar_views(),
            )
            strategy = model_runtime.strategy
        account = Account(
            cash=config.initial_cash,
            positions={
                info.symbol: Position(
                    symbol=info.symbol,
                    turnover_rule=rule_resolver.resolve(
                        info.symbol,
                        max(config.start_date, info.list_date),
                    ).turnover_rule,
                )
                for info in universe.etf_infos
            },
        )
        fill_model = FillModel(
            fee_model=FeeModel(config.fee),
            slippage_model=SlippageModel(config.slippage),
        )
        result = BacktestEngine(
            portal=portal,
            account=account,
            strategy=strategy,
            rule_resolver=rule_resolver,
            rule_engine=EtfRuleEngine(
                fill_model=fill_model,
                volume_participation_rate=config.volume_participation_rate,
            ),
            order_generator=OrderGenerator(),
            fill_model=fill_model,
        ).run(start_date=config.start_date, end_date=config.end_date)
        expected_dates = portal.trading_calendar.trading_dates(config.start_date, config.end_date)
        if tuple(row.trade_date for row in result.daily_snapshots) != expected_dates:
            raise RuntimeError("daily NAV does not exactly cover configured SSE dates")
        calculated_metrics = _metrics(config.initial_cash, result)
        daily_rows = tuple(
            DailyMetricRow(row.trade_date, row.cash, row.market_value, row.total_asset)
            for row in result.daily_snapshots
        )
        metadata["provenance"] = _provenance(
            config,
            dataset,
            portal,
            universe,
            rule_resolver,
        )
        metadata["rule_settings"] = (
            prepared.rule.settings.resolved_dict() if prepared.rule is not None else None
        )
        plots = render_backtest_plots(initial_cash=config.initial_cash, daily_rows=daily_rows)
        if model_runtime is None:
            run_dir = writer.write_success(
                run_id=run_id,
                run_metadata=metadata,
                result=result,
                metrics=calculated_metrics,
                plots=plots,
            )
        else:
            assert prepared.model is not None
            metadata["model"] = _model_metadata(model_runtime, prepared.model)
            with TemporaryDirectory(prefix="qmt-model-bundle-") as temp_directory:
                bundle_path = model_runtime.workflow.save(
                    Path(temp_directory) / "model_bundle.pt",
                    source_run_dir=runs_dir / run_id,
                )
                run_dir = writer.write_success(
                    run_id=run_id,
                    run_metadata=metadata,
                    result=result,
                    metrics=calculated_metrics,
                    plots=plots,
                    model_artifacts=ModelArtifacts(
                        bundle_path=bundle_path,
                        predictions=prediction_rows(model_runtime.strategy.predictions),
                    ),
                )
        return {
            "status": "success",
            "experiment": prepared.experiment.name,
            "case": prepared.experiment.case,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "daily_count": len(result.daily_snapshots),
            "order_count": len(result.orders),
            "trade_count": len(result.fills),
        }
    except Exception as exc:
        writer.write_failure(run_id=run_id, run_metadata=metadata, error=exc)
        raise
    finally:
        if sql_engine is not None:
            sql_engine.dispose()


__all__ = [
    "PreparedExperiment",
    "prepare_experiment",
    "run_experiment",
    "validate_experiment",
]
