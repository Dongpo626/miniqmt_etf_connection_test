"""Minimal shared assembly for one Rule strategy runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Connection, Engine

from etf_backtest.application.strategy_source import ModelStrategySource, RuleStrategySource
from etf_backtest.config.schema import (
    BacktestConfig,
    DatabaseConfig,
    RuleStrategyConfig,
    normalize_symbol,
)
from etf_backtest.core.effective_rules import (
    EffectiveDatedEtfRuleResolver,
    load_effective_rule_resolver,
)
from etf_backtest.data.mysql import QmtDailyDataset, QmtDailyRepository
from etf_backtest.data.portal import DailyDataPortal
from etf_backtest.universe.resolver import FrozenUniverse, FrozenUniverseResolver


@dataclass(frozen=True, slots=True)
class RuleRuntime:
    """Objects shared by Rule backtest and later daily strategy evaluation."""

    source: RuleStrategySource
    repository: QmtDailyRepository
    universe: FrozenUniverse
    dataset: QmtDailyDataset
    portal: DailyDataPortal
    rule_resolver: EffectiveDatedEtfRuleResolver
    universe_json: str
    universe_sha256: str


@dataclass(frozen=True, slots=True)
class RuleSignalRuntime:
    """Frozen-symbol data portal used by one live signal transaction."""

    source: RuleStrategySource
    repository: QmtDailyRepository
    dataset: QmtDailyDataset
    portal: DailyDataPortal


@dataclass(frozen=True, slots=True)
class ModelSignalRuntime:
    """Frozen-symbol daily data used by inference in one caller transaction."""

    source: ModelStrategySource
    repository: QmtDailyRepository
    dataset: QmtDailyDataset
    portal: DailyDataPortal


def create_database_engine(database: DatabaseConfig) -> Engine:
    """Create the project's existing read-only MySQL SQLAlchemy engine."""

    url = URL.create(
        "mysql+pymysql",
        username=database.user,
        password=database.resolved_password(),
        host=database.host,
        port=database.port,
        database=database.database,
        query={"charset": database.charset},
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": database.connect_timeout_seconds},
    )


def create_repository(
    config: BacktestConfig,
    engine: Engine,
    *,
    connection: Connection | None = None,
) -> QmtDailyRepository:
    """Create the existing repository, optionally bound to a caller-owned connection."""

    return QmtDailyRepository(
        engine,
        connection=connection,
        dataset_version=config.data_snapshot.dataset_version,
        trade_status_table=config.data_snapshot.trade_status_table,
        share_table=config.data_snapshot.share_table,
        index_table=config.data_snapshot.index_table,
        index_codes=config.data_snapshot.rule_index_codes,
        huijin_holders_csv=config.data_snapshot.huijin_holders_csv,
        huijin_holders_csv_sha256=config.data_snapshot.huijin_holders_csv_sha256,
    )


def resolve_universe(config: BacktestConfig, repository: QmtDailyRepository) -> FrozenUniverse:
    """Resolve the existing explicit-plus-pool universe once for this runtime."""

    return FrozenUniverseResolver(repository).resolve(
        explicit_symbols=config.universe.symbols,
        pools=config.universe.pools,
        start_date=config.start_date,
        end_date=config.end_date,
    )


def canonical_universe_identity(symbols: Sequence[str]) -> tuple[tuple[str, ...], str, str]:
    """Return sorted symbols, canonical JSON, and its UTF-8 SHA-256."""

    ordered = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return ordered, payload, digest


def required_project_resource(path: Path, project_root: Path, label: str) -> Path:
    """Resolve one configured resource while keeping it inside the project."""

    root = Path(project_root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FileNotFoundError(f"{label} is missing: {target}")
    return target


def build_rule_runtime(
    *,
    source: RuleStrategySource,
    config: BacktestConfig,
    project_root: Path,
    engine: Engine,
    load_start: date,
    load_end: date,
    connection: Connection | None = None,
) -> RuleRuntime:
    """Assemble Repository, frozen Universe, Portal, and effective Rule resources."""

    if not isinstance(config.strategy, RuleStrategyConfig):
        raise TypeError("Rule runtime requires RuleStrategyConfig")
    repository = create_repository(config, engine, connection=connection)
    universe = resolve_universe(config, repository)
    symbols, universe_json, universe_sha256 = canonical_universe_identity(universe.symbols)
    dataset = repository.load_daily_dataset(
        symbols,
        load_start,
        load_end,
        etf_infos=universe.etf_infos,
    )
    portal = DailyDataPortal(dataset)
    rule_resolver = load_effective_rule_resolver(
        universe.etf_infos,
        required_project_resource(config.limit_rules_csv, project_root, "limit rule CSV"),
        required_project_resource(
            config.limit_rules_manifest,
            project_root,
            "limit rule manifest",
        ),
    )
    return RuleRuntime(
        source=source,
        repository=repository,
        universe=universe,
        dataset=dataset,
        portal=portal,
        rule_resolver=rule_resolver,
        universe_json=universe_json,
        universe_sha256=universe_sha256,
    )


def build_rule_signal_runtime(
    *,
    source: RuleStrategySource,
    config: BacktestConfig,
    engine: Engine,
    frozen_symbols: Sequence[str],
    load_start: date,
    load_end: date,
    connection: Connection,
) -> RuleSignalRuntime:
    """Load one signal slice without resolving or changing the frozen Universe."""

    if not isinstance(config.strategy, RuleStrategyConfig):
        raise TypeError("Rule signal runtime requires RuleStrategyConfig")
    symbols, _, _ = canonical_universe_identity(frozen_symbols)
    repository = create_repository(config, engine, connection=connection)
    dataset = repository.load_daily_dataset(symbols, load_start, load_end)
    return RuleSignalRuntime(
        source=source,
        repository=repository,
        dataset=dataset,
        portal=DailyDataPortal(dataset),
    )


def build_model_signal_runtime(
    *,
    source: ModelStrategySource,
    config: BacktestConfig,
    engine: Engine,
    frozen_symbols: Sequence[str],
    load_start: date,
    load_end: date,
    connection: Connection,
) -> ModelSignalRuntime:
    """Load inference history without Universe resolution, fitting, or state writes."""

    symbols, _, _ = canonical_universe_identity(frozen_symbols)
    repository = create_repository(config, engine, connection=connection)
    dataset = repository.load_daily_dataset(symbols, load_start, load_end)
    return ModelSignalRuntime(
        source=source,
        repository=repository,
        dataset=dataset,
        portal=DailyDataPortal(dataset),
    )


__all__ = [
    "ModelSignalRuntime",
    "RuleRuntime",
    "RuleSignalRuntime",
    "build_model_signal_runtime",
    "build_rule_runtime",
    "build_rule_signal_runtime",
    "canonical_universe_identity",
    "create_database_engine",
    "create_repository",
    "required_project_resource",
    "resolve_universe",
]
