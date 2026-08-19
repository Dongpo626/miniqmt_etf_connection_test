"""Load one trusted local Rule experiment and its existing strategy adapter."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from etf_backtest.experiments.config import (
    SystemSettings,
    UserExperimentConfig,
    load_system_settings,
    load_user_experiment_config,
)
from etf_backtest.strategy.loader import load_user_rule
from etf_backtest.strategy.model import LoadedModelComponents, load_user_model_components
from etf_backtest.strategy.rule import SimpleRuleStrategy, UserRule


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one existing file without normalizing its bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RuleStrategySource:
    """Resolved Rule configuration, source identity, and strategy objects."""

    experiment_path: Path
    system_path: Path
    experiment: UserExperimentConfig
    system: SystemSettings
    rule: UserRule
    strategy: SimpleRuleStrategy
    strategy_source_sha256: str


@dataclass(frozen=True, slots=True)
class ModelStrategySource:
    """Resolved Model source components without training or loading market data."""

    experiment_path: Path
    system_path: Path
    experiment: UserExperimentConfig
    system: SystemSettings
    components: LoadedModelComponents
    strategy_source_sha256: str


def load_rule_strategy_source(
    experiment_path: Path,
    *,
    system_path: Path,
) -> RuleStrategySource:
    """Load the existing Rule extension contract from one experiment directory."""

    experiment_source = Path(experiment_path).resolve(strict=True)
    if not experiment_source.is_file() or experiment_source.suffix.casefold() not in {
        ".yaml",
        ".yml",
    }:
        raise ValueError("experiment must be one existing YAML file")
    system_source = Path(system_path).resolve(strict=True)
    experiment = load_user_experiment_config(experiment_source)
    if experiment.case != "rule":
        raise ValueError("Rule strategy source requires experiment case: rule")
    system = load_system_settings(system_source)
    rule_path = (experiment_source.parent / "rule.py").resolve(strict=True)
    rule = load_user_rule(rule_path, allowed_root=experiment_source.parent)
    return RuleStrategySource(
        experiment_path=experiment_source,
        system_path=system_source,
        experiment=experiment,
        system=system,
        rule=rule,
        strategy=SimpleRuleStrategy(rule=rule),
        strategy_source_sha256=sha256_file(rule_path),
    )


def load_model_strategy_source(
    experiment_path: Path,
    *,
    system_path: Path,
) -> ModelStrategySource:
    experiment_source = Path(experiment_path).resolve(strict=True)
    if not experiment_source.is_file() or experiment_source.suffix.casefold() not in {
        ".yaml",
        ".yml",
    }:
        raise ValueError("experiment must be one existing YAML file")
    experiment = load_user_experiment_config(experiment_source)
    if experiment.case != "model":
        raise ValueError("Model strategy source requires experiment case: model")
    system_source = Path(system_path).resolve(strict=True)
    components = load_user_model_components(
        experiment_source.parent / "model.py",
        allowed_root=experiment_source.parent,
    )
    return ModelStrategySource(
        experiment_path=experiment_source,
        system_path=system_source,
        experiment=experiment,
        system=load_system_settings(system_source),
        components=components,
        strategy_source_sha256=components.source_sha256,
    )


__all__ = [
    "ModelStrategySource",
    "RuleStrategySource",
    "load_model_strategy_source",
    "load_rule_strategy_source",
    "sha256_file",
]
