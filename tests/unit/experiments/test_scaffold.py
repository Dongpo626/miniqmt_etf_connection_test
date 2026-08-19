from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

from etf_backtest.experiments.config import load_user_experiment_config
from etf_backtest.experiments.scaffold import scaffold_experiment
from etf_backtest.strategy.loader import load_user_rule
from etf_backtest.strategy.model import (
    FeatureBuilder,
    ModelSettings,
    TopKPortfolio,
    TorchModelFactory,
)


@pytest.mark.unit
def test_scaffold_creates_one_complete_private_experiment(tmp_path: Path) -> None:
    private_root = tmp_path / "private_strategy"

    experiment_path, rule_path, model_path = scaffold_experiment(
        "demo_策略",
        private_strategy_root=private_root,
    )

    directory = private_root / "demo_策略"
    assert (experiment_path, rule_path, model_path) == (
        directory / "experiment.yaml",
        directory / "rule.py",
        directory / "model.py",
    )
    assert load_user_experiment_config(experiment_path).case == "rule"
    experiment_source = experiment_path.read_text(encoding="utf-8")
    assert "database" not in experiment_source
    assert "case: rule" in experiment_source
    assert "pools: [domestic_stock_etf, gold_etf]" in experiment_source
    for removed in ("cases:", "rule:", "model:", "target_weight", "epochs"):
        assert removed not in experiment_source

    rule_source = rule_path.read_text(encoding="utf-8")
    model_source = model_path.read_text(encoding="utf-8")
    ast.parse(rule_source)
    ast.parse(model_source)
    assert "RuleSettings(" in rule_source
    assert "class Strategy(UserRule)" in rule_source
    assert "def generate_weights(" in rule_source
    assert "class Features(FeatureBuilder)" in model_source
    assert "class Model(TorchModelFactory)" in model_source
    assert "MODEL_SETTINGS = ModelSettings(" in model_source
    assert "portfolio=TopKPortfolio(" in model_source
    assert "主要编辑 ``RuleSettings`` 和 ``generate_weights``" in rule_source
    assert "``current_weight(symbol)``" in rule_source
    assert "``index_bars(index_code)``" in rule_source
    assert "Optional Rule-only fund data" not in rule_source
    assert "Workflow" not in model_source

    loaded_rule = load_user_rule(rule_path, allowed_root=directory)
    assert type(loaded_rule).__name__ == "Strategy"
    model_namespace = runpy.run_path(str(model_path))
    settings = model_namespace["MODEL_SETTINGS"]
    assert isinstance(settings, ModelSettings)
    assert isinstance(settings.portfolio, TopKPortfolio)
    assert isinstance(model_namespace["Features"](**settings.feature_kwargs), FeatureBuilder)
    assert isinstance(model_namespace["Model"](**settings.model_kwargs), TorchModelFactory)


@pytest.mark.unit
def test_scaffold_preflights_all_targets_and_refuses_overwrite(tmp_path: Path) -> None:
    private_root = tmp_path / "private_strategy"
    targets = scaffold_experiment("demo", private_strategy_root=private_root)
    before = {target: target.read_bytes() for target in targets}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scaffold_experiment("demo", private_strategy_root=private_root)

    assert {target: target.read_bytes() for target in targets} == before


@pytest.mark.unit
@pytest.mark.parametrize("name", ["../escape", "nested/name", "nested\\name", "CON"])
def test_scaffold_rejects_unsafe_experiment_paths(tmp_path: Path, name: str) -> None:
    private_root = tmp_path / "private_strategy"
    with pytest.raises(ValueError):
        scaffold_experiment(name, private_strategy_root=private_root)
    assert not private_root.exists()
