"""Rule source loading and stable source identity tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from etf_backtest.application.strategy_source import load_rule_strategy_source
from etf_backtest.experiment import prepare_experiment
from etf_backtest.strategy.rule import SimpleRuleStrategy, UserRule

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SYSTEM_PATH = _PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"
_RULE_SOURCE = """
from collections.abc import Mapping

from etf_backtest.strategy.rule import RuleMarketData, RuleSettings, UserRule, WeightInput


class Strategy(UserRule):
    settings = RuleSettings(
        lookback_trading_days=2,
        rebalance_every_trading_days=3,
        target_weight="0.50",
    )

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        del data
        return {}
""".lstrip()


def _write_experiment(root: Path, *, case: str = "rule") -> Path:
    experiment_path = root / "experiment.yaml"
    experiment_path.write_text(
        "\n".join(
            (
                "name: stage_two_rule",
                "start_date: 2024-01-01",
                "end_date: 2024-01-31",
                'initial_cash: "1000000"',
                f"case: {case}",
                "universe:",
                "  symbols: [SH.510300]",
                "  pools: []",
                "",
            )
        ),
        encoding="utf-8",
    )
    (root / "rule.py").write_text(_RULE_SOURCE, encoding="utf-8")
    return experiment_path


@pytest.mark.unit
def test_load_rule_source_reuses_existing_user_rule_contract(tmp_path: Path) -> None:
    experiment_path = _write_experiment(tmp_path)

    source = load_rule_strategy_source(experiment_path, system_path=_SYSTEM_PATH)

    assert source.experiment.case == "rule"
    assert source.rule.lookback_trading_days == 2
    assert isinstance(source.strategy, SimpleRuleStrategy)
    assert source.strategy.rule is source.rule
    assert tuple(inspect.signature(UserRule.generate_weights).parameters) == ("self", "data")
    prepared = prepare_experiment(
        experiment_path,
        system_path=_SYSTEM_PATH,
        project_root=_PROJECT_ROOT,
        require_password=False,
    )
    assert prepared.rule_source is not None
    assert prepared.rule_source.strategy_source_sha256 == source.strategy_source_sha256


@pytest.mark.unit
def test_non_rule_case_is_rejected_before_loading_a_rule(tmp_path: Path) -> None:
    experiment_path = _write_experiment(tmp_path, case="model")

    with pytest.raises(ValueError, match="requires experiment case: rule"):
        load_rule_strategy_source(experiment_path, system_path=_SYSTEM_PATH)


@pytest.mark.unit
def test_rule_source_hash_is_stable_and_changes_with_source_bytes(tmp_path: Path) -> None:
    experiment_path = _write_experiment(tmp_path)

    first = load_rule_strategy_source(experiment_path, system_path=_SYSTEM_PATH)
    second = load_rule_strategy_source(experiment_path, system_path=_SYSTEM_PATH)
    assert first.strategy_source_sha256 == second.strategy_source_sha256

    rule_path = tmp_path / "rule.py"
    rule_path.write_text(_RULE_SOURCE + "\n# changed\n", encoding="utf-8")
    changed = load_rule_strategy_source(experiment_path, system_path=_SYSTEM_PATH)

    assert changed.strategy_source_sha256 != first.strategy_source_sha256
