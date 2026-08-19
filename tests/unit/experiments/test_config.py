from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from etf_backtest.config.schema import (
    BacktestConfig,
    DatabaseConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
)
from etf_backtest.experiments.config import (
    SystemSettings,
    UserExperimentConfig,
    load_system_settings,
    load_user_experiment_config,
)


def _system() -> SystemSettings:
    return SystemSettings(database=DatabaseConfig(user="reader", password="test-only"))


def _experiment(*, case: str = "rule") -> UserExperimentConfig:
    return UserExperimentConfig.model_validate(
        {
            "name": "momentum_demo",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_cash": "1000000",
            "universe": {"symbols": ["510300", "518880", "588000"]},
            "case": case,
        }
    )


@pytest.mark.unit
def test_user_config_contains_only_common_inputs_and_builds_resolved_case() -> None:
    experiment = _experiment()
    system = _system()
    dumped = experiment.model_dump(mode="json")

    assert set(dumped) == {
        "name",
        "start_date",
        "end_date",
        "initial_cash",
        "universe",
        "case",
    }
    assert experiment.case == "rule"

    rule = RuleStrategyConfig(
        lookback_trading_days=21,
        rebalance_every_trading_days=20,
        target_weight="0.90",
    )
    core = experiment.build_case(system, strategy=rule)
    assert isinstance(core, BacktestConfig)
    assert core.database is system.database
    assert core.strategy is rule
    assert core.runs_dir == Path("runs")

    model = ModelStrategyConfig(
        train_start=date(2021, 1, 1),
        train_end=date(2022, 12, 31),
        valid_start=date(2023, 1, 1),
        valid_end=date(2023, 12, 31),
        test_start=date(2024, 1, 1),
        test_end=date(2024, 12, 31),
        max_total_weight="0.85",
    )
    model_experiment = _experiment(case="model")
    assert model_experiment.build_case(system, strategy=model).strategy is model
    assert model.max_total_weight == Decimal("0.85")

    with pytest.raises(TypeError, match="decimal text"):
        ModelStrategyConfig(max_total_weight=0.85)


@pytest.mark.unit
def test_build_case_requires_matching_strategy_type() -> None:
    experiment = _experiment()
    system = _system()
    rule = RuleStrategyConfig()
    model = ModelStrategyConfig()

    with pytest.raises(TypeError, match="RuleStrategyConfig"):
        experiment.build_case(system, strategy=model)
    assert experiment.build_case(system, strategy=rule).strategy.kind == "rule"


@pytest.mark.unit
def test_load_user_yaml_accepts_only_common_document(tmp_path: Path) -> None:
    source = tmp_path / "experiment.yaml"
    source.write_text(
        """
name: yaml_demo
start_date: 2024-01-01
end_date: 2024-12-31
initial_cash: "1000000"
case: rule
universe:
  symbols: [SH.510300]
""".strip(),
        encoding="utf-8",
    )
    loaded = load_user_experiment_config(source)
    assert loaded.name == "yaml_demo"
    assert loaded.case == "rule"

    source.write_text(
        source.read_text(encoding="utf-8") + "\nrule:\n  lookback: 20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_user_experiment_config(source)


@pytest.mark.unit
def test_load_system_settings_reads_shared_execution_defaults(tmp_path: Path) -> None:
    source = tmp_path / "system.yaml"
    source.write_text(
        """
database:
  user: qmt_readonly
  password: test-only
fee:
  commission_rate: "0.0004"
  minimum_commission: "3"
  stamp_duty_rate: "0"
slippage:
  rate: "0.0006"
volume_participation_rate: "0.15"
runs_dir: qmt_example/results/user_experiments
""".strip(),
        encoding="utf-8",
    )
    settings = load_system_settings(source)
    assert settings.database.user == "qmt_readonly"
    assert settings.fee.commission_rate == Decimal("0.0004")
    assert settings.slippage.rate == Decimal("0.0006")
    assert settings.volume_participation_rate == Decimal("0.15")
    assert settings.runs_dir == Path("qmt_example/results/user_experiments")
