from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from etf_backtest.config.schema import (
    DatabaseConfig,
    RuleStrategyConfig,
)
from etf_backtest.experiments.config import SystemSettings, UserExperimentConfig


def _payload() -> dict[str, object]:
    return {
        "name": "code_owned_settings",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_cash": "1000000",
        "universe": {
            "symbols": ["SH.510300"],
            "pools": ["gold_etf"],
        },
        "case": "rule",
    }


@pytest.mark.unit
def test_experiment_yaml_contains_only_common_inputs_and_case_name() -> None:
    experiment = UserExperimentConfig.model_validate(_payload())

    assert experiment.case == "rule"
    assert set(experiment.model_dump()) == {
        "name",
        "start_date",
        "end_date",
        "initial_cash",
        "universe",
        "case",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "old_field",
    (
        "rule",
        "model",
        "fee",
        "slippage",
        "volume_participation_rate",
        "runs_dir",
    ),
)
def test_experiment_yaml_rejects_removed_parameter_blocks(old_field: str) -> None:
    payload = _payload()
    payload[old_field] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        UserExperimentConfig.model_validate(payload)


@pytest.mark.unit
def test_case_is_required_and_limited_to_supported_kinds() -> None:
    for case in (None, "other", ["rule"]):
        payload = _payload()
        payload["case"] = case
        with pytest.raises(ValidationError):
            UserExperimentConfig.model_validate(payload)


@pytest.mark.unit
def test_system_settings_own_shared_execution_defaults_and_build_case() -> None:
    system = SystemSettings(
        database=DatabaseConfig(user="reader", password=SecretStr("test-only")),
        runs_dir=Path("qmt_example/results/user_experiments"),
    )
    experiment = UserExperimentConfig.model_validate(_payload())
    strategy = RuleStrategyConfig(
        lookback_trading_days=37,
        rebalance_every_trading_days=7,
        target_weight="0.83",
    )

    core = experiment.build_case(system, strategy=strategy)

    assert core.strategy is strategy
    assert core.fee is system.fee
    assert core.slippage is system.slippage
    assert core.volume_participation_rate == Decimal("0.20")
    assert core.runs_dir == Path("qmt_example/results/user_experiments")
