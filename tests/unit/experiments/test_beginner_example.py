from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from etf_backtest.experiment import _table_checks, prepare_experiment
from etf_backtest.experiments.config import (
    load_system_settings,
    load_user_experiment_config,
)
from etf_backtest.strategy.loader import load_user_rule
from etf_backtest.strategy.model import TopKPortfolio, load_user_model_components

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_ROOT = _PROJECT_ROOT / "private_strategy" / "beginner_example"


@pytest.mark.unit
def test_checked_in_beginner_example_uses_code_owned_settings() -> None:
    experiment_path = _EXAMPLE_ROOT / "experiment.yaml"
    experiment = load_user_experiment_config(experiment_path)
    source = experiment_path.read_text(encoding="utf-8")

    assert experiment.case == "rule"
    assert experiment.universe.symbols == ("SH.510300", "SH.518880", "SH.588000")
    for removed in ("\nrule:", "\nmodel:", "\nfee:", "target_weight", "epochs"):
        assert removed not in source

    rule = load_user_rule(_EXAMPLE_ROOT / "rule.py", allowed_root=_EXAMPLE_ROOT)
    assert rule.lookback_trading_days == 21
    assert rule.rebalance_every_trading_days == 20
    assert rule.parameters["momentum_period"] == 20

    model = load_user_model_components(
        _EXAMPLE_ROOT / "model.py",
        allowed_root=_EXAMPLE_ROOT,
    )
    assert model.settings.training.max_epochs == 15
    assert model.settings.model_kwargs == {"hidden_dim": 16}
    assert isinstance(model.settings.portfolio, TopKPortfolio)
    assert model.settings.portfolio.top_k == 3
    assert model.settings.portfolio.max_total_weight == Decimal("0.90")
    assert model.settings.portfolio.weighting == "score_proportional"


@pytest.mark.unit
def test_beginner_rule_explains_current_editing_and_data_interfaces_in_chinese() -> None:
    source = (_EXAMPLE_ROOT / "rule.py").read_text(encoding="utf-8")

    assert "主要编辑 ``RuleSettings`` 和 ``generate_weights``" in source
    assert "``current_weight(symbol)``" in source
    assert "``index_bars(index_code)``" in source
    assert "Optional Rule-only fund data" not in source


@pytest.mark.unit
def test_beginner_model_uses_user_facing_framework_terms() -> None:
    source = (_EXAMPLE_ROOT / "model.py").read_text(encoding="utf-8")

    assert "模型运行器输入是每个“日期-证券”的一维特征向量" in source
    assert "模型运行器在调用前已经统一设置" in source
    assert "Workflow" not in source


@pytest.mark.unit
def test_checked_in_system_config_owns_shared_execution_defaults() -> None:
    system = load_system_settings(_PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml")

    assert system.fee.commission_rate == Decimal("0.0003")
    assert system.slippage.rate == Decimal("0.0005")
    assert system.volume_participation_rate == Decimal("0.20")
    assert system.runs_dir == Path("qmt_example/results/user_experiments")
    assert system.data_snapshot.share_table == "etf_share_daily"
    assert system.data_snapshot.huijin_holders_csv is not None
    assert system.data_snapshot.huijin_holders_csv.name == "huijin_combined.csv"


@pytest.mark.unit
def test_mysql_preflight_uses_each_table_business_date_column() -> None:
    prepared = prepare_experiment(
        _EXAMPLE_ROOT / "experiment.yaml",
        require_password=False,
    )
    checks = dict(_table_checks(prepared))

    assert checks["etf_share_daily"] == "SELECT asof_date FROM `etf_share_daily` LIMIT 1"
    assert checks["etf_quote_qmt_unadjusted_daily"] == (
        "SELECT trade_date FROM `etf_quote_qmt_unadjusted_daily` LIMIT 1"
    )
