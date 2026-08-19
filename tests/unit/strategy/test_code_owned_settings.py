from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from decimal import Decimal

import pytest

import etf_backtest.strategy.model as model_api
import etf_backtest.strategy.rule as rule_api
from etf_backtest.strategy.model_contracts import DateRange
from etf_backtest.strategy.model_training import TorchTrainingConfig


@pytest.mark.unit
def test_rule_settings_are_declared_in_rule_code_and_drive_the_adapter() -> None:
    assert hasattr(rule_api, "RuleSettings")
    settings_type = rule_api.RuleSettings
    configured = settings_type(
        lookback_trading_days=37,
        rebalance_every_trading_days=7,
        target_weight="0.83",
        parameters={"minimum_momentum": "0.0123"},
    )

    class CodeOwnedRule(rule_api.UserRule):
        settings = configured

        def generate_weights(
            self,
            data: rule_api.RuleMarketData,
        ) -> Mapping[str, rule_api.WeightInput]:
            del data
            return {}

    rule = CodeOwnedRule()
    strategy = rule_api.SimpleRuleStrategy(rule=rule)

    assert rule.lookback_trading_days == 37
    assert rule.rebalance_every_trading_days == 7
    assert rule.target_weight == Decimal("0.83")
    assert dict(rule.parameters) == {"minimum_momentum": "0.0123"}
    assert strategy.required_history_trading_days == 37
    assert strategy.should_generate_target(7)
    assert not strategy.should_generate_target(8)
    assert configured.resolved_dict()["target_weight"] == "0.83"


@pytest.mark.unit
def test_rule_settings_reject_invalid_code_owned_values() -> None:
    assert hasattr(rule_api, "RuleSettings")
    settings_type = rule_api.RuleSettings
    invalid = (
        {"lookback_trading_days": 1},
        {"rebalance_every_trading_days": 0},
        {"target_weight": "0"},
        {"parameters": {"database": "forbidden"}},
    )
    for changes in invalid:
        with pytest.raises((TypeError, ValueError)):
            settings_type(**changes)


@pytest.mark.unit
def test_model_settings_hold_splits_training_and_constructor_kwargs_in_code() -> None:
    assert hasattr(model_api, "ModelSettings")
    settings_type = model_api.ModelSettings
    portfolio = model_api.TopKPortfolio(
        top_k=3,
        total_weight="0.87",
        weighting="equal",
    )
    training = TorchTrainingConfig(
        seed=17,
        max_epochs=9,
        patience=2,
        batch_size=64,
        learning_rate=0.002,
    )
    settings = settings_type(
        train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
        valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
        portfolio=portfolio,
        training=training,
        feature_kwargs={"window": 21},
        model_kwargs={"hidden_dim": 32},
    )

    assert settings.portfolio is portfolio
    assert settings.training is training
    assert dict(settings.feature_kwargs) == {"window": 21}
    assert dict(settings.model_kwargs) == {"hidden_dim": 32}
    assert settings.resolved_dict()["training"]["max_epochs"] == 9
    assert settings.resolved_dict()["portfolio"] == {
        "type": "top_k",
        "top_k": 3,
        "total_weight": "0.87",
        "min_score": 0.0,
        "weighting": "equal",
        "softmax_temperature": 1.0,
    }


@pytest.mark.unit
def test_model_settings_reject_overlap_bad_portfolio_and_non_json_kwargs() -> None:
    assert hasattr(model_api, "ModelSettings")
    settings_type = model_api.ModelSettings
    train = DateRange(date(2021, 1, 1), date(2022, 12, 31))
    valid = DateRange(date(2022, 12, 31), date(2023, 12, 31))
    with pytest.raises(ValueError, match="non-overlapping"):
        settings_type(train_range=train, valid_range=valid)
    with pytest.raises(TypeError, match="portfolio"):
        settings_type(
            train_range=train,
            valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
            portfolio=object(),
        )
    with pytest.raises(ValueError, match="JSON"):
        settings_type(
            train_range=train,
            valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
            model_kwargs={"bad": object()},
        )
