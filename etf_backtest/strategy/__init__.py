"""Public extension points for trusted local Rule and Model experiments."""

from etf_backtest.strategy.loader import UserRuleLoadError, load_user_rule
from etf_backtest.strategy.model import (
    CustomPortfolio,
    DateRange,
    FeatureBuilder,
    LoadedModelComponents,
    ModelSettings,
    TopKPortfolio,
    TorchModelFactory,
    TorchTrainingConfig,
    load_user_model_components,
)
from etf_backtest.strategy.rule import (
    NO_REBALANCE,
    NoRebalance,
    RuleMarketData,
    RuleSettings,
    SimpleRuleStrategy,
    UserRule,
    WeightInput,
)

__all__ = [
    "NO_REBALANCE",
    "CustomPortfolio",
    "DateRange",
    "FeatureBuilder",
    "LoadedModelComponents",
    "ModelSettings",
    "NoRebalance",
    "RuleMarketData",
    "RuleSettings",
    "SimpleRuleStrategy",
    "TopKPortfolio",
    "TorchModelFactory",
    "TorchTrainingConfig",
    "UserRule",
    "UserRuleLoadError",
    "WeightInput",
    "load_user_model_components",
    "load_user_rule",
]
