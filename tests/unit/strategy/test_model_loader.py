"""User-facing Model loader stays framework-lazy."""

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.strategy.model import (
    ModelSettings,
    UserModelLoadError,
    load_user_model_components,
)


@pytest.mark.unit
def test_controlled_user_model_loader_rejects_path_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside_model.py"
    outside.write_text("class Anything: pass", encoding="utf-8")

    with pytest.raises(UserModelLoadError, match="inside allowed_root"):
        load_user_model_components(
            outside,
            allowed_root=tmp_path,
        )


@pytest.mark.unit
def test_model_loader_uses_module_settings_and_constructor_kwargs(tmp_path) -> None:
    source = tmp_path / "configured_model.py"
    source.write_text(
        """
from collections.abc import Mapping
from datetime import date
from decimal import Decimal

from etf_backtest.strategy.model import (
    DateRange,
    ModelSettings,
    TopKPortfolio,
    TorchTrainingConfig,
)

MODEL_SETTINGS = ModelSettings(
    train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
    valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
    portfolio=TopKPortfolio(top_k=3, total_weight="0.87", weighting="equal"),
    training=TorchTrainingConfig(
        seed=17,
        max_epochs=9,
        patience=2,
        batch_size=64,
        learning_rate=0.002,
    ),
    feature_kwargs={"value": "1.5"},
    model_kwargs={"width": 8},
)

class Features:
    feature_names = ("constant",)
    required_history_trading_days = 1

    def __init__(self, value):
        self.value = Decimal(value)

    def build_features(self, *, symbol, signal_date, history):
        return (self.value,)

class Model:
    model_id = "user.tiny"
    model_class_name = "Tiny"

    def __init__(self, width):
        self.model_parameters: Mapping[str, object] = {"width": width}

    def create(self, *, input_dim, seed):
        return object()
""".strip(),
        encoding="utf-8",
    )

    loaded = load_user_model_components(
        source.name,
        allowed_root=tmp_path,
    )

    assert isinstance(loaded.settings, ModelSettings)
    assert loaded.source_path == source.resolve()
    assert loaded.source_sha256
    assert loaded.settings.portfolio.max_total_weight == Decimal("0.87")
    assert loaded.settings.portfolio.resolved_dict()["top_k"] == 3
    assert loaded.feature_builder.build_features(
        symbol="SH.510300", signal_date=date(2024, 1, 2), history=()
    ) == (Decimal("1.5"),)
    assert loaded.model_factory.model_parameters == {"width": 8}


@pytest.mark.unit
@pytest.mark.parametrize(
    "declaration",
    (
        "",
        "MODEL_SETTINGS = {}",
    ),
)
def test_model_loader_requires_valid_module_settings(tmp_path, declaration: str) -> None:
    source = tmp_path / "invalid_settings.py"
    source.write_text(
        f"""
{declaration}

class Features:
    feature_names = ("constant",)
    required_history_trading_days = 1
    def build_features(self, *, symbol, signal_date, history):
        return (1,)

class Model:
    model_id = "bad"
    model_class_name = "Bad"
    model_parameters = {{}}
    def create(self, *, input_dim, seed):
        return object()
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(UserModelLoadError, match="MODEL_SETTINGS"):
        load_user_model_components(
            source.name,
            allowed_root=tmp_path,
        )
