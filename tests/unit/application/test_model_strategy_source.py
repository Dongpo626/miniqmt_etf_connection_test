from __future__ import annotations

from pathlib import Path

import pytest

from etf_backtest.application.strategy_source import load_model_strategy_source
from etf_backtest.strategy.model_training import DailyTorchWorkflow

ROOT = Path(__file__).parents[3]
SYSTEM = ROOT / "qmt_example/configs/system.yaml"
MODEL_SOURCE = '''
from datetime import date

from etf_backtest.strategy.model import DateRange, ModelSettings

MODEL_SETTINGS = ModelSettings(
    train_range=DateRange(date(2021, 1, 1), date(2021, 1, 31)),
    valid_range=DateRange(date(2021, 2, 1), date(2021, 2, 28)),
)

class Features:
    feature_names = ("close",)
    required_history_trading_days = 1

    def build_features(self, *, symbol, signal_date, history):
        del symbol, signal_date
        return None if not history else (history[-1].close,)

class Model:
    model_id = "tests.source.linear"
    model_class_name = "Linear"
    model_parameters = {"bias": True}

    def create(self, *, input_dim, seed):
        del seed
        import torch
        return torch.nn.Linear(input_dim, 1, bias=True)
'''.lstrip()


def _experiment(root: Path, case: str = "model") -> Path:
    path = root / "experiment.yaml"
    path.write_text(
        "\n".join(
            (
                "name: model_live_test",
                "start_date: 2021-03-01",
                "end_date: 2021-03-31",
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
    (root / "model.py").write_text(MODEL_SOURCE, encoding="utf-8")
    return path


@pytest.mark.unit
def test_model_source_loads_stably_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _experiment(tmp_path)
    monkeypatch.setattr(
        DailyTorchWorkflow,
        "fit",
        lambda *args, **kwargs: pytest.fail("Model source loader trained a model"),
    )

    first = load_model_strategy_source(path, system_path=SYSTEM)
    second = load_model_strategy_source(path, system_path=SYSTEM)

    assert first.experiment.case == "model"
    assert first.components.feature_builder.feature_names == ("close",)
    assert first.components.model_factory.model_id == "tests.source.linear"
    assert first.strategy_source_sha256 == second.strategy_source_sha256

    (tmp_path / "model.py").write_text(MODEL_SOURCE + "\n# changed\n", encoding="utf-8")
    changed = load_model_strategy_source(path, system_path=SYSTEM)
    assert changed.strategy_source_sha256 != first.strategy_source_sha256


@pytest.mark.unit
def test_non_model_case_is_rejected_before_model_loading(tmp_path: Path) -> None:
    path = _experiment(tmp_path, case="rule")
    with pytest.raises(ValueError, match="requires experiment case: model"):
        load_model_strategy_source(path, system_path=SYSTEM)
