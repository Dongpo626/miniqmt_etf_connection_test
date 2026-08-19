"""Optional PyTorch import boundary tests."""

import pytest

import etf_backtest.strategy.model_training as daily_torch
from etf_backtest.strategy.model_training import TorchUnavailableError


@pytest.mark.unit
def test_require_torch_is_lazy_and_actionable_when_optional_runtime_is_missing(
    monkeypatch,
) -> None:
    def missing_runtime(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(daily_torch.importlib, "import_module", missing_runtime)
    with pytest.raises(TorchUnavailableError, match="PyTorch is required"):
        daily_torch.require_torch()
