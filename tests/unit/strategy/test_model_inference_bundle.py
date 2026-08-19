"""Fixed Model bundle persistence and inference-only loading tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from etf_backtest.core.market import MarketBarView  # noqa: E402
from etf_backtest.strategy.model_contracts import (  # noqa: E402
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    DateRange,
    LabeledRecord,
    ModelDataIdentity,
    SampleKey,
)
from etf_backtest.strategy.model_training import (  # noqa: E402
    DailyTorchWorkflow,
    TorchBundleCompatibilityError,
    TorchTrainingConfig,
    bundle_sha256,
    load_daily_torch_bundle_for_inference,
)
from etf_backtest.strategy.portfolio import TopKPortfolio  # noqa: E402


class _Features:
    feature_names = ("x",)
    required_history_trading_days = 1

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> tuple[Decimal, ...] | None:
        del symbol, signal_date
        return None if not history else (history[-1].close,)


class _Model:
    model_id = "tests.live.linear"
    model_class_name = "Linear"
    model_parameters: Mapping[str, object] = {"bias": True}

    def create(self, *, input_dim: int, seed: int) -> object:
        del seed
        return torch.nn.Linear(input_dim, 1, bias=True)


def _records(start: date, values: tuple[int, ...]) -> tuple[LabeledRecord, ...]:
    return tuple(
        LabeledRecord(
            key=SampleKey(signal_date=start + timedelta(days=index), symbol="SH.510300"),
            features=(Decimal(value),),
            label=Decimal(value) / Decimal("100"),
        )
        for index, value in enumerate(values)
    )


def _dataset() -> DatasetSplits:
    return DatasetSplits(
        feature_names=("x",),
        label_name=DAILY_FORWARD_RETURN_LABEL,
        train_range=DateRange(date(2021, 1, 1), date(2021, 1, 4)),
        valid_range=DateRange(date(2021, 2, 1), date(2021, 2, 2)),
        test_range=DateRange(date(2021, 3, 1), date(2021, 3, 2)),
        train=_records(date(2021, 1, 1), (1, 2, 3, 4)),
        valid=_records(date(2021, 2, 1), (5, 6)),
        test=_records(date(2021, 3, 1), (7, 8)),
    )


@pytest.fixture
def saved_bundle(tmp_path: Path) -> tuple[Path, DatasetSplits]:
    dataset = _dataset()
    workflow = DailyTorchWorkflow(
        feature_builder=_Features(),
        model_factory=_Model(),
        data_identity=ModelDataIdentity(
            dataset_version="test",
            manifest_sha256="a" * 64,
            snapshot_started_at_utc=datetime(2026, 8, 1, tzinfo=UTC),
        ),
        training_config=TorchTrainingConfig(
            seed=7,
            max_epochs=2,
            patience=1,
            batch_size=4,
            learning_rate=0.01,
        ),
        portfolio=TopKPortfolio(),
    )
    workflow.fit(dataset)
    path = workflow.save(tmp_path / "model_bundle.pt", source_run_dir=tmp_path / "run-1")
    return path, dataset


def _payload(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(value, dict)
    return value


def _save_payload(path: Path, payload: dict[str, object]) -> None:
    torch.save(payload, path)


@pytest.mark.unit
def test_inference_bundle_metadata_round_trip_is_deterministic_and_never_trains(
    saved_bundle: tuple[Path, DatasetSplits], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, dataset = saved_bundle
    payload = _payload(path)
    inference = payload["inference"]
    assert isinstance(inference, dict)
    assert payload["bundle_format_version"] == 1
    assert inference["input_dim"] == 1
    assert inference["feature_order"] == ("x",)
    assert inference["required_history_trading_days"] == 1
    assert "portfolio_json" in inference and "factor_schema" in inference
    assert "file_sha256" not in payload and "model_bundle_sha256" not in payload

    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: pytest.fail("trained"))
    loaded = load_daily_torch_bundle_for_inference(
        path,
        feature_builder=_Features(),
        model_factory=_Model(),
        portfolio=TopKPortfolio(),
        signal_date=date(2022, 1, 1),
        expected_sha256=bundle_sha256(path),
        expected_model_id="tests.live.linear",
    )
    first = loaded.bundle.predict(dataset.test)
    second = loaded.bundle.predict(dataset.test)
    np.testing.assert_allclose(
        [item.score for item in first], [item.score for item in second], rtol=0, atol=0
    )
    assert loaded.bundle.scaler.mean_.tolist() == [2.5]
    assert loaded.source_run_dir.endswith("run-1")


@pytest.mark.unit
def test_inference_bundle_rejects_hash_and_model_id_mismatch(
    saved_bundle: tuple[Path, DatasetSplits],
) -> None:
    path, _ = saved_bundle
    with pytest.raises(TorchBundleCompatibilityError, match="SHA-256"):
        load_daily_torch_bundle_for_inference(
            path,
            feature_builder=_Features(),
            model_factory=_Model(),
            portfolio=TopKPortfolio(),
            signal_date=date(2022, 1, 1),
            expected_sha256="0" * 64,
        )
    with pytest.raises(TorchBundleCompatibilityError, match="model_id"):
        load_daily_torch_bundle_for_inference(
            path,
            feature_builder=_Features(),
            model_factory=_Model(),
            portfolio=TopKPortfolio(),
            signal_date=date(2022, 1, 1),
            expected_model_id="other",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("feature_order", "feature_order"),
        ("input_dim", "input_dim"),
        ("fingerprint", "fingerprint"),
        ("trained_through", "trained_through"),
        ("scaler", "scaler"),
        ("state_dict", "state_dict"),
    ],
)
def test_inference_bundle_rejects_incompatible_or_incomplete_payload(
    saved_bundle: tuple[Path, DatasetSplits], mutation: str, message: str
) -> None:
    path, _ = saved_bundle
    payload = _payload(path)
    inference = payload["inference"]
    metadata = payload["metadata"]
    assert isinstance(inference, dict) and isinstance(metadata, dict)
    if mutation == "feature_order":
        inference["feature_order"] = ("other",)
    elif mutation == "input_dim":
        inference["input_dim"] = 2
    elif mutation == "fingerprint":
        metadata["feature_fingerprint"] = "0" * 64
    elif mutation == "trained_through":
        metadata["trained_through"] = "2022-01-01"
    else:
        payload.pop(mutation)
    _save_payload(path, payload)

    with pytest.raises(ValueError, match=message):
        load_daily_torch_bundle_for_inference(
            path,
            feature_builder=_Features(),
            model_factory=_Model(),
            portfolio=TopKPortfolio(),
            signal_date=date(2022, 1, 1),
        )
