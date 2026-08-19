"""PyTorch fit-once, early-stopping, and state_dict bundle tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

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
)


class _OneFeatureBuilder:
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
        if not history:
            return None
        return (history[-1].close,)


class _LinearFactory:
    model_id = "tests.linear"
    model_class_name = "Linear"
    model_parameters: Mapping[str, object] = {"bias": True}

    def create(self, *, input_dim: int, seed: int) -> object:
        del seed
        return torch.nn.Linear(input_dim, 1, bias=True)


def _records(start: date, values: tuple[int, ...]) -> tuple[LabeledRecord, ...]:
    return tuple(
        LabeledRecord(
            key=SampleKey(
                signal_date=start + timedelta(days=index),
                symbol="SH.510300",
            ),
            features=(Decimal(value),),
            label=Decimal(value) / Decimal("100") + Decimal("0.01"),
        )
        for index, value in enumerate(values)
    )


def _dataset() -> DatasetSplits:
    return DatasetSplits(
        feature_names=("x",),
        label_name=DAILY_FORWARD_RETURN_LABEL,
        train_range=DateRange(date(2021, 1, 1), date(2021, 1, 8)),
        valid_range=DateRange(date(2021, 2, 1), date(2021, 2, 3)),
        test_range=DateRange(date(2021, 3, 1), date(2021, 3, 3)),
        train=_records(date(2021, 1, 1), (1, 2, 3, 4, 5, 6, 7, 8)),
        valid=_records(date(2021, 2, 1), (9, 10, 11)),
        test=_records(date(2021, 3, 1), (12, 13, 14)),
    )


def _identity(manifest: str = "a") -> ModelDataIdentity:
    return ModelDataIdentity(
        dataset_version="test-snapshot",
        manifest_sha256=manifest * 64,
        snapshot_started_at_utc=datetime(2026, 8, 12, 8, 48, 57, tzinfo=UTC),
    )


def _training() -> TorchTrainingConfig:
    return TorchTrainingConfig(
        seed=17,
        max_epochs=20,
        patience=3,
        batch_size=3,
        learning_rate=0.01,
        min_delta=1_000_000.0,
    )


def _workflow(identity: ModelDataIdentity | None = None) -> DailyTorchWorkflow:
    return DailyTorchWorkflow(
        feature_builder=_OneFeatureBuilder(),
        model_factory=_LinearFactory(),
        data_identity=_identity() if identity is None else identity,
        training_config=_training(),
    )


@pytest.mark.unit
def test_torch_training_config_requires_strict_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TorchTrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        TorchTrainingConfig(batch_size=2.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_torch_workflow_uses_train_only_scaler_seeded_batches_and_early_stopping() -> None:
    dataset = _dataset()
    workflow = _workflow()
    result = workflow.fit(dataset)

    assert result.best_epoch == 1
    assert result.epochs_trained == 4
    assert result.bundle.scaler.mean_.tolist() == [4.5]
    parameters = json.loads(result.bundle.metadata.training_parameters_json)
    assert parameters["batch_size"] == 3
    assert parameters["batch_mode"] == "DETERMINISTIC_SEQUENTIAL_MINI_BATCH"
    with pytest.raises(RuntimeError, match="fit or load only once"):
        workflow.fit(dataset)

    repeated = _workflow().fit(dataset)
    np.testing.assert_allclose(
        [value.score for value in result.test_predictions],
        [value.score for value in repeated.test_predictions],
        rtol=0,
        atol=0,
    )


@pytest.mark.unit
def test_torch_state_dict_bundle_round_trip_and_metadata_validation(tmp_path) -> None:
    dataset = _dataset()
    workflow = _workflow()
    result = workflow.fit(dataset)
    target = tmp_path / "model_bundle.pt"

    assert workflow.save(target) == target
    assert target.is_file()
    assert not tuple(tmp_path.glob(".*.tmp"))

    loaded = _workflow().load(target, dataset=dataset)
    loaded_scores = [value.score for value in loaded.predict(dataset.test)]
    np.testing.assert_allclose(
        loaded_scores,
        [value.score for value in result.test_predictions],
        rtol=0,
        atol=0,
    )

    with pytest.raises(TorchBundleCompatibilityError, match="data_identity"):
        _workflow(_identity("b")).load(target, dataset=dataset)
