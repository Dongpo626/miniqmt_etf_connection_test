"""Pluggable, deterministic PyTorch workflow for daily ETF predictions.

Importing this module never imports PyTorch.  The framework is required only
when fitting, predicting, saving, or loading a torch state-dict bundle.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    MODEL_BUNDLE_SCHEMA_VERSION,
    DatasetSplits,
    DateRange,
    FeatureBuilder,
    FeatureRecord,
    LabeledRecord,
    ModelDataIdentity,
    ModelMetadata,
    PredictionRecord,
    RegressionMetricReport,
    TorchModelFactory,
    build_feature_record,
    canonical_json,
    feature_fingerprint,
    feature_records_for_signal,
    validate_feature_builder,
)
from etf_backtest.strategy.portfolio import ModelPortfolioPolicy, TopKPortfolio

TORCH_BUNDLE_FORMAT: Final = "ETF_DAILY_TORCH_STATE_DICT_V2"
INFERENCE_BUNDLE_FORMAT_VERSION: Final = 1
FOUR_FACTOR_FEATURE_NAMES: Final = (
    "ret5",
    "ret20",
    "return_volatility20",
    "volume_mean5_to_mean20",
)


class TorchUnavailableError(ImportError):
    """The optional PyTorch runtime is not installed."""


class TorchBundleCompatibilityError(ValueError):
    """A saved bundle does not match the requested workflow identity."""


def require_torch() -> Any:
    """Lazily import PyTorch or raise an actionable optional-dependency error."""

    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise TorchUnavailableError(
                "PyTorch is required for the daily torch model workflow; "
                "install a Python 3.12 compatible torch build before fitting, "
                "predicting, saving, or loading a torch bundle"
            ) from exc
        raise


class DailyFourFactorFeatureBuilder:
    """The fixed daily four-factor schema packaged as a pluggable builder."""

    __slots__ = ()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FOUR_FACTOR_FEATURE_NAMES

    @property
    def required_history_trading_days(self) -> int:
        return 21

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> Sequence[Decimal] | None:
        del symbol, signal_date
        rows = tuple(history)
        if len(rows) < self.required_history_trading_days:
            return None
        closes = tuple(view.close for view in rows[-21:])
        ret5 = closes[-1] / closes[-6] - Decimal("1")
        ret20 = closes[-1] / closes[-21] - Decimal("1")
        daily_returns = tuple(
            closes[index] / closes[index - 1] - Decimal("1") for index in range(1, 21)
        )
        mean_return = sum(daily_returns, start=Decimal("0")) / Decimal("20")
        variance = sum(
            ((value - mean_return) ** 2 for value in daily_returns),
            start=Decimal("0"),
        ) / Decimal("20")
        volumes = tuple(Decimal(view.volume) for view in rows[-20:])
        mean_volume20 = sum(volumes, start=Decimal("0")) / Decimal("20")
        if mean_volume20 == 0:
            return None
        mean_volume5 = sum(volumes[-5:], start=Decimal("0")) / Decimal("5")
        return ret5, ret20, variance.sqrt(), mean_volume5 / mean_volume20


class DailyTorchDatasetBuilder:
    """Construct D-aligned samples while owning the fixed forward label."""

    __slots__ = ("_feature_builder", "_feature_names", "_lookback")

    def __init__(self, feature_builder: FeatureBuilder) -> None:
        names, lookback = validate_feature_builder(feature_builder)
        self._feature_builder = feature_builder
        self._feature_names = names
        self._lookback = lookback

    @property
    def feature_builder(self) -> FeatureBuilder:
        return self._feature_builder

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def build(
        self,
        *,
        market_views: Iterable[MarketBarView],
        train_range: DateRange,
        valid_range: DateRange,
        test_range: DateRange,
    ) -> DatasetSplits:
        for value, field_name in (
            (train_range, "train_range"),
            (valid_range, "valid_range"),
            (test_range, "test_range"),
        ):
            if not isinstance(value, DateRange):
                raise TypeError(f"{field_name} must be DateRange")
        if not train_range.end_date < valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not valid_range.end_date < test_range.start_date:
            raise ValueError("valid_range must precede test_range")

        by_symbol: dict[str, dict[date, MarketBarView]] = {}
        for view in market_views:
            if not isinstance(view, MarketBarView):
                raise TypeError("market_views may contain only MarketBarView values")
            rows = by_symbol.setdefault(view.symbol, {})
            if view.trade_date in rows:
                raise ValueError(f"duplicate daily view for {view.symbol} on {view.trade_date}")
            rows[view.trade_date] = view
        if not by_symbol:
            raise ValueError("market_views must not be empty")

        split_rows: dict[str, list[LabeledRecord]] = {"train": [], "valid": [], "test": []}
        for symbol in sorted(by_symbol):
            ordered = tuple(by_symbol[symbol][day] for day in sorted(by_symbol[symbol]))
            for index in range(len(ordered) - 2):
                signal_date = ordered[index].trade_date
                split_name = _split_name(
                    signal_date,
                    train_range=train_range,
                    valid_range=valid_range,
                    test_range=test_range,
                )
                if split_name is None:
                    continue
                history = ordered[max(0, index - self._lookback + 1) : index + 1]
                feature_record = build_feature_record(
                    builder=self._feature_builder,
                    symbol=symbol,
                    signal_date=signal_date,
                    history=history,
                )
                if feature_record is None:
                    continue
                label = ordered[index + 2].close / ordered[index + 1].close - Decimal("1")
                split_rows[split_name].append(
                    LabeledRecord(
                        key=feature_record.key,
                        features=feature_record.features,
                        label=label,
                    )
                )
        return DatasetSplits(
            feature_names=self._feature_names,
            label_name=DAILY_FORWARD_RETURN_LABEL,
            train_range=train_range,
            valid_range=valid_range,
            test_range=test_range,
            train=tuple(split_rows["train"]),
            valid=tuple(split_rows["valid"]),
            test=tuple(split_rows["test"]),
        )

    def features_for_signal(
        self,
        *,
        market_views: Sequence[MarketBarView],
        signal_date: date,
    ) -> tuple[FeatureRecord, ...]:
        return feature_records_for_signal(
            builder=self._feature_builder,
            market_views=market_views,
            signal_date=signal_date,
        )


def _split_name(
    signal_date: date,
    *,
    train_range: DateRange,
    valid_range: DateRange,
    test_range: DateRange,
) -> str | None:
    if train_range.contains(signal_date):
        return "train"
    if valid_range.contains(signal_date):
        return "valid"
    if test_range.contains(signal_date):
        return "test"
    return None


@dataclass(frozen=True, slots=True)
class TorchTrainingConfig:
    """Deterministic sequential mini-batch and validation early-stopping policy."""

    seed: int = 20260803
    max_epochs: int = 500
    patience: int = 30
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.max_epochs) is not int or self.max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if type(self.patience) is not int or self.patience <= 0:
            raise ValueError("patience must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for field_name in ("learning_rate", "weight_decay", "min_delta"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0 or self.min_delta < 0:
            raise ValueError("weight_decay and min_delta must be non-negative")

    def to_parameters(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "min_delta": float(self.min_delta),
            "optimizer": "Adam",
            "loss": "MSELoss",
            "device": "cpu",
            "batch_mode": "DETERMINISTIC_SEQUENTIAL_MINI_BATCH",
        }


@dataclass(frozen=True, slots=True)
class DailyTorchWorkflowResult:
    bundle: DailyTorchBundle
    validation_metrics: RegressionMetricReport
    test_metrics: RegressionMetricReport
    validation_predictions: tuple[PredictionRecord, ...]
    test_predictions: tuple[PredictionRecord, ...]
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, DailyTorchBundle):
            raise TypeError("bundle must be DailyTorchBundle")
        for field_name in ("validation_metrics", "test_metrics"):
            if not isinstance(getattr(self, field_name), RegressionMetricReport):
                raise TypeError(f"{field_name} must be RegressionMetricReport")
        if type(self.best_epoch) is not int or self.best_epoch <= 0:
            raise ValueError("best_epoch must be a positive integer")
        if type(self.epochs_trained) is not int or self.epochs_trained < self.best_epoch:
            raise ValueError("epochs_trained must not precede best_epoch")
        if not isinstance(self.best_validation_loss, float) or not math.isfinite(
            self.best_validation_loss
        ):
            raise ValueError("best_validation_loss must be finite")


@dataclass(frozen=True, slots=True)
class LoadedInferenceBundle:
    """A hash-verified fixed bundle ready for DailyModelStrategy."""

    bundle: DailyTorchBundle
    file_sha256: str
    portfolio_json: str
    source_run_dir: str


class DailyTorchBundle:
    """Inference bundle backed by a validated CPU torch state_dict."""

    __slots__ = (
        "_best_epoch",
        "_best_validation_loss",
        "_epochs_trained",
        "_factory",
        "_metadata",
        "_scaler",
        "_state_dict",
    )

    def __init__(
        self,
        *,
        metadata: ModelMetadata,
        scaler: StandardScaler,
        factory: TorchModelFactory,
        state_dict: Mapping[str, np.ndarray[Any, Any]],
        best_epoch: int,
        epochs_trained: int,
        best_validation_loss: float,
    ) -> None:
        if not isinstance(metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if not isinstance(scaler, StandardScaler):
            raise TypeError("scaler must be StandardScaler")
        if getattr(scaler, "n_features_in_", None) != len(metadata.feature_names):
            raise ValueError("scaler feature width does not match model metadata")
        _validate_factory(factory)
        if metadata.model_id != factory.model_id:
            raise TorchBundleCompatibilityError("factory model_id does not match metadata")
        if metadata.model_class_name != factory.model_class_name:
            raise TorchBundleCompatibilityError("factory model_class_name does not match metadata")
        if metadata.model_parameters_json != canonical_json(factory.model_parameters):
            raise TorchBundleCompatibilityError("factory parameters do not match metadata")
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("state_dict must be a non-empty mapping")
        frozen_state: dict[str, np.ndarray[Any, Any]] = {}
        for raw_name, raw_value in state_dict.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError("state_dict keys must be nonblank strings")
            value = np.asarray(raw_value).copy()
            if value.dtype == object:
                raise ValueError("state_dict arrays may not use object dtype")
            value.setflags(write=False)
            frozen_state[raw_name] = value
        if type(best_epoch) is not int or best_epoch <= 0:
            raise ValueError("best_epoch must be a positive integer")
        if type(epochs_trained) is not int or epochs_trained < best_epoch:
            raise ValueError("epochs_trained must not precede best_epoch")
        best_loss = float(best_validation_loss)
        if not math.isfinite(best_loss) or best_loss < 0:
            raise ValueError("best_validation_loss must be finite and non-negative")
        self._metadata = metadata
        self._scaler = scaler
        self._factory = factory
        self._state_dict = MappingProxyType(dict(sorted(frozen_state.items())))
        self._best_epoch = best_epoch
        self._epochs_trained = epochs_trained
        self._best_validation_loss = best_loss
        self._validate_state_dict()

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def epochs_trained(self) -> int:
        return self._epochs_trained

    @property
    def best_validation_loss(self) -> float:
        return self._best_validation_loss

    @property
    def state_dict(self) -> Mapping[str, np.ndarray[Any, Any]]:
        return MappingProxyType({name: value.copy() for name, value in self._state_dict.items()})

    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]:
        supplied_records = cast(object, records)
        if isinstance(supplied_records, (str, bytes)) or not isinstance(supplied_records, Sequence):
            raise TypeError("records must be a sequence")
        ordered = tuple(sorted(records, key=lambda record: record.key))
        if not ordered:
            raise ValueError("records must not be empty")
        if any(not isinstance(record, FeatureRecord) for record in ordered):
            raise TypeError("records may contain only FeatureRecord values")
        keys = tuple(record.key for record in ordered)
        if len(keys) != len(set(keys)):
            raise ValueError("records contain duplicate sample keys")
        if any(len(record.features) != len(self._metadata.feature_names) for record in ordered):
            raise ValueError("record feature width does not match bundle metadata")
        matrix = _feature_matrix(ordered, len(self._metadata.feature_names))
        normalized = self._scaler.transform(matrix)
        torch = require_torch()
        model = self._new_model(torch)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device="cpu")
        model.eval()
        with torch.no_grad():
            output = _flat_output(model(tensor), expected_rows=len(ordered))
        scores = output.detach().cpu().numpy()
        return tuple(
            PredictionRecord(key=record.key, score=float(score))
            for record, score in zip(ordered, scores, strict=True)
        )

    def _new_model(self, torch: Any) -> Any:
        _seed_everything(torch, self._metadata.random_seed)
        model = self._factory.create(
            input_dim=len(self._metadata.feature_names),
            seed=self._metadata.random_seed,
        )
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchModelFactory.create must return torch.nn.Module")
        model.to("cpu")
        tensors = {name: torch.as_tensor(value.copy()) for name, value in self._state_dict.items()}
        model.load_state_dict(tensors, strict=True)
        return model

    def _validate_state_dict(self) -> None:
        torch = require_torch()
        self._new_model(torch)


class DailyTorchWorkflow:
    """Fit once, early-stop on validation, and persist a state_dict bundle."""

    __slots__ = (
        "_bundle",
        "_data_identity",
        "_factory",
        "_feature_builder",
        "_feature_names",
        "_portfolio",
        "_required_history",
        "_training_config",
    )

    def __init__(
        self,
        *,
        feature_builder: FeatureBuilder,
        model_factory: TorchModelFactory,
        data_identity: ModelDataIdentity,
        training_config: TorchTrainingConfig | None = None,
        portfolio: ModelPortfolioPolicy | None = None,
    ) -> None:
        names, required_history = validate_feature_builder(feature_builder)
        _validate_factory(model_factory)
        if not isinstance(data_identity, ModelDataIdentity):
            raise TypeError("data_identity must be ModelDataIdentity")
        if training_config is None:
            training_config = TorchTrainingConfig()
        if not isinstance(training_config, TorchTrainingConfig):
            raise TypeError("training_config must be TorchTrainingConfig")
        portfolio = portfolio or TopKPortfolio()
        if not isinstance(portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        self._feature_builder = feature_builder
        self._feature_names = names
        self._required_history = required_history
        self._portfolio = portfolio
        self._factory = model_factory
        self._data_identity = data_identity
        self._training_config = training_config
        self._bundle: DailyTorchBundle | None = None

    @property
    def bundle(self) -> DailyTorchBundle | None:
        return self._bundle

    def fit(self, dataset: DatasetSplits) -> DailyTorchWorkflowResult:
        if self._bundle is not None:
            raise RuntimeError("DailyTorchWorkflow may fit or load only once")
        self._validate_dataset_schema(dataset)
        torch = require_torch()
        seed = self._training_config.seed
        _seed_everything(torch, seed)

        train_matrix = _feature_matrix(dataset.train, len(self._feature_names))
        valid_matrix = _feature_matrix(dataset.valid, len(self._feature_names))
        scaler = StandardScaler()
        scaler.fit(train_matrix)
        normalized_train = scaler.transform(train_matrix)
        normalized_valid = scaler.transform(valid_matrix)
        train_targets = _target_vector(dataset.train)
        valid_targets = _target_vector(dataset.valid)

        model = self._factory.create(input_dim=len(self._feature_names), seed=seed)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchModelFactory.create must return torch.nn.Module")
        model.to("cpu")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self._training_config.learning_rate),
            weight_decay=float(self._training_config.weight_decay),
        )
        loss_function = torch.nn.MSELoss()
        train_x = torch.as_tensor(normalized_train, dtype=torch.float32, device="cpu")
        valid_x = torch.as_tensor(normalized_valid, dtype=torch.float32, device="cpu")
        train_y = torch.as_tensor(train_targets, dtype=torch.float32, device="cpu")
        valid_y = torch.as_tensor(valid_targets, dtype=torch.float32, device="cpu")

        best_loss = math.inf
        best_epoch = 0
        epochs_trained = 0
        stale_epochs = 0
        best_state: dict[str, Any] | None = None
        for epoch in range(1, self._training_config.max_epochs + 1):
            epochs_trained = epoch
            model.train()
            for batch_start in range(0, len(dataset.train), self._training_config.batch_size):
                batch_end = min(
                    batch_start + self._training_config.batch_size,
                    len(dataset.train),
                )
                batch_x = train_x[batch_start:batch_end]
                batch_y = train_y[batch_start:batch_end]
                optimizer.zero_grad(set_to_none=True)
                prediction = _flat_output(model(batch_x), expected_rows=batch_end - batch_start)
                loss = loss_function(prediction, batch_y)
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("training loss became non-finite")
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                valid_prediction = _flat_output(model(valid_x), expected_rows=len(dataset.valid))
                validation_loss = float(loss_function(valid_prediction, valid_y).item())
            if not math.isfinite(validation_loss):
                raise ValueError("validation loss became non-finite")
            if validation_loss < best_loss - float(self._training_config.min_delta):
                best_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= self._training_config.patience:
                    break
        if best_state is None or best_epoch <= 0 or not math.isfinite(best_loss):
            raise RuntimeError("validation early stopping did not produce a finite model")
        model.load_state_dict(best_state, strict=True)
        state_arrays = {
            name: value.detach().cpu().numpy().copy() for name, value in best_state.items()
        }
        metadata = ModelMetadata(
            schema_version=MODEL_BUNDLE_SCHEMA_VERSION,
            model_id=self._factory.model_id,
            model_class_name=self._factory.model_class_name,
            model_parameters_json=canonical_json(self._factory.model_parameters),
            training_parameters_json=canonical_json(self._training_config.to_parameters()),
            feature_names=self._feature_names,
            feature_fingerprint=feature_fingerprint(
                self._feature_names, DAILY_FORWARD_RETURN_LABEL
            ),
            label_name=DAILY_FORWARD_RETURN_LABEL,
            train_range=dataset.train_range,
            valid_range=dataset.valid_range,
            test_range=dataset.test_range,
            random_seed=seed,
            data_identity=self._data_identity,
            trained_through=dataset.trained_through,
            framework_name="torch",
            framework_version=str(torch.__version__),
        )
        bundle = DailyTorchBundle(
            metadata=metadata,
            scaler=scaler,
            factory=self._factory,
            state_dict=state_arrays,
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            best_validation_loss=best_loss,
        )
        self._bundle = bundle
        validation_predictions = bundle.predict(dataset.valid)
        test_predictions = bundle.predict(dataset.test)
        return DailyTorchWorkflowResult(
            bundle=bundle,
            validation_metrics=evaluate_predictions(
                samples=dataset.valid,
                predictions=validation_predictions,
            ),
            test_metrics=evaluate_predictions(
                samples=dataset.test,
                predictions=test_predictions,
            ),
            validation_predictions=validation_predictions,
            test_predictions=test_predictions,
            best_epoch=best_epoch,
            epochs_trained=epochs_trained,
            best_validation_loss=best_loss,
        )

    def save(self, path: Path, *, source_run_dir: Path | None = None) -> Path:
        if self._bundle is None:
            raise RuntimeError("DailyTorchWorkflow has no bundle to save")
        torch = require_torch()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        run_directory = Path(source_run_dir or target.parent).resolve()
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        payload = {
            "bundle_format": TORCH_BUNDLE_FORMAT,
            "bundle_format_version": INFERENCE_BUNDLE_FORMAT_VERSION,
            "metadata": self._bundle.metadata.to_dict(),
            "inference": {
                "input_dim": len(self._feature_names),
                "feature_order": self._feature_names,
                "required_history_trading_days": self._required_history,
                "portfolio_json": canonical_json(self._portfolio.resolved_dict()),
                "source_run_dir": str(run_directory),
                "factor_schema": {
                    "feature_names": self._feature_names,
                    "label_name": self._bundle.metadata.label_name,
                    "feature_fingerprint": self._bundle.metadata.feature_fingerprint,
                },
            },
            "scaler": _scaler_payload(self._bundle.scaler),
            "state_dict": {
                name: torch.as_tensor(value.copy())
                for name, value in self._bundle.state_dict.items()
            },
            "fit_summary": {
                "best_epoch": self._bundle.best_epoch,
                "epochs_trained": self._bundle.epochs_trained,
                "best_validation_loss": self._bundle.best_validation_loss,
            },
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def load(self, path: Path, *, dataset: DatasetSplits) -> DailyTorchBundle:
        if self._bundle is not None:
            raise RuntimeError("DailyTorchWorkflow may fit or load only once")
        self._validate_dataset_schema(dataset)
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        torch = require_torch()
        try:
            raw_payload = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older torch
            raw_payload = torch.load(source, map_location="cpu")
        if not isinstance(raw_payload, Mapping):
            raise TorchBundleCompatibilityError("torch bundle payload must be a mapping")
        if raw_payload.get("bundle_format") != TORCH_BUNDLE_FORMAT:
            raise TorchBundleCompatibilityError("unsupported torch bundle format")
        metadata = ModelMetadata.from_mapping(
            _required_mapping(raw_payload, "metadata", "metadata")
        )
        self._validate_metadata(metadata, dataset)
        scaler = _restore_scaler(_required_mapping(raw_payload, "scaler", "scaler"))
        raw_state = _required_mapping(raw_payload, "state_dict", "state_dict")
        state_arrays: dict[str, np.ndarray[Any, Any]] = {}
        for raw_name, raw_value in raw_state.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise TorchBundleCompatibilityError("state_dict keys must be nonblank strings")
            if not hasattr(raw_value, "detach"):
                raise TorchBundleCompatibilityError("state_dict values must be torch tensors")
            state_arrays[raw_name] = raw_value.detach().cpu().numpy().copy()
        summary = _required_mapping(raw_payload, "fit_summary", "fit_summary")
        bundle = DailyTorchBundle(
            metadata=metadata,
            scaler=scaler,
            factory=self._factory,
            state_dict=state_arrays,
            best_epoch=_required_int(summary, "best_epoch"),
            epochs_trained=_required_int(summary, "epochs_trained"),
            best_validation_loss=_required_float(summary, "best_validation_loss"),
        )
        self._bundle = bundle
        return bundle

    def _validate_dataset_schema(self, dataset: DatasetSplits) -> None:
        if not isinstance(dataset, DatasetSplits):
            raise TypeError("dataset must be DatasetSplits")
        if dataset.feature_names != self._feature_names:
            raise ValueError("dataset feature_names do not match FeatureBuilder")
        if dataset.label_name != DAILY_FORWARD_RETURN_LABEL:
            raise ValueError("dataset label does not match the daily forward-return contract")

    def _validate_metadata(self, metadata: ModelMetadata, dataset: DatasetSplits) -> None:
        expected = {
            "model_id": self._factory.model_id,
            "model_class_name": self._factory.model_class_name,
            "model_parameters_json": canonical_json(self._factory.model_parameters),
            "training_parameters_json": canonical_json(self._training_config.to_parameters()),
            "feature_names": self._feature_names,
            "feature_fingerprint": feature_fingerprint(
                self._feature_names, DAILY_FORWARD_RETURN_LABEL
            ),
            "label_name": DAILY_FORWARD_RETURN_LABEL,
            "train_range": dataset.train_range,
            "valid_range": dataset.valid_range,
            "test_range": dataset.test_range,
            "random_seed": self._training_config.seed,
            "data_identity": self._data_identity,
            "trained_through": dataset.trained_through,
        }
        for field_name, expected_value in expected.items():
            if getattr(metadata, field_name) != expected_value:
                raise TorchBundleCompatibilityError(f"bundle metadata mismatch for {field_name}")
        if metadata.framework_name != "torch":
            raise TorchBundleCompatibilityError("bundle framework_name must be torch")
        current_torch = require_torch()
        saved_major = metadata.framework_version.split(".", maxsplit=1)[0]
        current_major = str(current_torch.__version__).split(".", maxsplit=1)[0]
        if saved_major != current_major:
            raise TorchBundleCompatibilityError("bundle torch major version is incompatible")


def bundle_sha256(path: Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_daily_torch_bundle_for_inference(
    path: Path,
    *,
    feature_builder: FeatureBuilder,
    model_factory: TorchModelFactory,
    portfolio: ModelPortfolioPolicy,
    signal_date: date,
    expected_sha256: str | None = None,
    expected_model_id: str | None = None,
) -> LoadedInferenceBundle:
    """Load and strictly validate one fixed bundle without fitting any model."""

    source = Path(path)
    actual_sha256 = bundle_sha256(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
        raise TorchBundleCompatibilityError("model bundle SHA-256 does not match deployment")
    feature_names, required_history = validate_feature_builder(feature_builder)
    _validate_factory(model_factory)
    if not isinstance(portfolio, ModelPortfolioPolicy):
        raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
    if expected_model_id is not None and model_factory.model_id != expected_model_id:
        raise TorchBundleCompatibilityError("Model source model_id does not match deployment")
    torch = require_torch()
    try:
        raw_payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        raw_payload = torch.load(source, map_location="cpu")
    if not isinstance(raw_payload, Mapping):
        raise TorchBundleCompatibilityError("torch bundle payload must be a mapping")
    if raw_payload.get("bundle_format") != TORCH_BUNDLE_FORMAT:
        raise TorchBundleCompatibilityError(
            "unsupported torch bundle format; retrain the model to create a Live bundle"
        )
    if raw_payload.get("bundle_format_version") != INFERENCE_BUNDLE_FORMAT_VERSION:
        raise TorchBundleCompatibilityError("unsupported inference bundle format version")
    metadata = ModelMetadata.from_mapping(
        _required_mapping(raw_payload, "metadata", "metadata")
    )
    if metadata.model_id != model_factory.model_id:
        raise TorchBundleCompatibilityError("bundle model_id does not match Model source")
    if metadata.model_class_name != model_factory.model_class_name:
        raise TorchBundleCompatibilityError("bundle model type does not match Model source")
    if metadata.model_parameters_json != canonical_json(model_factory.model_parameters):
        raise TorchBundleCompatibilityError("bundle model parameters do not match Model source")
    if metadata.feature_names != feature_names:
        raise TorchBundleCompatibilityError("bundle feature order does not match Model source")
    if metadata.feature_fingerprint != feature_fingerprint(
        feature_names, DAILY_FORWARD_RETURN_LABEL
    ):
        raise TorchBundleCompatibilityError("bundle feature fingerprint does not match Model source")
    if metadata.trained_through >= signal_date:
        raise TorchBundleCompatibilityError("bundle trained_through must precede signal_date")

    inference = _required_mapping(raw_payload, "inference", "inference")
    if _required_int(inference, "input_dim") != len(feature_names):
        raise TorchBundleCompatibilityError("bundle input_dim does not match feature count")
    raw_order = inference.get("feature_order")
    if isinstance(raw_order, (str, bytes)) or not isinstance(raw_order, Sequence):
        raise TorchBundleCompatibilityError("bundle feature_order must be a sequence")
    if tuple(raw_order) != feature_names:
        raise TorchBundleCompatibilityError("bundle feature_order does not match Model source")
    if _required_int(inference, "required_history_trading_days") != required_history:
        raise TorchBundleCompatibilityError("bundle history requirement does not match Model source")
    portfolio_json = inference.get("portfolio_json")
    expected_portfolio_json = canonical_json(portfolio.resolved_dict())
    if portfolio_json != expected_portfolio_json:
        raise TorchBundleCompatibilityError("bundle portfolio does not match Model source")
    source_run_dir = inference.get("source_run_dir")
    if not isinstance(source_run_dir, str) or not source_run_dir.strip():
        raise TorchBundleCompatibilityError("bundle source_run_dir is missing")
    factor_schema = _required_mapping(inference, "factor_schema", "factor_schema")
    expected_schema = {
        "feature_names": feature_names,
        "label_name": metadata.label_name,
        "feature_fingerprint": metadata.feature_fingerprint,
    }
    if dict(factor_schema) != expected_schema:
        raise TorchBundleCompatibilityError("bundle factor schema does not match metadata")

    scaler = _restore_scaler(_required_mapping(raw_payload, "scaler", "scaler"))
    raw_state = _required_mapping(raw_payload, "state_dict", "state_dict")
    state_arrays: dict[str, np.ndarray[Any, Any]] = {}
    for raw_name, raw_value in raw_state.items():
        if not isinstance(raw_name, str) or not raw_name or not hasattr(raw_value, "detach"):
            raise TorchBundleCompatibilityError("bundle state_dict is invalid")
        state_arrays[raw_name] = raw_value.detach().cpu().numpy().copy()
    summary = _required_mapping(raw_payload, "fit_summary", "fit_summary")
    bundle = DailyTorchBundle(
        metadata=metadata,
        scaler=scaler,
        factory=model_factory,
        state_dict=state_arrays,
        best_epoch=_required_int(summary, "best_epoch"),
        epochs_trained=_required_int(summary, "epochs_trained"),
        best_validation_loss=_required_float(summary, "best_validation_loss"),
    )
    saved_major = metadata.framework_version.split(".", maxsplit=1)[0]
    current_major = str(torch.__version__).split(".", maxsplit=1)[0]
    if metadata.framework_name != "torch" or saved_major != current_major:
        raise TorchBundleCompatibilityError("bundle torch runtime is incompatible")
    return LoadedInferenceBundle(
        bundle=bundle,
        file_sha256=actual_sha256,
        portfolio_json=expected_portfolio_json,
        source_run_dir=source_run_dir,
    )


def _validate_factory(factory: TorchModelFactory) -> None:
    if not isinstance(factory, TorchModelFactory):
        raise TypeError("model_factory must satisfy TorchModelFactory")
    if not isinstance(factory.model_id, str) or not factory.model_id.strip():
        raise ValueError("model_factory.model_id must be nonblank")
    if not isinstance(factory.model_class_name, str) or not factory.model_class_name.strip():
        raise ValueError("model_factory.model_class_name must be nonblank")
    canonical_json(factory.model_parameters)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _flat_output(value: Any, *, expected_rows: int) -> Any:
    flattened = value.reshape(-1)
    if int(flattened.numel()) != expected_rows:
        raise ValueError("torch model must return exactly one score per input row")
    return flattened


def _feature_matrix(
    records: Sequence[FeatureRecord],
    feature_count: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    matrix = np.asarray(
        [[float(value) for value in record.features] for record in records],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape != (len(records), feature_count):
        raise ValueError("feature matrix has an invalid shape")
    if not np.isfinite(matrix).all():
        raise ValueError("feature matrix must be finite")
    return matrix


def _target_vector(
    records: Sequence[LabeledRecord],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    target = np.asarray([float(record.label) for record in records], dtype=np.float64)
    if target.shape != (len(records),) or not np.isfinite(target).all():
        raise ValueError("target vector must be finite and one-dimensional")
    return target


def evaluate_predictions(
    *,
    samples: Sequence[LabeledRecord],
    predictions: Sequence[PredictionRecord],
) -> RegressionMetricReport:
    ordered = tuple(sorted(samples, key=lambda sample: sample.key))
    if not ordered or any(not isinstance(sample, LabeledRecord) for sample in ordered):
        raise ValueError("samples must contain labeled records")
    by_key: dict[object, PredictionRecord] = {}
    for prediction in predictions:
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predictions may contain only PredictionRecord values")
        if prediction.key in by_key:
            raise ValueError("predictions contain duplicate keys")
        by_key[prediction.key] = prediction
    expected_keys = tuple(sample.key for sample in ordered)
    if frozenset(by_key) != frozenset(expected_keys):
        raise ValueError("predictions must exactly match sample keys")
    target = _target_vector(ordered)
    score = np.asarray([by_key[key].score for key in expected_keys], dtype=np.float64)
    error = score - target
    if np.std(score) == 0.0 or np.std(target) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(score, target)[0, 1])
    return RegressionMetricReport(
        sample_count=len(ordered),
        mean_squared_error=float(np.mean(error**2)),
        mean_absolute_error=float(np.mean(np.abs(error))),
        prediction_correlation=correlation,
    )


def _scaler_payload(scaler: StandardScaler) -> dict[str, object]:
    return {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "var": np.asarray(scaler.var_, dtype=np.float64),
        "n_features_in": int(scaler.n_features_in_),
        "n_samples_seen": np.asarray(scaler.n_samples_seen_),
    }


def _restore_scaler(value: Mapping[str, object]) -> StandardScaler:
    try:
        mean = np.asarray(value["mean"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
        variance = np.asarray(value["var"], dtype=np.float64)
        feature_count = _required_int(value, "n_features_in")
        samples_seen = np.asarray(value["n_samples_seen"])
    except KeyError as exc:
        raise TorchBundleCompatibilityError("scaler payload is incomplete") from exc
    if mean.shape != (feature_count,) or scale.shape != mean.shape or variance.shape != mean.shape:
        raise TorchBundleCompatibilityError("scaler arrays have incompatible shapes")
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(variance).all()
    ):
        raise TorchBundleCompatibilityError("scaler arrays must be finite")
    if (scale <= 0).any() or (variance < 0).any():
        raise TorchBundleCompatibilityError("scaler scale/variance are invalid")
    scaler = StandardScaler()
    scaler.mean_ = mean.copy()
    scaler.scale_ = scale.copy()
    scaler.var_ = variance.copy()
    scaler.n_features_in_ = feature_count
    scaler.n_samples_seen_ = samples_seen.item() if samples_seen.ndim == 0 else samples_seen.copy()
    return scaler


def _required_mapping(
    value: Mapping[str, object],
    key: str,
    field_name: str,
) -> Mapping[str, object]:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{field_name} is missing") from exc
    if not isinstance(result, Mapping):
        raise TorchBundleCompatibilityError(f"{field_name} must be a mapping")
    return result


def _required_int(value: Mapping[str, object], key: str) -> int:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{key} is missing") from exc
    if type(result) is not int:
        raise TorchBundleCompatibilityError(f"{key} must be int")
    return result


def _required_float(value: Mapping[str, object], key: str) -> float:
    try:
        result = value[key]
    except KeyError as exc:
        raise TorchBundleCompatibilityError(f"{key} is missing") from exc
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise TorchBundleCompatibilityError(f"{key} must be numeric")
    converted = float(result)
    if not math.isfinite(converted):
        raise TorchBundleCompatibilityError(f"{key} must be finite")
    return converted


__all__ = [
    "FOUR_FACTOR_FEATURE_NAMES",
    "INFERENCE_BUNDLE_FORMAT_VERSION",
    "TORCH_BUNDLE_FORMAT",
    "DailyFourFactorFeatureBuilder",
    "DailyTorchBundle",
    "DailyTorchDatasetBuilder",
    "DailyTorchWorkflow",
    "DailyTorchWorkflowResult",
    "LoadedInferenceBundle",
    "TorchBundleCompatibilityError",
    "TorchTrainingConfig",
    "TorchUnavailableError",
    "bundle_sha256",
    "evaluate_predictions",
    "load_daily_torch_bundle_for_inference",
    "require_torch",
]
