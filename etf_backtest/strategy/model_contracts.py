"""Framework-neutral contracts for daily supervised model workflows.

The execution engine intentionally knows only :class:`BaseStrategy`.  This
module defines the research-side boundary needed by pluggable models without
introducing a dependency on a numerical or deep-learning framework.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import MarketBarView

DAILY_FORWARD_RETURN_LABEL = "front_close[D+2]/front_close[D+1]-1"
MODEL_BUNDLE_SCHEMA_VERSION = "DAILY_MODEL_BUNDLE_V2"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _finite_decimal(value: object, field_name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field_name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def canonical_json(value: Mapping[str, object]) -> str:
    """Return a stable JSON identity for model or training parameters."""

    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata must be finite and JSON-serializable") from exc


def validate_feature_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("feature_names must be a sequence")
    names = tuple(_non_blank(value, "feature_name") for value in values)
    if not names:
        raise ValueError("feature_names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("feature_names must be unique")
    return names


def feature_fingerprint(feature_names: Sequence[str], label_name: str) -> str:
    names = validate_feature_names(feature_names)
    label = _non_blank(label_name, "label_name")
    encoded = json.dumps(
        {"feature_names": names, "label_name": label},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DateRange:
    """Inclusive signal-date range for one chronological split."""

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        start = _plain_date(self.start_date, "start_date")
        end = _plain_date(self.end_date, "end_date")
        if end < start:
            raise ValueError("end_date must not precede start_date")

    def contains(self, value: date) -> bool:
        return self.start_date <= value <= self.end_date

    def to_dict(self) -> dict[str, str]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], field_name: str) -> DateRange:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        try:
            start = date.fromisoformat(_non_blank(value["start_date"], f"{field_name}.start_date"))
            end = date.fromisoformat(_non_blank(value["end_date"], f"{field_name}.end_date"))
        except KeyError as exc:
            raise ValueError(f"{field_name} is incomplete") from exc
        return cls(start_date=start, end_date=end)


@dataclass(frozen=True, slots=True, order=True)
class SampleKey:
    """Stable identity for a model row aligned to signal date ``D``."""

    signal_date: date
    symbol: str

    def __post_init__(self) -> None:
        _plain_date(self.signal_date, "signal_date")
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """Framework-neutral Decimal feature vector for one sample key."""

    key: SampleKey
    features: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, SampleKey):
            raise TypeError("key must be SampleKey")
        values = tuple(self.features)
        if not values:
            raise ValueError("features must not be empty")
        for index, value in enumerate(values):
            _finite_decimal(value, f"features[{index}]")
        object.__setattr__(self, "features", values)


@dataclass(frozen=True, slots=True)
class LabeledRecord(FeatureRecord):
    """Feature vector plus the D+1-close-to-D+2-close return label."""

    label: Decimal

    def __post_init__(self) -> None:
        super(LabeledRecord, self).__post_init__()
        _finite_decimal(self.label, "label")


def _freeze_labeled_records(
    records: Sequence[LabeledRecord],
    *,
    field_name: str,
    feature_count: int,
    date_range: DateRange,
) -> tuple[LabeledRecord, ...]:
    supplied_records = cast(object, records)
    if isinstance(supplied_records, (str, bytes)) or not isinstance(supplied_records, Sequence):
        raise TypeError(f"{field_name} must be a sequence")
    values = tuple(records)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(record, LabeledRecord) for record in values):
        raise TypeError(f"{field_name} may contain only LabeledRecord values")
    ordered = tuple(sorted(values, key=lambda record: record.key))
    keys = tuple(record.key for record in ordered)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} contains duplicate sample keys")
    for record in ordered:
        if len(record.features) != feature_count:
            raise ValueError(f"{field_name} has a feature-width mismatch")
        if not date_range.contains(record.key.signal_date):
            raise ValueError(f"{field_name} contains a signal outside its date range")
    return ordered


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Qlib-style immutable train/valid/test dataset contract."""

    feature_names: tuple[str, ...]
    label_name: str
    train_range: DateRange
    valid_range: DateRange
    test_range: DateRange
    train: tuple[LabeledRecord, ...]
    valid: tuple[LabeledRecord, ...]
    test: tuple[LabeledRecord, ...]

    def __post_init__(self) -> None:
        names = validate_feature_names(self.feature_names)
        label = _non_blank(self.label_name, "label_name")
        for field_name in ("train_range", "valid_range", "test_range"):
            if not isinstance(getattr(self, field_name), DateRange):
                raise TypeError(f"{field_name} must be DateRange")
        if not self.train_range.end_date < self.valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not self.valid_range.end_date < self.test_range.start_date:
            raise ValueError("valid_range must precede test_range")
        train = _freeze_labeled_records(
            self.train,
            field_name="train",
            feature_count=len(names),
            date_range=self.train_range,
        )
        valid = _freeze_labeled_records(
            self.valid,
            field_name="valid",
            feature_count=len(names),
            date_range=self.valid_range,
        )
        test = _freeze_labeled_records(
            self.test,
            field_name="test",
            feature_count=len(names),
            date_range=self.test_range,
        )
        keys = tuple(record.key for split in (train, valid, test) for record in split)
        if len(keys) != len(set(keys)):
            raise ValueError("dataset splits contain overlapping sample keys")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "label_name", label)
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "test", test)

    @property
    def trained_through(self) -> date:
        return max(record.key.signal_date for record in self.train)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One finite model score aligned to a sample key."""

    key: SampleKey
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, SampleKey):
            raise TypeError("key must be SampleKey")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be numeric")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score must be finite")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True, slots=True)
class RegressionMetricReport:
    """Finite regression metrics for one exact prediction alignment."""

    sample_count: int
    mean_squared_error: float
    mean_absolute_error: float
    prediction_correlation: float

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        for field_name in (
            "mean_squared_error",
            "mean_absolute_error",
            "prediction_correlation",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite float")


@dataclass(frozen=True, slots=True)
class ModelDataIdentity:
    """Frozen data identity that a saved model bundle must match on load."""

    dataset_version: str
    manifest_sha256: str
    snapshot_started_at_utc: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_version", _non_blank(self.dataset_version, "dataset_version")
        )
        manifest = _non_blank(self.manifest_sha256, "manifest_sha256").lower()
        if _SHA256.fullmatch(manifest) is None:
            raise ValueError("manifest_sha256 must contain 64 lowercase hex characters")
        snapshot_started = self.snapshot_started_at_utc
        if not isinstance(snapshot_started, datetime):
            raise TypeError("snapshot_started_at_utc must be datetime")
        if snapshot_started.tzinfo is None or snapshot_started.utcoffset() is None:
            raise ValueError("snapshot_started_at_utc must be timezone-aware")
        object.__setattr__(self, "manifest_sha256", manifest)
        object.__setattr__(self, "snapshot_started_at_utc", snapshot_started.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Self-describing compatibility identity embedded in every bundle."""

    schema_version: str
    model_id: str
    model_class_name: str
    model_parameters_json: str
    training_parameters_json: str
    feature_names: tuple[str, ...]
    feature_fingerprint: str
    label_name: str
    train_range: DateRange
    valid_range: DateRange
    test_range: DateRange
    random_seed: int
    data_identity: ModelDataIdentity
    trained_through: date
    framework_name: str
    framework_version: str

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported model bundle schema_version")
        for field_name in (
            "model_id",
            "model_class_name",
            "model_parameters_json",
            "training_parameters_json",
            "label_name",
            "framework_name",
            "framework_version",
        ):
            object.__setattr__(self, field_name, _non_blank(getattr(self, field_name), field_name))
        names = validate_feature_names(self.feature_names)
        fingerprint = _non_blank(self.feature_fingerprint, "feature_fingerprint").lower()
        if _SHA256.fullmatch(fingerprint) is None:
            raise ValueError("feature_fingerprint must be a SHA-256 hex string")
        if fingerprint != feature_fingerprint(names, self.label_name):
            raise ValueError("feature_fingerprint does not match feature/label schema")
        for json_field in ("model_parameters_json", "training_parameters_json"):
            try:
                parsed = json.loads(getattr(self, json_field))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{json_field} must contain valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{json_field} must encode a JSON object")
            if canonical_json(parsed) != getattr(self, json_field):
                raise ValueError(f"{json_field} must use canonical JSON encoding")
        for range_name in ("train_range", "valid_range", "test_range"):
            if not isinstance(getattr(self, range_name), DateRange):
                raise TypeError(f"{range_name} must be DateRange")
        if not self.train_range.end_date < self.valid_range.start_date:
            raise ValueError("train_range must precede valid_range")
        if not self.valid_range.end_date < self.test_range.start_date:
            raise ValueError("valid_range must precede test_range")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not isinstance(self.data_identity, ModelDataIdentity):
            raise TypeError("data_identity must be ModelDataIdentity")
        trained = _plain_date(self.trained_through, "trained_through")
        if not self.train_range.contains(trained):
            raise ValueError("trained_through must be within train_range")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_fingerprint", fingerprint)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_class_name": self.model_class_name,
            "model_parameters_json": self.model_parameters_json,
            "training_parameters_json": self.training_parameters_json,
            "feature_names": self.feature_names,
            "feature_fingerprint": self.feature_fingerprint,
            "label_name": self.label_name,
            "train_range": self.train_range.to_dict(),
            "valid_range": self.valid_range.to_dict(),
            "test_range": self.test_range.to_dict(),
            "random_seed": self.random_seed,
            "data_identity": {
                "dataset_version": self.data_identity.dataset_version,
                "manifest_sha256": self.data_identity.manifest_sha256,
                "snapshot_started_at_utc": (self.data_identity.snapshot_started_at_utc.isoformat()),
            },
            "trained_through": self.trained_through.isoformat(),
            "framework_name": self.framework_name,
            "framework_version": self.framework_version,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ModelMetadata:
        if not isinstance(value, Mapping):
            raise TypeError("model metadata must be a mapping")
        try:
            raw_names = value["feature_names"]
            raw_identity = value["data_identity"]
            if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
                raise TypeError("feature_names must be a sequence")
            if any(not isinstance(item, str) for item in raw_names):
                raise TypeError("feature_names may contain only strings")
            if not isinstance(raw_identity, Mapping):
                raise TypeError("data_identity must be a mapping")
            snapshot_started = datetime.fromisoformat(
                _non_blank(
                    raw_identity["snapshot_started_at_utc"],
                    "snapshot_started_at_utc",
                )
            )
            return cls(
                schema_version=_non_blank(value["schema_version"], "schema_version"),
                model_id=_non_blank(value["model_id"], "model_id"),
                model_class_name=_non_blank(value["model_class_name"], "model_class_name"),
                model_parameters_json=_non_blank(
                    value["model_parameters_json"], "model_parameters_json"
                ),
                training_parameters_json=_non_blank(
                    value["training_parameters_json"], "training_parameters_json"
                ),
                feature_names=tuple(raw_names),
                feature_fingerprint=_non_blank(value["feature_fingerprint"], "feature_fingerprint"),
                label_name=_non_blank(value["label_name"], "label_name"),
                train_range=DateRange.from_mapping(
                    _mapping_value(value["train_range"], "train_range"), "train_range"
                ),
                valid_range=DateRange.from_mapping(
                    _mapping_value(value["valid_range"], "valid_range"), "valid_range"
                ),
                test_range=DateRange.from_mapping(
                    _mapping_value(value["test_range"], "test_range"), "test_range"
                ),
                random_seed=_integer_value(value["random_seed"], "random_seed"),
                data_identity=ModelDataIdentity(
                    dataset_version=_non_blank(raw_identity["dataset_version"], "dataset_version"),
                    manifest_sha256=_non_blank(raw_identity["manifest_sha256"], "manifest_sha256"),
                    snapshot_started_at_utc=snapshot_started,
                ),
                trained_through=date.fromisoformat(
                    _non_blank(value["trained_through"], "trained_through")
                ),
                framework_name=_non_blank(value["framework_name"], "framework_name"),
                framework_version=_non_blank(value["framework_version"], "framework_version"),
            )
        except KeyError as exc:
            raise ValueError("model metadata is incomplete") from exc


def _mapping_value(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _integer_value(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be int")
    return value


@runtime_checkable
class FeatureBuilder(Protocol):
    """User-pluggable, label-free feature calculation visible through D."""

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def required_history_trading_days(self) -> int: ...

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> Sequence[Decimal] | None: ...


@runtime_checkable
class TorchModelFactory(Protocol):
    """Framework-lazy factory for one torch.nn.Module architecture."""

    @property
    def model_id(self) -> str: ...

    @property
    def model_class_name(self) -> str: ...

    @property
    def model_parameters(self) -> Mapping[str, object]: ...

    def create(self, *, input_dim: int, seed: int) -> object: ...


@runtime_checkable
class PredictorBundle(Protocol):
    """Inference-only bundle consumed by the generic daily model strategy."""

    @property
    def metadata(self) -> ModelMetadata: ...

    def predict(self, records: Sequence[FeatureRecord]) -> tuple[PredictionRecord, ...]: ...


class Dataset(Protocol):
    """Minimal Qlib-style dataset interface."""

    feature_names: tuple[str, ...]
    label_name: str
    train: tuple[LabeledRecord, ...]
    valid: tuple[LabeledRecord, ...]
    test: tuple[LabeledRecord, ...]


class Model(Protocol):
    """Minimal Qlib-style fit/predict/save/load model interface."""

    @property
    def bundle(self) -> PredictorBundle | None: ...

    def fit(self, dataset: DatasetSplits) -> object: ...

    def save(self, path: object) -> object: ...

    def load(self, path: object, *, dataset: DatasetSplits) -> PredictorBundle: ...


class ModelArtifactWriter(Protocol):
    def write_json_artifact(self, filename: str, payload: Mapping[str, object]) -> object: ...

    def write_csv_artifact(
        self,
        filename: str,
        *,
        fieldnames: Sequence[str],
        rows: Sequence[Mapping[str, object]],
    ) -> object: ...


class Record(Protocol):
    """Qlib-style record boundary for model-specific local artifacts."""

    def write(self, writer: ModelArtifactWriter) -> None: ...


def validate_feature_builder(builder: FeatureBuilder) -> tuple[tuple[str, ...], int]:
    if not isinstance(builder, FeatureBuilder):
        raise TypeError("feature_builder must satisfy FeatureBuilder")
    names = validate_feature_names(builder.feature_names)
    lookback = builder.required_history_trading_days
    if type(lookback) is not int or lookback <= 0:
        raise ValueError("required_history_trading_days must be a positive integer")
    return names, lookback


def build_feature_record(
    *,
    builder: FeatureBuilder,
    symbol: str,
    signal_date: date,
    history: Sequence[MarketBarView],
) -> FeatureRecord | None:
    """Validate one plugin call and freeze its result without future views."""

    names, lookback = validate_feature_builder(builder)
    canonical = normalize_symbol(symbol)
    signal = _plain_date(signal_date, "signal_date")
    supplied_history = cast(object, history)
    if isinstance(supplied_history, (str, bytes)) or not isinstance(supplied_history, Sequence):
        raise TypeError("history must be a sequence")
    ordered = tuple(sorted(history, key=lambda view: view.trade_date))
    if any(not isinstance(view, MarketBarView) for view in ordered):
        raise TypeError("history may contain only MarketBarView values")
    if not ordered or ordered[-1].trade_date != signal:
        return None
    if any(view.symbol != canonical for view in ordered):
        raise ValueError("history contains a different symbol")
    if any(view.trade_date > signal for view in ordered):
        raise ValueError("history contains a future view")
    dates = tuple(view.trade_date for view in ordered)
    if len(dates) != len(set(dates)):
        raise ValueError("history contains duplicate daily views")
    visible = ordered[-lookback:]
    values = builder.build_features(symbol=canonical, signal_date=signal, history=visible)
    if values is None:
        return None
    supplied_values = cast(object, values)
    if isinstance(supplied_values, (str, bytes)) or not isinstance(supplied_values, Sequence):
        raise TypeError("FeatureBuilder must return a sequence or None")
    features = tuple(values)
    if len(features) != len(names):
        raise ValueError("FeatureBuilder returned the wrong feature width")
    return FeatureRecord(key=SampleKey(signal_date=signal, symbol=canonical), features=features)


def feature_records_for_signal(
    *,
    builder: FeatureBuilder,
    market_views: Sequence[MarketBarView],
    signal_date: date,
) -> tuple[FeatureRecord, ...]:
    """Build label-free records for every symbol visible through signal D."""

    signal = _plain_date(signal_date, "signal_date")
    supplied_views = cast(object, market_views)
    if isinstance(supplied_views, (str, bytes)) or not isinstance(supplied_views, Sequence):
        raise TypeError("market_views must be a sequence")
    by_symbol: dict[str, dict[date, MarketBarView]] = {}
    for view in market_views:
        if not isinstance(view, MarketBarView):
            raise TypeError("market_views may contain only MarketBarView values")
        if view.trade_date > signal:
            raise ValueError("market_views contains data after signal_date")
        rows = by_symbol.setdefault(view.symbol, {})
        if view.trade_date in rows:
            raise ValueError(f"duplicate daily view for {view.symbol} on {view.trade_date}")
        rows[view.trade_date] = view
    records: list[FeatureRecord] = []
    for symbol in sorted(by_symbol):
        history = tuple(by_symbol[symbol][day] for day in sorted(by_symbol[symbol]))
        record = build_feature_record(
            builder=builder,
            symbol=symbol,
            signal_date=signal,
            history=history,
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def prediction_rows(
    predictions: Sequence[PredictionRecord],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    seen: set[SampleKey] = set()
    for prediction in sorted(predictions, key=lambda value: value.key):
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predictions may contain only PredictionRecord values")
        if prediction.key in seen:
            raise ValueError("predictions contain duplicate keys")
        seen.add(prediction.key)
        rows.append(
            MappingProxyType(
                {
                    "signal_date": prediction.key.signal_date.isoformat(),
                    "symbol": prediction.key.symbol,
                    "score": prediction.score,
                }
            )
        )
    return tuple(rows)


__all__ = [
    "DAILY_FORWARD_RETURN_LABEL",
    "MODEL_BUNDLE_SCHEMA_VERSION",
    "Dataset",
    "DatasetSplits",
    "DateRange",
    "FeatureBuilder",
    "FeatureRecord",
    "LabeledRecord",
    "Model",
    "ModelArtifactWriter",
    "ModelDataIdentity",
    "ModelMetadata",
    "PredictionRecord",
    "PredictorBundle",
    "Record",
    "RegressionMetricReport",
    "SampleKey",
    "TorchModelFactory",
    "build_feature_record",
    "canonical_json",
    "feature_fingerprint",
    "feature_records_for_signal",
    "prediction_rows",
    "validate_feature_builder",
    "validate_feature_names",
]
