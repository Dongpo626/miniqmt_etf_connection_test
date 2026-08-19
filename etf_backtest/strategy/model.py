"""Fixed user Model extension contract and its shared concrete workflow."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType, ModuleType

from etf_backtest.strategy.model_contracts import (
    DAILY_FORWARD_RETURN_LABEL,
    DatasetSplits,
    DateRange,
    FeatureBuilder,
    FeatureRecord,
    LabeledRecord,
    ModelDataIdentity,
    ModelMetadata,
    PredictionRecord,
    PredictorBundle,
    SampleKey,
    TorchModelFactory,
    canonical_json,
    validate_feature_builder,
)
from etf_backtest.strategy.model_runtime import DailyModelStrategy
from etf_backtest.strategy.model_training import (
    DailyFourFactorFeatureBuilder,
    DailyTorchBundle,
    DailyTorchDatasetBuilder,
    DailyTorchWorkflow,
    DailyTorchWorkflowResult,
    TorchTrainingConfig,
    TorchUnavailableError,
)
from etf_backtest.strategy.portfolio import (
    AllocationFunction,
    CustomPortfolio,
    ModelPortfolioPolicy,
    PortfolioWeightInput,
    TopKPortfolio,
    WeightingMode,
)

_MODEL_MODULE_PREFIX = "_etf_backtest_user_model_"


class UserModelLoadError(ValueError):
    """A trusted local model file violates the controlled loading contract."""


def _settings_kwargs(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    parameters = dict(value)
    if any(not isinstance(key, str) or not key.isidentifier() for key in parameters):
        raise ValueError(f"{field_name} keys must be Python identifiers")
    try:
        canonical_json(parameters)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and JSON-compatible") from exc
    return MappingProxyType(dict(sorted(parameters.items())))


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Model-specific data, training and construction settings owned by ``model.py``.

    The backtest test range still comes from the common experiment YAML.  Users
    may edit every field here; the shared Workflow continues to own labels,
    train-only scaling, optimization, early stopping and daily inference.
    """

    train_range: DateRange = field(
        default_factory=lambda: DateRange(date(2021, 1, 1), date(2022, 12, 31))
    )
    valid_range: DateRange = field(
        default_factory=lambda: DateRange(date(2023, 1, 1), date(2023, 12, 31))
    )
    portfolio: ModelPortfolioPolicy = field(default_factory=TopKPortfolio)
    training: TorchTrainingConfig = field(default_factory=TorchTrainingConfig)
    feature_kwargs: Mapping[str, object] = field(default_factory=dict)
    model_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.train_range, DateRange):
            raise TypeError("train_range must be DateRange")
        if not isinstance(self.valid_range, DateRange):
            raise TypeError("valid_range must be DateRange")
        if self.train_range.end_date >= self.valid_range.start_date:
            raise ValueError(
                "train_range and valid_range must be chronological and non-overlapping"
            )
        if not isinstance(self.training, TorchTrainingConfig):
            raise TypeError("training must be TorchTrainingConfig")
        if not isinstance(self.portfolio, ModelPortfolioPolicy):
            raise TypeError("portfolio must satisfy ModelPortfolioPolicy")
        if (
            not isinstance(self.portfolio.max_total_weight, Decimal)
            or not self.portfolio.max_total_weight.is_finite()
            or not Decimal("0") < self.portfolio.max_total_weight <= Decimal("1")
        ):
            raise ValueError("portfolio max_total_weight must be in (0, 1]")
        try:
            canonical_json(self.portfolio.resolved_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError("portfolio provenance must be finite and JSON-compatible") from exc
        object.__setattr__(
            self,
            "feature_kwargs",
            _settings_kwargs(self.feature_kwargs, "feature_kwargs"),
        )
        object.__setattr__(
            self,
            "model_kwargs",
            _settings_kwargs(self.model_kwargs, "model_kwargs"),
        )

    def resolved_dict(self) -> dict[str, object]:
        """Return the exact code-owned values written to workflow provenance."""

        return {
            "train_range": self.train_range.to_dict(),
            "valid_range": self.valid_range.to_dict(),
            "portfolio": self.portfolio.resolved_dict(),
            "training": self.training.to_parameters(),
            "feature_kwargs": dict(self.feature_kwargs),
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(frozen=True, slots=True)
class LoadedModelComponents:
    """Auditable pair returned by :func:`load_user_model_components`."""

    settings: ModelSettings
    feature_builder: FeatureBuilder
    model_factory: TorchModelFactory
    source_path: Path
    source_sha256: str


def load_user_model_components(
    path: str | Path,
    *,
    allowed_root: str | Path,
) -> LoadedModelComponents:
    """Load explicitly named components from one trusted local Python file.

    The selected file must define one module-level ``MODEL_SETTINGS`` value.
    Its constructor kwargs are passed to the selected feature and model classes.
    This is a controlled loader, not a sandbox: executing an allowed file grants
    it normal Python process permissions.
    """

    root = _resolved_model_root(allowed_root)
    source_path = _resolved_model_source(path, root=root)
    feature_name, model_name = "Features", "Model"
    source = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    module_name = (
        f"{_MODEL_MODULE_PREFIX}"
        f"{hashlib.sha256(str(source_path).encode('utf-8') + b'\0' + source).hexdigest()[:24]}"
    )
    module = _execute_model_module(
        module_name=module_name,
        source_path=source_path,
        source=source,
    )
    feature_type = _selected_component_class(module, feature_name)
    model_type = _selected_component_class(module, model_name)
    settings = vars(module).get("MODEL_SETTINGS")
    if not isinstance(settings, ModelSettings):
        raise UserModelLoadError("model file must define MODEL_SETTINGS as ModelSettings")
    feature_instance = _instantiate_component(
        feature_type,
        settings.feature_kwargs,
        component_name=feature_name,
    )
    model_instance = _instantiate_component(
        model_type,
        settings.model_kwargs,
        component_name=model_name,
    )
    if not isinstance(feature_instance, FeatureBuilder):
        raise UserModelLoadError(f"{feature_name} must satisfy FeatureBuilder")
    if not isinstance(model_instance, TorchModelFactory):
        raise UserModelLoadError(f"{model_name} must satisfy TorchModelFactory")
    try:
        validate_feature_builder(feature_instance)
        canonical_json(model_instance.model_parameters)
    except (TypeError, ValueError) as exc:
        raise UserModelLoadError(f"loaded model component validation failed: {exc}") from exc
    return LoadedModelComponents(
        settings=settings,
        feature_builder=feature_instance,
        model_factory=model_instance,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _resolved_model_root(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UserModelLoadError("allowed_root must be an existing directory") from exc
    if not root.is_dir():
        raise UserModelLoadError("allowed_root must be an existing directory")
    return root


def _resolved_model_source(value: str | Path, *, root: Path) -> Path:
    supplied = Path(value)
    unresolved = supplied if supplied.is_absolute() else root / supplied
    try:
        source_path = unresolved.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UserModelLoadError("user Model file does not exist") from exc
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise UserModelLoadError("user Model file must stay inside allowed_root") from exc
    if source_path.suffix.casefold() != ".py" or not source_path.is_file():
        raise UserModelLoadError("user Model source must be one regular .py file")
    return source_path


def _execute_model_module(*, module_name: str, source_path: Path, source: bytes) -> ModuleType:
    del source
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise UserModelLoadError(f"cannot load model module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise UserModelLoadError(
            f"user Model module execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
    return module


def _selected_component_class(module: ModuleType, class_name: str) -> type[object]:
    value = vars(module).get(class_name)
    if not isinstance(value, type):
        raise UserModelLoadError(f"model.py must define class {class_name}")
    return value


def _instantiate_component(
    component_type: type[object],
    parameters: Mapping[str, object],
    *,
    component_name: str,
) -> object:
    try:
        return component_type(**parameters)
    except Exception as exc:
        raise UserModelLoadError(
            f"{component_name} initialization failed: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "DAILY_FORWARD_RETURN_LABEL",
    "AllocationFunction",
    "CustomPortfolio",
    "DailyFourFactorFeatureBuilder",
    "DailyModelStrategy",
    "DailyTorchBundle",
    "DailyTorchDatasetBuilder",
    "DailyTorchWorkflow",
    "DailyTorchWorkflowResult",
    "DatasetSplits",
    "DateRange",
    "FeatureBuilder",
    "FeatureRecord",
    "LabeledRecord",
    "LoadedModelComponents",
    "ModelDataIdentity",
    "ModelMetadata",
    "ModelPortfolioPolicy",
    "ModelSettings",
    "PortfolioWeightInput",
    "PredictionRecord",
    "PredictorBundle",
    "SampleKey",
    "TopKPortfolio",
    "TorchModelFactory",
    "TorchTrainingConfig",
    "TorchUnavailableError",
    "UserModelLoadError",
    "WeightingMode",
    "load_user_model_components",
]
