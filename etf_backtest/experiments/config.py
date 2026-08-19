"""Strict, secret-free user experiment configuration.

The public experiment document intentionally contains only choices owned by a
strategy author.  Database credentials, snapshot identity, and effective rule
resources live in :class:`SystemSettings` and are merged only when a concrete
``BacktestConfig`` is built.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from etf_backtest.config.schema import (
    CALENDAR_POLICY,
    BacktestConfig,
    DatabaseConfig,
    DataSnapshotConfig,
    FeeConfig,
    ModelStrategyConfig,
    RuleStrategyConfig,
    SlippageConfig,
    UniverseConfig,
)

_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_FORBIDDEN_NAME_CHARACTERS = frozenset('<>:"/\\|?*')


class _StrictExperimentModel(BaseModel):
    """Immutable, typo-rejecting base shared by experiment documents."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


def _reject_float(value: object, field_name: str) -> object:
    if isinstance(value, float):
        raise TypeError(f"{field_name} must be supplied as decimal text")
    return value


def _safe_component(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if normalized != value or not normalized:
        raise ValueError(f"{field_name} must be nonblank without surrounding whitespace")
    if len(normalized) > 80:
        raise ValueError(f"{field_name} must not exceed 80 characters")
    if normalized in {".", ".."} or any(
        character in _FORBIDDEN_NAME_CHARACTERS or ord(character) < 32 for character in normalized
    ):
        raise ValueError(f"{field_name} must be one safe path component")
    if normalized.endswith((".", " ")):
        raise ValueError(f"{field_name} must not end with a dot or space")
    if normalized.partition(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError(f"{field_name} uses a reserved Windows filename")
    return normalized


def safe_relative_path(
    value: object,
    field_name: str,
    *,
    suffix: str | None = None,
) -> Path:
    """Return a normalized project-relative path that cannot escape its root."""

    if not isinstance(value, str | Path):
        raise TypeError(f"{field_name} must be a path string")
    normalized = Path(str(value).replace("\\", "/"))
    if (
        not normalized.parts
        or normalized.is_absolute()
        or normalized.drive
        or any(part in {"", ".", ".."} for part in normalized.parts)
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    for part in normalized.parts:
        _safe_component(part, field_name)
    if suffix is not None and normalized.suffix.casefold() != suffix.casefold():
        raise ValueError(f"{field_name} must end in {suffix}")
    return normalized


class SystemSettings(_StrictExperimentModel):
    """Operator-owned data, execution defaults, output root and rule resources."""

    database: DatabaseConfig
    data_snapshot: DataSnapshotConfig = Field(default_factory=DataSnapshotConfig)
    limit_rules_csv: Path = Path("resources/limit_rules/etf_price_limit_20pct.csv")
    limit_rules_manifest: Path = Path("resources/limit_rules/manifest.json")
    calendar_policy: Literal["SSE_FOR_ALL"] = CALENDAR_POLICY
    fee: FeeConfig = Field(default_factory=FeeConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    volume_participation_rate: Decimal = Decimal("0.20")
    runs_dir: Path = Path("runs")

    @field_validator("limit_rules_csv", mode="before")
    @classmethod
    def _rule_csv_path(cls, value: object) -> Path:
        return safe_relative_path(value, "limit_rules_csv", suffix=".csv")

    @field_validator("limit_rules_manifest", mode="before")
    @classmethod
    def _rule_manifest_path(cls, value: object) -> Path:
        return safe_relative_path(value, "limit_rules_manifest", suffix=".json")

    @field_validator("volume_participation_rate", mode="before")
    @classmethod
    def _volume_rate(cls, value: object) -> object:
        return _reject_float(value, "volume_participation_rate")

    @field_validator("runs_dir", mode="before")
    @classmethod
    def _runs_path(cls, value: object) -> Path:
        return safe_relative_path(value, "runs_dir")

    @model_validator(mode="after")
    def _valid_execution_defaults(self) -> Self:
        if not self.volume_participation_rate.is_finite() or not Decimal(
            "0"
        ) < self.volume_participation_rate <= Decimal("1"):
            raise ValueError("volume_participation_rate must be in (0, 1]")
        return self

    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> Self:
        """Project system-owned values from an already validated run config."""

        if not isinstance(config, BacktestConfig):
            raise TypeError("config must be BacktestConfig")
        return cls(
            database=config.database,
            data_snapshot=config.data_snapshot,
            limit_rules_csv=config.limit_rules_csv,
            limit_rules_manifest=config.limit_rules_manifest,
            calendar_policy=config.calendar_policy,
            fee=config.fee,
            slippage=config.slippage,
            volume_participation_rate=config.volume_participation_rate,
            runs_dir=config.runs_dir,
        )


class UserExperimentConfig(_StrictExperimentModel):
    """Common experiment inputs; all strategy-specific settings live in Python."""

    name: str
    start_date: date
    end_date: date
    initial_cash: Decimal = Decimal("1000000")
    universe: UniverseConfig
    case: Literal["rule", "model"]

    @field_validator("name", mode="before")
    @classmethod
    def _experiment_name(cls, value: object) -> str:
        return _safe_component(value, "name")

    @field_validator("initial_cash", mode="before")
    @classmethod
    def _decimal_core_values(cls, value: object, info: object) -> object:
        field_name = getattr(info, "field_name", "Decimal")
        return _reject_float(value, field_name)

    @model_validator(mode="after")
    def _valid_experiment(self) -> Self:
        if self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date")
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        return self

    def build_case(
        self,
        system: SystemSettings,
        *,
        strategy: RuleStrategyConfig | ModelStrategyConfig,
    ) -> BacktestConfig:
        """Merge code-resolved strategy settings with common and system inputs."""

        if not isinstance(system, SystemSettings):
            raise TypeError("system must be SystemSettings")
        if self.case == "rule" and not isinstance(strategy, RuleStrategyConfig):
            raise TypeError("rule case requires RuleStrategyConfig")
        if self.case == "model" and not isinstance(strategy, ModelStrategyConfig):
            raise TypeError("model case requires ModelStrategyConfig")
        return BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            initial_cash=self.initial_cash,
            database=system.database,
            data_snapshot=system.data_snapshot,
            universe=self.universe,
            strategy=strategy,
            fee=system.fee,
            slippage=system.slippage,
            volume_participation_rate=system.volume_participation_rate,
            runs_dir=system.runs_dir,
            limit_rules_csv=system.limit_rules_csv,
            limit_rules_manifest=system.limit_rules_manifest,
            calendar_policy=system.calendar_policy,
        )


def load_user_experiment_config(path: Path) -> UserExperimentConfig:
    """Load one UTF-8 user YAML document without system-setting translation."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment configuration root must be a mapping")
    return UserExperimentConfig.model_validate(payload)


def load_system_settings(path: Path) -> SystemSettings:
    """Load one UTF-8 operator-owned system settings YAML document."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("system settings root must be a mapping")
    return SystemSettings.model_validate(payload)


__all__ = [
    "SystemSettings",
    "UserExperimentConfig",
    "load_system_settings",
    "load_user_experiment_config",
    "safe_relative_path",
]
