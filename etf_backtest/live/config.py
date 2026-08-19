"""Strict Rule PAPER configuration with a secret-free stable identity."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, StrictInt, field_validator, model_validator

from etf_backtest.experiments.config import safe_relative_path


class _LiveModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)


class LiveDeploymentConfig(_LiveModel):
    deployment_id: str
    mode: Literal["PAPER"]
    case: Literal["rule", "model"]
    experiment_path: Path
    system_path: Path
    account_id_env: str
    account_type: Literal["STOCK"] = "STOCK"
    schedule_anchor_date: date

    @field_validator("experiment_path", "system_path", mode="before")
    @classmethod
    def _project_path(cls, value: object, info: object) -> Path:
        return safe_relative_path(value, str(getattr(info, "field_name", "path")))

    @field_validator("deployment_id", "account_id_env")
    @classmethod
    def _text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("deployment text fields must not be blank")
        return normalized

    def account_id(self) -> str:
        try:
            value = os.environ[self.account_id_env].strip()
        except KeyError:
            raise ValueError(
                f"account environment variable is not set: {self.account_id_env}"
            ) from None
        if not value:
            raise ValueError("configured account environment variable is blank")
        return value


class MiniQmtConfig(_LiveModel):
    userdata_path: Path
    session_id: StrictInt
    reconnect_interval_seconds: StrictInt

    @model_validator(mode="after")
    def _positive_values(self) -> Self:
        if self.session_id <= 0 or self.reconnect_interval_seconds <= 0:
            raise ValueError("MiniQMT session and reconnect interval must be positive")
        return self


class SignalConfig(_LiveModel):
    run_time: time


class ExecutionConfig(_LiveModel):
    policy: Literal["NEAR_CLOSE_LIMIT"]
    submit_start: time
    stop_new_orders: time
    cancel_open_orders: time
    price_offset_ticks: StrictInt
    quote_stale_seconds: StrictInt
    lot_size: StrictInt
    order_type: Literal["FIX_PRICE"]

    @model_validator(mode="after")
    def _validate_execution(self) -> Self:
        if not self.submit_start < self.stop_new_orders < self.cancel_open_orders:
            raise ValueError(
                "execution times must satisfy submit_start < stop_new_orders < cancel_open_orders"
            )
        if self.price_offset_ticks < 0 or self.quote_stale_seconds <= 0 or self.lot_size <= 0:
            raise ValueError("execution offsets, staleness and lot size are invalid")
        return self


class LiveStateDatabaseConfig(_LiveModel):
    host: str = "127.0.0.1"
    port: StrictInt = 3306
    database: str
    user: str
    password_env: str

    @model_validator(mode="after")
    def _validate_database(self) -> Self:
        if not 1 <= self.port <= 65535:
            raise ValueError("state database port must be between 1 and 65535")
        if not all(value.strip() for value in (self.host, self.database, self.user, self.password_env)):
            raise ValueError("state database text fields must not be blank")
        return self

    def resolved_password(self) -> str:
        try:
            return os.environ[self.password_env]
        except KeyError:
            raise ValueError(
                f"state database password environment variable is not set: {self.password_env}"
            ) from None


class LiveRiskConfig(_LiveModel):
    max_single_order_notional: Decimal
    max_daily_order_notional: Decimal
    min_order_notional: Decimal
    max_total_target_weight: Decimal

    @field_validator("*", mode="before")
    @classmethod
    def _reject_float(cls, value: object) -> object:
        if isinstance(value, float):
            raise TypeError("risk Decimal values must be supplied as decimal text")
        return value

    @model_validator(mode="after")
    def _positive_risk(self) -> Self:
        values = (
            self.max_single_order_notional,
            self.max_daily_order_notional,
            self.min_order_notional,
            self.max_total_target_weight,
        )
        if any(not value.is_finite() or value <= 0 for value in values):
            raise ValueError("risk limits must be finite and positive")
        if self.max_total_target_weight > 1:
            raise ValueError("max_total_target_weight must not exceed one")
        return self


class ModelLiveConfig(_LiveModel):
    bundle_path: Path

    @field_validator("bundle_path", mode="before")
    @classmethod
    def _bundle_path(cls, value: object) -> Path:
        return safe_relative_path(value, "model.bundle_path", suffix=".pt")


class LiveConfig(_LiveModel):
    config_version: Literal["1.0"]
    deployment: LiveDeploymentConfig
    miniqmt: MiniQmtConfig
    signal: SignalConfig
    execution: ExecutionConfig
    state_database: LiveStateDatabaseConfig
    risk: LiveRiskConfig
    model: ModelLiveConfig | None = None

    @model_validator(mode="after")
    def _validate_case(self) -> Self:
        if self.deployment.case == "rule" and self.model is not None:
            raise ValueError("Rule live configuration must not contain model")
        if self.deployment.case == "model" and self.model is None:
            raise ValueError("Model live configuration requires model.bundle_path")
        return self

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def project_path(self, path: Path, project_root: Path) -> Path:
        root = Path(project_root).resolve()
        result = (root / path).resolve()
        if not result.is_relative_to(root):
            raise ValueError("configured project path escapes the project root")
        return result


def load_live_config(path: Path) -> LiveConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("live configuration root must be a mapping")
    return LiveConfig.model_validate(payload)


__all__ = ["LiveConfig", "LiveStateDatabaseConfig", "ModelLiveConfig", "load_live_config"]
