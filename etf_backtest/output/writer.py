"""Atomic writer for the small, fixed public backtest output contract."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import uuid4

from etf_backtest.core.account import DailySnapshot
from etf_backtest.core.engine import BacktestResult
from etf_backtest.core.order import FillResult, Order, RuleCheckResult
from etf_backtest.evaluation.backtest_metrics import BacktestMetricResult

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_SECRET_FRAGMENTS = ("password", "passwd", "secret", "token", "credential", "api_key")
_PLOT_FILENAMES = frozenset(("cumulative_return.png", "drawdown.png", "cash.png"))
_COMMON_FILENAMES = frozenset(
    (
        "run.json",
        "daily_nav.csv",
        "daily_positions.csv",
        "orders.csv",
        "trades.csv",
        "metrics.json",
        "final_account.json",
        *_PLOT_FILENAMES,
    )
)
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    """The only two Model-specific public artifacts."""

    bundle_path: Path
    predictions: Sequence[Mapping[str, object]]


def _safe_run_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("run_id must be str")
    normalized = value.strip()
    if _RUN_ID.fullmatch(normalized) is None:
        raise ValueError("run_id contains unsafe path characters or has invalid length")
    return normalized


def _is_secret_key(key: object) -> bool:
    folded = str(key).casefold()
    return any(fragment in folded for fragment in _SECRET_FRAGMENTS)


def _json_value(value: object, *, key_hint: object | None = None) -> object:
    if key_hint is not None and _is_secret_key(key_hint):
        return "***"
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("output contains a non-finite float")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("output contains a non-finite Decimal")
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(nested, key_hint=key)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    if callable(getattr(value, "get_secret_value", None)):
        return "***"
    raise TypeError(f"unsupported output value: {type(value).__qualname__}")


def _secret_values(value: object, *, key_hint: object | None = None) -> set[str]:
    if key_hint is not None and _is_secret_key(key_hint):
        if isinstance(value, str) and value:
            return {value}
        reveal = getattr(value, "get_secret_value", None)
        if callable(reveal):
            secret = reveal()
            return {secret} if isinstance(secret, str) and secret else set()
        return set()
    if isinstance(value, Mapping):
        secrets: set[str] = set()
        for key, nested in value.items():
            secrets.update(_secret_values(nested, key_hint=key))
        return secrets
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        secrets = set()
        for nested in value:
            secrets.update(_secret_values(nested))
        return secrets
    return set()


def _json_text(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _write_json(directory: Path, filename: str, payload: Mapping[str, object]) -> None:
    (directory / filename).write_text(_json_text(payload), encoding="utf-8", newline="\n")


def _write_csv(
    directory: Path,
    filename: str,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    with (directory / filename).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_value(row.get(name)) for name in fieldnames})


def _daily_row(snapshot: DailySnapshot) -> Mapping[str, object]:
    return {
        "trade_date": snapshot.trade_date,
        "cash": snapshot.cash,
        "market_value": snapshot.market_value,
        "total_asset": snapshot.total_asset,
    }


def _position_rows(snapshot: DailySnapshot) -> tuple[Mapping[str, object], ...]:
    account = snapshot.account_snapshot
    rows: list[Mapping[str, object]] = []
    for symbol, position in account.positions.items():
        market_value = account.position_values.get(symbol, _ZERO)
        rows.append(
            {
                "trade_date": snapshot.trade_date,
                "symbol": symbol,
                "turnover_rule": position.turnover_rule,
                "total_quantity": position.total_quantity,
                "available_quantity": position.available_quantity,
                "today_buy_quantity": position.today_buy_quantity,
                "raw_close": account.mark_close_prices.get(symbol),
                "market_value": market_value,
                "portfolio_weight": (
                    market_value / account.total_asset if account.total_asset > _ZERO else _ZERO
                ),
            }
        )
    return tuple(rows)


def _order_rows(result: BacktestResult) -> tuple[Mapping[str, object], ...]:
    approvals: dict[str, RuleCheckResult] = {}
    for approval in result.approvals:
        if approval.order_id in approvals:
            raise ValueError("each order must have exactly one approval")
        approvals[approval.order_id] = approval
    if set(approvals) != {order.order_id for order in result.orders}:
        raise ValueError("each order must have exactly one approval")
    return tuple(_order_row(order, approvals[order.order_id]) for order in result.orders)


def _order_row(order: Order, approval: RuleCheckResult) -> Mapping[str, object]:
    return {
        "order_id": order.order_id,
        "signal_date": order.signal_date,
        "execution_date": order.execution_date,
        "symbol": order.symbol,
        "side": order.side,
        "target_value_gap": order.target_value_gap,
        "requested_quantity": order.requested_quantity,
        "approved_quantity": approval.approved_quantity,
        "passed": approval.passed,
        "reason_code": approval.reason_code,
        "message": approval.message,
        "base_trade_price": approval.base_trade_price,
        "price_limit_down": approval.price_limit_down,
        "price_limit_up": approval.price_limit_up,
        "price_limit_source": approval.price_limit_source,
        "price_limit_fallback_reason": approval.price_limit_fallback_reason,
    }


def _trade_row(fill: FillResult) -> Mapping[str, object]:
    return {
        "order_id": fill.order_id,
        "signal_date": fill.signal_date,
        "execution_date": fill.execution_date,
        "trade_time": fill.trade_time,
        "source_record_key": fill.source_record_key,
        "symbol": fill.symbol,
        "side": fill.side,
        "requested_quantity": fill.requested_quantity,
        "fill_quantity": fill.fill_quantity,
        "base_trade_price": fill.base_trade_price,
        "fill_price": fill.fill_price,
        "trade_amount": fill.trade_amount,
        "fee": fill.fee,
        "status": fill.status,
        "price_source": fill.price_source,
    }


def _metrics_payload(metrics: BacktestMetricResult) -> Mapping[str, object]:
    return {
        "daily_returns": metrics.daily_returns,
        "cumulative_return": metrics.cumulative_return,
        "annualized_return": metrics.annualized_return,
        "max_drawdown": metrics.max_drawdown,
        "sharpe": metrics.sharpe,
        "sharpe_reason": metrics.sharpe_reason,
        "total_fee": metrics.total_fee,
        "total_slippage_cost": metrics.total_slippage_cost,
        "total_transaction_cost": metrics.total_transaction_cost,
        "transaction_cost_rate": metrics.transaction_cost_rate,
        "trade_count": metrics.trade_count,
        "turnover": metrics.turnover,
    }


def _final_account(snapshot: DailySnapshot) -> Mapping[str, object]:
    return {
        **_daily_row(snapshot),
        "positions": tuple(
            {key: value for key, value in row.items() if key != "trade_date"}
            for row in _position_rows(snapshot)
        ),
    }


class BacktestOutputWriter:
    """Write one committed result directory with no optional research files."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = Path(runs_dir).resolve()

    def write_success(
        self,
        *,
        run_id: str,
        run_metadata: Mapping[str, object],
        result: BacktestResult,
        metrics: BacktestMetricResult,
        plots: Mapping[str, bytes],
        model_artifacts: ModelArtifacts | None = None,
    ) -> Path:
        normalized_id = _safe_run_id(run_id)
        if not isinstance(run_metadata, Mapping):
            raise TypeError("run_metadata must be a mapping")
        if not isinstance(result, BacktestResult):
            raise TypeError("result must be BacktestResult")
        if not isinstance(metrics, BacktestMetricResult):
            raise TypeError("metrics must be BacktestMetricResult")
        if set(plots) != _PLOT_FILENAMES:
            raise ValueError("plot filenames must exactly match the fixed output contract")
        if any(
            not isinstance(content, bytes) or not content.startswith(b"\x89PNG\r\n\x1a\n")
            for content in plots.values()
        ):
            raise ValueError("each plot must contain PNG bytes")
        if not result.daily_snapshots:
            raise ValueError("a successful run requires daily snapshots")
        order_rows = _order_rows(result)
        expected = set(_COMMON_FILENAMES)
        if model_artifacts is not None:
            expected.update(("model_bundle.pt", "predictions.csv"))

        def populate(directory: Path) -> None:
            self._write_result_files(
                directory,
                result=result,
                metrics=metrics,
                plots=plots,
                order_rows=order_rows,
                model_artifacts=model_artifacts,
            )
            if {path.name for path in directory.iterdir()} != expected - {"run.json"}:
                raise RuntimeError("staged output files do not match the fixed contract")
            _write_json(
                directory,
                "run.json",
                {
                    "run_id": normalized_id,
                    "status": "success",
                    "daily_count": len(result.daily_snapshots),
                    "order_count": len(result.orders),
                    "trade_count": len(result.fills),
                    "metadata": dict(run_metadata),
                },
            )

        return self._commit(normalized_id, populate)

    def write_failure(
        self,
        *,
        run_id: str,
        run_metadata: Mapping[str, object],
        error: Exception,
    ) -> Path:
        normalized_id = _safe_run_id(run_id)
        if not isinstance(run_metadata, Mapping):
            raise TypeError("run_metadata must be a mapping")
        if not isinstance(error, Exception):
            raise TypeError("error must be Exception")
        message = str(error)
        for secret in sorted(_secret_values(run_metadata), key=len, reverse=True):
            message = message.replace(secret, "***")

        def populate(directory: Path) -> None:
            _write_json(
                directory,
                "run.json",
                {
                    "run_id": normalized_id,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": message,
                    "metadata": dict(run_metadata),
                },
            )

        return self._commit(normalized_id, populate)

    def _write_result_files(
        self,
        directory: Path,
        *,
        result: BacktestResult,
        metrics: BacktestMetricResult,
        plots: Mapping[str, bytes],
        order_rows: Sequence[Mapping[str, object]],
        model_artifacts: ModelArtifacts | None,
    ) -> None:
        ordered_daily = tuple(sorted(result.daily_snapshots, key=lambda row: row.trade_date))
        _write_csv(
            directory,
            "daily_nav.csv",
            ("trade_date", "cash", "market_value", "total_asset"),
            tuple(_daily_row(row) for row in ordered_daily),
        )
        _write_csv(
            directory,
            "daily_positions.csv",
            (
                "trade_date",
                "symbol",
                "turnover_rule",
                "total_quantity",
                "available_quantity",
                "today_buy_quantity",
                "raw_close",
                "market_value",
                "portfolio_weight",
            ),
            tuple(row for daily in ordered_daily for row in _position_rows(daily)),
        )
        _write_csv(
            directory,
            "orders.csv",
            (
                "order_id",
                "signal_date",
                "execution_date",
                "symbol",
                "side",
                "target_value_gap",
                "requested_quantity",
                "approved_quantity",
                "passed",
                "reason_code",
                "message",
                "base_trade_price",
                "price_limit_down",
                "price_limit_up",
                "price_limit_source",
                "price_limit_fallback_reason",
            ),
            order_rows,
        )
        _write_csv(
            directory,
            "trades.csv",
            (
                "order_id",
                "signal_date",
                "execution_date",
                "trade_time",
                "source_record_key",
                "symbol",
                "side",
                "requested_quantity",
                "fill_quantity",
                "base_trade_price",
                "fill_price",
                "trade_amount",
                "fee",
                "status",
                "price_source",
            ),
            tuple(_trade_row(fill) for fill in result.fills),
        )
        _write_json(directory, "metrics.json", _metrics_payload(metrics))
        _write_json(directory, "final_account.json", _final_account(ordered_daily[-1]))
        for filename, content in plots.items():
            (directory / filename).write_bytes(content)
        if model_artifacts is not None:
            bundle = Path(model_artifacts.bundle_path).resolve(strict=True)
            if not bundle.is_file():
                raise FileNotFoundError(bundle)
            shutil.copyfile(bundle, directory / "model_bundle.pt")
            predictions = tuple(model_artifacts.predictions)
            _write_csv(
                directory,
                "predictions.csv",
                ("signal_date", "symbol", "score"),
                predictions,
            )

    def _commit(self, run_id: str, populate: Callable[[Path], None]) -> Path:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        target = self._runs_dir / run_id
        if target.exists():
            raise FileExistsError(target)
        temporary = self._runs_dir / f".{run_id}.{uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            populate(temporary)
            temporary.rename(target)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        return target


__all__ = ["BacktestOutputWriter", "ModelArtifacts"]
