from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pytest

from etf_backtest.evaluation.backtest_metrics import (
    BacktestMetrics,
    DailyMetricRow,
    TradeMetricRow,
)
from etf_backtest.output.writer import BacktestOutputWriter
from tests.unit.core.conftest import DATES

_PNG = b"\x89PNG\r\n\x1a\nfixture"


def _metrics(result):
    return BacktestMetrics.calculate(
        initial_cash=Decimal("10000"),
        daily_rows=tuple(
            DailyMetricRow(
                trade_date=row.trade_date,
                cash=row.cash,
                market_value=row.market_value,
                total_asset=row.total_asset,
            )
            for row in result.daily_snapshots
        ),
        trade_rows=tuple(
            TradeMetricRow(
                trade_amount=fill.trade_amount,
                fee=fill.fee,
                base_trade_price=fill.base_trade_price,
                fill_price=fill.fill_price,
                fill_quantity=fill.fill_quantity,
            )
            for fill in result.fills
        ),
    )


@pytest.mark.unit
def test_success_writes_only_fixed_outputs_and_combines_order_approval(
    tmp_path: Path,
    engine_components,
) -> None:
    engine, *_ = engine_components
    result = engine.run(start_date=DATES[0], end_date=DATES[-1])
    plots = {
        "cumulative_return.png": _PNG,
        "drawdown.png": _PNG,
        "cash.png": _PNG,
    }

    run_dir = BacktestOutputWriter(tmp_path).write_success(
        run_id="run-1",
        run_metadata={
            "case": "rule",
            "database": {"user": "reader", "password": "must-not-leak"},
        },
        result=result,
        metrics=_metrics(result),
        plots=plots,
    )

    assert {path.name for path in run_dir.iterdir()} == {
        "run.json",
        "daily_nav.csv",
        "daily_positions.csv",
        "orders.csv",
        "trades.csv",
        "metrics.json",
        "final_account.json",
        "cumulative_return.png",
        "drawdown.png",
        "cash.png",
    }
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["metadata"]["database"]["password"] == "***"
    assert "must-not-leak" not in "".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in run_dir.iterdir()
    )

    with (run_dir / "orders.csv").open(encoding="utf-8", newline="") as stream:
        orders = list(csv.DictReader(stream))
    assert len(orders) == len(result.orders) == len(result.approvals)
    assert {
        "requested_quantity",
        "approved_quantity",
        "passed",
        "reason_code",
        "price_limit_down",
        "price_limit_up",
    } <= set(orders[0])
    assert orders[0]["reason_code"] == "APPROVED"


@pytest.mark.unit
def test_writer_never_overwrites_and_failure_publishes_only_run_json(
    tmp_path: Path,
    engine_components,
) -> None:
    writer = BacktestOutputWriter(tmp_path)
    failed = writer.write_failure(
        run_id="failed-run",
        run_metadata={"database": {"password": "must-not-leak"}},
        error=RuntimeError("database unavailable"),
    )
    assert {path.name for path in failed.iterdir()} == {"run.json"}
    manifest = json.loads((failed / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "RuntimeError"

    with pytest.raises(FileExistsError):
        writer.write_failure(
            run_id="failed-run",
            run_metadata={},
            error=RuntimeError("again"),
        )

    engine, *_ = engine_components
    result = engine.run(start_date=DATES[0], end_date=DATES[-1])
    with pytest.raises(ValueError, match="plot filenames"):
        writer.write_success(
            run_id="bad-plots",
            run_metadata={},
            result=result,
            metrics=_metrics(result),
            plots={"cash.png": _PNG},
        )
    assert not (tmp_path / "bad-plots").exists()
