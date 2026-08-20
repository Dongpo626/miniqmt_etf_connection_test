import json
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

import etf_backtest.live_cli as live_cli
from etf_backtest.live.broker.base import BrokerGateway
from etf_backtest.live.config import load_live_config
from etf_backtest.live.engine import LiveTradingEngine
from etf_backtest.live.jobs import LiveDailyJobs
from etf_backtest.live.market.base import QuoteProvider
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.scheduler import LiveScheduler
from etf_backtest.live.state import SnapshotType
from etf_backtest.live_cli import LiveCommandRuntime, main

ROOT = Path(__file__).parents[3]
CONFIG = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def _record[ValueT](events: list[str], label: str, result: ValueT) -> ValueT:
    events.append(label)
    return result


def _runtime() -> tuple[LiveCommandRuntime, Mock, Mock, Mock]:
    connection = Mock(spec=Connection)
    state_engine = Mock(spec=Engine)
    state_engine.connect.return_value = connection
    broker = Mock(spec=BrokerGateway)
    jobs = Mock(spec=LiveDailyJobs)
    runtime = LiveCommandRuntime(
        state_engine=cast(Engine, state_engine),
        repository=cast(LiveStateRepository, Mock(spec=LiveStateRepository)),
        broker=cast(BrokerGateway, broker),
        quote_provider=cast(QuoteProvider, Mock(spec=QuoteProvider)),
        jobs=cast(LiveDailyJobs, jobs),
        scheduler=cast(LiveScheduler, Mock(spec=LiveScheduler)),
        engine=cast(LiveTradingEngine, Mock(spec=LiveTradingEngine)),
    )
    return runtime, connection, broker, jobs


def test_manual_write_uses_account_lock_and_same_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    runtime, connection, broker, jobs = _runtime()
    events: list[str] = []
    monkeypatch.setattr(
        "etf_backtest.live_cli.acquire_account_lock",
        lambda *args: _record(events, "lock", True),
    )
    monkeypatch.setattr(
        "etf_backtest.live_cli.release_account_lock",
        lambda *args: events.append("unlock"),
    )
    broker.connect.side_effect = lambda: events.append("connect")
    jobs.prepare_signal.side_effect = lambda *args, **kwargs: events.append("job")
    broker.disconnect.side_effect = lambda: events.append("disconnect")

    result = main(
        ["signal", "--date", "2026-08-19", "--config", str(CONFIG)],
        runtime_builder=lambda config: runtime,
    )
    assert result == 0
    assert events == ["lock", "connect", "job", "disconnect", "unlock"]
    connection.close.assert_called_once()


def test_read_commands_do_not_take_account_lock_and_no_model_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr("etf_backtest.live_cli._print_jobs", lambda *args: called.append("jobs"))
    monkeypatch.setattr(
        "etf_backtest.live_cli.acquire_account_lock",
        lambda *args: pytest.fail("read command requested account lock"),
    )
    assert main(["jobs", "--date", "2026-08-19", "--config", str(CONFIG)]) == 0
    assert called == ["jobs"]
    with pytest.raises(SystemExit):
        main(["model", "--config", str(CONFIG)])


def test_db_init_dispatch_does_not_build_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    monkeypatch.setattr("etf_backtest.live_cli._db_init", lambda config: called.append("migration"))
    assert main(["db", "init", "--config", str(CONFIG)]) == 0
    assert called == ["migration"]


def test_validate_works_without_external_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("etf_backtest.live_cli._validate", lambda config: None)
    assert main(["validate", "--config", str(CONFIG)]) == 0


def test_status_reads_current_and_eod_snapshots_separately(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "account-1")
    config = load_live_config(CONFIG)
    engine = Mock(spec=Engine)
    repository = Mock(spec=LiveStateRepository)
    repository.get_deployment.return_value = {"deployment_id": "deployment-1"}
    repository.latest_decision_for_deployment.return_value = None
    repository.latest_account_snapshot.side_effect = [
        {
            "trade_date": "2026-08-19",
            "snapshot_type": SnapshotType.CURRENT,
        },
        {"trade_date": "2026-08-19", "snapshot_type": SnapshotType.EOD},
    ]
    repository.load_position_snapshots.return_value = ()
    monkeypatch.setattr(live_cli, "_read_repository", lambda config: (engine, repository))

    live_cli._print_status(config)

    payload = json.loads(capsys.readouterr().out)
    assert set(payload["snapshots"]) == {"CURRENT", "EOD"}
    assert repository.latest_account_snapshot.call_args_list[0].args[1] is SnapshotType.CURRENT
    assert repository.latest_account_snapshot.call_args_list[1].args[1] is SnapshotType.EOD
