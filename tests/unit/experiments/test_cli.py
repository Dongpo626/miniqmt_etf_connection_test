"""Private Rule/Model experiment CLI tests without external MySQL."""

import json
from pathlib import Path

import pytest

import etf_backtest.cli as cli
import etf_backtest.experiment as experiment_runner


@pytest.mark.unit
def test_cli_dispatches_user_experiment_run(monkeypatch, capsys) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_run(path: Path, *, system_path: Path):
        calls.append((path, system_path))
        return {"status": "success", "experiment": "demo", "runs": ()}

    monkeypatch.setattr(experiment_runner, "run_experiment", fake_run)
    result = cli.main(["run", "private/demo/experiment.yaml", "--system", "system.yaml"])

    assert result == 0
    assert calls == [(Path("private/demo/experiment.yaml"), Path("system.yaml"))]
    assert json.loads(capsys.readouterr().out)["experiment"] == "demo"


@pytest.mark.unit
def test_cli_dispatches_user_experiment_validation(monkeypatch, capsys) -> None:
    calls: list[tuple[Path, Path]] = []

    def fake_validate(path: Path, *, system_path: Path):
        calls.append((path, system_path))
        return {"status": "valid", "name": "demo", "case": "rule"}

    monkeypatch.setattr(experiment_runner, "validate_experiment", fake_validate)
    result = cli.main(["validate", "experiment.yaml", "--system", "operator.yaml"])

    assert result == 0
    assert calls == [(Path("experiment.yaml"), Path("operator.yaml"))]
    assert json.loads(capsys.readouterr().out)["status"] == "valid"


@pytest.mark.unit
def test_cli_new_experiment_creates_full_non_overwriting_scaffold(tmp_path, capsys) -> None:
    private_root = tmp_path / "private_strategy"
    arguments = ["new", "experiment", "demo", "--private-root", str(private_root)]

    assert cli.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "created"
    assert {Path(path).name for path in payload["paths"]} == {
        "experiment.yaml",
        "rule.py",
        "model.py",
    }
    assert (private_root / "demo" / "experiment.yaml").is_file()

    assert cli.main(arguments) == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "failed"
    assert failure["error_type"] == "FileExistsError"


@pytest.mark.unit
def test_cli_user_command_failure_is_json(monkeypatch, capsys) -> None:
    def fail_validation(path: Path, *, system_path: Path):
        del path, system_path
        raise ValueError("invalid private experiment")

    monkeypatch.setattr(experiment_runner, "validate_experiment", fail_validation)

    assert cli.main(["validate", "experiment.yaml", "--system", "system.yaml"]) == 1
    failure = json.loads(capsys.readouterr().err)
    assert failure == {
        "status": "failed",
        "error_type": "ValueError",
        "message": "invalid private experiment",
    }
