from __future__ import annotations

import importlib
from pathlib import Path


def test_main_runs_selected_experiment_from_project_root(monkeypatch, capsys) -> None:
    entrypoint = importlib.import_module("run_backtest")
    calls: list[tuple[Path, Path, Path]] = []

    def fake_run_experiment(
        experiment_path: Path,
        *,
        system_path: Path,
        project_root: Path,
    ) -> dict[str, object]:
        calls.append((experiment_path, system_path, project_root))
        return {"status": "success", "run_dir": "example-output"}

    monkeypatch.setattr(entrypoint, "run_experiment", fake_run_experiment)

    result = entrypoint.main()

    expected_root = Path(entrypoint.__file__).resolve().parent
    assert calls == [
        (
            expected_root / "private_strategy" / "beginner_example" / "experiment.yaml",
            expected_root / "qmt_example" / "configs" / "system.yaml",
            expected_root,
        )
    ]
    assert result == {"status": "success", "run_dir": "example-output"}
    assert '"status": "success"' in capsys.readouterr().out


def test_import_does_not_start_a_backtest(monkeypatch) -> None:
    entrypoint = importlib.import_module("run_backtest")

    def fail_if_called(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("importing run_backtest must not start a backtest")

    monkeypatch.setattr(entrypoint, "run_experiment", fail_if_called)
    importlib.reload(entrypoint)
