from contextlib import nullcontext
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import Connection, Engine

import etf_backtest.live.jobs as jobs_module
from etf_backtest.application.strategy_source import ModelStrategySource
from etf_backtest.live.config import LiveConfig
from etf_backtest.live.jobs import ModelDeploymentSpecFactory
from etf_backtest.live.persistence.repository import LiveStateRepository
from etf_backtest.live.state import DeploymentStatus
from etf_backtest.strategy.model_training import LoadedInferenceBundle, bundle_sha256


def _result(first: dict[str, object] | None = None) -> Mock:
    result = Mock()
    result.mappings.return_value.first.return_value = first
    return result


def _repository(connection: Mock) -> LiveStateRepository:
    engine = Mock(spec=Engine)
    engine.begin.return_value = nullcontext(connection)
    engine.connect.return_value = nullcontext(connection)
    return LiveStateRepository(cast(Engine, engine))


def _values(**model: str | None) -> dict[str, object]:
    return {
        "deployment_id": "model-deployment",
        "bound_account_id": "account-1",
        "mode": "PAPER",
        "experiment_path": "experiment.yaml",
        "experiment_sha256": "a" * 64,
        "strategy_source_sha256": "b" * 64,
        "schedule_anchor_date": date(2026, 1, 1),
        "universe_json": '["SH.510300"]',
        "universe_hash": "c" * 64,
        "config_hash": "d" * 64,
        **model,
    }


@pytest.mark.unit
def test_model_deployment_persists_bundle_identity_and_accepts_exact_restart() -> None:
    expected = {
        **_values(
            model_bundle_path="results/model_bundle.pt",
            model_bundle_sha256="e" * 64,
            model_id="approved-model",
        ),
        "status": DeploymentStatus.ACTIVE,
    }
    connection = Mock(spec=Connection)
    connection.execute.side_effect = [
        _result(),
        _result(),
        _result(),
        _result(expected),
    ]
    repository = _repository(connection)
    assert repository.ensure_deployment(**_values(  # type: ignore[arg-type]
        model_bundle_path="results/model_bundle.pt",
        model_bundle_sha256="e" * 64,
        model_id="approved-model",
    )) == expected
    insert_params = connection.execute.call_args_list[2].args[0].compile().params
    assert insert_params["model_bundle_path"] == "results/model_bundle.pt"
    assert insert_params["model_bundle_sha256"] == "e" * 64
    assert insert_params["model_id"] == "approved-model"

    connection.reset_mock()
    connection.execute.side_effect = [_result(expected), _result(expected), _result(expected)]
    restarted = repository.ensure_deployment(**_values(  # type: ignore[arg-type]
        model_bundle_path="results/model_bundle.pt",
        model_bundle_sha256="e" * 64,
        model_id="approved-model",
    ))
    assert restarted == expected
    assert connection.execute.call_count == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    "changed",
    [
        {"model_bundle_sha256": "f" * 64},
        {"model_bundle_path": "results/replacement.pt"},
        {"model_id": "replacement-model"},
    ],
)
def test_model_deployment_cannot_silently_replace_bundle(
    changed: dict[str, str],
) -> None:
    original = {
        **_values(
            model_bundle_path="results/model_bundle.pt",
            model_bundle_sha256="e" * 64,
            model_id="approved-model",
        ),
        "status": DeploymentStatus.ACTIVE,
    }
    requested = {
        "model_bundle_path": "results/model_bundle.pt",
        "model_bundle_sha256": "e" * 64,
        "model_id": "approved-model",
        **changed,
    }
    connection = Mock(spec=Connection)
    connection.execute.return_value = _result(original)
    repository = _repository(connection)
    with pytest.raises(ValueError, match="immutable fields differ"):
        repository.ensure_deployment(**_values(**requested))  # type: ignore[arg-type]
    connection.execute.assert_called_once()


@pytest.mark.unit
def test_rule_deployment_keeps_model_columns_null() -> None:
    connection = Mock(spec=Connection)
    rule_row = {**_values(), "status": DeploymentStatus.ACTIVE}
    connection.execute.side_effect = [_result(), _result(), _result(), _result(rule_row)]
    repository = _repository(connection)
    repository.ensure_deployment(**_values())  # type: ignore[arg-type]
    insert_params = connection.execute.call_args_list[2].args[0].compile().params
    assert insert_params["model_bundle_path"] is None
    assert insert_params["model_bundle_sha256"] is None
    assert insert_params["model_id"] is None


@pytest.mark.unit
def test_model_spec_rehashes_bundle_after_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_path = tmp_path / "model_bundle.pt"
    bundle_path.write_bytes(b"approved")
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("case: model\n", encoding="utf-8")
    loaded = SimpleNamespace(
        file_sha256=bundle_sha256(bundle_path),
        bundle=SimpleNamespace(metadata=SimpleNamespace(model_id="approved-model")),
    )
    deployment = SimpleNamespace(
        deployment_id="model-deployment",
        account_id=lambda: "account-1",
        mode="PAPER",
        schedule_anchor_date=date(2026, 1, 1),
    )
    config = SimpleNamespace(deployment=deployment, config_hash="d" * 64)
    source = SimpleNamespace(
        experiment_path=experiment_path,
        strategy_source_sha256="b" * 64,
    )
    monkeypatch.setattr(jobs_module, "_model_backtest_config", lambda value: Mock())
    factory = ModelDeploymentSpecFactory(
        live_config=cast(LiveConfig, config),
        source=cast(ModelStrategySource, source),
        strategy_engine=cast(Engine, Mock(spec=Engine)),
        bundle_path=bundle_path,
        loaded_bundle=cast(LoadedInferenceBundle, loaded),
    )

    spec = factory(("SH.510300",))
    assert spec.model_bundle_path == str(bundle_path)
    assert spec.model_bundle_sha256 == loaded.file_sha256
    assert spec.model_id == "approved-model"

    bundle_path.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="changed after runtime construction"):
        factory(("SH.510300",))
