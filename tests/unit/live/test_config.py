from pathlib import Path

import pytest
import yaml

from etf_backtest.live.config import load_live_config

ROOT = Path(__file__).parents[3]
SAMPLE = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def test_rule_paper_config_and_hash_are_stable_and_secret_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QMT_LIVE_MYSQL_PASSWORD", "secret-one")
    first = load_live_config(SAMPLE)
    monkeypatch.setenv("QMT_LIVE_MYSQL_PASSWORD", "secret-two")
    second = load_live_config(SAMPLE)

    assert first.deployment.case == "rule" and first.deployment.mode == "PAPER"
    assert first.config_hash == second.config_hash
    assert "secret" not in first.config_hash


def test_model_section_and_invalid_time_order_are_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))
    payload["model"] = {"bundle_path": "model.pt"}
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Rule live"):
        load_live_config(model_path)

    payload.pop("model")
    payload["execution"]["stop_new_orders"] = "14:49:00"
    invalid_time = tmp_path / "time.yaml"
    invalid_time.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="submit_start"):
        load_live_config(invalid_time)


def test_only_named_secret_environment_fields_are_resolved(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_live_config(SAMPLE)
    monkeypatch.setenv("QMT_PAPER_ACCOUNT_ID", "paper-account")
    assert config.deployment.account_id() == "paper-account"
    assert str(config.miniqmt.userdata_path).startswith("C:")
