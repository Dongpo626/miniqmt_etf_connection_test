from pathlib import Path

import pytest
import yaml

from etf_backtest.live.config import load_live_config

ROOT = Path(__file__).parents[3]
RULE_SAMPLE = ROOT / "qmt_example/configs/live/beginner_example_paper.yaml"


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.unit
def test_rule_remains_valid_and_rejects_model_section(tmp_path: Path) -> None:
    payload = yaml.safe_load(RULE_SAMPLE.read_text(encoding="utf-8"))
    assert load_live_config(RULE_SAMPLE).deployment.case == "rule"
    payload["model"] = {"bundle_path": "results/model_bundle.pt"}
    with pytest.raises(ValueError, match="Rule live"):
        load_live_config(_write(tmp_path, payload))


@pytest.mark.unit
def test_model_requires_bundle_path_and_hash_is_stable(tmp_path: Path) -> None:
    payload = yaml.safe_load(RULE_SAMPLE.read_text(encoding="utf-8"))
    payload["deployment"]["case"] = "model"
    with pytest.raises(ValueError, match="bundle_path"):
        load_live_config(_write(tmp_path, payload))

    payload["model"] = {"bundle_path": "results/model_bundle.pt"}
    path = _write(tmp_path, payload)
    first = load_live_config(path)
    second = load_live_config(path)
    assert first.model is not None
    assert first.model.bundle_path == Path("results/model_bundle.pt")
    assert first.config_hash == second.config_hash
