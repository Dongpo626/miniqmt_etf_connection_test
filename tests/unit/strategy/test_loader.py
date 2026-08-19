"""Fixed-name trusted local Rule loader tests."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from etf_backtest.strategy import RuleSettings, UserRule, UserRuleLoadError, load_user_rule


def _write_rule(path: Path, *, class_name: str = "Strategy", settings: str = "") -> Path:
    path.write_text(
        f"""from decimal import Decimal
from etf_backtest.strategy import RuleSettings, UserRule

class {class_name}(UserRule):
    {settings}
    def generate_weights(self, data):
        return {{data.symbols[0]: Decimal("0.5")}}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_loader_uses_the_fixed_strategy_class_and_python_settings(tmp_path: Path) -> None:
    source = _write_rule(
        tmp_path / "rule.py",
        settings='settings = RuleSettings(lookback_trading_days=37, target_weight="0.83")',
    )

    rule = load_user_rule(source, allowed_root=tmp_path)

    assert isinstance(rule, UserRule)
    assert type(rule).__name__ == "Strategy"
    assert isinstance(rule.settings, RuleSettings)
    assert rule.lookback_trading_days == 37
    assert rule.target_weight == Decimal("0.83")


@pytest.mark.unit
def test_loader_rejects_path_escape_and_missing_fixed_class(tmp_path: Path) -> None:
    outside = _write_rule(tmp_path.parent / "outside_rule.py")
    with pytest.raises(UserRuleLoadError, match="inside"):
        load_user_rule(outside, allowed_root=tmp_path)

    wrong = _write_rule(tmp_path / "wrong.py", class_name="Other")
    with pytest.raises(UserRuleLoadError, match="class Strategy"):
        load_user_rule(wrong, allowed_root=tmp_path)


@pytest.mark.unit
def test_loader_reports_module_and_constructor_errors(tmp_path: Path) -> None:
    broken_module = tmp_path / "broken.py"
    broken_module.write_text('raise RuntimeError("broken module")\n', encoding="utf-8")
    with pytest.raises(UserRuleLoadError, match="broken module"):
        load_user_rule(broken_module, allowed_root=tmp_path)

    broken_init = tmp_path / "broken_init.py"
    broken_init.write_text(
        """from etf_backtest.strategy import UserRule
class Strategy(UserRule):
    def __init__(self): raise RuntimeError("broken init")
    def generate_weights(self, data): return {}
""",
        encoding="utf-8",
    )
    with pytest.raises(UserRuleLoadError, match="broken init"):
        load_user_rule(broken_init, allowed_root=tmp_path)


@pytest.mark.unit
def test_loader_rejects_wrong_settings_type(tmp_path: Path) -> None:
    source = _write_rule(tmp_path / "rule.py", settings='settings = {"lookback": 20}')
    with pytest.raises(UserRuleLoadError, match="RuleSettings"):
        load_user_rule(source, allowed_root=tmp_path)
