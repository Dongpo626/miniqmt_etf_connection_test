"""Load the fixed ``Strategy`` class from one trusted local Rule file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from etf_backtest.strategy.rule import RuleSettings, UserRule


class UserRuleLoadError(ValueError):
    """A trusted local Rule file does not expose the fixed extension contract."""


def _source_in_root(path: str | Path, allowed_root: str | Path, label: str) -> Path:
    root = Path(allowed_root).resolve(strict=True)
    if not root.is_dir():
        raise UserRuleLoadError("allowed_root must be an existing directory")
    supplied = Path(path)
    source = (supplied if supplied.is_absolute() else root / supplied).resolve(strict=True)
    if not source.is_relative_to(root):
        raise UserRuleLoadError(f"{label} file must stay inside allowed_root")
    if not source.is_file() or source.suffix.casefold() != ".py":
        raise UserRuleLoadError(f"{label} source must be one .py file")
    return source


def _load_module(source: Path, label: str) -> ModuleType:
    module_name = f"_qmt_{label.casefold()}_{source.stat().st_mtime_ns:x}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise UserRuleLoadError(f"cannot load {label} module: {source}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        raise UserRuleLoadError(
            f"{label} module execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)
    return module


def load_user_rule(path: str | Path, *, allowed_root: str | Path) -> UserRule:
    """Instantiate ``Strategy`` from a trusted local Python file."""

    source = _source_in_root(path, allowed_root, "Rule")
    module = _load_module(source, "Rule")
    strategy_class = getattr(module, "Strategy", None)
    if not isinstance(strategy_class, type) or not issubclass(strategy_class, UserRule):
        raise UserRuleLoadError("rule.py must define class Strategy(UserRule)")
    try:
        strategy = strategy_class()
    except Exception as exc:
        raise UserRuleLoadError(f"Strategy initialization failed: {exc}") from exc
    if not isinstance(strategy.settings, RuleSettings):
        raise UserRuleLoadError("Strategy.settings must be RuleSettings")
    return strategy


__all__ = ["UserRuleLoadError", "load_user_rule"]
