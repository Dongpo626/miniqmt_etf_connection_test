"""Command line entry point for one private Rule or Model experiment."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SYSTEM_CONFIG = _PROJECT_ROOT / "qmt_example" / "configs" / "system.yaml"
_DEFAULT_PRIVATE_STRATEGY_ROOT = _PROJECT_ROOT / "private_strategy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qmt-etf-backtest",
        description="Create, validate or run one daily ETF experiment.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument("experiment", type=Path)
        command_parser.add_argument(
            "--system",
            type=Path,
            default=_DEFAULT_SYSTEM_CONFIG,
            help="Shared MySQL and execution settings YAML.",
        )
    new_parser = commands.add_parser("new")
    new_parser.add_argument("template_kind", choices=("experiment",))
    new_parser.add_argument("name")
    new_parser.add_argument(
        "--private-root",
        type=Path,
        default=_DEFAULT_PRIVATE_STRATEGY_ROOT,
        help="Directory that contains private experiments.",
    )
    return parser


def _dispatch_command(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "run":
        from etf_backtest.experiment import run_experiment

        return run_experiment(args.experiment, system_path=args.system)
    if args.command == "validate":
        from etf_backtest.experiment import validate_experiment

        return validate_experiment(args.experiment, system_path=args.system)
    from etf_backtest.experiments.scaffold import scaffold_experiment

    created = scaffold_experiment(args.name, private_strategy_root=args.private_root)
    return {
        "status": "created",
        "kind": "experiment",
        "name": args.name,
        "paths": tuple(str(path) for path in created),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        outcome = _dispatch_command(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
