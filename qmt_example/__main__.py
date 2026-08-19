"""Delegate ``python -m qmt_example`` to the production CLI."""

from etf_backtest.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
