import re
from pathlib import Path

from sqlalchemy import UniqueConstraint

from etf_backtest.live.persistence.schema import LIVE_TABLES, metadata

MIGRATION = (
    Path(__file__).parents[4]
    / "etf_backtest/live/persistence/migrations/001_create_live_tables.sql"
)


def _unique_columns(table_name: str) -> set[tuple[str, ...]]:
    table = metadata.tables[table_name]
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def test_schema_contains_only_nine_core_tables_and_required_unique_keys() -> None:
    expected = {
        "live_deployment",
        "live_job_run",
        "live_decision",
        "live_target_position",
        "live_order_intent",
        "live_broker_order",
        "live_broker_trade",
        "live_account_snapshot",
        "live_position_snapshot",
    }
    assert set(metadata.tables) == expected
    assert {table.name for table in LIVE_TABLES} == expected
    assert ("deployment_id", "signal_date") in _unique_columns("live_decision")
    assert ("decision_id", "symbol") in _unique_columns("live_target_position")
    assert ("intent_key",) in _unique_columns("live_order_intent")
    assert ("remark_token",) in _unique_columns("live_order_intent")
    assert ("account_id", "broker_order_id") in _unique_columns("live_broker_order")
    assert ("account_id", "broker_trade_id") in _unique_columns("live_broker_trade")
    assert not metadata.tables["live_job_run"].foreign_keys


def test_migration_has_same_nine_tables_and_schema_columns() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    created = set(re.findall(r"CREATE TABLE\s+(\w+)", sql, flags=re.IGNORECASE))
    assert created == set(metadata.tables)
    for table in LIVE_TABLES:
        match = re.search(
            rf"CREATE TABLE\s+{table.name}\s*\((.*?)\)\s*ENGINE=",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert match is not None
        block = match.group(1)
        assert {column.name for column in table.columns} <= set(
            re.findall(r"^\s*(\w+)\s+", block, flags=re.MULTILINE)
        )
