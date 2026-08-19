"""Caller-owned SQLAlchemy Connection behavior of QmtDailyRepository."""

from __future__ import annotations

from datetime import date
from typing import cast
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy.engine import Connection, Engine

from etf_backtest.data.mysql import QmtDailyRepository


def _master_row() -> dict[str, object]:
    return {
        "etf_code": "510300",
        "qmt_symbol": "510300.SH",
        "exchange": "SSE",
        "fund_name": "沪深300ETF",
        "list_date": date(2012, 5, 28),
        "delist_date": None,
        "current_status": "LISTED",
        "primary_category": "纯境内",
        "fund_type": "股票型",
        "etf_type": "纯境内",
        "source_system": "TUSHARE",
    }


def _connection_with_master_row() -> Mock:
    connection = Mock(spec=Connection)
    result = Mock()
    mappings = Mock()
    mappings.all.return_value = [_master_row()]
    result.mappings.return_value = mappings
    connection.execute.return_value = result
    return connection


@pytest.mark.unit
def test_external_connection_is_reused_without_lifecycle_calls() -> None:
    engine = Mock(spec=Engine)
    connection = _connection_with_master_row()
    repository = QmtDailyRepository(
        cast(Engine, engine),
        connection=cast(Connection, connection),
        dataset_version="test-version",
    )

    records = repository.load_etf_master(("SH.510300",))

    assert records[0].symbol == "SH.510300"
    engine.connect.assert_not_called()
    connection.execute.assert_called_once()
    connection.close.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_not_called()


@pytest.mark.unit
def test_unbound_repository_keeps_engine_connection_context_path() -> None:
    engine = Mock(spec=Engine)
    connection = _connection_with_master_row()
    manager = MagicMock()
    manager.__enter__.return_value = connection
    engine.connect.return_value = manager
    repository = QmtDailyRepository(
        cast(Engine, engine),
        dataset_version="test-version",
    )

    records = repository.load_etf_master(("SH.510300",))

    assert records[0].symbol == "SH.510300"
    engine.connect.assert_called_once_with()
    manager.__enter__.assert_called_once_with()
    manager.__exit__.assert_called_once()
