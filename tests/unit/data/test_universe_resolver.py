"""Frozen explicit-plus-pool universe resolution tests."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pytest

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import Exchange
from etf_backtest.data.mysql import QmtEtfMasterRecord
from etf_backtest.universe.resolver import (
    FrozenUniverseResolver,
    UniverseResolutionError,
)


def _master(
    symbol: str,
    *,
    list_date: date = date(2020, 1, 2),
    delist_date: date | None = None,
    status: str = "LISTED",
    primary_category: str = "纯境内",
    fund_type: str = "股票型",
) -> QmtEtfMasterRecord:
    canonical = normalize_symbol(symbol)
    code = canonical.partition(".")[2]
    exchange = Exchange.SSE if canonical.startswith("SH.") else Exchange.SZSE
    return QmtEtfMasterRecord(
        symbol=canonical,
        etf_code=code,
        qmt_symbol=f"{code}.{'SH' if exchange is Exchange.SSE else 'SZ'}",
        exchange=exchange,
        name=canonical,
        list_date=list_date,
        delist_date=delist_date,
        current_status=status,
        primary_category=primary_category,
        fund_type=fund_type,
        etf_type="纯境内",
        source_system="TUSHARE",
    )


class _Repository:
    dataset_version = "qmt-20260803-sha"

    def __init__(
        self,
        records: Sequence[QmtEtfMasterRecord],
        pools: Mapping[str, Sequence[str]],
        last_dates: Mapping[str, date] | None = None,
    ) -> None:
        self._records = {record.symbol: record for record in records}
        self._pools = pools
        self._last_dates = dict(last_dates or {})

    def load_etf_master(self, symbols: Sequence[str]) -> tuple[QmtEtfMasterRecord, ...]:
        return tuple(self._records[normalize_symbol(symbol)] for symbol in symbols)

    def load_pool_etf_master(self, pool_name: str) -> tuple[QmtEtfMasterRecord, ...]:
        return tuple(self._records[normalize_symbol(symbol)] for symbol in self._pools[pool_name])

    def load_last_raw_trade_dates(self, symbols: Sequence[str]) -> Mapping[str, date]:
        return {
            canonical: self._last_dates[canonical]
            for symbol in symbols
            if (canonical := normalize_symbol(symbol)) in self._last_dates
        }


@pytest.mark.unit
def test_explicit_and_multiple_pools_form_a_deduplicated_union() -> None:
    stock = _master("510300")
    sz_stock = _master("159919")
    gold = _master("518880", fund_type="其他")
    repository = _Repository(
        [stock, sz_stock, gold],
        {
            "domestic_stock_etf": ["510300", "159919"],
            "gold_etf": ["518880"],
        },
    )

    universe = FrozenUniverseResolver(repository).resolve(
        explicit_symbols=["510300.SH"],
        pools=["gold_etf", "domestic_stock_etf"],
        start_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
    )

    assert universe.symbols == ("SH.510300", "SH.518880", "SZ.159919")
    stock_member = universe.member_by_symbol()["SH.510300"]
    assert stock_member.sources == ("explicit", "pool:domestic_stock_etf")
    assert universe.active_symbols(date(2023, 1, 3)) == universe.symbols
    assert universe.approximation_flags == ("CURRENT_MASTER_POOL_CLASSIFICATION",)


@pytest.mark.unit
def test_listing_and_delisting_dates_filter_each_daily_membership() -> None:
    old = _master("510050", list_date=date(2010, 1, 1), delist_date=date(2022, 6, 30))
    future = _master("510300", list_date=date(2025, 1, 2))
    repository = _Repository([old, future], {})

    universe = FrozenUniverseResolver(repository).resolve(
        explicit_symbols=["510050", "510300"],
        start_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
    )

    assert universe.symbols == ("SH.510050",)
    assert universe.active_symbols(date(2022, 6, 30)) == ("SH.510050",)
    assert universe.active_symbols(date(2022, 7, 1)) == ()
    excluded = next(item for item in universe.decisions if item.symbol == "SH.510300")
    assert not excluded.included and excluded.exclusion_reason == "LISTED_AFTER_RUN"


@pytest.mark.unit
def test_missing_delist_date_uses_last_raw_date_and_marks_approximation() -> None:
    delisted = _master("510050", list_date=date(2010, 1, 1), status="DELISTED")
    repository = _Repository(
        [delisted],
        {},
        last_dates={"SH.510050": date(2022, 6, 30)},
    )

    universe = FrozenUniverseResolver(repository).resolve(
        explicit_symbols=["510050"],
        start_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
    )

    member = universe.members[0]
    assert member.info.delist_date == date(2022, 6, 30)
    assert member.info.delist_date_approximated
    assert "MISSING_DELIST_DATE_LAST_RAW_DATE" in universe.approximation_flags


@pytest.mark.unit
def test_missing_delist_date_without_raw_evidence_fails_closed() -> None:
    delisted = _master("510050", status="DELISTED")

    with pytest.raises(UniverseResolutionError, match="lacks delist_date"):
        FrozenUniverseResolver(_Repository([delisted], {})).resolve(
            explicit_symbols=["510050"],
            start_date=date(2021, 1, 1),
            end_date=date(2024, 12, 31),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "record",
    [
        _master("513100", primary_category="跨境", fund_type="QDII"),
        _master("511010", fund_type="债券型"),
        _master("518880", primary_category="跨境", fund_type="其他"),
    ],
)
def test_explicit_unsupported_etf_categories_fail_closed(record: QmtEtfMasterRecord) -> None:
    with pytest.raises(UniverseResolutionError, match="unsupported ETF category"):
        FrozenUniverseResolver(_Repository([record], {})).resolve(
            explicit_symbols=[record.symbol],
            start_date=date(2021, 1, 1),
            end_date=date(2024, 12, 31),
        )


@pytest.mark.unit
def test_pool_returning_an_unsupported_member_fails_closed() -> None:
    unsupported = _master("513100", primary_category="跨境", fund_type="QDII")

    with pytest.raises(UniverseResolutionError, match="unsupported ETF category"):
        FrozenUniverseResolver(
            _Repository([unsupported], {"domestic_stock_etf": [unsupported.symbol]})
        ).resolve(
            pools=["domestic_stock_etf"],
            start_date=date(2021, 1, 1),
            end_date=date(2024, 12, 31),
        )


@pytest.mark.unit
def test_unknown_master_status_fails_closed() -> None:
    stock = _master("510300", status="TERMINATED")

    with pytest.raises(UniverseResolutionError, match="current_status"):
        FrozenUniverseResolver(_Repository([stock], {})).resolve(
            explicit_symbols=[stock.symbol],
            start_date=date(2021, 1, 1),
            end_date=date(2024, 12, 31),
        )


@pytest.mark.unit
def test_universe_csv_contains_sources_and_approximation_flags(tmp_path: Path) -> None:
    stock = _master("510300")
    universe = FrozenUniverseResolver(
        _Repository([stock], {"domestic_stock_etf": ["510300"]})
    ).resolve(
        pools=["domestic_stock_etf"],
        start_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
    )

    target = universe.write_csv(tmp_path / "run" / "universe.csv")

    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["symbol"] == "SH.510300"
    assert rows[0]["sources"] == "pool:domestic_stock_etf"
    assert rows[0]["pool_classification_approximated"] == "true"
    assert rows[0]["dataset_version"] == "qmt-20260803-sha"
