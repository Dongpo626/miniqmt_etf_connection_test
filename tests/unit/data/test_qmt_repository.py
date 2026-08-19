"""QMT daily repository business-key, status, mapping, and preflight tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from typing import Any, cast

import pytest
from sqlalchemy.engine import Engine

from etf_backtest.core.market import EtfInfo, Exchange
from etf_backtest.data.mysql import QmtDailyRepository, QmtDataQualityError


class _Result:
    def __init__(self, rows: Iterable[dict[str, object]]) -> None:
        self._rows = list(rows)

    def mappings(self) -> _Result:
        return self

    def all(self) -> list[dict[str, object]]:
        return self._rows


class _Connection:
    def __init__(self, batches: Iterable[Iterable[dict[str, object]]]) -> None:
        self._batches = iter(batches)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: Any, parameters: dict[str, object]) -> _Result:
        self.calls.append((str(statement), parameters))
        return _Result(next(self._batches))


class _Engine:
    def __init__(self, batches: Iterable[Iterable[dict[str, object]]]) -> None:
        self.connection = _Connection(batches)

    def connect(self) -> _Connection:
        return self.connection


TRADE_DATE = date(2024, 1, 2)


def _master_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
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
    row.update(overrides)
    return row


def _calendar_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "exchange": "SSE",
        "cal_date": TRADE_DATE,
        "is_open": 1,
        "previous_open_date": date(2023, 12, 29),
        "next_open_date": date(2024, 1, 3),
        "source_system": "AKSHARE_SINA",
    }
    row.update(overrides)
    return row


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "etf_code": "510300",
        "trade_date": TRADE_DATE,
        "open_price_cny": Decimal("4.00000000"),
        "high_price_cny": Decimal("4.20000000"),
        "low_price_cny": Decimal("3.80000000"),
        "close_price_cny": Decimal("4.10000000"),
        "pre_close_price_cny": Decimal("3.90000000"),
        "volume_share": Decimal("100.0000"),
        "amount_cny": Decimal("410.0000"),
        "source_system": "QMT",
    }
    row.update(overrides)
    return row


def _front_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "etf_code": "510300",
        "trade_date": TRADE_DATE,
        "open_price_cny": Decimal("2.00000000"),
        "high_price_cny": Decimal("2.10000000"),
        "low_price_cny": Decimal("1.90000000"),
        "close_price_cny": Decimal("2.05000000"),
        "pre_close_price_cny": Decimal("1.95000000"),
        "source_system": "QMT",
    }
    row.update(overrides)
    return row


def _share_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "etf_code": "510300",
        "asof_date": TRADE_DATE,
        "total_share": Decimal("123456.0000"),
        "source_system": "TUSHARE",
    }
    row.update(overrides)
    return row


def _index_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "index_code": "000001.SH",
        "trade_date": TRADE_DATE,
        "price_series_type": "PRICE",
        "open_value": Decimal("2965.25000000"),
        "high_value": Decimal("2976.27000000"),
        "low_value": Decimal("2930.09000000"),
        "close_value": Decimal("2962.28000000"),
        "pre_close_value": Decimal("2974.93000000"),
        "pct_chg": Decimal("-0.4252000000"),
        "source_system": "TUSHARE",
    }
    row.update(overrides)
    return row


def _status_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "etf_code": "510300",
        "trade_date": TRADE_DATE,
        "qmt_suspend_flag": 0,
        "up_limit_price_cny": Decimal("4.50000000"),
        "down_limit_price_cny": Decimal("3.50000000"),
        "qmt_source_system": "QMT",
        "price_limit_source_system": "TUSHARE",
    }
    row.update(overrides)
    return row


def _repository(
    raw_rows: Iterable[dict[str, object]],
    front_rows: Iterable[dict[str, object]],
    *,
    calendar_rows: Iterable[dict[str, object]] | None = None,
    status_rows: Iterable[dict[str, object]] | None = None,
) -> tuple[QmtDailyRepository, _Engine]:
    batches: list[Iterable[dict[str, object]]] = [
        [_master_row()],
        list(calendar_rows or [_calendar_row()]),
        raw_rows,
        front_rows,
        list([_status_row()] if status_rows is None else status_rows),
    ]
    engine = _Engine(batches)
    repository = QmtDailyRepository(
        cast(Engine, engine),
        dataset_version="qmt-20260803-sha",
    )
    return repository, engine


def _calendar_repository(
    rows: Iterable[dict[str, object]],
) -> QmtDailyRepository:
    return QmtDailyRepository(
        cast(Engine, _Engine([rows])),
        dataset_version="qmt-20260803-sha",
    )


@pytest.mark.unit
def test_load_daily_dataset_reads_primary_key_rows_and_uses_dataset_source_key() -> None:
    repository, engine = _repository([_raw_row()], [_front_row()])

    dataset = repository.load_daily_dataset(["510300.SH"], TRADE_DATE, TRADE_DATE)

    raw = dataset.frames[0].raw_by_symbol["SH.510300"]
    front = dataset.frames[0].front_by_symbol["SH.510300"]
    assert not hasattr(raw, "revision_no")
    assert not hasattr(front, "revision_no")
    assert raw.volume == 100 and type(raw.volume) is int
    market_bar = dataset.market_frames()[0].bar_for("SH.510300")
    view = dataset.front_market_bar_views()[0]
    expected_key = "QMT:qmt-20260803-sha:510300:2024-01-02"
    assert market_bar.source_record_key == expected_key
    assert view.source_record_key == expected_key
    assert market_bar.close == Decimal("4.10000000")
    assert view.close == Decimal("2.05000000")
    raw_sql, raw_params = engine.connection.calls[2]
    front_sql, front_params = engine.connection.calls[3]
    assert "ROW_NUMBER() OVER" not in raw_sql and "unadjusted_daily" in raw_sql
    assert "ROW_NUMBER() OVER" not in front_sql and "front_ratio_daily" in front_sql
    assert "revision_no" not in raw_sql and "available_at_utc" not in raw_sql
    assert "revision_no" not in front_sql and "available_at_utc" not in front_sql
    assert "snapshot_cutoff_utc" not in raw_params
    assert "snapshot_cutoff_utc" not in front_params


@pytest.mark.unit
def test_rule_fund_loaders_keep_decimal_share_and_aggregate_holder_percentages(
    tmp_path,
) -> None:
    csv_path = tmp_path / "huijin_combined.csv"
    csv_path.write_text(
        "\ufeff\ufeffHuijinEntity,Symbol,EndDate,HolderOfListing\n"
        "中央汇金投资有限责任公司,510300,2023-12-31,9.71\n"
        "中央汇金投资有限责任公司,510300,2023-12-31,0.29\n"
        "中央汇金资产管理有限责任公司,510300,2023-06-30,1.25\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    engine = _Engine([[_share_row()]])
    repository = QmtDailyRepository(
        cast(Engine, engine),
        dataset_version="qmt-20260803-sha",
        share_table="etf_share_daily",
        huijin_holders_csv=csv_path,
        huijin_holders_csv_sha256=digest,
    )

    shares = repository.load_etf_share_records(["SH.510300"], TRADE_DATE, TRADE_DATE)
    ratios = repository.load_huijin_ratio_records(["SH.510300"])

    assert shares[0].total_share == Decimal("123456.0000")
    assert shares[0].asof_date == TRADE_DATE
    assert {(row.entity, row.end_date): row.ratio for row in ratios} == {
        ("中央汇金投资有限责任公司", date(2023, 12, 31)): Decimal("0.10"),
        ("中央汇金资产管理有限责任公司", date(2023, 6, 30)): Decimal("0.0125"),
    }
    share_sql, share_params = engine.connection.calls[0]
    assert "ROW_NUMBER() OVER" not in share_sql
    assert "etf_share_daily" in share_sql
    assert "revision_no" not in share_sql
    assert "snapshot_cutoff_utc" not in share_params


@pytest.mark.unit
def test_rule_index_loader_keeps_price_series_decimal_and_source_identity() -> None:
    engine = _Engine([[_index_row()]])
    repository = QmtDailyRepository(
        cast(Engine, engine),
        dataset_version="qmt-20260803-sha",
        index_table="index_quote_daily",
    )

    records = repository.load_index_records(("000001.SH",), TRADE_DATE, TRADE_DATE)

    assert len(records) == 1
    assert records[0].index_code == "000001.SH"
    assert records[0].trade_date == TRADE_DATE
    assert records[0].high == Decimal("2976.27000000")
    assert records[0].close == Decimal("2962.28000000")
    assert records[0].pct_change == Decimal("-0.4252000000")
    index_sql, index_params = engine.connection.calls[0]
    assert "index_quote_daily" in index_sql
    assert "price_series_type = 'PRICE'" in index_sql
    assert index_params["index_codes"] == ["000001.SH"]


@pytest.mark.unit
def test_load_daily_dataset_attaches_explicit_price_limits_from_status_table() -> None:
    repository, engine = _repository(
        [_raw_row()],
        [_front_row()],
        status_rows=[_status_row()],
    )

    market_bar = (
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)
        .market_frames()[0]
        .bar_for("SH.510300")
    )

    assert market_bar.price_limit_down == Decimal("3.50000000")
    assert market_bar.price_limit_up == Decimal("4.50000000")
    assert market_bar.price_limit_source.value == "TUSHARE_EXPLICIT"
    status_sql, status_params = engine.connection.calls[4]
    assert "etf_trade_status_daily" in status_sql
    assert "qmt_suspend_flag" in status_sql
    assert status_params["etf_codes"] == ["510300"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "status_rows", [[], [_status_row(qmt_suspend_flag=None, qmt_source_system=None)]]
)
def test_complete_quotes_are_tradable_when_qmt_status_is_missing_or_unknown(
    status_rows: list[dict[str, object]],
) -> None:
    repository, _ = _repository([_raw_row()], [_front_row()], status_rows=status_rows)

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)

    assert not dataset.market_frames()[0].bar_for("SH.510300").suspended
    assert not dataset.front_market_bar_views()[0].suspended


@pytest.mark.unit
@pytest.mark.parametrize("flag", [-1, 0])
def test_resume_and_normal_status_are_tradable(flag: int) -> None:
    repository, _ = _repository(
        [_raw_row()], [_front_row()], status_rows=[_status_row(qmt_suspend_flag=flag)]
    )

    bar = (
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)
        .market_frames()[0]
        .bar_for("SH.510300")
    )

    assert not bar.suspended


@pytest.mark.unit
def test_status_flag_one_marks_complete_quotes_suspended() -> None:
    repository, _ = _repository(
        [_raw_row(volume_share=Decimal("0"), amount_cny=Decimal("0"))],
        [_front_row()],
        status_rows=[_status_row(qmt_suspend_flag=1)],
    )

    bar = (
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)
        .market_frames()[0]
        .bar_for("SH.510300")
    )

    assert bar.suspended


@pytest.mark.unit
@pytest.mark.parametrize(
    "status_rows",
    [[], [_status_row(qmt_suspend_flag=None, qmt_source_system=None)]],
)
def test_missing_quotes_require_explicit_suspension_status(
    status_rows: list[dict[str, object]],
) -> None:
    repository, _ = _repository([], [], status_rows=status_rows)

    with pytest.raises(QmtDataQualityError, match="coverage conflict"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_incomplete_raw_row_is_treated_as_missing_and_requires_suspension() -> None:
    repository, _ = _repository(
        [_raw_row(close_price_cny=None)],
        [_front_row()],
        status_rows=[_status_row(qmt_suspend_flag=0)],
    )

    with pytest.raises(QmtDataQualityError, match=r"one-to-one conflict|coverage conflict"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_cent_rounded_explicit_upper_limit_reconciles_to_qmt_limit_close() -> None:
    repository, _ = _repository(
        [
            _raw_row(
                open_price_cny=Decimal("4.20000000"),
                high_price_cny=Decimal("4.50300000"),
                low_price_cny=Decimal("4.10000000"),
                close_price_cny=Decimal("4.50300000"),
                pre_close_price_cny=Decimal("4.09300000"),
            )
        ],
        [
            _front_row(
                open_price_cny=Decimal("2.10000000"),
                high_price_cny=Decimal("2.25150000"),
                low_price_cny=Decimal("2.05000000"),
                close_price_cny=Decimal("2.25150000"),
                pre_close_price_cny=Decimal("2.04650000"),
            )
        ],
        status_rows=[
            _status_row(
                up_limit_price_cny=Decimal("4.50000000"),
                down_limit_price_cny=Decimal("3.68000000"),
            )
        ],
    )

    market_bar = (
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)
        .market_frames()[0]
        .bar_for("SH.510300")
    )

    assert market_bar.price_limit_down == Decimal("3.68000000")
    assert market_bar.price_limit_up == Decimal("4.50300000")


@pytest.mark.unit
def test_isolated_status_only_session_becomes_a_suspended_carry_bar() -> None:
    gap_date = date(2024, 1, 3)
    next_date = date(2024, 1, 4)
    repository, _ = _repository(
        [
            _raw_row(),
            _raw_row(
                trade_date=next_date,
                open_price_cny=Decimal("8.00000000"),
                high_price_cny=Decimal("8.40000000"),
                low_price_cny=Decimal("7.60000000"),
                close_price_cny=Decimal("8.20000000"),
                pre_close_price_cny=Decimal("7.80000000"),
            ),
        ],
        [
            _front_row(),
            _front_row(
                trade_date=next_date,
                open_price_cny=Decimal("2.00000000"),
                high_price_cny=Decimal("2.10000000"),
                low_price_cny=Decimal("1.90000000"),
                close_price_cny=Decimal("2.05000000"),
                pre_close_price_cny=Decimal("1.95000000"),
            ),
        ],
        calendar_rows=[
            _calendar_row(next_open_date=gap_date),
            _calendar_row(
                cal_date=gap_date,
                previous_open_date=TRADE_DATE,
                next_open_date=next_date,
            ),
            _calendar_row(
                cal_date=next_date,
                previous_open_date=gap_date,
                next_open_date=date(2024, 1, 5),
            ),
        ],
        status_rows=[
            _status_row(trade_date=TRADE_DATE),
            _status_row(
                trade_date=gap_date,
                qmt_suspend_flag=1,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
            _status_row(
                trade_date=next_date,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
        ],
    )

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, next_date)
    carry = dataset.market_frames()[1].bar_for("SH.510300")
    carry_view = dataset.front_market_bar_views()[1]

    assert carry.trade_date == gap_date
    assert carry.close == Decimal("4.10000000")
    assert carry.volume == 0
    assert carry.suspended
    assert carry.source_record_key == "CARRY:qmt-20260803-sha:510300:2024-01-03"
    assert carry_view.close == Decimal("2.05000000")
    assert carry_view.suspended
    assert dataset.suspension_carry_keys == (("SH.510300", gap_date),)


@pytest.mark.unit
def test_null_qmt_status_with_missing_quotes_carries_raw_and_front_separately() -> None:
    gap_date = date(2024, 1, 3)
    next_date = date(2024, 1, 4)
    repository, _ = _repository(
        [
            _raw_row(),
            _raw_row(
                trade_date=next_date,
                open_price_cny=Decimal("8.00000000"),
                high_price_cny=Decimal("8.40000000"),
                low_price_cny=Decimal("7.60000000"),
                close_price_cny=Decimal("8.20000000"),
                pre_close_price_cny=Decimal("7.80000000"),
            ),
        ],
        [
            _front_row(),
            _front_row(
                trade_date=next_date,
                open_price_cny=Decimal("2.00000000"),
                high_price_cny=Decimal("2.10000000"),
                low_price_cny=Decimal("1.90000000"),
                close_price_cny=Decimal("2.05000000"),
            ),
        ],
        calendar_rows=[
            _calendar_row(next_open_date=gap_date),
            _calendar_row(
                cal_date=gap_date,
                previous_open_date=TRADE_DATE,
                next_open_date=next_date,
            ),
            _calendar_row(
                cal_date=next_date,
                previous_open_date=gap_date,
                next_open_date=date(2024, 1, 5),
            ),
        ],
        status_rows=[
            _status_row(trade_date=TRADE_DATE),
            _status_row(
                trade_date=gap_date,
                qmt_suspend_flag=None,
                qmt_source_system=None,
            ),
            _status_row(
                trade_date=next_date,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
        ],
    )

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, next_date)
    carried_raw = dataset.market_frames()[1].bar_for("SH.510300")
    carried_front = dataset.front_market_bar_views()[1]

    assert carried_raw.close == Decimal("4.10000000")
    assert carried_front.close == Decimal("2.05000000")
    assert carried_raw.suspended
    assert carried_front.suspended
    assert dataset.suspension_carry_keys == (("SH.510300", gap_date),)


@pytest.mark.unit
def test_terminal_status_only_session_is_not_carried_without_a_complete_next_day() -> None:
    gap_date = date(2024, 1, 3)
    repository, _ = _repository(
        [_raw_row()],
        [_front_row()],
        calendar_rows=[
            _calendar_row(next_open_date=gap_date),
            _calendar_row(
                cal_date=gap_date,
                previous_open_date=TRADE_DATE,
                next_open_date=date(2024, 1, 4),
            ),
        ],
        status_rows=[
            _status_row(trade_date=TRADE_DATE),
            _status_row(
                trade_date=gap_date,
                qmt_suspend_flag=1,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
        ],
    )

    with pytest.raises(QmtDataQualityError, match="coverage conflict"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, gap_date)


@pytest.mark.unit
def test_first_requested_session_uses_complete_boundary_neighbors_for_carry() -> None:
    previous_date = date(2023, 12, 29)
    next_date = date(2024, 1, 3)
    repository, _ = _repository(
        [
            _raw_row(trade_date=previous_date),
            _raw_row(trade_date=next_date),
        ],
        [
            _front_row(trade_date=previous_date),
            _front_row(trade_date=next_date),
        ],
        calendar_rows=[
            _calendar_row(previous_open_date=previous_date, next_open_date=next_date),
            _calendar_row(
                cal_date=next_date,
                previous_open_date=TRADE_DATE,
                next_open_date=date(2024, 1, 4),
            ),
        ],
        status_rows=[
            _status_row(
                qmt_suspend_flag=1,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
            _status_row(
                trade_date=next_date,
                up_limit_price_cny=None,
                down_limit_price_cny=None,
                price_limit_source_system=None,
            ),
        ],
    )

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, next_date)

    first_bar = dataset.market_frames()[0].bar_for("SH.510300")
    assert first_bar.trade_date == TRADE_DATE
    assert first_bar.suspended
    assert first_bar.source_record_key == "CARRY:qmt-20260803-sha:510300:2024-01-02"
    assert dataset.suspension_carry_keys == (("SH.510300", TRADE_DATE),)


@pytest.mark.unit
def test_partial_explicit_price_limit_pair_fails_preflight() -> None:
    repository, _ = _repository(
        [_raw_row()],
        [_front_row()],
        status_rows=[_status_row(down_limit_price_cny=None)],
    )

    with pytest.raises(QmtDataQualityError, match="both be present or both be NULL"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_null_explicit_pair_preserves_derived_fallback() -> None:
    repository, _ = _repository(
        [_raw_row()],
        [_front_row()],
        status_rows=[
            _status_row(
                down_limit_price_cny=None,
                up_limit_price_cny=None,
                price_limit_source_system=None,
            )
        ],
    )

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)
    market_bar = dataset.market_frames()[0].bar_for("SH.510300")

    assert market_bar.price_limit_down is None
    assert market_bar.price_limit_up is None
    assert dataset.explicit_price_limit_count == 0
    assert dataset.derived_price_limit_fallback_count == 1


@pytest.mark.unit
def test_trade_status_table_rejects_unsafe_sql_identifier() -> None:
    with pytest.raises(ValueError, match="plain MySQL identifier"):
        QmtDailyRepository(
            cast(Engine, _Engine([])),
            dataset_version="qmt-20260803-sha",
            trade_status_table="status`; DROP TABLE dim_etf; --",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"up_limit_price_cny": Decimal("4.000")}, "raw close"),
        ({"up_limit_price_cny": Decimal("4.5005")}, "ETF tick"),
        ({"price_limit_source_system": "OTHER"}, "must be TUSHARE"),
    ],
)
def test_invalid_explicit_price_limit_row_fails_dataset_preflight(
    overrides: dict[str, object], message: str
) -> None:
    repository, _ = _repository(
        [_raw_row()],
        [_front_row()],
        status_rows=[_status_row(**overrides)],
    )

    with pytest.raises(QmtDataQualityError, match=message):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_raw_and_front_business_keys_must_be_one_to_one() -> None:
    repository, _ = _repository([_raw_row()], [])

    with pytest.raises(QmtDataQualityError, match="one-to-one conflict"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_active_open_date_requires_a_complete_daily_pair() -> None:
    repository, _ = _repository([], [])

    with pytest.raises(QmtDataQualityError, match="coverage conflict"):
        repository.preflight_daily_slice(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_volume_decimal_must_convert_to_integer_shares_without_loss() -> None:
    repository, _ = _repository(
        [_raw_row(volume_share=Decimal("100.5000"))],
        [_front_row()],
    )

    with pytest.raises(QmtDataQualityError, match="losslessly"):
        repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)


@pytest.mark.unit
def test_front_ohlc_is_validated_independently_without_ratio_audit() -> None:
    repository, _ = _repository(
        [_raw_row()],
        [_front_row(close_price_cny=Decimal("2.06000000"))],
    )

    dataset = repository.load_daily_dataset(["SH.510300"], TRADE_DATE, TRADE_DATE)

    assert dataset.front_market_bar_views()[0].close == Decimal("2.06000000")


@pytest.mark.unit
def test_calendar_requires_every_sse_natural_date_without_session_columns() -> None:
    next_date = date(2024, 1, 3)
    repository = _calendar_repository([_calendar_row()])

    with pytest.raises(QmtDataQualityError, match="natural-date coverage"):
        repository.load_sse_calendar(TRADE_DATE, next_date)

    loaded = _calendar_repository([_calendar_row()]).load_sse_calendar(TRADE_DATE, TRADE_DATE)
    assert loaded[0].cal_date == TRADE_DATE


@pytest.mark.unit
def test_approximated_delist_lifecycle_controls_expected_daily_coverage() -> None:
    second = date(2024, 1, 3)
    second_calendar = _calendar_row(
        cal_date=second,
        previous_open_date=TRADE_DATE,
        next_open_date=date(2024, 1, 4),
    )
    engine = _Engine(
        [
            [_master_row(current_status="DELISTED")],
            [_calendar_row(), second_calendar],
            [_raw_row()],
            [_front_row()],
            [_status_row()],
        ]
    )
    repository = QmtDailyRepository(
        cast(Engine, engine),
        dataset_version="qmt-20260803-sha",
    )
    approximated = EtfInfo(
        symbol="SH.510300",
        exchange=Exchange.SSE,
        name="沪深300ETF",
        primary_category="纯境内",
        fund_type="股票型",
        list_date=date(2012, 5, 28),
        delist_date=TRADE_DATE,
        current_status="DELISTED",
        delist_date_approximated=True,
    )

    dataset = repository.load_daily_dataset(
        ["SH.510300"],
        TRADE_DATE,
        second,
        etf_infos=[approximated],
    )

    assert tuple(frame.trade_date for frame in dataset.frames) == (TRADE_DATE,)
    assert dataset.etf_infos[0].delist_date_approximated
