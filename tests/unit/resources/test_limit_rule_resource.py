"""Integrity checks for the frozen ETF 20% price-limit resource."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_DIRECTORY = PROJECT_ROOT / "resources" / "limit_rules"
CSV_PATH = RESOURCE_DIRECTORY / "etf_price_limit_20pct.csv"
MANIFEST_PATH = RESOURCE_DIRECTORY / "manifest.json"
EXPECTED_COLUMNS = [
    "symbol",
    "exchange",
    "price_limit_ratio",
    "valid_from",
    "valid_to",
    "record_mode",
    "source_id",
]


def _manifest() -> dict[str, object]:
    loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames == EXPECTED_COLUMNS
        return list(reader)


def test_limit_rule_csv_matches_frozen_manifest() -> None:
    manifest = _manifest()
    csv_metadata = manifest["csv"]
    assert isinstance(csv_metadata, dict)
    digest = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert digest == csv_metadata["sha256"]

    rows = _rows()
    assert len(rows) == csv_metadata["row_count"] == 231
    source_counts = defaultdict(int)
    record_mode_counts = defaultdict(int)
    for row in rows:
        source_counts[row["source_id"]] += 1
        record_mode_counts[row["record_mode"]] += 1
    assert dict(source_counts) == csv_metadata["source_counts"]
    assert dict(record_mode_counts) == csv_metadata["record_mode_counts"]


def test_limit_rule_rows_are_typed_sorted_and_non_overlapping() -> None:
    rows = _rows()
    assert rows == sorted(rows, key=lambda row: (row["symbol"], row["valid_from"]))

    periods: dict[str, list[tuple[date, date | None]]] = defaultdict(list)
    for row in rows:
        prefix, code = row["symbol"].split(".")
        assert prefix in {"SH", "SZ"}
        assert len(code) == 6 and code.isdigit()
        assert row["exchange"] == ("SSE" if prefix == "SH" else "SZSE")
        assert Decimal(row["price_limit_ratio"]) == Decimal("0.20")
        start = date.fromisoformat(row["valid_from"])
        end = date.fromisoformat(row["valid_to"]) if row["valid_to"] else None
        assert end is None or end >= start
        periods[row["symbol"]].append((start, end))

    for symbol_periods in periods.values():
        previous_end: date | None = None
        for index, (start, end) in enumerate(symbol_periods):
            if index:
                assert previous_end is not None and previous_end < start
            previous_end = end


def test_limit_rule_resource_covers_required_representative_etfs() -> None:
    indexed = {(row["symbol"], row["valid_from"]): row for row in _rows()}
    assert indexed[("SH.588000", "2020-11-16")]["record_mode"] == (
        "CURRENT_SNAPSHOT_RULE_INFERENCE"
    )
    assert indexed[("SZ.159915", "2020-08-24")]["record_mode"] == (
        "EFFECTIVE_DATED_OFFICIAL_SEED_CONFIRMED_CURRENT"
    )
    removed_seed = indexed[("SZ.159955", "2020-08-24")]
    assert removed_seed["valid_to"] == "2026-08-03"
    assert removed_seed["record_mode"] == "SNAPSHOT_DERIVED_REMOVAL_CUTOFF"


def test_limit_rule_sources_are_official_and_caveats_are_frozen() -> None:
    manifest = _manifest()
    assert manifest["retrieved_at"] == "2026-08-04"
    assert manifest["rule_mode"] == "LATEST_SNAPSHOT_WITH_2020_SEED"
    sources = manifest["sources"]
    assert isinstance(sources, list)
    indexed_sources = {
        source["source_id"]: source for source in sources if isinstance(source, dict)
    }
    assert set(manifest["csv"]["source_counts"]) <= set(indexed_sources)
    assert indexed_sources["SSE_CURRENT_20260804"]["official_etf_rows"] == 146
    assert indexed_sources["SSE_CURRENT_20260804"]["included_etf_rows"] == 145
    assert indexed_sources["SSE_CURRENT_20260804"]["dataset_scope_exclusion"] == {
        "symbol": "SH.588530",
        "qmt_master_snapshot_date": "2026-08-03",
        "reason": (
            "Absent from qmt_etf_quant.dim_etf; retaining it would create a rule "
            "row outside the readable instrument universe."
        ),
    }
    assert all(row["symbol"] != "SH.588530" for row in _rows())
    assert indexed_sources["SZSE_CURRENT_20260804"]["included_etf_rows"] == 85
    assert len(indexed_sources["SZSE_CURRENT_20260804"]["response_pages"]) == 5
    urls = "\n".join(
        str(source.get("url", source.get("data_url", "")))
        for source in sources
        if isinstance(source, dict)
    )
    assert "sse.com.cn" in urls
    assert "szse.cn" in urls
    coverage = manifest["coverage"]
    assert isinstance(coverage, dict)
    current_snapshot = coverage["current_snapshot"]
    assert current_snapshot["sse_included_etfs"] == 145
    assert current_snapshot["szse_included_etfs"] == 85
    assert "effective-dated" in str(coverage["known_gap"])
    assert coverage["snapshot_derived_cutoff"]["symbol"] == "SZ.159955"
