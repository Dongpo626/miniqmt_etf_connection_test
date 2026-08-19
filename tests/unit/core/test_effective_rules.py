"""Manifest-backed effective ETF rule tests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from etf_backtest.core.effective_rules import (
    EffectiveDatedEtfRuleResolver,
    EtfRulePeriod,
    RuleProvenance,
    RuleResolutionError,
    RuleResourceValidationError,
    load_effective_rule_resolver,
)
from etf_backtest.core.market import EtfCategory, EtfInfo, Exchange, TurnoverRule

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESOURCE_DIR = PROJECT_ROOT / "resources" / "limit_rules"
CSV_PATH = RESOURCE_DIR / "etf_price_limit_20pct.csv"
MANIFEST_PATH = RESOURCE_DIR / "manifest.json"
CSV_COLUMNS = (
    "symbol",
    "exchange",
    "price_limit_ratio",
    "valid_from",
    "valid_to",
    "record_mode",
    "source_id",
)


def _info(
    symbol: str,
    *,
    list_date: date,
    delist_date: date | None = None,
    primary_category: str = "\u7eaf\u5883\u5185",
    fund_type: str = "\u80a1\u7968\u578b",
) -> EtfInfo:
    return EtfInfo(
        symbol=symbol,
        exchange=Exchange.SSE if symbol.startswith("SH.") else Exchange.SZSE,
        name=symbol,
        primary_category=primary_category,
        fund_type=fund_type,
        list_date=list_date,
        delist_date=delist_date,
        current_status="LISTED" if delist_date is None else "DELISTED",
    )


def _write_resource(tmp_path: Path, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    rows = sorted(rows, key=lambda row: (row["symbol"], row["valid_from"]))
    csv_path = tmp_path / "etf_price_limit_20pct.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    counts = Counter(row["source_id"] for row in rows)
    source_ids = sorted(counts)
    manifest = {
        "resource_name": "etf_price_limit_20pct",
        "resource_version": "test-v1",
        "rule_mode": "LATEST_SNAPSHOT_WITH_2020_SEED",
        "default_price_limit_ratio": "0.10",
        "covered_price_limit_ratio": "0.20",
        "csv": {
            "path": csv_path.name,
            "sha256": digest,
            "row_count": len(rows),
            "columns": list(CSV_COLUMNS),
            "source_counts": dict(counts),
            "record_mode_counts": dict(Counter(row["record_mode"] for row in rows)),
        },
        "sources": [{"source_id": source_id} for source_id in source_ids],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, manifest_path


def _sz_exception(
    *,
    symbol: str = "SZ.159955",
    valid_from: str = "2020-08-24",
    valid_to: str = "",
    record_mode: str = "EFFECTIVE_DATED_OFFICIAL_SEED_CONFIRMED_CURRENT",
) -> dict[str, str]:
    return {
        "symbol": symbol,
        "exchange": "SZSE",
        "price_limit_ratio": "0.20",
        "valid_from": valid_from,
        "valid_to": valid_to,
        "record_mode": record_mode,
        "source_id": "SZSE_CURRENT_20260804_WITH_2020_SEED",
    }


@pytest.mark.unit
def test_frozen_resource_builds_10_and_20_percent_periods_with_identity() -> None:
    infos = [
        _info("SH.588000", list_date=date(2020, 11, 16)),
        _info("SZ.159915", list_date=date(2019, 1, 1)),
        _info("SZ.159955", list_date=date(2011, 12, 13)),
        _info("SH.510300", list_date=date(2012, 5, 28)),
        _info("SH.518880", list_date=date(2013, 7, 29), fund_type="\u5176\u4ed6"),
    ]

    resolver = load_effective_rule_resolver(infos, CSV_PATH, MANIFEST_PATH)

    identity = resolver.resource_identity
    assert identity is not None
    assert identity.resource_name == "etf_price_limit_20pct"
    assert identity.csv_sha256 == hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    assert resolver.resolve("SH.510300", date(2024, 1, 2)).price_limit_ratio == Decimal("0.10")
    assert resolver.resolve("SZ.159915", date(2020, 8, 23)).price_limit_ratio == Decimal("0.10")
    sz_seed = resolver.resolve_with_provenance("SZ.159915", date(2020, 8, 24))
    assert sz_seed.rule.price_limit_ratio == Decimal("0.20")
    assert sz_seed.provenance.approximate
    sse_snapshot = resolver.resolve_with_provenance("SH.588000", date(2024, 1, 2))
    assert sse_snapshot.rule.price_limit_ratio == Decimal("0.20")
    assert sse_snapshot.provenance.approximate
    cutoff = resolver.resolve_with_provenance("SZ.159955", date(2026, 8, 3))
    assert cutoff.rule.price_limit_ratio == Decimal("0.20")
    assert cutoff.provenance.approximate
    assert resolver.resolve("SZ.159955", date(2026, 8, 4)).price_limit_ratio == Decimal("0.10")

    gold = resolver.resolve("SH.518880", date(2024, 1, 2))
    assert gold.etf_category is EtfCategory.GOLD_ETF
    assert gold.turnover_rule is TurnoverRule.T0
    assert gold.price_limit_ratio == Decimal("0.10")


@pytest.mark.unit
def test_finite_valid_to_is_inclusive_then_restores_10_percent(tmp_path: Path) -> None:
    csv_path, manifest_path = _write_resource(
        tmp_path,
        [_sz_exception(valid_to="2022-12-31")],
    )
    info = _info("SZ.159955", list_date=date(2018, 1, 1))

    resolver = load_effective_rule_resolver([info], csv_path, manifest_path)

    assert resolver.resolve(info.symbol, date(2020, 8, 23)).price_limit_ratio == Decimal("0.10")
    bounded = resolver.resolve_with_provenance(info.symbol, date(2022, 12, 31))
    assert bounded.rule.price_limit_ratio == Decimal("0.20")
    assert bounded.provenance.approximate
    assert bounded.provenance.method.endswith("WITH_VALID_TO")
    restored = resolver.resolve_with_provenance(info.symbol, date(2023, 1, 1))
    assert restored.rule.price_limit_ratio == Decimal("0.10")
    assert restored.effective_from == date(2023, 1, 1)


@pytest.mark.unit
def test_csv_digest_mismatch_fails_before_rule_loading(tmp_path: Path) -> None:
    csv_path, manifest_path = _write_resource(tmp_path, [_sz_exception()])
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuleResourceValidationError, match="SHA256 mismatch"):
        load_effective_rule_resolver(
            [_info("SZ.159955", list_date=date(2018, 1, 1))],
            csv_path,
            manifest_path,
        )


@pytest.mark.unit
def test_unknown_record_mode_and_overlapping_periods_fail_closed(tmp_path: Path) -> None:
    bad_mode_dir = tmp_path / "bad-mode"
    bad_mode_dir.mkdir()
    csv_path, manifest_path = _write_resource(
        bad_mode_dir,
        [_sz_exception(record_mode="PREFIX_INFERENCE")],
    )
    info = _info("SZ.159955", list_date=date(2018, 1, 1))
    with pytest.raises(RuleResourceValidationError, match="unsupported record modes"):
        load_effective_rule_resolver([info], csv_path, manifest_path)

    overlap_dir = tmp_path / "overlap"
    overlap_dir.mkdir()
    csv_path, manifest_path = _write_resource(
        overlap_dir,
        [
            _sz_exception(valid_to="2021-12-31"),
            _sz_exception(valid_from="2021-01-01"),
        ],
    )
    with pytest.raises(RuleResourceValidationError, match="overlapping"):
        load_effective_rule_resolver([info], csv_path, manifest_path)


@pytest.mark.unit
def test_unsupported_etf_category_is_rejected_even_when_not_in_resource() -> None:
    unsupported = _info(
        "SH.513100",
        list_date=date(2013, 5, 15),
        primary_category="\u8de8\u5883",
        fund_type="QDII",
    )

    with pytest.raises(ValueError, match="unsupported ETF category"):
        load_effective_rule_resolver([unsupported], CSV_PATH, MANIFEST_PATH)


@pytest.mark.unit
def test_rule_resolution_fails_closed_for_unknown_symbol_or_gap() -> None:
    provenance = RuleProvenance(
        source="test",
        version="v1",
        method="EXACT",
        approximate=False,
    )
    period = EtfRulePeriod(
        symbol="SH.510300",
        effective_from=date(2021, 1, 1),
        effective_to=date(2021, 12, 31),
        etf_category=EtfCategory.DOMESTIC_STOCK_ETF,
        turnover_rule=TurnoverRule.T1,
        price_limit_ratio=Decimal("0.10"),
        lot_size=100,
        tick_size=Decimal("0.001"),
        provenance=provenance,
    )
    resolver = EffectiveDatedEtfRuleResolver([period])

    with pytest.raises(RuleResolutionError, match="no ETF rules"):
        resolver.resolve("SH.510050", date(2021, 1, 4))
    with pytest.raises(RuleResolutionError, match="found 0"):
        resolver.resolve("SH.510300", date(2022, 1, 4))


@pytest.mark.unit
def test_overlapping_manual_effective_periods_are_rejected() -> None:
    provenance = RuleProvenance("test", "v1", "EXACT", False)
    common = {
        "symbol": "SH.510300",
        "etf_category": EtfCategory.DOMESTIC_STOCK_ETF,
        "turnover_rule": TurnoverRule.T1,
        "price_limit_ratio": Decimal("0.10"),
        "lot_size": 100,
        "tick_size": Decimal("0.001"),
        "provenance": provenance,
    }
    left = EtfRulePeriod(
        effective_from=date(2021, 1, 1),
        effective_to=date(2021, 6, 30),
        **common,  # type: ignore[arg-type]
    )
    right = EtfRulePeriod(
        effective_from=date(2021, 6, 30),
        effective_to=None,
        **common,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="overlapping"):
        EffectiveDatedEtfRuleResolver([left, right])
