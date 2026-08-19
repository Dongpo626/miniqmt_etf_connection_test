"""Manifest-verified effective-dated ETF turnover and price-limit rules."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import (
    EtfCategory,
    EtfInfo,
    EtfTradingRule,
    TurnoverRule,
)

_TEN: Final = Decimal("0.10")
_TWENTY: Final = Decimal("0.20")
_ONE_DAY: Final = timedelta(days=1)
_DOMESTIC_CATEGORY: Final = "\u7eaf\u5883\u5185"
_STOCK_FUND_TYPE: Final = "\u80a1\u7968\u578b"
_RESOURCE_NAME: Final = "etf_price_limit_20pct"
_RULE_MODE: Final = "LATEST_SNAPSHOT_WITH_2020_SEED"
_RECORD_MODES: Final = frozenset(
    {
        "CURRENT_SNAPSHOT_RULE_INFERENCE",
        "EFFECTIVE_DATED_OFFICIAL_SEED_CONFIRMED_CURRENT",
        "SNAPSHOT_DERIVED_REMOVAL_CUTOFF",
    }
)
_EXPECTED_COLUMNS: Final = (
    "symbol",
    "exchange",
    "price_limit_ratio",
    "valid_from",
    "valid_to",
    "record_mode",
    "source_id",
)
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class RuleResolutionError(LookupError):
    """No unambiguous effective rule exists for a symbol/date."""


class RuleResourceValidationError(ValueError):
    """The frozen CSV and manifest do not form a valid rule resource."""


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _non_blank(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _manifest_mapping(
    container: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise RuleResourceValidationError(f"manifest {key} must be an object")
    return cast(dict[str, object], value)


def _manifest_text(container: Mapping[str, object], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuleResourceValidationError(f"manifest {key} must be a nonblank string")
    return value.strip()


def _manifest_int(container: Mapping[str, object], key: str) -> int:
    value = container.get(key)
    if type(value) is not int or value < 0:
        raise RuleResourceValidationError(f"manifest {key} must be a non-negative integer")
    return value


def _manifest_string_tuple(container: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = container.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RuleResourceValidationError(f"manifest {key} must be a string array")
    return tuple(cast(list[str], value))


@dataclass(frozen=True, slots=True)
class RuleProvenance:
    """Frozen identity and approximation status of one rule assertion."""

    source: str
    version: str
    method: str
    approximate: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _non_blank(self.source, "source"))
        object.__setattr__(self, "version", _non_blank(self.version, "version"))
        object.__setattr__(self, "method", _non_blank(self.method, "method"))
        if not isinstance(self.approximate, bool):
            raise TypeError("approximate must be bool")


@dataclass(frozen=True, slots=True)
class RuleResourceIdentity:
    """Integrity identity suitable for inclusion in local run metadata."""

    resource_name: str
    resource_version: str
    rule_mode: str
    manifest_sha256: str
    csv_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_name",
            _non_blank(self.resource_name, "resource_name"),
        )
        object.__setattr__(
            self,
            "resource_version",
            _non_blank(self.resource_version, "resource_version"),
        )
        object.__setattr__(self, "rule_mode", _non_blank(self.rule_mode, "rule_mode"))
        for field_name in ("manifest_sha256", "csv_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EtfRulePeriod:
    """One inclusive effective interval for all execution-facing ETF rules."""

    symbol: str
    effective_from: date
    effective_to: date | None
    etf_category: EtfCategory
    turnover_rule: TurnoverRule
    price_limit_ratio: Decimal
    lot_size: int
    tick_size: Decimal
    provenance: RuleProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        start = _plain_date(self.effective_from, "effective_from")
        if self.effective_to is not None:
            end = _plain_date(self.effective_to, "effective_to")
            if end < start:
                raise ValueError("effective_to must not precede effective_from")
        if self.price_limit_ratio not in {_TEN, _TWENTY}:
            raise ValueError("ETF price_limit_ratio must be exactly 0.10 or 0.20")
        if not isinstance(self.provenance, RuleProvenance):
            raise TypeError("provenance must be RuleProvenance")
        self.to_rule()

    def contains(self, trade_date: date) -> bool:
        value = _plain_date(trade_date, "trade_date")
        return self.effective_from <= value and (
            self.effective_to is None or value <= self.effective_to
        )

    def to_rule(self) -> EtfTradingRule:
        return EtfTradingRule(
            symbol=self.symbol,
            etf_category=self.etf_category,
            turnover_rule=self.turnover_rule,
            price_limit_ratio=self.price_limit_ratio,
            lot_size=self.lot_size,
            tick_size=self.tick_size,
        )


@dataclass(frozen=True, slots=True)
class ResolvedEtfRule:
    """Domain rule plus the effective-period evidence used to construct it."""

    rule: EtfTradingRule
    effective_from: date
    effective_to: date | None
    provenance: RuleProvenance


class EffectiveDatedEtfRuleResolver:
    """Strict, deterministic resolver used by the engine once per symbol/date."""

    def __init__(
        self,
        periods: Sequence[EtfRulePeriod],
        *,
        resource_identity: RuleResourceIdentity | None = None,
    ) -> None:
        by_symbol: dict[str, list[EtfRulePeriod]] = {}
        for period in periods:
            if not isinstance(period, EtfRulePeriod):
                raise TypeError("periods must contain EtfRulePeriod")
            by_symbol.setdefault(period.symbol, []).append(period)
        if not by_symbol:
            raise ValueError("periods must not be empty")
        if resource_identity is not None and not isinstance(
            resource_identity, RuleResourceIdentity
        ):
            raise TypeError("resource_identity must be RuleResourceIdentity")
        frozen: dict[str, tuple[EtfRulePeriod, ...]] = {}
        for symbol, symbol_periods in by_symbol.items():
            ordered = tuple(sorted(symbol_periods, key=lambda item: item.effective_from))
            for left, right in pairwise(ordered):
                if left.effective_to is None or left.effective_to >= right.effective_from:
                    raise ValueError(f"overlapping rule periods for {symbol}")
            frozen[symbol] = ordered
        self._periods_by_symbol: Mapping[str, tuple[EtfRulePeriod, ...]] = MappingProxyType(
            dict(sorted(frozen.items()))
        )
        self._resource_identity = resource_identity

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._periods_by_symbol)

    @property
    def periods(self) -> tuple[EtfRulePeriod, ...]:
        return tuple(
            period for symbol in self.symbols for period in self._periods_by_symbol[symbol]
        )

    @property
    def resource_identity(self) -> RuleResourceIdentity | None:
        return self._resource_identity

    def resolve(self, symbol: str, trade_date: date) -> EtfTradingRule:
        """Resolve exactly one rule or fail closed on an uncovered date."""

        return self.resolve_with_provenance(symbol, trade_date).rule

    def resolve_with_provenance(self, symbol: str, trade_date: date) -> ResolvedEtfRule:
        canonical = normalize_symbol(symbol)
        value = _plain_date(trade_date, "trade_date")
        try:
            periods = self._periods_by_symbol[canonical]
        except KeyError:
            raise RuleResolutionError(f"no ETF rules configured for {canonical}") from None
        matches = tuple(period for period in periods if period.contains(value))
        if len(matches) != 1:
            raise RuleResolutionError(
                f"expected exactly one ETF rule for {canonical} on {value}; found {len(matches)}"
            )
        period = matches[0]
        return ResolvedEtfRule(
            rule=period.to_rule(),
            effective_from=period.effective_from,
            effective_to=period.effective_to,
            provenance=period.provenance,
        )


@dataclass(frozen=True, slots=True)
class _TwentyPercentResourcePeriod:
    symbol: str
    exchange: str
    valid_from: date
    valid_to: date | None
    record_mode: str
    source_id: str


@dataclass(frozen=True, slots=True)
class _LoadedRuleResource:
    identity: RuleResourceIdentity
    periods: tuple[_TwentyPercentResourcePeriod, ...]


def _read_manifest(path: Path) -> tuple[Mapping[str, object], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuleResourceValidationError(f"cannot read rule manifest: {path}") from error
    try:
        loaded: object = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuleResourceValidationError("rule manifest is not valid UTF-8 JSON") from error
    if not isinstance(loaded, dict):
        raise RuleResourceValidationError("rule manifest root must be an object")
    return cast(dict[str, object], loaded), hashlib.sha256(payload).hexdigest()


def _manifest_source_ids(manifest: Mapping[str, object]) -> frozenset[str]:
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuleResourceValidationError("manifest sources must be a non-empty array")
    source_ids: set[str] = set()
    for source in sources:
        if not isinstance(source, dict):
            raise RuleResourceValidationError("manifest source entries must be objects")
        source_id = _manifest_text(cast(dict[str, object], source), "source_id")
        if source_id in source_ids:
            raise RuleResourceValidationError(f"duplicate manifest source_id: {source_id}")
        source_ids.add(source_id)
    return frozenset(source_ids)


def _manifest_count_mapping(
    container: Mapping[str, object],
    key: str,
) -> Mapping[str, int]:
    raw = _manifest_mapping(container, key)
    counts: dict[str, int] = {}
    for source_id, value in raw.items():
        if type(value) is not int or value < 0:
            raise RuleResourceValidationError(
                f"manifest csv.{key} values must be non-negative integers"
            )
        counts[source_id] = value
    return MappingProxyType(counts)


def _parse_iso_date(value: str, field_name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise RuleResourceValidationError(f"invalid {field_name}: {value!r}") from error
    if parsed.isoformat() != value:
        raise RuleResourceValidationError(f"{field_name} must use canonical YYYY-MM-DD")
    return parsed


def _read_csv_periods(
    *,
    csv_path: Path,
    expected_digest: str,
    expected_rows: int,
    expected_source_counts: Mapping[str, int],
    expected_record_mode_counts: Mapping[str, int],
    known_source_ids: frozenset[str],
) -> tuple[_TwentyPercentResourcePeriod, ...]:
    try:
        payload = csv_path.read_bytes()
    except OSError as error:
        raise RuleResourceValidationError(f"cannot read rule CSV: {csv_path}") from error
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        raise RuleResourceValidationError(
            f"rule CSV SHA256 mismatch: expected {expected_digest}, got {actual_digest}"
        )

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuleResourceValidationError("rule CSV is not valid UTF-8") from error

    reader = csv.DictReader(text.splitlines())
    if tuple(reader.fieldnames or ()) != _EXPECTED_COLUMNS:
        raise RuleResourceValidationError("rule CSV columns do not match the manifest schema")
    periods: list[_TwentyPercentResourcePeriod] = []
    source_counts: Counter[str] = Counter()
    record_mode_counts: Counter[str] = Counter()
    raw_order: list[tuple[str, date]] = []
    for line_number, raw_row in enumerate(reader, start=2):
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise RuleResourceValidationError(f"malformed rule CSV row {line_number}")
        row = cast(dict[str, str], raw_row)
        try:
            supplied_symbol = row["symbol"].strip()
            symbol = normalize_symbol(supplied_symbol)
        except (TypeError, ValueError) as error:
            raise RuleResourceValidationError(
                f"invalid symbol on rule CSV row {line_number}"
            ) from error
        if supplied_symbol != symbol:
            raise RuleResourceValidationError("rule CSV symbols must use canonical SH./SZ. form")
        exchange = row["exchange"].strip()
        expected_exchange = "SSE" if symbol.startswith("SH.") else "SZSE"
        if exchange != expected_exchange:
            raise RuleResourceValidationError(f"exchange mismatch for {symbol}")
        try:
            ratio = Decimal(row["price_limit_ratio"].strip())
        except Exception as error:
            raise RuleResourceValidationError(f"invalid price ratio for {symbol}") from error
        if ratio != _TWENTY:
            raise RuleResourceValidationError(f"resource ratio for {symbol} must be 0.20")

        valid_from = _parse_iso_date(row["valid_from"].strip(), "valid_from")
        raw_valid_to = row["valid_to"].strip()
        valid_to = _parse_iso_date(raw_valid_to, "valid_to") if raw_valid_to else None
        if valid_to is not None and valid_to < valid_from:
            raise RuleResourceValidationError(f"valid_to precedes valid_from for {symbol}")
        record_mode = row["record_mode"].strip()
        if record_mode not in _RECORD_MODES:
            raise RuleResourceValidationError(f"unknown record_mode for {symbol}: {record_mode}")
        if record_mode != "CURRENT_SNAPSHOT_RULE_INFERENCE" and exchange != "SZSE":
            raise RuleResourceValidationError(f"record_mode conflicts with exchange for {symbol}")
        source_id = row["source_id"].strip()
        if source_id not in known_source_ids:
            raise RuleResourceValidationError(f"unknown source_id for {symbol}: {source_id}")

        periods.append(
            _TwentyPercentResourcePeriod(
                symbol=symbol,
                exchange=exchange,
                valid_from=valid_from,
                valid_to=valid_to,
                record_mode=record_mode,
                source_id=source_id,
            )
        )
        raw_order.append((symbol, valid_from))
        source_counts[source_id] += 1
        record_mode_counts[record_mode] += 1

    if len(periods) != expected_rows:
        raise RuleResourceValidationError(
            f"rule CSV row count mismatch: expected {expected_rows}, got {len(periods)}"
        )
    if dict(source_counts) != dict(expected_source_counts):
        raise RuleResourceValidationError("rule CSV source counts do not match manifest")
    if dict(record_mode_counts) != dict(expected_record_mode_counts):
        raise RuleResourceValidationError("rule CSV record-mode counts do not match manifest")
    if raw_order != sorted(raw_order):
        raise RuleResourceValidationError("rule CSV rows must be sorted by symbol and valid_from")

    by_symbol: dict[str, list[_TwentyPercentResourcePeriod]] = {}
    for period in periods:
        by_symbol.setdefault(period.symbol, []).append(period)
    for symbol, symbol_periods in by_symbol.items():
        for left, right in pairwise(symbol_periods):
            if left.valid_to is None or left.valid_to >= right.valid_from:
                raise RuleResourceValidationError(
                    f"duplicate or overlapping 20% periods for {symbol}"
                )
    return tuple(periods)


def _load_rule_resource(csv_path: Path, manifest_path: Path) -> _LoadedRuleResource:
    manifest, manifest_digest = _read_manifest(manifest_path)
    resource_name = _manifest_text(manifest, "resource_name")
    if resource_name != _RESOURCE_NAME:
        raise RuleResourceValidationError(f"unexpected resource_name: {resource_name}")
    resource_version = _manifest_text(manifest, "resource_version")
    rule_mode = _manifest_text(manifest, "rule_mode")
    if rule_mode != _RULE_MODE:
        raise RuleResourceValidationError(f"unsupported rule_mode: {rule_mode}")
    try:
        default_ratio = Decimal(_manifest_text(manifest, "default_price_limit_ratio"))
        covered_ratio = Decimal(_manifest_text(manifest, "covered_price_limit_ratio"))
    except Exception as error:
        raise RuleResourceValidationError("manifest price-limit ratios are invalid") from error
    if default_ratio != _TEN or covered_ratio != _TWENTY:
        raise RuleResourceValidationError("manifest must define a 10% base and 20% exception")

    csv_metadata = _manifest_mapping(manifest, "csv")
    manifest_csv_path = _manifest_text(csv_metadata, "path")
    if Path(manifest_csv_path).name != manifest_csv_path:
        raise RuleResourceValidationError("manifest csv.path must be a local filename")
    expected_csv_path = (manifest_path.parent / manifest_csv_path).resolve()
    if expected_csv_path != csv_path.resolve():
        raise RuleResourceValidationError("CSV path does not match manifest csv.path")
    expected_digest = _manifest_text(csv_metadata, "sha256")
    if _SHA256.fullmatch(expected_digest) is None:
        raise RuleResourceValidationError("manifest csv.sha256 is invalid")
    if _manifest_string_tuple(csv_metadata, "columns") != _EXPECTED_COLUMNS:
        raise RuleResourceValidationError("manifest CSV columns are unsupported")
    expected_rows = _manifest_int(csv_metadata, "row_count")
    expected_source_counts = _manifest_count_mapping(csv_metadata, "source_counts")
    expected_record_mode_counts = _manifest_count_mapping(csv_metadata, "record_mode_counts")
    if not set(expected_record_mode_counts).issubset(_RECORD_MODES):
        raise RuleResourceValidationError("manifest declares unsupported record modes")
    source_ids = _manifest_source_ids(manifest)
    if not set(expected_source_counts).issubset(source_ids):
        raise RuleResourceValidationError("manifest source_counts references unknown sources")

    periods = _read_csv_periods(
        csv_path=csv_path,
        expected_digest=expected_digest,
        expected_rows=expected_rows,
        expected_source_counts=expected_source_counts,
        expected_record_mode_counts=expected_record_mode_counts,
        known_source_ids=source_ids,
    )
    return _LoadedRuleResource(
        identity=RuleResourceIdentity(
            resource_name=resource_name,
            resource_version=resource_version,
            rule_mode=rule_mode,
            manifest_sha256=manifest_digest,
            csv_sha256=expected_digest,
        ),
        periods=periods,
    )


def _is_gold(info: EtfInfo) -> bool:
    return info.primary_category == _DOMESTIC_CATEGORY and info.symbol.partition(".")[2].startswith(
        "518"
    )


def _is_domestic_stock(info: EtfInfo) -> bool:
    return info.primary_category == _DOMESTIC_CATEGORY and info.fund_type == _STOCK_FUND_TYPE


def _period(
    *,
    info: EtfInfo,
    effective_from: date,
    effective_to: date | None,
    category: EtfCategory,
    turnover: TurnoverRule,
    ratio: Decimal,
    provenance: RuleProvenance,
    lot_size: int,
    tick_size: Decimal,
) -> EtfRulePeriod:
    return EtfRulePeriod(
        symbol=info.symbol,
        effective_from=effective_from,
        effective_to=effective_to,
        etf_category=category,
        turnover_rule=turnover,
        price_limit_ratio=ratio,
        lot_size=lot_size,
        tick_size=tick_size,
        provenance=provenance,
    )


def _stock_periods(
    *,
    info: EtfInfo,
    exceptions: Sequence[_TwentyPercentResourcePeriod],
    identity: RuleResourceIdentity,
    lot_size: int,
    tick_size: Decimal,
) -> tuple[EtfRulePeriod, ...]:
    base_provenance = RuleProvenance(
        source=identity.resource_name,
        version=identity.resource_version,
        method="DEFAULT_10PCT_RESOURCE_FALLBACK",
        approximate=True,
    )
    periods: list[EtfRulePeriod] = []
    cursor = info.list_date
    lifecycle_end = info.delist_date

    for exception in exceptions:
        exception_start = max(info.list_date, exception.valid_from)
        if lifecycle_end is not None and exception_start > lifecycle_end:
            continue
        exception_end = exception.valid_to
        if lifecycle_end is not None and (exception_end is None or exception_end > lifecycle_end):
            exception_end = lifecycle_end
        if exception_end is not None and exception_end < info.list_date:
            continue
        if exception_start > cursor:
            periods.append(
                _period(
                    info=info,
                    effective_from=cursor,
                    effective_to=exception_start - _ONE_DAY,
                    category=EtfCategory.DOMESTIC_STOCK_ETF,
                    turnover=TurnoverRule.T1,
                    ratio=_TEN,
                    provenance=base_provenance,
                    lot_size=lot_size,
                    tick_size=tick_size,
                )
            )
        twenty_provenance = RuleProvenance(
            source=f"{identity.resource_name}:{exception.source_id}",
            version=identity.resource_version,
            method=(
                exception.record_mode
                if exception.valid_to is None
                else f"{exception.record_mode}_WITH_VALID_TO"
            ),
            approximate=True,
        )
        periods.append(
            _period(
                info=info,
                effective_from=exception_start,
                effective_to=exception_end,
                category=EtfCategory.DOMESTIC_STOCK_ETF,
                turnover=TurnoverRule.T1,
                ratio=_TWENTY,
                provenance=twenty_provenance,
                lot_size=lot_size,
                tick_size=tick_size,
            )
        )
        if exception_end is None or exception_end == lifecycle_end:
            return tuple(periods)
        cursor = exception_end + _ONE_DAY

    if lifecycle_end is None or cursor <= lifecycle_end:
        periods.append(
            _period(
                info=info,
                effective_from=cursor,
                effective_to=lifecycle_end,
                category=EtfCategory.DOMESTIC_STOCK_ETF,
                turnover=TurnoverRule.T1,
                ratio=_TEN,
                provenance=base_provenance,
                lot_size=lot_size,
                tick_size=tick_size,
            )
        )
    return tuple(periods)


def load_effective_rule_resolver(
    etf_infos: Sequence[EtfInfo],
    csv_path: Path,
    manifest_path: Path,
    *,
    lot_size: int = 100,
    tick_size: Decimal = Decimal("0.001"),
) -> EffectiveDatedEtfRuleResolver:
    """Verify the frozen resource and build complete lifecycle rule periods.

    The CSV contains only 20% exceptions.  Every uncovered domestic stock ETF
    date receives the auditable 10% fallback.  A finite ``valid_to`` is
    inclusive and is followed by a restored 10% period.  Gold ETFs remain 10%
    regardless of any accidental resource membership.
    """

    if not isinstance(etf_infos, Sequence):
        raise TypeError("etf_infos must be a sequence")
    if not isinstance(csv_path, Path) or not isinstance(manifest_path, Path):
        raise TypeError("csv_path and manifest_path must be pathlib.Path")
    info_by_symbol: dict[str, EtfInfo] = {}
    for info in etf_infos:
        if not isinstance(info, EtfInfo):
            raise TypeError("etf_infos must contain EtfInfo")
        if info.symbol in info_by_symbol:
            raise ValueError(f"duplicate EtfInfo for {info.symbol}")
        info_by_symbol[info.symbol] = info
    if not info_by_symbol:
        raise ValueError("etf_infos must not be empty")

    resource = _load_rule_resource(csv_path, manifest_path)
    exceptions_by_symbol: dict[str, list[_TwentyPercentResourcePeriod]] = {}
    for exception in resource.periods:
        if exception.symbol in info_by_symbol:
            exceptions_by_symbol.setdefault(exception.symbol, []).append(exception)

    periods: list[EtfRulePeriod] = []
    gold_provenance = RuleProvenance(
        source=resource.identity.resource_name,
        version=resource.identity.resource_version,
        method="GOLD_ETF_FIXED_10PCT",
        approximate=False,
    )
    for symbol in sorted(info_by_symbol):
        info = info_by_symbol[symbol]
        exceptions = tuple(exceptions_by_symbol.get(symbol, ()))
        if _is_gold(info):
            if exceptions:
                raise RuleResourceValidationError(
                    f"20% resource must not contain supported gold ETF {symbol}"
                )
            periods.append(
                _period(
                    info=info,
                    effective_from=info.list_date,
                    effective_to=info.delist_date,
                    category=EtfCategory.GOLD_ETF,
                    turnover=TurnoverRule.T0,
                    ratio=_TEN,
                    provenance=gold_provenance,
                    lot_size=lot_size,
                    tick_size=tick_size,
                )
            )
            continue
        if not _is_domestic_stock(info):
            raise ValueError(f"unsupported ETF category for rule resolution: {symbol}")
        periods.extend(
            _stock_periods(
                info=info,
                exceptions=exceptions,
                identity=resource.identity,
                lot_size=lot_size,
                tick_size=tick_size,
            )
        )
    return EffectiveDatedEtfRuleResolver(
        periods,
        resource_identity=resource.identity,
    )


__all__ = [
    "EffectiveDatedEtfRuleResolver",
    "EtfRulePeriod",
    "ResolvedEtfRule",
    "RuleProvenance",
    "RuleResolutionError",
    "RuleResourceIdentity",
    "RuleResourceValidationError",
    "load_effective_rule_resolver",
]
