"""Resolve explicit symbols and named current-master pools into one run universe."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.market import EtfInfo
from etf_backtest.data.mysql import QmtEtfMasterRecord

_SUPPORTED_POOLS = frozenset({"domestic_stock_etf", "gold_etf", "all_supported_etf"})
_CURRENT_STATUSES = frozenset({"LISTED", "DELISTED"})
_DOMESTIC_CATEGORY = "\u7eaf\u5883\u5185"
_STOCK_FUND_TYPE = "\u80a1\u7968\u578b"


class UniverseResolutionError(ValueError):
    """The requested universe cannot be frozen without silent assumptions."""


class UniverseRepository(Protocol):
    """Read-only master-data capabilities required by the resolver."""

    @property
    def dataset_version(self) -> str: ...

    def load_etf_master(self, symbols: Sequence[str]) -> tuple[QmtEtfMasterRecord, ...]: ...

    def load_pool_etf_master(self, pool_name: str) -> tuple[QmtEtfMasterRecord, ...]: ...

    def load_last_raw_trade_dates(self, symbols: Sequence[str]) -> Mapping[str, date]: ...


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


def _normalize_pools(pools: Sequence[str]) -> tuple[str, ...]:
    if isinstance(pools, (str, bytes)) or not isinstance(pools, Sequence):
        raise TypeError("pools must be a sequence of strings")
    normalized = tuple(sorted({pool.strip() for pool in pools}))
    if any(not pool for pool in normalized):
        raise ValueError("pool names must not be blank")
    unsupported = tuple(pool for pool in normalized if pool not in _SUPPORTED_POOLS)
    if unsupported:
        raise ValueError(f"unsupported ETF pools: {unsupported!r}")
    return normalized


def _normalize_explicit(symbols: Sequence[str]) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, Sequence):
        raise TypeError("explicit_symbols must be a sequence")
    return tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))


@dataclass(frozen=True, slots=True)
class FrozenUniverseMember:
    """One master record and its deterministic run-interval decision."""

    info: EtfInfo
    sources: tuple[str, ...]
    effective_from: date
    effective_to: date
    included: bool
    exclusion_reason: str | None
    pool_classification_approximated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.info, EtfInfo):
            raise TypeError("info must be EtfInfo")
        if not self.sources or any(not source.strip() for source in self.sources):
            raise ValueError("sources must contain nonblank identities")
        if tuple(sorted(set(self.sources))) != self.sources:
            raise ValueError("sources must be sorted and unique")
        _plain_date(self.effective_from, "effective_from")
        _plain_date(self.effective_to, "effective_to")
        if not isinstance(self.included, bool):
            raise TypeError("included must be bool")
        if self.included and self.effective_from > self.effective_to:
            raise ValueError("included member has an empty effective interval")
        if self.included and self.exclusion_reason is not None:
            raise ValueError("included member cannot have an exclusion reason")
        if not self.included and not self.exclusion_reason:
            raise ValueError("excluded member requires an exclusion reason")

    @property
    def symbol(self) -> str:
        return self.info.symbol

    def is_active(self, trade_date: date) -> bool:
        value = _plain_date(trade_date, "trade_date")
        return self.included and self.effective_from <= value <= self.effective_to


@dataclass(frozen=True, slots=True)
class FrozenUniverse:
    """Resolved union plus every inclusion/exclusion decision for audit output."""

    dataset_version: str
    run_start_date: date
    run_end_date: date
    explicit_symbols: tuple[str, ...]
    pools: tuple[str, ...]
    decisions: tuple[FrozenUniverseMember, ...]
    approximation_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dataset_version.strip():
            raise ValueError("dataset_version must not be blank")
        start = _plain_date(self.run_start_date, "run_start_date")
        end = _plain_date(self.run_end_date, "run_end_date")
        if start > end:
            raise ValueError("run_start_date must not follow run_end_date")
        symbols = tuple(decision.symbol for decision in self.decisions)
        if symbols != tuple(sorted(set(symbols))):
            raise ValueError("universe decisions must have unique sorted symbols")

    @property
    def members(self) -> tuple[FrozenUniverseMember, ...]:
        return tuple(decision for decision in self.decisions if decision.included)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members)

    @property
    def etf_infos(self) -> tuple[EtfInfo, ...]:
        return tuple(member.info for member in self.members)

    def active_symbols(self, trade_date: date) -> tuple[str, ...]:
        return tuple(member.symbol for member in self.members if member.is_active(trade_date))

    def member_by_symbol(self) -> Mapping[str, FrozenUniverseMember]:
        return MappingProxyType({member.symbol: member for member in self.members})

    def csv_rows(self) -> tuple[Mapping[str, str], ...]:
        rows: list[Mapping[str, str]] = []
        for decision in self.decisions:
            info = decision.info
            rows.append(
                {
                    "symbol": info.symbol,
                    "sources": "|".join(decision.sources),
                    "exchange": info.exchange.value,
                    "name": info.name,
                    "primary_category": info.primary_category,
                    "fund_type": info.fund_type,
                    "list_date": info.list_date.isoformat(),
                    "delist_date": "" if info.delist_date is None else info.delist_date.isoformat(),
                    "current_status": info.current_status,
                    "delist_date_approximated": str(info.delist_date_approximated).lower(),
                    "pool_classification_approximated": str(
                        decision.pool_classification_approximated
                    ).lower(),
                    "effective_from": decision.effective_from.isoformat(),
                    "effective_to": decision.effective_to.isoformat(),
                    "included": str(decision.included).lower(),
                    "exclusion_reason": decision.exclusion_reason or "",
                    "dataset_version": self.dataset_version,
                }
            )
        return tuple(rows)

    def write_csv(self, path: Path) -> Path:
        """Write the stable ``universe.csv`` audit artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = (
            "symbol",
            "sources",
            "exchange",
            "name",
            "primary_category",
            "fund_type",
            "list_date",
            "delist_date",
            "current_status",
            "delist_date_approximated",
            "pool_classification_approximated",
            "effective_from",
            "effective_to",
            "included",
            "exclusion_reason",
            "dataset_version",
        )
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.csv_rows())
        return target


class FrozenUniverseResolver:
    """Resolve an explicit-plus-pools union against one frozen current master."""

    def __init__(self, repository: UniverseRepository) -> None:
        self._repository = repository

    def resolve(
        self,
        *,
        explicit_symbols: Sequence[str] = (),
        pools: Sequence[str] = (),
        start_date: date,
        end_date: date,
    ) -> FrozenUniverse:
        start = _plain_date(start_date, "start_date")
        end = _plain_date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date must not follow end_date")
        explicit = _normalize_explicit(explicit_symbols)
        normalized_pools = _normalize_pools(pools)
        if not explicit and not normalized_pools:
            raise ValueError("universe requires explicit symbols or a named pool")

        records: dict[str, QmtEtfMasterRecord] = {}
        sources: dict[str, set[str]] = {}
        if explicit:
            for record in self._repository.load_etf_master(explicit):
                self._merge_record(records, record)
                sources.setdefault(record.symbol, set()).add("explicit")
        for pool in normalized_pools:
            for record in self._repository.load_pool_etf_master(pool):
                self._merge_record(records, record)
                sources.setdefault(record.symbol, set()).add(f"pool:{pool}")

        for record in records.values():
            self._validate_supported_record(record)

        missing_delist_symbols = tuple(
            sorted(
                record.symbol
                for record in records.values()
                if record.delist_date is None
                and record.current_status.strip().upper() == "DELISTED"
            )
        )
        last_trade_dates = (
            self._repository.load_last_raw_trade_dates(missing_delist_symbols)
            if missing_delist_symbols
            else MappingProxyType({})
        )

        decisions: list[FrozenUniverseMember] = []
        flags: set[str] = set()
        if normalized_pools:
            flags.add("CURRENT_MASTER_POOL_CLASSIFICATION")
        for symbol in sorted(records):
            record = records[symbol]
            delist_date = record.delist_date
            approximated = False
            if symbol in missing_delist_symbols:
                try:
                    delist_date = last_trade_dates[symbol]
                except KeyError:
                    raise UniverseResolutionError(
                        f"delisted ETF {symbol} lacks delist_date and raw last date"
                    ) from None
                if delist_date < record.list_date:
                    raise UniverseResolutionError(
                        f"approximated delist date precedes listing for {symbol}"
                    )
                approximated = True
                flags.add("MISSING_DELIST_DATE_LAST_RAW_DATE")

            info = self._to_etf_info(
                record,
                delist_date=delist_date,
                delist_date_approximated=approximated,
            )
            effective_from = max(start, info.list_date)
            effective_to = min(end, info.delist_date or end)
            included = effective_from <= effective_to
            if included:
                exclusion_reason = None
            elif info.list_date > end:
                exclusion_reason = "LISTED_AFTER_RUN"
            else:
                exclusion_reason = "DELISTED_BEFORE_RUN"
            member_sources = tuple(sorted(sources[symbol]))
            decisions.append(
                FrozenUniverseMember(
                    info=info,
                    sources=member_sources,
                    effective_from=effective_from,
                    effective_to=effective_to,
                    included=included,
                    exclusion_reason=exclusion_reason,
                    pool_classification_approximated=any(
                        source.startswith("pool:") for source in member_sources
                    ),
                )
            )

        if not any(decision.included for decision in decisions):
            raise UniverseResolutionError("resolved universe has no member active in the run")
        return FrozenUniverse(
            dataset_version=self._repository.dataset_version,
            run_start_date=start,
            run_end_date=end,
            explicit_symbols=explicit,
            pools=normalized_pools,
            decisions=tuple(decisions),
            approximation_flags=tuple(sorted(flags)),
        )

    @staticmethod
    def _merge_record(records: dict[str, QmtEtfMasterRecord], record: QmtEtfMasterRecord) -> None:
        existing = records.get(record.symbol)
        if existing is not None and existing != record:
            raise UniverseResolutionError(f"conflicting current-master rows for {record.symbol}")
        records[record.symbol] = record

    @staticmethod
    def _validate_supported_record(record: QmtEtfMasterRecord) -> None:
        status = record.current_status.strip().upper()
        if status not in _CURRENT_STATUSES:
            raise UniverseResolutionError(
                f"unsupported dim_etf current_status for {record.symbol}: {record.current_status!r}"
            )
        domestic = record.primary_category == _DOMESTIC_CATEGORY
        stock = domestic and record.fund_type == _STOCK_FUND_TYPE
        gold = domestic and record.etf_code.startswith("518")
        if not (stock or gold):
            raise UniverseResolutionError(
                f"unsupported ETF category for universe member {record.symbol}"
            )

    @staticmethod
    def _to_etf_info(
        record: QmtEtfMasterRecord,
        *,
        delist_date: date | None,
        delist_date_approximated: bool,
    ) -> EtfInfo:
        if record.primary_category is None or record.fund_type is None:
            raise UniverseResolutionError(f"supported ETF {record.symbol} lacks category metadata")
        return EtfInfo(
            symbol=record.symbol,
            exchange=record.exchange,
            name=record.name,
            primary_category=record.primary_category,
            fund_type=record.fund_type,
            list_date=record.list_date,
            delist_date=delist_date,
            current_status=record.current_status,
            delist_date_approximated=delist_date_approximated,
        )


__all__ = [
    "FrozenUniverse",
    "FrozenUniverseMember",
    "FrozenUniverseResolver",
    "UniverseRepository",
    "UniverseResolutionError",
]
