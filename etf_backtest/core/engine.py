"""Pure daily D-close signal to D+1-close execution engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal
from itertools import pairwise
from types import MappingProxyType
from typing import Protocol

from etf_backtest.application.contracts import DecisionStatus, StrategyDataPortal
from etf_backtest.application.daily_decision import DailyDecisionService
from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import Account, DailySnapshot
from etf_backtest.core.fill import FillModel
from etf_backtest.core.market import (
    EtfInfo,
    EtfTradingRule,
    IndexBarView,
    MarketBarView,
    MarketFrame,
    PriceLimitSource,
)
from etf_backtest.core.order import (
    ExecutionEstimate,
    FillResult,
    Order,
    RuleCheckResult,
    TradePriceQuote,
)
from etf_backtest.core.order_generator import OrderGenerator
from etf_backtest.core.pricing import TradePriceQuoteCache
from etf_backtest.core.rule_resolver import RuleResolver
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy.base import BaseStrategy
from etf_backtest.strategy.context import AccountView


def _plain_date(value: object, field_name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{field_name} must be datetime.date")
    return value


class DailyPortal(StrategyDataPortal, Protocol):
    @property
    def symbols(self) -> tuple[str, ...]: ...

    @property
    def etf_infos(self) -> tuple[EtfInfo, ...]: ...

    @property
    def trading_calendar(self) -> object: ...

    def execution_frames(self, start_date: date, end_date: date) -> tuple[MarketFrame, ...]: ...

    def views_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
        lookback_trading_days: int | None = None,
    ) -> tuple[MarketBarView, ...]: ...

    def share_history_through(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[date, Decimal]]: ...

    def huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, Mapping[str, tuple[date, Decimal]]]: ...

    def index_history_through(
        self,
        as_of_date: date,
        *,
        lookback_trading_days: int | None = None,
    ) -> Mapping[str, tuple[IndexBarView, ...]]: ...

    def combined_huijin_ratios_as_of(
        self,
        as_of_date: date,
        *,
        symbols: Sequence[str] | None = None,
    ) -> Mapping[str, tuple[date, Decimal]]: ...


class QuantityRuleEngine(Protocol):
    def approve_batch(
        self,
        *,
        frame: MarketFrame,
        orders: Sequence[Order],
        quotes: Mapping[str, TradePriceQuote],
        estimates: Mapping[str, ExecutionEstimate],
        account: Account,
        trading_rules: Mapping[str, EtfTradingRule],
        etf_infos: Mapping[str, EtfInfo],
    ) -> Sequence[RuleCheckResult]: ...


@dataclass(frozen=True, slots=True)
class TargetDecision:
    signal_date: date
    execution_date: date
    target_portfolio: TargetPortfolio

    def __post_init__(self) -> None:
        signal = _plain_date(self.signal_date, "signal_date")
        execution = _plain_date(self.execution_date, "execution_date")
        if execution <= signal:
            raise ValueError("execution_date must follow signal_date")
        if not isinstance(self.target_portfolio, TargetPortfolio):
            raise TypeError("target_portfolio must be TargetPortfolio")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    etf_infos: tuple[EtfInfo, ...]
    daily_snapshots: tuple[DailySnapshot, ...]
    orders: tuple[Order, ...]
    fills: tuple[FillResult, ...]
    decisions: tuple[TargetDecision, ...]
    approvals: tuple[RuleCheckResult, ...]


class BacktestEngine:
    """Advance complete SSE daily frames in the one permitted event order."""

    __slots__ = (
        "_account",
        "_decision_service",
        "_etf_infos",
        "_fill_model",
        "_order_generator",
        "_portal",
        "_rule_engine",
        "_rule_resolver",
        "_strategy",
    )

    def __init__(
        self,
        *,
        portal: DailyPortal,
        account: Account,
        strategy: BaseStrategy,
        rule_resolver: RuleResolver,
        rule_engine: QuantityRuleEngine,
        order_generator: OrderGenerator,
        fill_model: FillModel,
        decision_service: DailyDecisionService | None = None,
    ) -> None:
        for method_name in (
            "execution_frames",
            "views_through",
            "share_history_through",
            "huijin_ratios_as_of",
            "index_history_through",
            "combined_huijin_ratios_as_of",
        ):
            if not callable(getattr(portal, method_name, None)):
                raise TypeError("portal must satisfy the daily data boundary")
        if not isinstance(account, Account):
            raise TypeError("account must be Account")
        if not isinstance(strategy, BaseStrategy):
            raise TypeError("strategy must be BaseStrategy")
        if not isinstance(rule_resolver, RuleResolver):
            raise TypeError("rule_resolver must satisfy RuleResolver")
        if not callable(getattr(rule_engine, "approve_batch", None)):
            raise TypeError("rule_engine must provide approve_batch")
        if not isinstance(order_generator, OrderGenerator):
            raise TypeError("order_generator must be OrderGenerator")
        if not isinstance(fill_model, FillModel):
            raise TypeError("fill_model must be FillModel")

        symbols = tuple(sorted(normalize_symbol(symbol) for symbol in portal.symbols))
        if set(symbols) != set(account.positions):
            raise ValueError("account positions must exactly cover portal symbols")
        infos: dict[str, EtfInfo] = {}
        for info in portal.etf_infos:
            if not isinstance(info, EtfInfo):
                raise TypeError("portal.etf_infos may contain only EtfInfo")
            if info.symbol in infos:
                raise ValueError("duplicate EtfInfo")
            infos[info.symbol] = info
        if set(infos) != set(symbols):
            raise ValueError("EtfInfo must exactly cover portal symbols")

        self._portal = portal
        self._account = account
        self._strategy = strategy
        self._decision_service = decision_service or DailyDecisionService()
        self._rule_resolver = rule_resolver
        self._rule_engine = rule_engine
        self._order_generator = order_generator
        self._fill_model = fill_model
        self._etf_infos = MappingProxyType(dict(sorted(infos.items())))

    @property
    def account(self) -> Account:
        return self._account

    def run(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        start = _plain_date(start_date, "start_date")
        end = _plain_date(end_date, "end_date")
        if end < start:
            raise ValueError("end_date must not precede start_date")
        frames = tuple(self._portal.execution_frames(start, end))
        if not frames:
            raise ValueError("backtest interval contains no complete execution frame")
        self._validate_frame_sequence(frames)

        pending: TargetDecision | None = None
        daily_values: list[DailySnapshot] = []
        all_fills: list[FillResult] = []
        all_orders: list[Order] = []
        decisions: list[TargetDecision] = []
        approvals: list[RuleCheckResult] = []

        for frame_index, frame in enumerate(frames):
            # 1. The date transition releases T+1 inventory before any order.
            self._account.on_new_trade_date()

            # Rules are resolved afresh for this symbol/date; no static map is
            # allowed to leak across an effective-date boundary.
            trading_rules = self._resolve_rules(frame)
            raw_closes = {symbol: bar.close for symbol, bar in frame.bars_by_symbol.items()}
            if pending is not None:
                if pending.execution_date != frame.trade_date:
                    raise RuntimeError("pending target did not reach its bound D+1 frame")
                frame_fills, frame_approvals, frame_orders = self._execute_pending(
                    decision=pending,
                    frame=frame,
                    raw_closes=raw_closes,
                    trading_rules=trading_rules,
                )
                all_fills.extend(frame_fills)
                all_orders.extend(frame_orders)
                approvals.extend(frame_approvals)
                pending = None

            # 3. NAV is recorded only after formal fills at the same raw close.
            daily = DailySnapshot(
                trade_date=frame.trade_date,
                account_snapshot=self._account.snapshot(raw_closes),
            )
            daily_values.append(daily)

            # 4. The final frame never creates a dangling target.
            if frame_index == len(frames) - 1:
                continue
            next_frame = frames[frame_index + 1]
            account_view = AccountView.from_account(self._account)
            signal_snapshot = daily.account_snapshot
            current_weights = {
                symbol: (
                    signal_snapshot.position_values.get(symbol, Decimal("0"))
                    / signal_snapshot.total_asset
                    if signal_snapshot.total_asset > Decimal("0")
                    else Decimal("0")
                )
                for symbol in self._account.positions
            }
            decision_result = self._decision_service.evaluate(
                strategy=self._strategy,
                portal=self._portal,
                signal_date=frame.trade_date,
                execution_date=next_frame.trade_date,
                schedule_index=frame_index,
                symbols=tuple(self._account.positions),
                account_view=account_view,
                current_weights_by_symbol=current_weights,
            )
            if decision_result.status is not DecisionStatus.TARGET_CREATED:
                continue
            target = decision_result.target_portfolio
            assert target is not None
            pending = TargetDecision(
                signal_date=decision_result.signal_date,
                execution_date=decision_result.execution_date,
                target_portfolio=target,
            )
            decisions.append(pending)

        return BacktestResult(
            etf_infos=tuple(self._etf_infos.values()),
            daily_snapshots=tuple(daily_values),
            orders=tuple(all_orders),
            fills=tuple(all_fills),
            decisions=tuple(decisions),
            approvals=tuple(approvals),
        )

    def _execute_pending(
        self,
        *,
        decision: TargetDecision,
        frame: MarketFrame,
        raw_closes: Mapping[str, Decimal],
        trading_rules: Mapping[str, EtfTradingRule],
    ) -> tuple[
        tuple[FillResult, ...],
        tuple[RuleCheckResult, ...],
        tuple[Order, ...],
    ]:
        valuation = self._account.snapshot(raw_closes)
        orders = self._order_generator.generate(
            target_portfolio=decision.target_portfolio,
            valuation_snapshot=valuation,
            signal_date=decision.signal_date,
            execution_date=frame.trade_date,
        )
        if not orders:
            return (), (), ()
        quote_cache = TradePriceQuoteCache(frame=frame, trading_rules=trading_rules)
        quotes = quote_cache.resolve_all(order.symbol for order in orders)
        estimates = {
            order.order_id: self._fill_model.create_estimate(
                order=order,
                quote=quotes[order.symbol],
                tick_size=trading_rules[order.symbol].tick_size,
            )
            for order in orders
        }
        raw_approvals = tuple(
            self._rule_engine.approve_batch(
                frame=frame,
                orders=orders,
                quotes=quotes,
                estimates=estimates,
                account=self._account,
                trading_rules=trading_rules,
                etf_infos=self._etf_infos,
            )
        )
        raw_by_order = self._validate_approvals(orders=orders, approvals=raw_approvals)
        evidenced_approvals = tuple(
            replace(
                raw_by_order[order.order_id],
                base_trade_price=quotes[order.symbol].base_trade_price,
                price_limit_down=quotes[order.symbol].price_limit_down,
                price_limit_up=quotes[order.symbol].price_limit_up,
                price_limit_source=quotes[order.symbol].price_limit_source,
                price_limit_fallback_reason=(
                    "NO_EXPLICIT_PRICE_LIMIT"
                    if quotes[order.symbol].price_limit_source
                    is PriceLimitSource.DERIVED_RULE_FALLBACK
                    else None
                ),
            )
            for order in orders
        )
        by_order = self._validate_approvals(orders=orders, approvals=evidenced_approvals)
        fills: list[FillResult] = []
        for order in orders:
            fill = self._fill_model.create_fill(
                order=order,
                quote=quotes[order.symbol],
                estimate=estimates[order.order_id],
                approval=by_order[order.order_id],
            )
            if fill is not None:
                self._account.apply_fill(fill)
                fills.append(fill)
        return tuple(fills), evidenced_approvals, orders

    def _resolve_rules(self, frame: MarketFrame) -> Mapping[str, EtfTradingRule]:
        resolved: dict[str, EtfTradingRule] = {}
        for symbol in frame.canonical_symbols:
            rule = self._rule_resolver.resolve(symbol, frame.trade_date)
            if not isinstance(rule, EtfTradingRule) or rule.symbol != symbol:
                raise TypeError("RuleResolver returned an invalid EtfTradingRule")
            registered = self._account.position_for(symbol)
            if registered.turnover_rule is not rule.turnover_rule:
                raise ValueError("effective turnover rule conflicts with registered position")
            resolved[symbol] = rule
        return MappingProxyType(resolved)

    def _validate_frame_sequence(self, frames: tuple[MarketFrame, ...]) -> None:
        if any(not isinstance(frame, MarketFrame) for frame in frames):
            raise TypeError("portal returned a non-MarketFrame value")
        if any(left.trade_date >= right.trade_date for left, right in pairwise(frames)):
            raise ValueError("execution frames must be strictly chronological")
        calendar = self._portal.trading_calendar
        next_trading_day = getattr(calendar, "next_trading_day", None)
        if not callable(next_trading_day):
            raise TypeError("portal.trading_calendar must provide next_trading_day")
        for left, right in pairwise(frames):
            if next_trading_day(left.trade_date) != right.trade_date:
                raise ValueError("execution frames must be adjacent SSE trading dates")

    @staticmethod
    def _validate_approvals(
        *,
        orders: Sequence[Order],
        approvals: Sequence[RuleCheckResult],
    ) -> Mapping[str, RuleCheckResult]:
        if any(not isinstance(value, RuleCheckResult) for value in approvals):
            raise TypeError("rule_engine returned a non-RuleCheckResult value")
        by_order = {value.order_id: value for value in approvals}
        order_ids = {order.order_id for order in orders}
        if len(by_order) != len(approvals) or set(by_order) != order_ids:
            raise ValueError("rule_engine must return exactly one approval per order")
        return MappingProxyType(by_order)


__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "DailyPortal",
    "QuantityRuleEngine",
    "TargetDecision",
]
