"""Pure daily-close ETF order approval.

The rule engine owns the one final approved quantity for every order.  It
simulates cash, sellable holdings, and daily volume consumption without
mutating the account; only a later formal fill may change economic state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Final, Protocol, runtime_checkable

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.account import Account
from etf_backtest.core.market import (
    EtfInfo,
    EtfTradingRule,
    MarketBar,
    MarketFrame,
    resolve_legal_price_limits,
)
from etf_backtest.core.order import (
    ExecutionEstimate,
    Order,
    OrderSide,
    RuleCheckResult,
    RuleReasonCode,
    TradePriceQuote,
)

_ZERO: Final = Decimal("0")
_DEFAULT_VOLUME_PARTICIPATION_RATE: Final = Decimal("0.20")


class _ExecutionCost(Protocol):
    @property
    def trade_amount(self) -> Decimal: ...

    @property
    def fee(self) -> Decimal: ...

    @property
    def total_cash_required(self) -> Decimal: ...


@runtime_checkable
class _FillCostModel(Protocol):
    def estimate_cost(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        quantity: int,
    ) -> _ExecutionCost: ...

    def max_affordable_buy_quantity(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        available_cash: Decimal,
        lot_size: int,
        upper_quantity: int | None = None,
    ) -> int: ...


def _approved_result(
    order: Order,
    approved_quantity: int,
    reductions: Sequence[str] = (),
) -> RuleCheckResult:
    if approved_quantity <= 0:
        raise ValueError("an approval must have positive quantity")
    message = (
        "approved" if not reductions else "approved with " + " and ".join(reductions) + " reduction"
    )
    return RuleCheckResult(
        order_id=order.order_id,
        requested_quantity=order.requested_quantity,
        approved_quantity=approved_quantity,
        passed=True,
        reason_code=RuleReasonCode.APPROVED,
        message=message,
    )


def _rejected_result(
    order: Order,
    reason_code: RuleReasonCode,
    message: str,
) -> RuleCheckResult:
    if reason_code is RuleReasonCode.APPROVED:
        raise ValueError("a rejection cannot use APPROVED")
    return RuleCheckResult(
        order_id=order.order_id,
        requested_quantity=order.requested_quantity,
        approved_quantity=0,
        passed=False,
        reason_code=reason_code,
        message=message,
    )


def _normalize_quotes(values: Mapping[str, TradePriceQuote]) -> dict[str, TradePriceQuote]:
    if not isinstance(values, Mapping):
        raise TypeError("quotes must be a mapping")
    normalized: dict[str, TradePriceQuote] = {}
    for supplied_symbol, quote in values.items():
        symbol = normalize_symbol(supplied_symbol)
        if not isinstance(quote, TradePriceQuote):
            raise TypeError("quotes may contain only TradePriceQuote")
        if quote.symbol != symbol:
            raise ValueError("quote mapping key must equal TradePriceQuote.symbol")
        if symbol in normalized:
            raise ValueError("quotes contains a duplicate canonical symbol")
        normalized[symbol] = quote
    return normalized


def _normalize_rules(values: Mapping[str, EtfTradingRule]) -> dict[str, EtfTradingRule]:
    if not isinstance(values, Mapping):
        raise TypeError("trading_rules must be a mapping")
    normalized: dict[str, EtfTradingRule] = {}
    for supplied_symbol, rule in values.items():
        symbol = normalize_symbol(supplied_symbol)
        if not isinstance(rule, EtfTradingRule):
            raise TypeError("trading_rules may contain only EtfTradingRule")
        if rule.symbol != symbol:
            raise ValueError("trading rule mapping key must equal EtfTradingRule.symbol")
        if symbol in normalized:
            raise ValueError("trading_rules contains a duplicate canonical symbol")
        normalized[symbol] = rule
    return normalized


def _normalize_infos(values: Mapping[str, EtfInfo]) -> dict[str, EtfInfo]:
    if not isinstance(values, Mapping):
        raise TypeError("etf_infos must be a mapping")
    normalized: dict[str, EtfInfo] = {}
    for supplied_symbol, info in values.items():
        symbol = normalize_symbol(supplied_symbol)
        if not isinstance(info, EtfInfo):
            raise TypeError("etf_infos may contain only EtfInfo")
        if info.symbol != symbol:
            raise ValueError("ETF info mapping key must equal EtfInfo.symbol")
        if symbol in normalized:
            raise ValueError("etf_infos contains a duplicate canonical symbol")
        normalized[symbol] = info
    return normalized


def _normalize_estimates(
    values: Mapping[str, ExecutionEstimate],
) -> dict[str, ExecutionEstimate]:
    if not isinstance(values, Mapping):
        raise TypeError("estimates must be a mapping")
    normalized: dict[str, ExecutionEstimate] = {}
    for order_id, estimate in values.items():
        if not isinstance(order_id, str):
            raise TypeError("estimate mapping keys must be strings")
        if not isinstance(estimate, ExecutionEstimate):
            raise TypeError("estimates may contain only ExecutionEstimate")
        if order_id != estimate.order_id:
            raise ValueError("estimate mapping key must equal ExecutionEstimate.order_id")
        normalized[order_id] = estimate
    return normalized


class EtfRuleEngine:
    """Approve daily-close orders in deterministic SELL-then-BUY order."""

    __slots__ = ("_fill_model", "_volume_participation_rate")

    def __init__(
        self,
        *,
        fill_model: _FillCostModel,
        volume_participation_rate: Decimal = _DEFAULT_VOLUME_PARTICIPATION_RATE,
    ) -> None:
        if not isinstance(fill_model, _FillCostModel):
            raise TypeError("fill_model must provide execution-cost and affordability methods")
        if not isinstance(volume_participation_rate, Decimal):
            raise TypeError("volume_participation_rate must be Decimal")
        if (
            not volume_participation_rate.is_finite()
            or volume_participation_rate <= _ZERO
            or volume_participation_rate > Decimal("1")
        ):
            raise ValueError("volume_participation_rate must be in (0, 1]")
        self._fill_model = fill_model
        self._volume_participation_rate = volume_participation_rate

    @property
    def volume_participation_rate(self) -> Decimal:
        return self._volume_participation_rate

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
    ) -> list[RuleCheckResult]:
        """Return exactly one immutable approval result per supplied order.

        SELL orders are evaluated first so their net proceeds may fund BUY
        orders.  Within each side, sorting makes the result independent of
        caller input order.  BUY priority follows descending target-value gap.
        """

        if not isinstance(frame, MarketFrame):
            raise TypeError("frame must be MarketFrame")
        if not isinstance(orders, Sequence):
            raise TypeError("orders must be a sequence")
        if not isinstance(account, Account):
            raise TypeError("account must be Account")

        order_list = list(orders)
        if any(not isinstance(order, Order) for order in order_list):
            raise TypeError("orders may contain only Order")
        order_ids = [order.order_id for order in order_list]
        if len(order_ids) != len(set(order_ids)):
            raise ValueError("order_id must be unique within an approval batch")

        normalized_quotes = _normalize_quotes(quotes)
        normalized_estimates = _normalize_estimates(estimates)
        normalized_rules = _normalize_rules(trading_rules)
        normalized_infos = _normalize_infos(etf_infos)

        sells = sorted(
            (order for order in order_list if order.side is OrderSide.SELL),
            key=lambda order: (order.symbol, order.order_id),
        )
        buys = sorted(
            (order for order in order_list if order.side is OrderSide.BUY),
            key=lambda order: (-order.target_value_gap, order.symbol, order.order_id),
        )

        projected_cash = account.cash
        projected_available = {
            symbol: position.available_quantity for symbol, position in account.positions.items()
        }
        used_volume = {symbol: 0 for symbol in frame.canonical_symbols}
        results: list[RuleCheckResult] = []

        for order in sells:
            rejection = self._common_precheck(
                frame=frame,
                order=order,
                quotes=normalized_quotes,
                estimates=normalized_estimates,
                account=account,
                trading_rules=normalized_rules,
                etf_infos=normalized_infos,
            )
            if rejection is not None:
                results.append(rejection)
                continue

            rule = normalized_rules[order.symbol]
            estimate = normalized_estimates[order.order_id]
            available = projected_available[order.symbol]
            if available < rule.lot_size:
                results.append(
                    _rejected_result(
                        order,
                        RuleReasonCode.TURNOVER_RULE,
                        "no sellable whole lot is available under the effective turnover rule",
                    )
                )
                continue

            turnover_cap = min(order.requested_quantity, available)
            turnover_cap = (turnover_cap // rule.lot_size) * rule.lot_size
            volume_remaining = self._volume_remaining(
                bar=frame.bar_for(order.symbol),
                rule=rule,
                already_used=used_volume[order.symbol],
            )
            if volume_remaining < rule.lot_size:
                results.append(
                    _rejected_result(
                        order,
                        RuleReasonCode.VOLUME_LIMIT,
                        "daily volume participation limit is exhausted",
                    )
                )
                continue

            approved = min(turnover_cap, volume_remaining)
            approved = (approved // rule.lot_size) * rule.lot_size
            reductions: list[str] = []
            if turnover_cap < order.requested_quantity:
                reductions.append("sellable-holdings")
            if approved < turnover_cap:
                reductions.append("daily-volume")

            cost = self._fill_model.estimate_cost(
                order=order,
                estimate=estimate,
                quantity=approved,
            )
            next_cash = projected_cash + cost.trade_amount - cost.fee
            if next_cash < _ZERO:
                results.append(
                    _rejected_result(
                        order,
                        RuleReasonCode.INSUFFICIENT_CASH,
                        "sell fee would make projected cash negative",
                    )
                )
                continue

            projected_cash = next_cash
            projected_available[order.symbol] -= approved
            used_volume[order.symbol] += approved
            results.append(_approved_result(order, approved, reductions))

        for order in buys:
            rejection = self._common_precheck(
                frame=frame,
                order=order,
                quotes=normalized_quotes,
                estimates=normalized_estimates,
                account=account,
                trading_rules=normalized_rules,
                etf_infos=normalized_infos,
            )
            if rejection is not None:
                results.append(rejection)
                continue

            rule = normalized_rules[order.symbol]
            estimate = normalized_estimates[order.order_id]
            volume_remaining = self._volume_remaining(
                bar=frame.bar_for(order.symbol),
                rule=rule,
                already_used=used_volume[order.symbol],
            )
            if volume_remaining < rule.lot_size:
                results.append(
                    _rejected_result(
                        order,
                        RuleReasonCode.VOLUME_LIMIT,
                        "daily volume participation limit is exhausted",
                    )
                )
                continue

            non_cash_quantity = min(order.requested_quantity, volume_remaining)
            non_cash_quantity = (non_cash_quantity // rule.lot_size) * rule.lot_size
            affordable = self._fill_model.max_affordable_buy_quantity(
                order=order,
                estimate=estimate,
                available_cash=projected_cash,
                lot_size=rule.lot_size,
                upper_quantity=non_cash_quantity,
            )
            if (
                type(affordable) is not int
                or affordable < 0
                or affordable > non_cash_quantity
                or affordable % rule.lot_size != 0
            ):
                raise ValueError("fill model returned an invalid affordable quantity")
            if affordable == 0:
                results.append(
                    _rejected_result(
                        order,
                        RuleReasonCode.INSUFFICIENT_CASH,
                        "projected cash cannot fund one whole lot",
                    )
                )
                continue

            reductions = []
            if non_cash_quantity < order.requested_quantity:
                reductions.append("daily-volume")
            if affordable < non_cash_quantity:
                reductions.append("available-cash")
            cost = self._fill_model.estimate_cost(
                order=order,
                estimate=estimate,
                quantity=affordable,
            )
            if cost.total_cash_required > projected_cash:
                raise ValueError("fill model affordability result exceeds projected cash")
            projected_cash -= cost.total_cash_required
            used_volume[order.symbol] += affordable
            results.append(_approved_result(order, affordable, reductions))

        return results

    def _common_precheck(
        self,
        *,
        frame: MarketFrame,
        order: Order,
        quotes: Mapping[str, TradePriceQuote],
        estimates: Mapping[str, ExecutionEstimate],
        account: Account,
        trading_rules: Mapping[str, EtfTradingRule],
        etf_infos: Mapping[str, EtfInfo],
    ) -> RuleCheckResult | None:
        if (
            order.execution_date != frame.trade_date
            or order.symbol not in frame.bars_by_symbol
            or order.symbol not in trading_rules
            or order.symbol not in etf_infos
            or order.symbol not in account.positions
        ):
            return _rejected_result(
                order,
                RuleReasonCode.LISTING_OR_WINDOW,
                "order is outside the registered daily execution window",
            )

        info = etf_infos[order.symbol]
        rule = trading_rules[order.symbol]
        if not info.is_active(frame.trade_date):
            return _rejected_result(
                order,
                RuleReasonCode.LISTING_OR_WINDOW,
                "ETF is not active on the execution date",
            )
        if account.position_for(order.symbol).turnover_rule is not rule.turnover_rule:
            return _rejected_result(
                order,
                RuleReasonCode.TURNOVER_RULE,
                "registered position turnover rule conflicts with the effective ETF rule",
            )

        bar = frame.bar_for(order.symbol)
        if bar.suspended:
            return _rejected_result(
                order,
                RuleReasonCode.SUSPENDED,
                "raw daily execution bar is suspended",
            )

        quote = quotes.get(order.symbol)
        estimate = estimates.get(order.order_id)
        if not self._execution_chain_is_valid(
            frame=frame,
            order=order,
            bar=bar,
            rule=rule,
            quote=quote,
            estimate=estimate,
        ):
            return _rejected_result(
                order,
                RuleReasonCode.QUOTE_UNAVAILABLE,
                "quote or execution estimate is missing or inconsistent",
            )
        if quote is None:  # pragma: no cover - narrowed by the validation above
            raise AssertionError("validated quote unexpectedly missing")

        if (order.side is OrderSide.BUY and quote.base_trade_price >= quote.price_limit_up) or (
            order.side is OrderSide.SELL and quote.base_trade_price <= quote.price_limit_down
        ):
            return _rejected_result(
                order,
                RuleReasonCode.PRICE_LIMIT,
                "order direction is blocked at the legal daily price limit",
            )
        if order.requested_quantity <= 0 or order.requested_quantity % rule.lot_size != 0:
            return _rejected_result(
                order,
                RuleReasonCode.LOT_SIZE,
                "requested quantity must be a positive whole lot",
            )
        return None

    @classmethod
    def _execution_chain_is_valid(
        cls,
        *,
        frame: MarketFrame,
        order: Order,
        bar: MarketBar,
        rule: EtfTradingRule,
        quote: TradePriceQuote | None,
        estimate: ExecutionEstimate | None,
    ) -> bool:
        if not isinstance(quote, TradePriceQuote) or not isinstance(estimate, ExecutionEstimate):
            return False
        lower, upper, source = resolve_legal_price_limits(
            execution_bar=bar,
            trading_rule=rule,
        )
        direction_is_valid = (
            estimate.fill_price >= quote.base_trade_price
            if order.side is OrderSide.BUY
            else estimate.fill_price <= quote.base_trade_price
        )
        return (
            quote.symbol == order.symbol
            and quote.trade_date == frame.trade_date
            and quote.trade_date == order.execution_date
            and quote.source_record_key == bar.source_record_key
            and quote.base_trade_price == bar.close
            and quote.price_limit_down == lower
            and quote.price_limit_up == upper
            and quote.price_limit_source is source
            and estimate.order_id == order.order_id
            and estimate.requested_quantity == order.requested_quantity
            and estimate.base_trade_price == quote.base_trade_price
            and quote.price_limit_down <= estimate.fill_price <= quote.price_limit_up
            and direction_is_valid
        )

    def _volume_remaining(
        self,
        *,
        bar: MarketBar,
        rule: EtfTradingRule,
        already_used: int,
    ) -> int:
        raw_limit = Decimal(bar.volume) * self._volume_participation_rate
        volume_limit = int(raw_limit // rule.lot_size) * rule.lot_size
        return max(0, volume_limit - already_used)


__all__ = ["EtfRuleEngine"]
