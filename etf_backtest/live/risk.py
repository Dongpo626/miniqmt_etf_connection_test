"""Minimal order and target risk checks for planned live intents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.order import OrderSide
from etf_backtest.live.state import OrderIntent


@dataclass(frozen=True, slots=True)
class RiskCheckResult:
    approved: bool
    reason: str | None = None


class LiveRiskManager:
    """Validate an intent without inspecting deployment or infrastructure state."""

    def check(
        self,
        intent: OrderIntent,
        *,
        symbols: Sequence[str],
        target_weights: Mapping[str, Decimal],
        max_total_target_weight: Decimal,
        available_cash: Decimal,
        available_quantity: int,
        lot_size: int,
        max_single_order_notional: Decimal,
        max_daily_order_notional: Decimal,
        daily_planned_notional: Decimal,
        min_order_notional: Decimal,
        quote_valid: bool,
    ) -> RiskCheckResult:
        frozen_symbols = {normalize_symbol(symbol) for symbol in symbols}
        if intent.symbol not in frozen_symbols:
            return RiskCheckResult(False, "SYMBOL_OUTSIDE_UNIVERSE")
        if any(not weight.is_finite() or weight < 0 for weight in target_weights.values()):
            return RiskCheckResult(False, "INVALID_TARGET_WEIGHT")
        total_weight = sum(target_weights.values(), Decimal("0"))
        if total_weight > max_total_target_weight:
            return RiskCheckResult(False, "TOTAL_TARGET_WEIGHT_EXCEEDED")
        if not quote_valid:
            return RiskCheckResult(False, "INVALID_QUOTE")
        if intent.requested_quantity <= 0:
            return RiskCheckResult(False, "NON_POSITIVE_QUANTITY")
        if lot_size <= 0 or intent.requested_quantity % lot_size:
            return RiskCheckResult(False, "INVALID_LOT_SIZE")
        if not intent.limit_price.is_finite() or intent.limit_price <= 0:
            return RiskCheckResult(False, "INVALID_LIMIT_PRICE")

        notional = intent.requested_quantity * intent.limit_price
        if notional < min_order_notional:
            return RiskCheckResult(False, "BELOW_MIN_ORDER_NOTIONAL")
        if notional > max_single_order_notional:
            return RiskCheckResult(False, "MAX_SINGLE_ORDER_NOTIONAL_EXCEEDED")
        if daily_planned_notional + notional > max_daily_order_notional:
            return RiskCheckResult(False, "MAX_DAILY_ORDER_NOTIONAL_EXCEEDED")
        if intent.side is OrderSide.SELL and intent.requested_quantity > available_quantity:
            return RiskCheckResult(False, "INSUFFICIENT_AVAILABLE_QUANTITY")
        if intent.side is OrderSide.BUY and notional > available_cash:
            return RiskCheckResult(False, "INSUFFICIENT_CASH")
        return RiskCheckResult(True)


__all__ = ["LiveRiskManager", "RiskCheckResult"]
