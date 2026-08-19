"""Plan one deterministic batch of live order intents from a complete target."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import ROUND_FLOOR, Decimal

from etf_backtest.config.schema import normalize_symbol
from etf_backtest.core.order import OrderSide
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.state import BrokerOrderSnapshot, BrokerPositionSnapshot, OrderIntent


def generate_intent_key(
    *,
    deployment_id: str,
    execution_date: date,
    decision_id: str,
    symbol: str,
    side: OrderSide,
) -> str:
    canonical_symbol = normalize_symbol(symbol)
    fields = (deployment_id, execution_date.strftime("%Y%m%d"), decision_id, canonical_symbol, side.value)
    if any(not field or "|" in field for field in fields):
        raise ValueError("intent key fields must be non-empty and cannot contain '|'")
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


def generate_remark_token(intent_key: str) -> str:
    try:
        digest = bytes.fromhex(intent_key)
    except ValueError as error:
        raise ValueError("intent_key must be a SHA-256 hex digest") from error
    if len(digest) != 32:
        raise ValueError("intent_key must be a SHA-256 hex digest")
    return "L" + base64.b32encode(digest).decode("ascii")[:20]


def _positive_price(prices: Mapping[str, Decimal], symbol: str, name: str) -> Decimal:
    price = prices.get(symbol)
    if price is None or not price.is_finite() or price <= 0:
        raise ValueError(f"{name} is unavailable for {symbol}")
    return price


def _whole_lots(quantity: int, lot_size: int) -> int:
    return max(0, quantity // lot_size * lot_size)


class LiveRebalancePlanner:
    """Create a single sell-first batch without submission or persistence."""

    def plan(
        self,
        *,
        deployment_id: str,
        decision_id: str,
        execution_date: date,
        symbols: Sequence[str],
        target: TargetPortfolio,
        total_asset: Decimal,
        available_cash: Decimal,
        positions: Mapping[str, BrokerPositionSnapshot],
        active_orders: Sequence[BrokerOrderSnapshot],
        valuation_prices: Mapping[str, Decimal],
        limit_prices: Mapping[str, Decimal],
        lot_size: int,
    ) -> tuple[OrderIntent, ...]:
        frozen_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in symbols}))
        if not frozen_symbols:
            raise ValueError("symbols must not be empty")
        if any(symbol not in frozen_symbols for symbol in target.weights):
            raise ValueError("target contains a symbol outside the frozen universe")
        if lot_size <= 0 or total_asset <= 0 or available_cash < 0:
            raise ValueError("lot_size and total_asset must be positive; cash cannot be negative")

        projected = {
            symbol: positions[symbol].total_quantity if symbol in positions else 0
            for symbol in frozen_symbols
        }
        for order in active_orders:
            if order.symbol not in projected or not order.status.is_active:
                continue
            direction = 1 if order.side is OrderSide.BUY else -1
            projected[order.symbol] += direction * order.remaining_quantity

        deltas: dict[str, int] = {}
        for symbol in frozen_symbols:
            weight = target.weight_for(symbol)
            if weight == 0:
                target_quantity = 0
            else:
                valuation_price = _positive_price(
                    valuation_prices, symbol, "valuation_price"
                )
                raw_lots = total_asset * weight / valuation_price / lot_size
                target_quantity = int(raw_lots.to_integral_value(rounding=ROUND_FLOOR)) * lot_size
            deltas[symbol] = target_quantity - projected[symbol]

        intents: list[OrderIntent] = []
        for side in (OrderSide.SELL, OrderSide.BUY):
            for symbol in frozen_symbols:
                delta = deltas[symbol]
                desired = -delta if side is OrderSide.SELL else delta
                if desired <= 0:
                    continue
                valuation_price = _positive_price(valuation_prices, symbol, "valuation_price")
                limit_price = _positive_price(limit_prices, symbol, "limit_price")
                quantity = _whole_lots(desired, lot_size)
                if side is OrderSide.SELL:
                    available = positions[symbol].available_quantity if symbol in positions else 0
                    quantity = _whole_lots(min(quantity, available), lot_size)
                else:
                    affordable = int(
                        (available_cash / limit_price / lot_size).to_integral_value(
                            rounding=ROUND_FLOOR
                        )
                    ) * lot_size
                    quantity = min(quantity, affordable)
                if quantity <= 0:
                    continue
                intent_key = generate_intent_key(
                    deployment_id=deployment_id,
                    execution_date=execution_date,
                    decision_id=decision_id,
                    symbol=symbol,
                    side=side,
                )
                intents.append(
                    OrderIntent(
                        intent_key=intent_key,
                        remark_token=generate_remark_token(intent_key),
                        deployment_id=deployment_id,
                        decision_id=decision_id,
                        execution_date=execution_date,
                        symbol=symbol,
                        side=side,
                        requested_quantity=quantity,
                        target_weight=target.weight_for(symbol),
                        valuation_price=valuation_price,
                        limit_price=limit_price,
                    )
                )
                if side is OrderSide.BUY:
                    available_cash -= quantity * limit_price
        return tuple(intents)


__all__ = ["LiveRebalancePlanner", "generate_intent_key", "generate_remark_token"]
