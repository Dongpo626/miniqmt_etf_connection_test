"""Execution estimates, affordability checks, and formal fill creation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from etf_backtest.core.fee import FeeModel
from etf_backtest.core.order import (
    ExecutionEstimate,
    FillResult,
    Order,
    OrderSide,
    RuleCheckResult,
    TradePriceQuote,
)
from etf_backtest.core.slippage import SlippageModel


def _quantity(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _money(value: object, field_name: str) -> Decimal:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field_name} must be Decimal")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    quantity: int
    trade_amount: Decimal
    fee: Decimal
    total_cash_required: Decimal

    def __post_init__(self) -> None:
        quantity = _quantity(self.quantity, "quantity")
        amount = _money(self.trade_amount, "trade_amount")
        fee = _money(self.fee, "fee")
        total = _money(self.total_cash_required, "total_cash_required")
        if quantity == 0 and any(value != 0 for value in (amount, fee, total)):
            raise ValueError("zero quantity requires zero cost")
        if quantity > 0 and amount <= 0:
            raise ValueError("positive quantity requires positive trade amount")
        if total != amount + fee:
            raise ValueError("total cash required is inconsistent")


class FillModel:
    """Apply slippage exactly once, then reuse its price at every quantity."""

    __slots__ = ("_fee_model", "_slippage_model")

    def __init__(self, *, fee_model: FeeModel, slippage_model: SlippageModel) -> None:
        if not isinstance(fee_model, FeeModel):
            raise TypeError("fee_model must be FeeModel")
        if not isinstance(slippage_model, SlippageModel):
            raise TypeError("slippage_model must be SlippageModel")
        self._fee_model = fee_model
        self._slippage_model = slippage_model

    def create_estimate(
        self,
        *,
        order: Order,
        quote: TradePriceQuote,
        tick_size: Decimal,
    ) -> ExecutionEstimate:
        if not isinstance(order, Order):
            raise TypeError("order must be Order")
        if not isinstance(quote, TradePriceQuote):
            raise TypeError("quote must be TradePriceQuote")
        if order.symbol != quote.symbol or order.execution_date != quote.trade_date:
            raise ValueError("quote identity does not match order symbol and execution date")
        fill_price = self._slippage_model.apply(
            base_trade_price=quote.base_trade_price,
            side=order.side,
            tick_size=tick_size,
            price_limit_down=quote.price_limit_down,
            price_limit_up=quote.price_limit_up,
        )
        cost = self._cost(side=order.side, fill_price=fill_price, quantity=order.requested_quantity)
        return ExecutionEstimate(
            order_id=order.order_id,
            requested_quantity=order.requested_quantity,
            base_trade_price=quote.base_trade_price,
            fill_price=fill_price,
            estimated_trade_amount=cost.trade_amount,
            estimated_fee=cost.fee,
            estimated_total_cash_required=cost.total_cash_required,
        )

    def estimate_cost(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        quantity: int,
    ) -> ExecutionCost:
        self._validate_estimate(order=order, estimate=estimate)
        requested = _quantity(quantity, "quantity")
        if requested > order.requested_quantity:
            raise ValueError("quantity exceeds the original order request")
        return self._cost(side=order.side, fill_price=estimate.fill_price, quantity=requested)

    def max_affordable_buy_quantity(
        self,
        *,
        order: Order,
        estimate: ExecutionEstimate,
        available_cash: Decimal,
        lot_size: int,
        upper_quantity: int | None = None,
    ) -> int:
        self._validate_estimate(order=order, estimate=estimate)
        if order.side is not OrderSide.BUY:
            raise ValueError("affordability search is only defined for BUY")
        cash = _money(available_cash, "available_cash")
        lot = _quantity(lot_size, "lot_size")
        if lot <= 0:
            raise ValueError("lot_size must be positive")
        upper = (
            order.requested_quantity
            if upper_quantity is None
            else _quantity(upper_quantity, "upper_quantity")
        )
        if upper > order.requested_quantity:
            raise ValueError("upper_quantity exceeds the original order request")
        upper = (upper // lot) * lot

        low_lots = 0
        high_lots = upper // lot
        while low_lots < high_lots:
            middle = (low_lots + high_lots + 1) // 2
            candidate = middle * lot
            cost = self.estimate_cost(order=order, estimate=estimate, quantity=candidate)
            if cost.total_cash_required <= cash:
                low_lots = middle
            else:
                high_lots = middle - 1
        return low_lots * lot

    def create_fill(
        self,
        *,
        order: Order,
        quote: TradePriceQuote,
        estimate: ExecutionEstimate,
        approval: RuleCheckResult,
    ) -> FillResult | None:
        if not isinstance(approval, RuleCheckResult):
            raise TypeError("approval must be RuleCheckResult")
        self._validate_estimate(order=order, estimate=estimate)
        if approval.order_id != order.order_id:
            raise ValueError("approval identity does not match order")
        if not approval.passed:
            return None
        cost = self.estimate_cost(
            order=order,
            estimate=estimate,
            quantity=approval.approved_quantity,
        )
        return FillResult.from_approved(
            order=order,
            quote=quote,
            estimate=estimate,
            approval=approval,
            trade_amount=cost.trade_amount,
            fee=cost.fee,
        )

    def _cost(self, *, side: OrderSide, fill_price: Decimal, quantity: int) -> ExecutionCost:
        requested = _quantity(quantity, "quantity")
        amount = fill_price * requested
        fee = self._fee_model.calculate(trade_amount=amount, side=side).total
        return ExecutionCost(
            quantity=requested,
            trade_amount=amount,
            fee=fee,
            total_cash_required=amount + fee,
        )

    @staticmethod
    def _validate_estimate(*, order: Order, estimate: ExecutionEstimate) -> None:
        if not isinstance(order, Order):
            raise TypeError("order must be Order")
        if not isinstance(estimate, ExecutionEstimate):
            raise TypeError("estimate must be ExecutionEstimate")
        if (
            order.order_id != estimate.order_id
            or order.requested_quantity != estimate.requested_quantity
        ):
            raise ValueError("execution estimate identity does not match order")


__all__ = ["ExecutionCost", "FillModel"]
