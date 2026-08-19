import inspect
from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.core.order import OrderSide
from etf_backtest.live.risk import LiveRiskManager
from etf_backtest.live.state import OrderIntent


def _intent(*, side: OrderSide = OrderSide.BUY, quantity: int = 100) -> OrderIntent:
    return OrderIntent(
        intent_key="a" * 64,
        remark_token="L" + "A" * 20,
        deployment_id="deployment-1",
        decision_id="decision-1",
        execution_date=date(2026, 8, 19),
        symbol="510300.SH",
        side=side,
        requested_quantity=quantity,
        target_weight=Decimal("0.5"),
        valuation_price=Decimal("10"),
        limit_price=Decimal("10"),
    )


def _check(intent: OrderIntent, **overrides: object) -> str | None:
    inputs: dict[str, object] = {
        "symbols": ["510300.SH"],
        "target_weights": {"SH.510300": Decimal("0.5")},
        "max_total_target_weight": Decimal("1"),
        "available_cash": Decimal("10000"),
        "available_quantity": 1000,
        "lot_size": 100,
        "max_single_order_notional": Decimal("10000"),
        "max_daily_order_notional": Decimal("20000"),
        "daily_planned_notional": Decimal("0"),
        "min_order_notional": Decimal("100"),
        "quote_valid": True,
    }
    inputs.update(overrides)
    result = LiveRiskManager().check(intent, **inputs)  # type: ignore[arg-type]
    return result.reason


@pytest.mark.parametrize(
    ("intent", "overrides", "reason"),
    [
        (_intent(), {"available_cash": Decimal("999")}, "INSUFFICIENT_CASH"),
        (
            _intent(side=OrderSide.SELL),
            {"available_quantity": 99},
            "INSUFFICIENT_AVAILABLE_QUANTITY",
        ),
        (_intent(quantity=150), {}, "INVALID_LOT_SIZE"),
        (
            _intent(),
            {"max_single_order_notional": Decimal("999")},
            "MAX_SINGLE_ORDER_NOTIONAL_EXCEEDED",
        ),
    ],
)
def test_risk_rejects_key_order_constraints(
    intent: OrderIntent, overrides: dict[str, object], reason: str
) -> None:
    assert _check(intent, **overrides) == reason


def test_risk_accepts_valid_intent_and_has_no_infrastructure_inputs() -> None:
    assert _check(_intent()) is None
    parameters = inspect.signature(LiveRiskManager.check).parameters
    assert not ({"deployment_status", "account_lock", "connection_status"} & parameters.keys())
