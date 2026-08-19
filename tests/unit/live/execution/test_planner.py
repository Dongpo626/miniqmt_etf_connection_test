import re
from datetime import date, datetime
from decimal import Decimal

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.market import TurnoverRule
from etf_backtest.core.order import OrderSide
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.live.execution.planner import (
    LiveRebalancePlanner,
    generate_intent_key,
    generate_remark_token,
)
from etf_backtest.live.state import (
    BrokerOrderSnapshot,
    BrokerOrderStatus,
    BrokerPositionSnapshot,
)

NOW = datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE)


def _position(symbol: str, total: int, available: int) -> BrokerPositionSnapshot:
    return BrokerPositionSnapshot(
        symbol=symbol,
        total_quantity=total,
        available_quantity=available,
        today_buy_quantity=total - available,
        market_value=Decimal("0"),
        turnover_rule=TurnoverRule.T1,
        captured_at=NOW,
    )


def _order(symbol: str, side: OrderSide, requested: int, filled: int) -> BrokerOrderSnapshot:
    return BrokerOrderSnapshot(
        broker_order_id=f"{symbol}-{side}",
        symbol=symbol,
        side=side,
        requested_quantity=requested,
        filled_quantity=filled,
        limit_price=Decimal("10"),
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        captured_at=NOW,
    )


def test_planner_uses_valuation_for_target_limit_for_cash_and_active_remaining() -> None:
    intents = LiveRebalancePlanner().plan(
        deployment_id="deployment-1",
        decision_id="decision-1",
        execution_date=date(2026, 8, 19),
        symbols=["510300.SH", "518880.SH"],
        target=TargetPortfolio({"518880.SH": Decimal("0.5")}),
        total_asset=Decimal("10000"),
        available_cash=Decimal("5000"),
        positions={"SH.510300": _position("510300.SH", 500, 500)},
        active_orders=[
            _order("510300.SH", OrderSide.SELL, 200, 100),
            _order("518880.SH", OrderSide.BUY, 200, 100),
        ],
        valuation_prices={"SH.510300": Decimal("5"), "SH.518880": Decimal("10")},
        limit_prices={"SH.510300": Decimal("4"), "SH.518880": Decimal("20")},
        lot_size=100,
    )

    assert [(intent.side, intent.symbol, intent.requested_quantity) for intent in intents] == [
        (OrderSide.SELL, "SH.510300", 400),
        (OrderSide.BUY, "SH.518880", 200),
    ]
    assert intents[1].valuation_price == Decimal("10")
    assert intents[1].limit_price == Decimal("20")


def test_intent_key_is_stable_distinct_and_remark_is_short_base32() -> None:
    def key(symbol: str, side: OrderSide) -> str:
        return generate_intent_key(
            deployment_id="deployment-1",
            execution_date=date(2026, 8, 19),
            decision_id="decision-1",
            symbol=symbol,
            side=side,
        )

    first = key("510300.SH", OrderSide.BUY)
    same = key("SH.510300", OrderSide.BUY)
    other_symbol = key("518880.SH", OrderSide.BUY)
    other_side = key("510300.SH", OrderSide.SELL)
    token = generate_remark_token(first)

    assert first == same
    assert len({first, other_symbol, other_side}) == 3
    assert re.fullmatch(r"L[A-Z2-7]{20}", token)
    assert len(token) <= 24
