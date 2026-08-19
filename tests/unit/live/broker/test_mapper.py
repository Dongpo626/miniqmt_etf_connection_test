from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from etf_backtest.core.order import OrderSide
from etf_backtest.live.broker.mapper import (
    external_to_internal_symbol,
    internal_to_external_symbol,
    map_asset,
    map_order,
    map_position,
    map_submit_result,
    map_trade,
    order_status_from_xt,
    side_from_xt,
    side_to_xt,
)
from etf_backtest.live.state import BrokerOrderStatus, SubmitOrderStatus

NOW = datetime(2026, 8, 19, 14, 50, tzinfo=ZoneInfo("Asia/Shanghai"))
CONSTANTS = SimpleNamespace(STOCK_BUY=23, STOCK_SELL=24)


def test_symbol_side_and_all_order_status_mappings() -> None:
    assert internal_to_external_symbol("SH.510300") == "510300.SH"
    assert external_to_internal_symbol("159915.SZ") == "SZ.159915"
    assert side_to_xt(OrderSide.BUY, CONSTANTS) == 23
    assert side_from_xt(24, constants=CONSTANTS) is OrderSide.SELL
    assert side_from_xt(None, offset_flag=48) is OrderSide.BUY
    expected = {
        48: BrokerOrderStatus.PENDING,
        49: BrokerOrderStatus.PENDING,
        50: BrokerOrderStatus.PENDING,
        51: BrokerOrderStatus.PENDING,
        52: BrokerOrderStatus.PARTIALLY_FILLED,
        53: BrokerOrderStatus.CANCELED,
        54: BrokerOrderStatus.CANCELED,
        55: BrokerOrderStatus.PARTIALLY_FILLED,
        56: BrokerOrderStatus.FILLED,
        57: BrokerOrderStatus.REJECTED,
        255: BrokerOrderStatus.UNKNOWN,
    }
    assert {key: order_status_from_xt(key) for key in expected} == expected
    assert order_status_from_xt(999) is BrokerOrderStatus.UNKNOWN
    assert map_submit_result(101).status is SubmitOrderStatus.ACCEPTED
    assert map_submit_result(-1).status is SubmitOrderStatus.REJECTED
    assert map_submit_result(None).status is SubmitOrderStatus.UNKNOWN


def test_asset_position_order_and_trade_mapping_uses_explicit_sdk_fields() -> None:
    asset = map_asset(
        SimpleNamespace(
            account_id="paper-1",
            cash=100.1,
            frozen_cash=2.2,
            market_value=300.3,
            total_asset=400.4,
        ),
        captured_at=NOW,
    )
    assert asset.available_cash == Decimal("100.1")
    assert asset.frozen_cash == Decimal("2.2")

    position = map_position(
        SimpleNamespace(
            account_id="paper-1",
            stock_code="510300.SH",
            volume=1000,
            can_use_volume=600,
            market_value=4012.5,
            frozen_volume=100,
            on_road_volume=300,
            yesterday_volume=700,
            avg_price=3.999,
        ),
        captured_at=NOW,
    )
    assert position.today_buy_quantity == 300
    assert position.available_quantity == 600
    assert position.average_cost == Decimal("3.999")

    order = map_order(
        SimpleNamespace(
            account_id="paper-1",
            order_id=101,
            order_sysid="sys-101",
            stock_code="510300.SH",
            order_type=23,
            offset_flag=48,
            order_volume=1000,
            traded_volume=200,
            price=4.001,
            traded_price=4.0,
            order_status=55,
            order_remark="L123",
            order_time=1724049000,
        ),
        constants=CONSTANTS,
    )
    assert order.limit_price == Decimal("4.001")
    assert order.traded_price == Decimal("4.0")
    assert order.remark_token == "L123"

    trade = map_trade(
        SimpleNamespace(
            account_id="paper-1",
            traded_id="trade-1",
            order_id=101,
            order_sysid="sys-101",
            stock_code="510300.SH",
            order_type=23,
            offset_flag=48,
            traded_volume=200,
            traded_price=4.0,
            traded_time=1724049001,
            order_remark="L123",
        ),
        constants=CONSTANTS,
    )
    assert trade.price == Decimal("4.0")
    assert trade.broker_order_id == "101"
    assert trade.traded_at.tzinfo is not None
