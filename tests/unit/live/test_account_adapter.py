from datetime import datetime
from decimal import Decimal

from etf_backtest.config.schema import MARKET_TIMEZONE
from etf_backtest.core.market import TurnoverRule
from etf_backtest.live.account_adapter import adapt_broker_account
from etf_backtest.live.state import BrokerAssetSnapshot, BrokerPositionSnapshot


def _position(symbol: str, market_value: str) -> BrokerPositionSnapshot:
    return BrokerPositionSnapshot(
        symbol=symbol,
        total_quantity=100,
        available_quantity=60,
        today_buy_quantity=40,
        market_value=Decimal(market_value),
        turnover_rule=TurnoverRule.T1,
        captured_at=datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE),
    )


def test_adapter_adds_zero_view_calculates_weights_and_preserves_full_positions() -> None:
    captured_at = datetime(2026, 8, 19, 14, 50, tzinfo=MARKET_TIMEZONE)
    result = adapt_broker_account(
        asset=BrokerAssetSnapshot(
            total_asset=Decimal("1000"),
            available_cash=Decimal("200"),
            captured_at=captured_at,
        ),
        positions=[_position("510300.SH", "400"), _position("159915.SZ", "100")],
        symbols=["510300.SH", "518880.SH"],
    )

    zero = result.account_view.positions["SH.518880"]
    assert (zero.total_quantity, zero.available_quantity, zero.today_buy_quantity) == (0, 0, 0)
    assert result.current_weights_by_symbol == {
        "SH.510300": Decimal("0.4"),
        "SH.518880": Decimal("0"),
    }
    assert set(result.positions_by_symbol) == {"SH.510300", "SH.518880", "SZ.159915"}
    assert result.total_asset == Decimal("1000")
