"""Fixed proportional close-price slippage tests."""

from decimal import Decimal
from inspect import signature

import pytest

from etf_backtest.config.schema import SlippageConfig
from etf_backtest.core.order import OrderSide
from etf_backtest.core.slippage import SlippageModel


@pytest.mark.unit
def test_slippage_is_proportional_and_rounded_against_the_order() -> None:
    model = SlippageModel(SlippageConfig(rate=Decimal("0.001")))
    kwargs = {
        "base_trade_price": Decimal("10.001"),
        "tick_size": Decimal("0.001"),
        "price_limit_down": Decimal("9.000"),
        "price_limit_up": Decimal("11.000"),
    }

    assert model.apply(side=OrderSide.BUY, **kwargs) == Decimal("10.012")
    assert model.apply(side=OrderSide.SELL, **kwargs) == Decimal("9.990")


@pytest.mark.unit
def test_slippage_clamps_at_legal_limits_not_observed_high_low() -> None:
    model = SlippageModel(SlippageConfig(rate=Decimal("0.50")))

    assert model.apply(
        base_trade_price=Decimal("10"),
        side=OrderSide.BUY,
        tick_size=Decimal("0.001"),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("11.000"),
    ) == Decimal("11.000")
    assert model.apply(
        base_trade_price=Decimal("10"),
        side=OrderSide.SELL,
        tick_size=Decimal("0.001"),
        price_limit_down=Decimal("9.000"),
        price_limit_up=Decimal("11.000"),
    ) == Decimal("9.000")
    assert "high" not in signature(model.apply).parameters
    assert "low" not in signature(model.apply).parameters


@pytest.mark.unit
def test_slippage_rejects_invalid_boundaries() -> None:
    model = SlippageModel(SlippageConfig())

    with pytest.raises(ValueError, match="inside"):
        model.apply(
            base_trade_price=Decimal("12"),
            side=OrderSide.BUY,
            tick_size=Decimal("0.001"),
            price_limit_down=Decimal("9"),
            price_limit_up=Decimal("11"),
        )
    with pytest.raises(ValueError, match="tick aligned"):
        model.apply(
            base_trade_price=Decimal("10"),
            side=OrderSide.BUY,
            tick_size=Decimal("0.01"),
            price_limit_down=Decimal("9.005"),
            price_limit_up=Decimal("11"),
        )
