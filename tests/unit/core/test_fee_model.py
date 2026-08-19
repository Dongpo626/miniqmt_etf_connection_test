"""ETF fee calculations stay Decimal and per fill."""

from decimal import Decimal

import pytest

from etf_backtest.config.schema import FeeConfig
from etf_backtest.core.fee import FeeModel
from etf_backtest.core.order import OrderSide


@pytest.mark.unit
def test_minimum_commission_is_applied_independently_per_fill() -> None:
    model = FeeModel(FeeConfig())

    assert model.calculate(trade_amount=Decimal("1000"), side=OrderSide.BUY).total == Decimal(
        "5.000"
    )
    assert model.calculate(trade_amount=Decimal("1000"), side=OrderSide.SELL).total == Decimal(
        "5.000"
    )


@pytest.mark.unit
def test_rate_commission_uses_three_decimal_money_precision() -> None:
    model = FeeModel(
        FeeConfig(
            commission_rate=Decimal("0.0003"),
            minimum_commission=Decimal("0"),
        )
    )

    fee = model.calculate(trade_amount=Decimal("899937"), side=OrderSide.SELL)

    assert fee.total == Decimal("269.981")
    assert fee.stamp_duty == 0
    assert fee.total.as_tuple().exponent == -3


@pytest.mark.unit
def test_fee_rejects_non_decimal_amount() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        FeeModel(FeeConfig()).calculate(trade_amount=1000.0, side=OrderSide.BUY)  # type: ignore[arg-type]
