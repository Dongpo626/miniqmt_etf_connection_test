"""Contract tests for the strategy's immutable target-portfolio output."""

from decimal import Decimal

import pytest

from etf_backtest.core.target import TargetPortfolio


@pytest.mark.unit
@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"SH.510300": Decimal("0")},
        {"SH.510300": Decimal("1")},
        {"SH.510300": Decimal("0.60"), "SH.518880": Decimal("0.30")},
    ],
)
def test_target_portfolio_accepts_valid_decimal_weights(
    weights: dict[str, Decimal],
) -> None:
    target = TargetPortfolio(weights=weights)

    assert dict(target.weights) == weights


@pytest.mark.unit
def test_target_portfolio_returns_zero_for_an_omitted_symbol() -> None:
    target = TargetPortfolio(weights={"510300": Decimal("0.75")})

    assert target.weight_for("518880") == Decimal("0")
    assert tuple(target.weights) == ("SH.510300",)


@pytest.mark.unit
def test_target_portfolio_defensively_copies_and_freezes_weights() -> None:
    source = {"510300": Decimal("0.60")}
    target = TargetPortfolio(weights=source)

    source["510300"] = Decimal("0.10")
    assert target.weight_for("510300") == Decimal("0.60")

    with pytest.raises(TypeError):
        target.weights["518880"] = Decimal("0.20")  # type: ignore[index]

    with pytest.raises((AttributeError, TypeError)):
        target.weights = {}  # type: ignore[misc]


@pytest.mark.unit
def test_target_portfolio_preserves_explicit_cash_weight() -> None:
    target = TargetPortfolio(weights={"510300": Decimal("0.40"), "518880": Decimal("0.30")})

    assert sum(target.weights.values(), Decimal("0")) == Decimal("0.70")


@pytest.mark.unit
@pytest.mark.parametrize(
    "weights",
    [
        {"": Decimal("0.10")},
        {"   ": Decimal("0.10")},
        {"510300": Decimal("-0.01")},
        {"510300": Decimal("1.01")},
        {"510300": Decimal("0.60"), "518880": Decimal("0.50")},
        {"510300": Decimal("NaN")},
        {"510300": Decimal("Infinity")},
        {"510300": Decimal("-Infinity")},
        {"510300": 0.5},
    ],
)
def test_target_portfolio_rejects_invalid_symbols_or_weights(
    weights: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError, ArithmeticError)):
        TargetPortfolio(weights=weights)  # type: ignore[arg-type]


@pytest.mark.unit
def test_target_portfolio_rejects_non_string_symbol() -> None:
    with pytest.raises((TypeError, ValueError)):
        TargetPortfolio(weights={123: Decimal("0.1")})  # type: ignore[dict-item]
