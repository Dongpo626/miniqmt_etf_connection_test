"""Framework-neutral Model portfolio-policy tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from etf_backtest.strategy.model_contracts import PredictionRecord, SampleKey
from etf_backtest.strategy.portfolio import CustomPortfolio, TopKPortfolio


def _prediction(symbol: str, score: float, *, day: int = 2) -> PredictionRecord:
    return PredictionRecord(
        key=SampleKey(signal_date=date(2024, 1, day), symbol=symbol),
        score=score,
    )


@pytest.mark.unit
def test_score_proportional_top_k_allocates_exact_total_weight() -> None:
    policy = TopKPortfolio(
        top_k=2,
        total_weight="0.90",
        min_score=0.0,
        weighting="score_proportional",
    )

    weights = policy.allocate(
        (
            _prediction("SH.510300", 0.01),
            _prediction("SH.518880", 0.02),
            _prediction("SH.588000", -0.01),
        )
    )

    assert dict(weights) == {
        "SH.510300": Decimal("0.30"),
        "SH.518880": Decimal("0.60"),
    }
    assert sum(weights.values()) == Decimal("0.90")


@pytest.mark.unit
def test_equal_top_k_uses_fewer_assets_when_only_some_scores_qualify() -> None:
    policy = TopKPortfolio(top_k=3, total_weight="0.90", weighting="equal")

    weights = policy.allocate(
        (
            _prediction("SH.510300", 0.03),
            _prediction("SH.518880", 0.02),
            _prediction("SH.588000", 0.0),
        )
    )

    assert dict(weights) == {
        "SH.510300": Decimal("0.45"),
        "SH.518880": Decimal("0.45"),
    }


@pytest.mark.unit
def test_softmax_top_k_is_dynamic_and_preserves_exact_exposure() -> None:
    policy = TopKPortfolio(
        top_k=3,
        total_weight="0.90",
        weighting="softmax",
        softmax_temperature=0.1,
    )

    weights = policy.allocate(
        (
            _prediction("SH.510300", 0.01),
            _prediction("SH.518880", 0.02),
            _prediction("SH.588000", 0.03),
        )
    )

    assert sum(weights.values()) == Decimal("0.90")
    assert weights["SH.588000"] > weights["SH.518880"] > weights["SH.510300"]


@pytest.mark.unit
def test_top_k_returns_cash_when_no_score_exceeds_threshold() -> None:
    policy = TopKPortfolio(top_k=2, min_score=0.01)

    weights = policy.allocate(
        (
            _prediction("SH.510300", 0.01),
            _prediction("SH.518880", -0.02),
        )
    )

    assert dict(weights) == {}


@pytest.mark.unit
def test_top_k_tie_break_uses_symbol_descending() -> None:
    policy = TopKPortfolio(top_k=1, weighting="equal")

    weights = policy.allocate(
        (
            _prediction("SH.510300", 0.02),
            _prediction("SH.588000", 0.02),
        )
    )

    assert dict(weights) == {"SH.588000": Decimal("0.90")}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"top_k": 0}, "top_k"),
        ({"top_k": True}, "top_k"),
        ({"total_weight": "0"}, r"\(0, 1\]"),
        ({"total_weight": True}, "boolean"),
        ({"min_score": float("nan")}, "min_score"),
        ({"weighting": "unknown"}, "weighting"),
        ({"softmax_temperature": 0.0}, "temperature"),
        ({"softmax_temperature": float("inf")}, "temperature"),
    ),
)
def test_top_k_rejects_invalid_parameters(changes: dict[str, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        TopKPortfolio(**changes)  # type: ignore[arg-type]


@pytest.mark.unit
def test_portfolio_rejects_duplicate_prediction_keys_and_mixed_dates() -> None:
    policy = TopKPortfolio()
    duplicate = _prediction("SH.510300", 0.01)

    with pytest.raises(ValueError, match="duplicate"):
        policy.allocate((duplicate, duplicate))
    with pytest.raises(ValueError, match="same signal_date"):
        policy.allocate(
            (
                duplicate,
                _prediction("SH.518880", 0.02, day=3),
            )
        )


@pytest.mark.unit
def test_custom_portfolio_normalizes_and_freezes_valid_output() -> None:
    predictions = (
        _prediction("SH.510300", 0.01),
        _prediction("SH.518880", 0.02),
    )
    policy = CustomPortfolio(
        name="risk_scaled",
        total_weight="0.80",
        allocator=lambda records: {
            "510300": "0.30",
            records[1].key.symbol: Decimal("0.50"),
        },
    )

    weights = policy.allocate(predictions)

    assert dict(weights) == {
        "SH.510300": Decimal("0.30"),
        "SH.518880": Decimal("0.50"),
    }
    assert policy.resolved_dict() == {
        "type": "custom",
        "name": "risk_scaled",
        "total_weight": "0.80",
    }
    with pytest.raises(TypeError):
        weights["SH.510300"] = Decimal("0")  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("allocator", "error"),
    (
        (lambda records: {"SH.588000": "0.5"}, "current prediction"),
        (lambda records: {"SH.510300": "-0.1"}, "between zero"),
        (lambda records: {"SH.510300": float("inf")}, "finite"),
        (lambda records: {"SH.510300": "0.81"}, "exposure cap"),
        (lambda records: {"510300": "0.2", "SH.510300": "0.3"}, "unique"),
        (lambda records: [("SH.510300", "0.2")], "mapping"),
    ),
)
def test_custom_portfolio_rejects_invalid_output(allocator: object, error: str) -> None:
    policy = CustomPortfolio(
        name="invalid",
        total_weight="0.80",
        allocator=allocator,  # type: ignore[arg-type]
    )

    with pytest.raises((TypeError, ValueError), match=error):
        policy.allocate((_prediction("SH.510300", 0.01),))


@pytest.mark.unit
def test_custom_portfolio_rejects_invalid_definition() -> None:
    with pytest.raises(ValueError, match="name"):
        CustomPortfolio(name=" ", total_weight="0.8", allocator=lambda records: {})
    with pytest.raises(TypeError, match="callable"):
        CustomPortfolio(name="bad", total_weight="0.8", allocator=object())  # type: ignore[arg-type]
