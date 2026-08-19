from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from etf_backtest.core.market import IndexBarView, MarketBarView, TurnoverRule
from etf_backtest.core.target import NO_REBALANCE
from etf_backtest.experiments.config import load_user_experiment_config
from etf_backtest.strategy.context import AccountPositionView
from etf_backtest.strategy.loader import load_user_rule
from etf_backtest.strategy.rule import RuleMarketData

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STRATEGY_ROOT = _PROJECT_ROOT / "private_strategy" / "huijin_multi_example"
_INDEX = "000001.SH"
_HS300 = "SH.510300"
_SIGNAL_DATE = date(2024, 4, 5)
_SYMBOLS = (
    "SH.510050",
    _HS300,
    "SH.510500",
    "SH.512100",
    "SZ.159915",
    "SH.510230",
    "SH.588080",
)
_ZERO = Decimal("0")
_TOLERANCE = Decimal("0.00000000000000000000000001")


def _trading_dates(end_date: date, observations: int) -> tuple[date, ...]:
    result: list[date] = []
    current = end_date
    while len(result) < observations:
        if current.weekday() < 5:
            result.append(current)
        current -= timedelta(days=1)
    return tuple(reversed(result))


def _next_trading_date(value: date) -> date:
    result = value + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def _etf_bar(symbol: str, trade_date: date, close: Decimal) -> MarketBarView:
    return MarketBarView(
        source_record_key=f"QMT:{symbol.split('.')[1]}:{trade_date.isoformat()}:1:1",
        symbol=symbol,
        trade_date=trade_date,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1_000_000,
        suspended=False,
    )


def _index_bar(trade_date: date, drawdown_points: Decimal) -> IndexBarView:
    stage_high = Decimal("4000")
    close = stage_high - drawdown_points
    return IndexBarView(
        index_code=_INDEX,
        trade_date=trade_date,
        open=close,
        high=stage_high,
        low=close - Decimal("10"),
        close=close,
        pre_close=close,
        pct_change=Decimal("0"),
        source_system="TUSHARE",
    )


def _share_series(
    observations: int,
    *,
    current_share: Decimal,
    tail_increments: tuple[Decimal, ...],
) -> tuple[Decimal, ...]:
    if len(tail_increments) > observations - 1:
        raise ValueError("tail increments exceed available observations")
    increments = [Decimal("100")] * (observations - 1 - len(tail_increments))
    increments.extend(tail_increments)
    first_share = current_share - sum(increments, start=_ZERO)
    values = [first_share]
    for increment in increments:
        values.append(values[-1] + increment)
    return tuple(values)


def _market_data(
    *,
    drawdown_points: Decimal,
    tail_increments_by_symbol: dict[str, tuple[Decimal, ...]] | None = None,
    current_weights: dict[str, Decimal] | None = None,
    current_shares: dict[str, Decimal] | None = None,
    huijin_ratios: dict[str, Decimal | None] | None = None,
    price_up_symbols: set[str] | None = None,
    missing_signal_share_symbols: set[str] | None = None,
) -> RuleMarketData:
    dates = _trading_dates(_SIGNAL_DATE, 90)
    disclosure_date = date(2023, 11, 30)
    supplied_increments = {} if tail_increments_by_symbol is None else tail_increments_by_symbol
    supplied_current_shares = {} if current_shares is None else current_shares
    supplied_ratios = {} if huijin_ratios is None else huijin_ratios
    price_up = set() if price_up_symbols is None else price_up_symbols
    missing_shares = set() if missing_signal_share_symbols is None else missing_signal_share_symbols
    weights = {symbol: _ZERO for symbol in _SYMBOLS} if current_weights is None else current_weights
    positions = {
        symbol: AccountPositionView(
            symbol=symbol,
            turnover_rule=TurnoverRule.T1,
            total_quantity=0,
            available_quantity=0,
            today_buy_quantity=0,
        )
        for symbol in _SYMBOLS
    }
    bars: dict[str, tuple[MarketBarView, ...]] = {}
    share_histories: dict[str, dict[date, Decimal]] = {}
    for symbol in _SYMBOLS:
        closes = [Decimal("4")] * len(dates)
        if symbol in price_up:
            closes[-1] = Decimal("5")
        bars[symbol] = tuple(
            _etf_bar(symbol, trade_date, close)
            for trade_date, close in zip(dates, closes, strict=True)
        )
        tail = supplied_increments.get(symbol, (Decimal("0"),) * 5)
        shares = _share_series(
            len(dates),
            current_share=supplied_current_shares.get(symbol, Decimal("1000000")),
            tail_increments=tail,
        )
        history = {disclosure_date: Decimal("1000000")}
        history.update(dict(zip(dates, shares, strict=True)))
        if symbol in missing_shares:
            history.pop(_SIGNAL_DATE)
        share_histories[symbol] = history
    index_bars = tuple(_index_bar(trade_date, drawdown_points) for trade_date in dates)
    ratios = {
        symbol: (disclosure_date, supplied_ratios.get(symbol, Decimal("0.20")))
        for symbol in _SYMBOLS
        if supplied_ratios.get(symbol, Decimal("0.20")) is not None
    }
    return RuleMarketData(
        signal_date=_SIGNAL_DATE,
        execution_date=_next_trading_date(_SIGNAL_DATE),
        frame_index=89,
        symbols=_SYMBOLS,
        cash=Decimal("100000"),
        positions=positions,
        _bars_by_symbol=bars,
        _current_weights_by_symbol=weights,
        _share_history_by_symbol=share_histories,
        _index_history_by_code={_INDEX: index_bars},
        _combined_huijin_ratio_by_symbol=ratios,
    )


def _first_small_signal_tail() -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in (-200, 0, 0, 0, 300))


def _persistent_ten_x_tail() -> tuple[Decimal, ...]:
    return tuple(Decimal(value) for value in (1000, 3000, 10000, 40000, 200000))


def _assert_decimal_close(actual: Decimal, expected: Decimal) -> None:
    assert abs(actual - expected) <= _TOLERANCE


@pytest.mark.unit
def test_multi_experiment_uses_exact_seven_etfs_and_100k_cash() -> None:
    experiment = load_user_experiment_config(_STRATEGY_ROOT / "experiment.yaml")

    assert experiment.start_date == date(2020, 1, 1)
    assert experiment.end_date == date(2026, 8, 7)
    assert experiment.initial_cash == Decimal("100000.0000")
    assert experiment.case == "rule"
    assert set(experiment.universe.symbols) == set(_SYMBOLS)


@pytest.mark.unit
def test_multi_rule_uses_universe_and_target_weight_as_single_sources() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    source = (_STRATEGY_ROOT / "rule.py").read_text(encoding="utf-8")

    assert rule.target_weight == Decimal("0.80")
    assert "trade_etfs" not in rule.parameters
    assert "big_max_position" not in rule.parameters
    assert "trade_etfs = data.symbols" in source
    assert "big_max_position = self.target_weight" in source
    for unused_name in ("_REDUCE", "_BIG_BUY", "_SMALL_BUY", "_WAIT"):
        assert unused_name not in source


@pytest.mark.unit
@pytest.mark.parametrize("drawdown_points", [Decimal("100"), Decimal("750")])
def test_multi_rule_wait_zones_do_not_establish_a_base_position(
    drawdown_points: Decimal,
) -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    increments = {symbol: _first_small_signal_tail() for symbol in _SYMBOLS}

    result = rule.generate_weights(
        _market_data(drawdown_points=drawdown_points, tail_increments_by_symbol=increments)
    )

    assert result is NO_REBALANCE


@pytest.mark.unit
def test_multi_rule_requires_four_of_seven_for_small_buy() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    contributors = _SYMBOLS[:4]
    increments = {symbol: _first_small_signal_tail() for symbol in contributors}

    result = rule.generate_weights(
        _market_data(drawdown_points=Decimal("450"), tail_increments_by_symbol=increments)
    )

    assert result is not NO_REBALANCE
    assert set(result) == set(_SYMBOLS)
    _assert_decimal_close(sum(result.values(), start=_ZERO), Decimal("0.20") / 3)


@pytest.mark.unit
def test_multi_rule_three_of_seven_does_not_trigger_small_buy() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    increments = {symbol: _first_small_signal_tail() for symbol in _SYMBOLS[:3]}

    result = rule.generate_weights(
        _market_data(drawdown_points=Decimal("450"), tail_increments_by_symbol=increments)
    )

    assert result is NO_REBALANCE


@pytest.mark.unit
def test_multi_rule_requires_six_persistent_ten_x_flows_for_big_buy() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    increments = {symbol: _persistent_ten_x_tail() for symbol in _SYMBOLS[:6]}

    result = rule.generate_weights(
        _market_data(drawdown_points=Decimal("1000"), tail_increments_by_symbol=increments)
    )

    assert result is not NO_REBALANCE
    _assert_decimal_close(sum(result.values(), start=_ZERO), Decimal("0.04"))


@pytest.mark.unit
@pytest.mark.parametrize("use_divergence", [False, True])
def test_multi_rule_reduces_proportionally_when_four_of_seven_exit(
    use_divergence: bool,
) -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    current_weights = {symbol: Decimal("0.10") for symbol in _SYMBOLS}
    if use_divergence:
        exit_tail = tuple(Decimal(value) for value in (100, 100, 100, 100, -100))
        price_up_symbols = set(_SYMBOLS[:4])
    else:
        exit_tail = (Decimal("-100"),) * 5
        price_up_symbols = set()
    increments = {symbol: exit_tail for symbol in _SYMBOLS[:4]}

    result = rule.generate_weights(
        _market_data(
            drawdown_points=Decimal("1000"),
            tail_increments_by_symbol=increments,
            current_weights=current_weights,
            price_up_symbols=price_up_symbols,
        )
    )

    assert result is not NO_REBALANCE
    _assert_decimal_close(sum(result.values(), start=_ZERO), Decimal("0.20"))
    for target in result.values():
        _assert_decimal_close(target, Decimal("0.20") / 7)


@pytest.mark.unit
def test_multi_rule_uses_ymin_median_when_hs300_ymin_is_unavailable() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    contributors = _SYMBOLS[:4]
    increments = {symbol: _first_small_signal_tail() for symbol in contributors}
    current_weights = {symbol: _ZERO for symbol in _SYMBOLS}
    current_weights[_HS300] = Decimal("0.01")

    result = rule.generate_weights(
        _market_data(
            drawdown_points=Decimal("450"),
            tail_increments_by_symbol=increments,
            current_weights=current_weights,
            huijin_ratios={_HS300: None},
        )
    )

    assert result is not NO_REBALANCE
    assert result[_HS300] == Decimal("0.01")
    _assert_decimal_close(sum(result.values(), start=_ZERO), Decimal("0.20") / 3)


@pytest.mark.unit
def test_multi_rule_clamps_negative_ymin_and_never_sells_during_buy() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    contributors = _SYMBOLS[:4]
    increments = {symbol: _first_small_signal_tail() for symbol in contributors}
    current_weights = {symbol: _ZERO for symbol in _SYMBOLS}
    current_weights["SH.510050"] = Decimal("0.05")

    result = rule.generate_weights(
        _market_data(
            drawdown_points=Decimal("450"),
            tail_increments_by_symbol=increments,
            current_weights=current_weights,
            current_shares={"SH.588080": Decimal("500000")},
            huijin_ratios={"SH.588080": Decimal("0.10")},
        )
    )

    assert result is not NO_REBALANCE
    assert result["SH.510050"] >= Decimal("0.05")
    assert result["SH.588080"] == _ZERO
    _assert_decimal_close(sum(result.values(), start=_ZERO), Decimal("0.20") / 3)


@pytest.mark.unit
def test_multi_rule_blocks_new_buys_when_a_signal_share_is_missing() -> None:
    rule = load_user_rule(_STRATEGY_ROOT / "rule.py", allowed_root=_STRATEGY_ROOT)
    increments = {symbol: _first_small_signal_tail() for symbol in _SYMBOLS}

    result = rule.generate_weights(
        _market_data(
            drawdown_points=Decimal("450"),
            tail_increments_by_symbol=increments,
            missing_signal_share_symbols={"SH.588080"},
        )
    )

    assert result is NO_REBALANCE
