"""Unit tests for the beginner-facing Rule adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import cast

import pytest

from etf_backtest.core.account import Account
from etf_backtest.core.market import IndexBarView, MarketBarView, TurnoverRule
from etf_backtest.core.position import Position
from etf_backtest.core.target import TargetPortfolio
from etf_backtest.strategy import (
    NO_REBALANCE,
    RuleMarketData,
    RuleSettings,
    SimpleRuleStrategy,
    UserRule,
    WeightInput,
)
from etf_backtest.strategy.context import AccountView, StrategyContext

SIGNAL_DATE = date(2024, 1, 5)
SYMBOLS = ("SH.510300", "SH.518880")


def _view(symbol: str, *, days_before_signal: int, close: str, volume: int) -> MarketBarView:
    trade_date = SIGNAL_DATE - timedelta(days=days_before_signal)
    value = Decimal(close)
    return MarketBarView(
        source_record_key=f"QMT:{symbol.split('.')[1]}:{trade_date.isoformat()}:1:1",
        symbol=symbol,
        trade_date=trade_date,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=volume,
        suspended=False,
    )


def _inputs() -> tuple[tuple[MarketBarView, ...], AccountView, StrategyContext]:
    account = Account(
        cash=Decimal("1000"),
        positions={
            "SH.510300": Position(
                symbol="SH.510300",
                turnover_rule=TurnoverRule.T1,
                total_quantity=200,
                available_quantity=100,
                today_buy_quantity=100,
            ),
            "SH.518880": Position(
                symbol="SH.518880",
                turnover_rule=TurnoverRule.T0,
                total_quantity=300,
                available_quantity=300,
            ),
        },
    )
    account_view = AccountView.from_account(account)
    context = StrategyContext(
        signal_date=SIGNAL_DATE,
        execution_date=date(2024, 1, 8),
        frame_index=4,
        symbols=SYMBOLS,
        account_view=account_view,
        current_weights_by_symbol={
            "SH.510300": Decimal("0.25"),
            "SH.518880": Decimal("0.50"),
        },
    )
    history = (
        _view("SH.518880", days_before_signal=2, close="20", volume=300),
        _view("SH.510300", days_before_signal=2, close="10", volume=100),
        _view("SH.510300", days_before_signal=0, close="11", volume=200),
        _view("SH.518880", days_before_signal=0, close="19", volume=400),
    )
    return history, account_view, context


class _CapturingRule(UserRule):
    settings = RuleSettings(
        lookback_trading_days=21,
        rebalance_every_trading_days=4,
    )

    def __init__(self) -> None:
        self.seen: RuleMarketData | None = None

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        self.seen = data
        return {"510300": "0.60", "518880": Decimal("0.30")}


def _generate(strategy: SimpleRuleStrategy) -> TargetPortfolio:
    history, account_view, context = _inputs()
    return strategy.generate_target(
        signal_date=SIGNAL_DATE,
        market_history=history,
        account_view=account_view,
        context=context,
    )


@pytest.mark.unit
def test_adapter_owns_schedule_lookback_and_target_conversion() -> None:
    rule = _CapturingRule()
    strategy = SimpleRuleStrategy(rule=rule)

    assert strategy.rule is rule
    assert strategy.required_history_trading_days == 21
    assert strategy.should_generate_target(0)
    assert not strategy.should_generate_target(3)
    assert strategy.should_generate_target(4)
    assert dict(_generate(strategy).weights) == {
        "SH.510300": Decimal("0.60"),
        "SH.518880": Decimal("0.30"),
    }


@pytest.mark.unit
def test_rule_can_explicitly_decline_to_create_a_rebalance_target() -> None:
    class _NoRebalanceRule(UserRule):
        def generate_weights(self, data: RuleMarketData):
            del data
            return NO_REBALANCE

    history, account_view, context = _inputs()
    result = SimpleRuleStrategy(rule=_NoRebalanceRule()).generate_target(
        signal_date=SIGNAL_DATE,
        market_history=history,
        account_view=account_view,
        context=context,
    )

    assert result is NO_REBALANCE


@pytest.mark.unit
def test_rule_market_data_has_friendly_immutable_accessors() -> None:
    rule = _CapturingRule()
    strategy = SimpleRuleStrategy(rule=rule)
    _generate(strategy)

    data = rule.seen
    assert data is not None
    assert data.signal_date == SIGNAL_DATE
    assert data.execution_date == date(2024, 1, 8)
    assert data.frame_index == 4
    assert data.symbols == SYMBOLS
    assert data.cash == Decimal("1000.0000")
    assert data.closes("510300") == (Decimal("10"), Decimal("11"))
    assert data.volumes("SH.510300") == (100, 200)
    assert data.latest("SH.510300") is not None
    assert data.latest("SH.510300").trade_date == SIGNAL_DATE  # type: ignore[union-attr]
    assert data.has_history("510300", 2)
    assert data.close_return("510300", 1) == Decimal("0.1")
    assert data.close_return("510300", 2) is None
    assert data.position_quantity("510300") == 200
    assert data.available_quantity("510300") == 100
    assert data.current_weight("510300") == Decimal("0.25")
    assert data.current_weight("SH.518880") == Decimal("0.50")
    with pytest.raises(TypeError):
        data.positions["SH.510300"] = data.positions["SH.510300"]  # type: ignore[index]
    with pytest.raises(ValueError, match="outside"):
        data.bars("SH.510500")


@pytest.mark.unit
def test_rule_market_data_exposes_daily_share_and_latest_prior_huijin_ratio() -> None:
    history, account_view, context = _inputs()
    enriched = replace(
        context,
        share_history_by_symbol={
            "SH.510300": {
                date(2024, 1, 4): Decimal("1000000.0000"),
                SIGNAL_DATE: Decimal("1100000.0000"),
            }
        },
        huijin_ratios_by_symbol={
            "SH.510300": {
                "中央汇金投资有限责任公司": (
                    date(2023, 12, 31),
                    Decimal("0.0971"),
                ),
                "中央汇金资产管理有限责任公司": (
                    date(2023, 6, 30),
                    Decimal("0.0125"),
                ),
            }
        },
        combined_huijin_ratio_by_symbol={"SH.510300": (date(2023, 12, 31), Decimal("0.1096"))},
    )

    data = RuleMarketData.from_strategy_inputs(
        market_history=history,
        account_view=account_view,
        context=enriched,
    )

    assert data.share_on("510300", SIGNAL_DATE) == Decimal("1100000.0000")
    assert data.share_history("SH.510300") == (
        (date(2024, 1, 4), Decimal("1000000.0000")),
        (SIGNAL_DATE, Decimal("1100000.0000")),
    )
    assert data.latest_huijin_ratio("510300", "中央汇金投资有限责任公司") == Decimal("0.0971")
    assert data.latest_huijin_ratio("510300", "未知公司") is None
    assert data.latest_combined_huijin_ratio("510300") == (
        date(2023, 12, 31),
        Decimal("0.1096"),
    )
    assert data.latest_combined_huijin_ratio("518880") is None
    with pytest.raises(ValueError, match="future"):
        data.share_on("510300", date(2024, 1, 6))


@pytest.mark.unit
def test_rule_market_data_exposes_date_gated_index_bars_outside_etf_universe() -> None:
    history, account_view, context = _inputs()
    index_bars = (
        IndexBarView(
            index_code="000001.SH",
            trade_date=date(2024, 1, 4),
            open=Decimal("3000"),
            high=Decimal("3020"),
            low=Decimal("2980"),
            close=Decimal("3010"),
            pre_close=Decimal("2990"),
            pct_change=Decimal("0.6689"),
            source_system="TUSHARE",
        ),
        IndexBarView(
            index_code="000001.SH",
            trade_date=SIGNAL_DATE,
            open=Decimal("3010"),
            high=Decimal("3030"),
            low=Decimal("2995"),
            close=Decimal("3005"),
            pre_close=Decimal("3010"),
            pct_change=Decimal("-0.1661"),
            source_system="TUSHARE",
        ),
    )
    enriched = replace(context, index_history_by_code={"000001.SH": index_bars})

    data = RuleMarketData.from_strategy_inputs(
        market_history=history,
        account_view=account_view,
        context=enriched,
    )

    assert data.index_bars("000001.SH") == index_bars
    with pytest.raises(ValueError, match="index code"):
        data.index_bars("000300.SH")


class _FixedWeightsRule(UserRule):
    def __init__(self, weights: object) -> None:
        self._weights = weights

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        del data
        return cast(Mapping[str, WeightInput], self._weights)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, object()])
def test_adapter_rejects_unsafe_weight_types(value: object) -> None:
    strategy = SimpleRuleStrategy(rule=_FixedWeightsRule({"510300": value}))
    with pytest.raises(TypeError, match="weights"):
        _generate(strategy)


@pytest.mark.unit
def test_adapter_accepts_finite_float_weight_and_rejects_nonfinite_float() -> None:
    strategy = SimpleRuleStrategy(rule=_FixedWeightsRule({"510300": 0.2}))
    assert dict(_generate(strategy).weights) == {"SH.510300": Decimal("0.2")}

    for value in (float("inf"), float("-inf"), float("nan")):
        strategy = SimpleRuleStrategy(rule=_FixedWeightsRule({"510300": value}))
        with pytest.raises(ValueError, match="finite"):
            _generate(strategy)


@pytest.mark.unit
def test_adapter_rejects_non_mapping_unknown_duplicate_and_excess_weights() -> None:
    cases: tuple[tuple[object, type[Exception], str], ...] = (
        ([], TypeError, "mapping"),
        ({"510500": "0.5"}, ValueError, "outside"),
        ({"510300": "0.5", "SH.510300": "0.4"}, ValueError, "unique"),
        ({"510300": "0.8", "518880": "0.3"}, ValueError, "sum"),
    )
    for weights, error_type, message in cases:
        strategy = SimpleRuleStrategy(rule=_FixedWeightsRule(weights))
        with pytest.raises(error_type, match=message):
            _generate(strategy)


@pytest.mark.unit
def test_adapter_constructor_rejects_invalid_rule_and_settings() -> None:
    with pytest.raises(TypeError, match="UserRule"):
        SimpleRuleStrategy(rule=object())  # type: ignore[arg-type]

    class BadSettingsRule(_CapturingRule):
        settings = object()  # type: ignore[assignment]

    with pytest.raises(TypeError, match="settings"):
        SimpleRuleStrategy(rule=BadSettingsRule())


@pytest.mark.unit
def test_rule_market_data_rejects_non_chronological_and_future_history() -> None:
    history, account_view, context = _inputs()
    with pytest.raises(ValueError, match="chronological"):
        RuleMarketData(
            signal_date=SIGNAL_DATE,
            execution_date=context.execution_date,
            frame_index=0,
            symbols=SYMBOLS,
            cash=account_view.cash,
            positions=account_view.positions,
            _bars_by_symbol={"SH.510300": (history[2], history[1])},
            _current_weights_by_symbol=context.current_weights_by_symbol,
        )
    future = _view("SH.510300", days_before_signal=-1, close="12", volume=200)
    with pytest.raises(ValueError, match="future"):
        RuleMarketData(
            signal_date=SIGNAL_DATE,
            execution_date=context.execution_date,
            frame_index=0,
            symbols=SYMBOLS,
            cash=account_view.cash,
            positions=account_view.positions,
            _bars_by_symbol={"SH.510300": (future,)},
            _current_weights_by_symbol=context.current_weights_by_symbol,
        )


@pytest.mark.unit
def test_rule_market_data_rejects_duplicate_and_outside_history_symbols() -> None:
    history, account_view, context = _inputs()
    with pytest.raises(ValueError, match="unique"):
        RuleMarketData(
            signal_date=SIGNAL_DATE,
            execution_date=context.execution_date,
            frame_index=0,
            symbols=SYMBOLS,
            cash=account_view.cash,
            positions=account_view.positions,
            _bars_by_symbol={
                "510300": (history[1], history[2]),
                "SH.510300": (history[1], history[2]),
            },
            _current_weights_by_symbol=context.current_weights_by_symbol,
        )
    outside = _view("SH.510500", days_before_signal=0, close="12", volume=200)
    with pytest.raises(ValueError, match="outside"):
        RuleMarketData.from_strategy_inputs(
            market_history=(*history, outside),
            account_view=account_view,
            context=context,
        )


@pytest.mark.unit
def test_rule_market_data_constructor_validation_guards() -> None:
    history, account_view, context = _inputs()
    data = RuleMarketData.from_strategy_inputs(
        market_history=history,
        account_view=account_view,
        context=context,
    )
    cases: tuple[tuple[dict[str, object], type[Exception], str], ...] = (
        ({"signal_date": datetime(2024, 1, 5)}, TypeError, "datetime.date"),
        ({"execution_date": SIGNAL_DATE}, ValueError, "follow"),
        ({"frame_index": -1}, ValueError, "frame_index"),
        ({"symbols": "SH.510300"}, TypeError, "sequence"),
        ({"symbols": ()}, ValueError, "non-empty"),
        ({"cash": 1000}, TypeError, "Decimal"),
        ({"cash": Decimal("-1")}, ValueError, "non-negative"),
        ({"positions": []}, TypeError, "mapping"),
        ({"positions": {}}, ValueError, "cover"),
        ({"_bars_by_symbol": []}, TypeError, "mapping"),
        (
            {"_bars_by_symbol": {"SH.510500": ()}},
            ValueError,
            "outside",
        ),
        (
            {"_bars_by_symbol": {"SH.510300": "not-bars"}},
            TypeError,
            "sequence",
        ),
        (
            {"_bars_by_symbol": {"SH.510300": (object(),)}},
            TypeError,
            "MarketBarView",
        ),
        (
            {"_bars_by_symbol": {"SH.510300": (history[0],)}},
            ValueError,
            "disagree",
        ),
    )
    replace_market_data = cast(Callable[..., RuleMarketData], replace)
    for changes, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            replace_market_data(data, **changes)


@pytest.mark.unit
def test_rule_market_data_factory_validates_engine_boundary_types() -> None:
    history, account_view, context = _inputs()
    other_view = AccountView(cash=account_view.cash, positions=account_view.positions)
    cases: tuple[tuple[object, object, object, type[Exception], str], ...] = (
        (history, object(), context, TypeError, "AccountView"),
        (history, account_view, object(), TypeError, "StrategyContext"),
        (history, other_view, context, ValueError, "stored"),
        (
            "not-history",
            account_view,
            context,
            TypeError,
            "sequence",
        ),
        (
            (object(),),
            account_view,
            context,
            TypeError,
            "MarketBarView",
        ),
    )
    for supplied_history, supplied_account, supplied_context, error_type, message in cases:
        with pytest.raises(error_type, match=message):
            RuleMarketData.from_strategy_inputs(
                market_history=cast(Sequence[MarketBarView], supplied_history),
                account_view=cast(AccountView, supplied_account),
                context=cast(StrategyContext, supplied_context),
            )


@pytest.mark.unit
def test_adapter_converts_integer_and_rejects_bad_symbols_or_decimal_text() -> None:
    zero_strategy = SimpleRuleStrategy(rule=_FixedWeightsRule({"510300": 0}))
    assert dict(_generate(zero_strategy).weights) == {"SH.510300": Decimal("0")}

    cases = (
        ({123: "0.5"}, TypeError, "symbols"),
        ({"510300": " "}, ValueError, "blank"),
        ({"510300": "not-decimal"}, ValueError, "valid decimal"),
    )
    for weights, error_type, message in cases:
        strategy = SimpleRuleStrategy(rule=_FixedWeightsRule(weights))
        with pytest.raises(error_type, match=message):
            _generate(strategy)


@pytest.mark.unit
def test_positive_integer_helpers_reject_non_integer_values() -> None:
    strategy = SimpleRuleStrategy(rule=_CapturingRule())
    with pytest.raises(TypeError, match="integer"):
        strategy.should_generate_target(True)
    history, account_view, context = _inputs()
    data = RuleMarketData.from_strategy_inputs(
        market_history=history,
        account_view=account_view,
        context=context,
    )
    with pytest.raises(TypeError, match="integer"):
        data.has_history("510300", True)


class _ConfiguredRule(UserRule):
    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        del data
        return {}


@pytest.mark.unit
def test_user_rule_exposes_validated_target_weight_and_immutable_parameters() -> None:
    defaults = _ConfiguredRule()
    assert defaults.target_weight == Decimal("0.90")
    assert dict(defaults.parameters) == {}

    supplied = {"window": 5}

    class ConfiguredRule(_ConfiguredRule):
        settings = RuleSettings(target_weight=0.625, parameters=supplied)

    configured = ConfiguredRule()
    supplied["window"] = 99
    assert configured.target_weight == Decimal("0.625")
    assert dict(configured.parameters) == {"window": 5}
    with pytest.raises(TypeError):
        configured.parameters["window"] = 10  # type: ignore[index]


@pytest.mark.unit
def test_user_rule_rejects_invalid_target_weight_and_parameters() -> None:
    with pytest.raises(TypeError, match="boolean"):
        RuleSettings(target_weight=True)
    with pytest.raises(ValueError, match="finite"):
        RuleSettings(target_weight=float("inf"))
    for value in ("0", "1.01"):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            RuleSettings(target_weight=value)
    with pytest.raises(TypeError, match="mapping"):
        RuleSettings(parameters=[])  # type: ignore[arg-type]
    for parameters in ({"": 1}, {" leading": 1}, {"trailing ": 1}, {1: "value"}):
        with pytest.raises(ValueError, match="nonblank strings"):
            RuleSettings(parameters=parameters)  # type: ignore[arg-type]
