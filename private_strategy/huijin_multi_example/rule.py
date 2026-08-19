# ruff: noqa: RUF002, RUF003
"""助力汇金：七只 ETF 共同确认信号，510300 的 Ymin 锚定组合仓位。

上证指数的 60 日高点回撤仅用于区分小周期和大周期；七只 ETF 的份额
变化共同形成申购、赎回、十倍量和价格—份额背离宽度。组合产生统一的
买入或减仓状态后，七只 ETF 按各自正 ``Ymin`` 比例分配目标仓位。

当前权重来自框架提供的 ``data.current_weight(symbol)`` 通用接口；行情截断、
D+1 执行、费用、滑点、停牌、涨跌停和账户记账仍全部由框架负责。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import pairwise

from etf_backtest.core.market import IndexBarView, MarketBarView
from etf_backtest.strategy.rule import (
    NO_REBALANCE,
    RuleMarketData,
    RuleOutput,
    RuleSettings,
    UserRule,
)

_ZERO = Decimal("0")
_ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class _FlowMetrics:
    signal_valid: bool
    share_change_1d: Decimal | None
    share_change_5d: Decimal | None
    normal_positive_flow: Decimal | None
    flow_multiple_1d: Decimal | None
    buy_divergence: bool
    sell_divergence: bool


@dataclass(frozen=True, slots=True)
class _EtfSnapshot:
    current_weight: Decimal
    metrics: _FlowMetrics
    ymin: Decimal | None
    ymax: Decimal | None


@dataclass(frozen=True, slots=True)
class _Widths:
    all_signal_valid: bool
    subscription_5d: Decimal
    redemption_5d: Decimal
    ten_x: Decimal
    buy_divergence: Decimal
    sell_divergence: Decimal
    aggregate_share_change_5d: Decimal


def _decimal_parameter(parameters: Mapping[str, object], key: str) -> Decimal:
    value = parameters[key]
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{key} must be finite")
    return result


def _integer_parameter(parameters: Mapping[str, object], key: str) -> int:
    value = parameters[key]
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _last_share_on_or_before(shares: Mapping[date, Decimal], cutoff: date) -> Decimal | None:
    eligible = tuple(asof_date for asof_date in shares if asof_date <= cutoff)
    return None if not eligible else shares[max(eligible)]


def _drawdown_at(
    index_bars: Sequence[IndexBarView],
    trade_date: date,
    stage_high_days: int,
) -> Decimal | None:
    positions = {bar.trade_date: index for index, bar in enumerate(index_bars)}
    position = positions.get(trade_date)
    if position is None or position + 1 < stage_high_days:
        return None
    window = index_bars[position - stage_high_days + 1 : position + 1]
    return max(bar.high for bar in window) - index_bars[position].close


def _median(values: Sequence[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")


def _positive_flow_average(
    *,
    bars: Sequence[MarketBarView],
    shares: Mapping[date, Decimal],
    position: int,
    lookback_days: int,
) -> Decimal | None:
    # The current day's increment is excluded. The window contains the 20
    # completed one-day share changes immediately preceding the signal day.
    if position < lookback_days + 1:
        return None
    positive_changes: list[Decimal] = []
    for end_position in range(position - lookback_days, position):
        current_share = shares.get(bars[end_position].trade_date)
        previous_share = shares.get(bars[end_position - 1].trade_date)
        if current_share is None or previous_share is None:
            return None
        change = current_share - previous_share
        if change > _ZERO:
            positive_changes.append(change)
    if not positive_changes:
        return None
    return sum(positive_changes, start=_ZERO) / Decimal(len(positive_changes))


def _flow_metrics_at(
    *,
    bars: Sequence[MarketBarView],
    shares: Mapping[date, Decimal],
    trade_date: date,
    normal_flow_lookback_days: int,
) -> _FlowMetrics:
    invalid = _FlowMetrics(
        signal_valid=False,
        share_change_1d=None,
        share_change_5d=None,
        normal_positive_flow=None,
        flow_multiple_1d=None,
        buy_divergence=False,
        sell_divergence=False,
    )
    positions = {bar.trade_date: index for index, bar in enumerate(bars)}
    position = positions.get(trade_date)
    if position is None or position < 5:
        return invalid
    current_bar = bars[position]
    if current_bar.suspended:
        return invalid
    current_share = shares.get(current_bar.trade_date)
    previous_share = shares.get(bars[position - 1].trade_date)
    five_day_share = shares.get(bars[position - 5].trade_date)
    if (
        current_share is None
        or previous_share is None
        or five_day_share is None
        or current_share <= _ZERO
        or previous_share <= _ZERO
        or five_day_share <= _ZERO
        or bars[position - 1].close <= _ZERO
    ):
        return invalid
    share_change_1d = current_share - previous_share
    share_change_5d = current_share - five_day_share
    normal_positive_flow = _positive_flow_average(
        bars=bars,
        shares=shares,
        position=position,
        lookback_days=normal_flow_lookback_days,
    )
    flow_multiple = (
        None
        if normal_positive_flow is None or normal_positive_flow <= _ZERO
        else share_change_1d / normal_positive_flow
    )
    price_change_1d = current_bar.close / bars[position - 1].close - _ONE
    return _FlowMetrics(
        signal_valid=True,
        share_change_1d=share_change_1d,
        share_change_5d=share_change_5d,
        normal_positive_flow=normal_positive_flow,
        flow_multiple_1d=flow_multiple,
        buy_divergence=price_change_1d < _ZERO and share_change_1d > _ZERO,
        sell_divergence=price_change_1d > _ZERO and share_change_1d < _ZERO,
    )


def _widths(
    metrics_by_symbol: Mapping[str, _FlowMetrics],
    *,
    symbol_count: int,
    ten_x_multiplier: Decimal,
) -> _Widths:
    denominator = Decimal(symbol_count)
    metrics = tuple(metrics_by_symbol.values())
    subscription_count = sum(
        metric.signal_valid
        and metric.share_change_5d is not None
        and metric.share_change_5d > _ZERO
        for metric in metrics
    )
    redemption_count = sum(
        metric.signal_valid
        and metric.share_change_5d is not None
        and metric.share_change_5d < _ZERO
        for metric in metrics
    )
    ten_x_count = sum(
        metric.signal_valid
        and metric.flow_multiple_1d is not None
        and metric.flow_multiple_1d >= ten_x_multiplier
        for metric in metrics
    )
    buy_divergence_count = sum(metric.signal_valid and metric.buy_divergence for metric in metrics)
    sell_divergence_count = sum(
        metric.signal_valid and metric.sell_divergence for metric in metrics
    )
    aggregate_share_change_5d = sum(
        (
            metric.share_change_5d
            for metric in metrics
            if metric.signal_valid and metric.share_change_5d is not None
        ),
        start=_ZERO,
    )
    return _Widths(
        all_signal_valid=all(metric.signal_valid for metric in metrics),
        subscription_5d=Decimal(subscription_count) / denominator,
        redemption_5d=Decimal(redemption_count) / denominator,
        ten_x=Decimal(ten_x_count) / denominator,
        buy_divergence=Decimal(buy_divergence_count) / denominator,
        sell_divergence=Decimal(sell_divergence_count) / denominator,
        aggregate_share_change_5d=aggregate_share_change_5d,
    )


class Strategy(UserRule):
    """七 ETF 汇金宽度策略。"""

    settings = RuleSettings(
        lookback_trading_days=90,
        rebalance_every_trading_days=1,
        # 大周期最多持有 80%；修改这个值即可调整策略总仓位上限。
        target_weight="0.80",
        parameters={
            # 锚定 510300 的汇金持仓下限，并用上证指数划分回撤周期。
            "position_anchor_etf": "SH.510300",
            "market_index": "000001.SH",
            # 回撤区间，单位为上证指数点数。
            "stage_high_days": 60,
            "small_drawdown_low_points": "400",
            "small_drawdown_high_points": "500",
            "big_drawdown_points": "1000",
            # ETF 份额流量和七只 ETF 的信号宽度阈值。
            "normal_flow_lookback_days": 20,
            "ten_x_multiplier": "10",
            "small_width": "0.50",
            "strong_width": "0.75",
            # 连续确认和分批建仓天数。
            "big_confirm_days": 5,
            "small_build_days": 3,
            "big_build_days": 5,
            # 小周期仓位、减仓保留仓位及单日买卖上限。
            "small_max_position": "0.30",
            "base_keep_position": "0.10",
            "daily_buy_cap": "0.20",
            "daily_sell_cap": "0.50",
            # 510300 的 Ymin 连续下降天数达到该值时触发减仓。
            "ymin_decline_days": 3,
        },
    )

    def generate_weights(self, data: RuleMarketData) -> RuleOutput:
        parameters = self.parameters
        # 交易标的只在 experiment.yaml 的 universe 中维护，避免两处名单不一致。
        trade_etfs = data.symbols
        market_index = parameters["market_index"]
        anchor_etf = parameters["position_anchor_etf"]
        if not isinstance(market_index, str) or not isinstance(anchor_etf, str):
            raise TypeError("market_index and position_anchor_etf must be strings")
        if anchor_etf not in trade_etfs:
            raise ValueError("position_anchor_etf must belong to trade_etfs")

        stage_high_days = _integer_parameter(parameters, "stage_high_days")
        normal_flow_days = _integer_parameter(parameters, "normal_flow_lookback_days")
        big_confirm_days = _integer_parameter(parameters, "big_confirm_days")
        small_build_days = _integer_parameter(parameters, "small_build_days")
        big_build_days = _integer_parameter(parameters, "big_build_days")
        ymin_decline_days = _integer_parameter(parameters, "ymin_decline_days")
        small_drawdown_low = _decimal_parameter(parameters, "small_drawdown_low_points")
        small_drawdown_high = _decimal_parameter(parameters, "small_drawdown_high_points")
        big_drawdown = _decimal_parameter(parameters, "big_drawdown_points")
        ten_x_multiplier = _decimal_parameter(parameters, "ten_x_multiplier")
        small_width = _decimal_parameter(parameters, "small_width")
        strong_width = _decimal_parameter(parameters, "strong_width")
        small_max_position = _decimal_parameter(parameters, "small_max_position")
        big_max_position = self.target_weight
        base_keep_position = _decimal_parameter(parameters, "base_keep_position")
        daily_buy_cap = _decimal_parameter(parameters, "daily_buy_cap")
        daily_sell_cap = _decimal_parameter(parameters, "daily_sell_cap")

        bars_by_symbol = {symbol: data.bars(symbol) for symbol in trade_etfs}
        shares_by_symbol = {symbol: dict(data.share_history(symbol)) for symbol in trade_etfs}
        index_bars = data.index_bars(market_index)
        metrics_cache: dict[tuple[str, date], _FlowMetrics] = {}

        def metrics_for(symbol: str, trade_date: date) -> _FlowMetrics:
            key = (symbol, trade_date)
            cached = metrics_cache.get(key)
            if cached is not None:
                return cached
            result = _flow_metrics_at(
                bars=bars_by_symbol[symbol],
                shares=shares_by_symbol[symbol],
                trade_date=trade_date,
                normal_flow_lookback_days=normal_flow_days,
            )
            metrics_cache[key] = result
            return result

        def widths_for(trade_date: date) -> _Widths:
            return _widths(
                {symbol: metrics_for(symbol, trade_date) for symbol in trade_etfs},
                symbol_count=len(trade_etfs),
                ten_x_multiplier=ten_x_multiplier,
            )

        snapshots = {
            symbol: self._snapshot(
                symbol=symbol,
                data=data,
                metrics=metrics_for(symbol, data.signal_date),
                shares=shares_by_symbol[symbol],
            )
            for symbol in trade_etfs
        }
        current_widths = widths_for(data.signal_date)
        reference_dates = tuple(bar.trade_date for bar in bars_by_symbol[trade_etfs[0]])

        anchor_declines = self._anchor_ymin_declines(
            anchor_etf=anchor_etf,
            data=data,
            bars=bars_by_symbol[anchor_etf],
            shares=shares_by_symbol[anchor_etf],
            decline_days=ymin_decline_days,
        )
        reduce_signal = (
            current_widths.redemption_5d >= small_width
            or current_widths.sell_divergence >= small_width
            or anchor_declines
        )
        if reduce_signal:
            return self._reduce_targets(
                trade_etfs=trade_etfs,
                snapshots=snapshots,
                anchor_etf=anchor_etf,
                base_keep_position=base_keep_position,
                daily_sell_cap=daily_sell_cap,
            )

        drawdown = _drawdown_at(index_bars, data.signal_date, stage_high_days)
        ten_x_streak = self._consecutive_width_days(
            reference_dates=reference_dates,
            signal_date=data.signal_date,
            condition=lambda trade_date: (
                widths_for(trade_date).all_signal_valid
                and widths_for(trade_date).ten_x >= strong_width
            ),
            maximum_days=big_confirm_days + big_build_days - 1,
        )
        big_buy = (
            drawdown is not None
            and drawdown >= big_drawdown
            and current_widths.all_signal_valid
            and current_widths.subscription_5d >= strong_width
            and ten_x_streak >= big_confirm_days
        )
        if big_buy:
            build_day = min(ten_x_streak - big_confirm_days + 1, big_build_days)
            return self._buy_targets(
                trade_etfs=trade_etfs,
                snapshots=snapshots,
                anchor_etf=anchor_etf,
                maximum_position=big_max_position,
                build_fraction=Decimal(build_day) / Decimal(big_build_days),
                daily_buy_cap=daily_buy_cap,
                use_median_floor=True,
            )

        def small_condition(trade_date: date) -> bool:
            return self._is_small_buy_day(
                trade_date=trade_date,
                index_bars=index_bars,
                stage_high_days=stage_high_days,
                small_drawdown_low=small_drawdown_low,
                small_drawdown_high=small_drawdown_high,
                widths=widths_for(trade_date),
                small_width=small_width,
            )

        if small_condition(data.signal_date):
            small_streak = self._consecutive_width_days(
                reference_dates=reference_dates,
                signal_date=data.signal_date,
                condition=small_condition,
                maximum_days=small_build_days,
            )
            return self._buy_targets(
                trade_etfs=trade_etfs,
                snapshots=snapshots,
                anchor_etf=anchor_etf,
                maximum_position=small_max_position,
                build_fraction=Decimal(small_streak) / Decimal(small_build_days),
                daily_buy_cap=daily_buy_cap,
                use_median_floor=False,
            )

        return NO_REBALANCE

    @staticmethod
    def _snapshot(
        *,
        symbol: str,
        data: RuleMarketData,
        metrics: _FlowMetrics,
        shares: Mapping[date, Decimal],
    ) -> _EtfSnapshot:
        current_weight = data.current_weight(symbol)
        current_share = shares.get(data.signal_date)
        huijin_disclosure = data.latest_combined_huijin_ratio(symbol)
        if current_share is None or current_share <= _ZERO or huijin_disclosure is None:
            return _EtfSnapshot(current_weight, metrics, None, None)
        disclosure_end_date, huijin_ratio = huijin_disclosure
        disclosure_share = _last_share_on_or_before(shares, disclosure_end_date)
        if disclosure_share is None or disclosure_share <= _ZERO:
            return _EtfSnapshot(current_weight, metrics, None, None)
        share_ratio_to_disclosure = current_share / disclosure_share
        ymin = max(_ZERO, share_ratio_to_disclosure - (_ONE - huijin_ratio))
        return _EtfSnapshot(
            current_weight=current_weight,
            metrics=metrics,
            ymin=ymin,
            ymax=share_ratio_to_disclosure,
        )

    @staticmethod
    def _anchor_ymin_declines(
        *,
        anchor_etf: str,
        data: RuleMarketData,
        bars: Sequence[MarketBarView],
        shares: Mapping[date, Decimal],
        decline_days: int,
    ) -> bool:
        disclosure = data.latest_combined_huijin_ratio(anchor_etf)
        if disclosure is None:
            return False
        disclosure_end_date, huijin_ratio = disclosure
        disclosure_share = _last_share_on_or_before(shares, disclosure_end_date)
        positions = {bar.trade_date: index for index, bar in enumerate(bars)}
        position = positions.get(data.signal_date)
        if disclosure_share is None or disclosure_share <= _ZERO or position is None:
            return False
        observations = decline_days + 1
        if position + 1 < observations:
            return False
        ymins: list[Decimal] = []
        for bar in bars[position - observations + 1 : position + 1]:
            current_share = shares.get(bar.trade_date)
            if current_share is None or current_share <= _ZERO:
                return False
            ymins.append(max(_ZERO, current_share / disclosure_share - (_ONE - huijin_ratio)))
        return all(current < previous for previous, current in pairwise(ymins))

    @staticmethod
    def _consecutive_width_days(
        *,
        reference_dates: Sequence[date],
        signal_date: date,
        condition: Callable[[date], bool],
        maximum_days: int,
    ) -> int:
        positions = {trade_date: index for index, trade_date in enumerate(reference_dates)}
        position = positions.get(signal_date)
        if position is None:
            return 0
        streak = 0
        for trade_date in reversed(reference_dates[: position + 1]):
            if streak >= maximum_days or not condition(trade_date):
                break
            streak += 1
        return streak

    @staticmethod
    def _is_small_buy_day(
        *,
        trade_date: date,
        index_bars: Sequence[IndexBarView],
        stage_high_days: int,
        small_drawdown_low: Decimal,
        small_drawdown_high: Decimal,
        widths: _Widths,
        small_width: Decimal,
    ) -> bool:
        drawdown = _drawdown_at(index_bars, trade_date, stage_high_days)
        return (
            drawdown is not None
            and small_drawdown_low <= drawdown < small_drawdown_high
            and widths.all_signal_valid
            and widths.subscription_5d >= small_width
            and widths.aggregate_share_change_5d > _ZERO
        )

    @staticmethod
    def _position_anchor(
        *,
        snapshots: Mapping[str, _EtfSnapshot],
        anchor_etf: str,
    ) -> tuple[Decimal | None, Decimal | None]:
        valid_ymins = tuple(
            snapshot.ymin for snapshot in snapshots.values() if snapshot.ymin is not None
        )
        median_ymin = _median(valid_ymins)
        anchor_ymin = snapshots[anchor_etf].ymin
        if anchor_ymin is None:
            anchor_ymin = median_ymin
        return anchor_ymin, median_ymin

    @classmethod
    def _buy_targets(
        cls,
        *,
        trade_etfs: Sequence[str],
        snapshots: Mapping[str, _EtfSnapshot],
        anchor_etf: str,
        maximum_position: Decimal,
        build_fraction: Decimal,
        daily_buy_cap: Decimal,
        use_median_floor: bool,
    ) -> RuleOutput:
        anchor_ymin, median_ymin = cls._position_anchor(
            snapshots=snapshots,
            anchor_etf=anchor_etf,
        )
        if anchor_ymin is None:
            return NO_REBALANCE
        cap_source = (
            max(anchor_ymin, median_ymin)
            if use_median_floor and median_ymin is not None
            else anchor_ymin
        )
        full_position_cap = min(maximum_position, max(_ZERO, cap_source))
        raw_ymins = {
            symbol: max(_ZERO, snapshot.ymin) if snapshot.ymin is not None else _ZERO
            for symbol, snapshot in snapshots.items()
        }
        raw_total = sum(raw_ymins.values(), start=_ZERO)
        if raw_total <= _ZERO or full_position_cap <= _ZERO:
            return NO_REBALANCE
        full_target_total = min(raw_total, full_position_cap)
        stage_total = full_target_total * min(_ONE, max(_ZERO, build_fraction))
        desired_targets = {
            symbol: stage_total * raw_ymins[symbol] / raw_total for symbol in trade_etfs
        }
        current_targets = {symbol: snapshots[symbol].current_weight for symbol in trade_etfs}
        current_total = sum(current_targets.values(), start=_ZERO)
        available_in_stage = max(_ZERO, stage_total - current_total)
        available_in_portfolio = max(_ZERO, _ONE - current_total)
        buy_capacity = min(available_in_stage, available_in_portfolio, daily_buy_cap)
        gaps = {
            symbol: max(_ZERO, desired_targets[symbol] - current_targets[symbol])
            for symbol in trade_etfs
        }
        total_gap = sum(gaps.values(), start=_ZERO)
        if buy_capacity <= _ZERO or total_gap <= _ZERO:
            return NO_REBALANCE
        scale = min(_ONE, buy_capacity / total_gap)
        targets = {symbol: current_targets[symbol] + gaps[symbol] * scale for symbol in trade_etfs}
        if sum(targets.values(), start=_ZERO) > _ONE:
            raise ValueError("target weights exceed 100%")
        return targets

    @classmethod
    def _reduce_targets(
        cls,
        *,
        trade_etfs: Sequence[str],
        snapshots: Mapping[str, _EtfSnapshot],
        anchor_etf: str,
        base_keep_position: Decimal,
        daily_sell_cap: Decimal,
    ) -> RuleOutput:
        anchor_ymin, _ = cls._position_anchor(snapshots=snapshots, anchor_etf=anchor_etf)
        reduce_total = (
            base_keep_position
            if anchor_ymin is None
            else min(base_keep_position, max(_ZERO, anchor_ymin))
        )
        current_total = sum(
            (snapshot.current_weight for snapshot in snapshots.values()),
            start=_ZERO,
        )
        if current_total <= reduce_total:
            return NO_REBALANCE
        target_total_today = max(reduce_total, current_total - daily_sell_cap)
        reducible_symbols = tuple(
            symbol
            for symbol in trade_etfs
            if snapshots[symbol].metrics.signal_valid and snapshots[symbol].ymin is not None
        )
        fixed_symbols = tuple(symbol for symbol in trade_etfs if symbol not in reducible_symbols)
        fixed_total = sum(
            (snapshots[symbol].current_weight for symbol in fixed_symbols),
            start=_ZERO,
        )
        reducible_total = sum(
            (snapshots[symbol].current_weight for symbol in reducible_symbols),
            start=_ZERO,
        )
        if reducible_total <= _ZERO:
            return NO_REBALANCE
        desired_reducible_total = max(_ZERO, target_total_today - fixed_total)
        scale = min(_ONE, desired_reducible_total / reducible_total)
        targets = {
            symbol: (
                snapshots[symbol].current_weight * scale
                if symbol in reducible_symbols
                else snapshots[symbol].current_weight
            )
            for symbol in trade_etfs
        }
        if targets == {symbol: snapshots[symbol].current_weight for symbol in trade_etfs}:
            return NO_REBALANCE
        return targets
