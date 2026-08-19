# ruff: noqa: RUF002, RUF003
"""初学者 Rule 示例：每 20 个交易日选择 20 日动量最强的 ETF。

用户可以做什么：
- 在 ``generate_weights`` 中实现打分、过滤、轮动、择时、多资产配置或空仓规则；
- 在本文件的 ``RuleSettings`` 中修改回看长度、调仓间隔、目标仓位和任意策略参数；
- 从 ``self.parameters``/``self.target_weight`` 读取已经校验且不可变的代码内设置；
- 读取截至信号日 D 收盘的前复权日行情、现金和不可变持仓快照。

``RuleMarketData`` 常用接口：
- ``signal_date``、``execution_date``、``frame_index``、``symbols``、``cash``、``positions``；
- ``latest(symbol)``、``bars(symbol)``、``closes(symbol)``、``volumes(symbol)``；
- ``has_history(symbol, observations)``、``close_return(symbol, periods)``；
- ``position_quantity(symbol)``、``available_quantity(symbol)``、``current_weight(symbol)``；
- ``share_on(symbol, date)``、``share_history(symbol)``、``index_bars(index_code)``；
- ``latest_huijin_ratio(symbol, company)``、``latest_combined_huijin_ratio(symbol)``。

``positions[symbol]`` 可读 ``turnover_rule/total_quantity/available_quantity`` 和
``today_buy_quantity``。接口不提供成本价、盈亏、成交历史、挂单或账户修改方法。

``latest``/``bars`` 中每个 MarketBarView 常用字段为 ``symbol/trade_date/open/high/low/close``、
``volume/suspended``。价格是表内前复权策略视图，成交量是原始份额成交量；source key、
revision 和 adjustment 字段是审计元数据，通常不应作为交易信号。接口不提供原始执行价、
pre_close、成交额、ETF 名称/类别或未来行情。

``bars`` 按日期升序、只含 D 及以前且受 RuleSettings 的 lookback 限制；新上市证券可能不足。
``has_history`` 只统计 bar 数量，``close_return`` 不会自动过滤停牌。冻结资产池中的证券在
某个信号日可能没有当日行情，所以必须检查 ``latest.trade_date == data.signal_date``。

返回值必须是“资产池内证券 -> 目标权重”的映射：单个权重在 0 到 1 之间，总和不超过 1；
省略的证券目标权重为 0，空映射表示目标全部持币。不返回订单、买卖方向或交易份额。
权重可使用 Decimal、小数字符串、整数或有限 float，新手优先使用 Decimal/字符串；禁止
负权重、杠杆和资产池外证券。目标权重是完整组合目标，不是增量买卖信号。
系统固定在 D+1 合法交易日收盘执行目标，并统一处理停牌、涨跌停、T+0/T+1、整手、
成交量、SELL 优先、现金、费用和滑点。因此“目标全部持币”不保证当日一定全部成交清仓。
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from etf_backtest.strategy.rule import RuleMarketData, RuleSettings, UserRule, WeightInput

# 份额、汇金和指数数据也已经由框架截断到信号日：share_on 不向前填充缺失日期，
# share_history 只含 D 及以前的精确观察，汇金接口只返回在 D 之前已经披露的数据。
# 这些基金专项接口只提供给 Rule，不提供给 Model。


class Strategy(UserRule):
    """用户主要编辑 ``RuleSettings`` 和 ``generate_weights``。"""

    # 所有 Rule 个性参数只在这里维护，无需在 experiment.yaml 重复填写。
    # N 日收益需要 N+1 个观察值，因此 20 日动量对应 lookback=21。
    # rebalance=1 表示每日决策；目标权重可小于 1，把剩余部分保留为现金。
    # parameters 可加入周期、阈值、开关或分组配置，但不能放数据库密码等系统设置。
    settings = RuleSettings(
        lookback_trading_days=21,
        rebalance_every_trading_days=20,
        target_weight="0.90",
        parameters={
            "momentum_period": 20,
            "minimum_momentum": "0",
        },
    )

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        # parameters 是只读映射，适合保存周期、阈值和开关等个性化参数。
        period_value = self.parameters["momentum_period"]
        if type(period_value) is not int or period_value <= 0:
            raise ValueError("momentum_period must be a positive integer")
        period = period_value
        threshold = Decimal(str(self.parameters.get("minimum_momentum", "0")))
        scores: list[tuple[Decimal, str]] = []

        # data.symbols 是本次运行冻结后的全体证券，不保证每只在当前信号日都有行情。
        for symbol in data.symbols:
            latest = data.latest(symbol)
            if latest is None or latest.trade_date != data.signal_date or latest.suspended:
                continue

            # 确认 latest 属于 D 后，close_return(20) 才表示最后 21 个观察值的 20 日收益。
            # 数据不足时返回 None；它不会自动排除窗口内的历史停牌日。
            momentum = data.close_return(symbol, periods=period)
            if momentum is not None:
                scores.append((momentum, symbol))

        # 无可用标的、或最高动量不为正时，返回空映射即目标全部为现金。
        # 交易限制可能使实际持仓不能立即全部卖出。
        if not scores:
            return {}
        best_score, winner = max(scores, key=lambda item: (item[0], item[1]))
        if best_score <= threshold:
            return {}

        # 也可以返回多只 ETF，例如 {symbol_a: "0.45", symbol_b: "0.45"}。
        # 所有未返回的证券目标权重自动为 0。
        return {winner: self.target_weight}
