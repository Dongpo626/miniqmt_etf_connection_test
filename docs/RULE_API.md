# Rule 策略接口手册

本文档说明用户在 `private_strategy/<策略名>/rule.py` 中可以编辑的内容，以及
`generate_weights()` 收到的 `RuleMarketData` 全部公开接口。Rule 只负责根据截至信号日 D
可见的数据返回目标权重；数据库读取、订单生成、D+1 撮合、交易限制、费用、滑点和账户记账
由框架统一完成。

## 1. 最小 Rule 结构

```python
from collections.abc import Mapping

from etf_backtest.strategy.rule import (
    RuleMarketData,
    RuleSettings,
    UserRule,
    WeightInput,
)


class Strategy(UserRule):
    settings = RuleSettings(
        lookback_trading_days=21,
        rebalance_every_trading_days=20,
        target_weight="0.90",
        parameters={"momentum_period": 20},
    )

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        period = int(self.parameters["momentum_period"])
        scores = {
            symbol: data.close_return(symbol, periods=period)
            for symbol in data.symbols
        }
        available = {symbol: score for symbol, score in scores.items() if score is not None}
        if not available:
            return {}
        winner = max(available, key=lambda symbol: (available[symbol], symbol))
        return {winner: self.target_weight}
```

文件必须定义名为 `Strategy` 的 `UserRule` 子类。实验选择 `case: rule` 时，框架动态加载
该文件，不会固定加载任何示例策略。

## 2. RuleSettings

| 字段 | 类型 | 作用 |
|---|---|---|
| `lookback_trading_days` | 正整数 | 每只证券最多向 Rule 提供多少条截至 D 日的历史行情 |
| `rebalance_every_trading_days` | 正整数 | 每隔多少个回测交易日调用一次 `generate_weights()` |
| `target_weight` | `Decimal`、字符串、整数或有限浮点数 | 策略代码可通过 `self.target_weight` 读取的默认目标总仓位 |
| `parameters` | 只读映射 | 策略的周期、阈值和开关等自定义参数 |

权重必须在 0 到 1 之间。建议用 `"0.90"` 或 `Decimal("0.90")` 表示小数，避免二进制
浮点误差。`parameters` 应只放策略参数，不要放数据库账号、密码或输出路径。

Rule 实例可以读取：

```python
self.lookback_trading_days
self.rebalance_every_trading_days
self.target_weight
self.parameters
```

## 3. RuleMarketData 基本字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `data.signal_date` | `datetime.date` | 当前形成目标组合的信号日 D |
| `data.execution_date` | `datetime.date` | 目标最早尝试成交的 D+1 合法交易日 |
| `data.frame_index` | `int` | 当前回测交易日从 0 开始的序号 |
| `data.symbols` | `tuple[str, ...]` | 当前实验解析后的全部证券代码 |
| `data.cash` | `Decimal` | 信号日可见的账户现金 |
| `data.positions` | 只读映射 | 每只证券的只读持仓状态 |

`data.positions[symbol]` 的公开字段：

| 字段 | 含义 |
|---|---|
| `symbol` | 规范化证券代码 |
| `turnover_rule` | `T0` 或 `T1` |
| `total_quantity` | 总持仓份额 |
| `available_quantity` | 当前允许卖出的份额 |
| `today_buy_quantity` | 当日买入、尚未按 T+1 释放的份额 |

这些对象都是只读快照，Rule 不能直接修改账户。

## 4. 行情接口

所有价格都是数据库表提供的等比前复权日行情，只用于策略信号。`volume` 保留原始成交量
语义。成交、估值和记账使用框架内部的原始未复权行情，不会把前复权价格用于实际成交。

### `data.bars(symbol)`

```python
bars = data.bars("SH.510300")
```

返回 `tuple[MarketBarView, ...]`。数据按日期升序，只包含 D 日及以前的数据，并受
`lookback_trading_days` 限制；没有数据时返回空元组。

每个 `MarketBarView` 可以读取：

| 字段 | 类型 | 含义 |
|---|---|---|
| `symbol` | `str` | 证券代码 |
| `trade_date` | `date` | 交易日 |
| `open`、`high`、`low`、`close` | `Decimal` | 等比前复权 OHLC |
| `volume` | `int` | 成交量 |
| `suspended` | `bool` | 当日是否停牌 |

### `data.latest(symbol)`

返回该证券当前可见的最后一条 `MarketBarView`；没有历史时返回 `None`。新上市证券或当日
缺行情时，最后一条数据不一定属于 `signal_date`，因此通常应检查：

```python
latest = data.latest(symbol)
if latest is None or latest.trade_date != data.signal_date or latest.suspended:
    continue
```

### `data.closes(symbol)` / `data.volumes(symbol)`

分别返回与 `bars(symbol)` 顺序一致的收盘价元组和成交量元组。

### `data.has_history(symbol, observations)`

判断可见行情是否至少有 `observations` 条。参数必须是正整数。

### `data.close_return(symbol, periods)`

数据足够时返回：

```text
close[D] / close[D-periods] - 1
```

返回类型是 `Decimal | None`。例如 20 日收益需要 21 条收盘价；数据不足时返回 `None`。
此函数不会自动过滤历史窗口中的停牌日。

## 5. 账户与持仓接口

### `data.current_weight(symbol)`

返回信号日当前实际持仓权重 `Decimal`。权重按框架的原始收盘市值除以账户总资产计算。

### `data.position_quantity(symbol)`

返回该证券当前总持仓份额 `int`。

### `data.available_quantity(symbol)`

返回该证券当前允许卖出的份额 `int`，已经包含 T+0/T+1 可用性。

## 6. ETF 份额接口

### `data.share_on(symbol, asof_date)`

返回指定日期的精确 ETF 总份额 `Decimal`；该日没有记录时返回 `None`，不会向前填充。
不得查询晚于 `signal_date` 的日期。

### `data.share_history(symbol)`

返回：

```python
tuple[tuple[date, Decimal], ...]
```

结果按日期升序，只包含信号日 D 及以前的精确份额观测。

## 7. 汇金持有比例接口

汇金数据是可选 Rule 数据。普通策略可以完全不调用这些接口。

### `data.latest_huijin_ratio(symbol, company)`

按汇金公司名称查询严格满足 `EndDate < signal_date` 的最近一期持有比例：

```python
ratio = data.latest_huijin_ratio(
    "SH.510300",
    "中央汇金资产管理有限责任公司",
)
```

返回 `Decimal | None`。例如 CSV 中 `HolderOfListing=11.9`，接口返回
`Decimal("0.119")`。没有可见报告期或公司名称不匹配时返回 `None`。公司名称会去除首尾
空白，但必须与数据中的 `HuijinEntity` 名称一致。

### `data.latest_combined_huijin_ratio(symbol)`

```python
disclosure = data.latest_combined_huijin_ratio("SH.510300")
if disclosure is not None:
    report_end_date, combined_ratio = disclosure
```

返回 `tuple[date, Decimal] | None`。框架选择严格早于 D 的最新报告期，只合计该报告期实际
存在的汇金记录，不跨报告期拼接。返回报告期日期，便于与当期 ETF 份额进行比较。

## 8. 指数接口

### `data.index_bars(index_code)`

返回 `tuple[IndexBarView, ...]`，包含配置指数截至 D 日的日行情，按日期升序并受 Rule 回看
长度限制。指数只用于信号，不能成为持仓或成交标的。

指数必须先列在 `qmt_example/configs/system.yaml` 的 `rule_index_codes` 中，否则调用会报错。

`IndexBarView` 可读取：

```text
index_code, trade_date, open, high, low, close,
pre_close, pct_change, source_system
```

## 9. generate_weights 返回值

可以返回以下两类结果：

### 目标权重映射

```python
return {
    "SH.510300": "0.45",
    "SH.518880": "0.45",
}
```

- 证券必须位于 `data.symbols`；
- 单个权重必须在 0 到 1 之间；
- 权重总和不得超过 1；
- 未返回的证券目标权重视为 0；
- 空字典 `{}` 表示目标组合全部持有现金。

这是完整的目标组合，不是增量买卖信号。框架会比较目标和实际持仓后生成订单。

### `NO_REBALANCE`

```python
from etf_backtest.strategy.rule import NO_REBALANCE

return NO_REBALANCE
```

表示本次不创建新的 D+1 目标，维持现状。它与 `{}` 的“目标全部持币”含义不同。

## 10. 时序与使用边界

固定时序是：

```text
D 日收盘数据进入 Rule
→ Rule 返回目标权重
→ 目标进入 pending
→ D+1 合法交易日收盘尝试成交
→ 账户和结果文件更新
```

Rule 不应自行连接数据库、读取未来数据、生成订单或修改账户。停牌、涨跌停、整手、成交量
上限、T+0/T+1、现金、费用和滑点可能使实际成交与目标权重不同。

传入各接口的证券代码必须属于本实验的 `data.symbols`，否则会抛出 `ValueError`。需要控制
证券池时，应编辑同目录的 `experiment.yaml`，而不是在核心代码中写死证券池。

完整普通策略示例见 `private_strategy/beginner_example/rule.py`；汇金多 ETF 示例见
`private_strategy/huijin_multi_example/rule.py`。
