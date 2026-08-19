# Model 策略接口手册

本文档说明用户在 `private_strategy/<策略名>/model.py` 中可以编辑的 Model 扩展接口。用户负责
定义特征、PyTorch 网络和模型个性参数；框架统一负责样本标签、数据切分、训练集标准化、
训练、验证集早停、逐日预测、组合权重、订单、撮合和账户记账。

运行 Model 需要安装带 PyTorch 的依赖：

```powershell
python -m pip install -e ".[deep]"
```

## 1. model.py 必须提供什么

文件必须定义以下三个模块级对象：

```python
MODEL_SETTINGS = ModelSettings(...)

class Features(FeatureBuilder):
    ...

class Model(TorchModelFactory):
    ...
```

实验的 `experiment.yaml` 必须使用：

```yaml
case: model
```

框架会动态加载该实验目录中的 `model.py`，不会固定加载示例模型。

## 2. MODEL_SETTINGS

```python
MODEL_SETTINGS = ModelSettings(
    train_range=DateRange(date(2021, 1, 1), date(2022, 12, 31)),
    valid_range=DateRange(date(2023, 1, 1), date(2023, 12, 31)),
    portfolio=TopKPortfolio(
        top_k=3,
        total_weight="0.90",
        min_score=0.0,
        weighting="score_proportional",
    ),
    training=TorchTrainingConfig(
        seed=42,
        max_epochs=100,
        patience=10,
        batch_size=64,
        learning_rate=0.001,
    ),
    feature_kwargs={},
    model_kwargs={"hidden_dim": 16},
)
```

| 字段 | 作用 |
|---|---|
| `train_range` | 训练样本的信号日闭区间 |
| `valid_range` | 验证和早停样本的信号日闭区间 |
| `portfolio` | 每日预测分数如何转换为目标权重 |
| `training` | PyTorch 训练参数 |
| `feature_kwargs` | 传给 `Features(...)` 构造函数的关键字参数 |
| `model_kwargs` | 传给 `Model(...)` 构造函数的关键字参数 |

训练区间必须早于验证区间且不能重叠。测试/回测区间不在 `MODEL_SETTINGS` 中重复填写，直接
使用同目录 `experiment.yaml` 的 `start_date` 和 `end_date`，并且必须晚于验证区间。

`feature_kwargs` 和 `model_kwargs` 的键必须是合法 Python 参数名，值必须有限且可以记录为
JSON。它们用于构造对象，例如：

```python
feature_kwargs={"short_window": 5, "long_window": 20}
model_kwargs={"hidden_dim": 32, "dropout": 0.1}
```

对应类应接受同名构造参数。

## 3. Features 接口

### `feature_names`

返回非空、无重复的特征名称元组，数量和 `build_features()` 每次返回的数值数量完全一致：

```python
@property
def feature_names(self) -> tuple[str, ...]:
    return ("return_5d", "return_20d", "volume_ratio_5_20")
```

修改特征含义或顺序时应同步修改这里。特征名称和顺序会写入模型 bundle，用于阻止不兼容的
模型和特征被混用。

### `required_history_trading_days`

返回正整数，表示计算一个样本最多需要多少条单证券历史行情：

```python
@property
def required_history_trading_days(self) -> int:
    return 21
```

计算 N 日收盘收益通常需要 N+1 条收盘价。

### `build_features()`

固定签名：

```python
def build_features(
    self,
    *,
    symbol: str,
    signal_date: date,
    history: Sequence[MarketBarView],
) -> Sequence[Decimal] | None:
    ...
```

参数含义：

| 参数 | 含义 |
|---|---|
| `symbol` | 当前样本的证券代码 |
| `signal_date` | 当前样本信号日 D |
| `history` | 该证券截至 D 日、按日期升序的历史行情 |

`history` 只包含当前这一只证券，最多包含 `required_history_trading_days` 条。OHLC 是等比前
复权价格，`volume` 是成交量。每个 `MarketBarView` 可以读取：

```text
symbol, trade_date, open, high, low, close, volume, suspended
```

特征有效时返回有限的 `Decimal` 序列；无法可靠计算时返回 `None`，框架会跳过当前
“日期 × 证券”样本。不要用 0 替代缺失特征，也不要在函数内部自行读取数据库、文件、网络
或未来数据。

示例：

```python
class Features(FeatureBuilder):
    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("return_5d", "return_20d")

    @property
    def required_history_trading_days(self) -> int:
        return 21

    def build_features(self, *, symbol, signal_date, history):
        del symbol, signal_date
        if len(history) < 21:
            return None
        if any(bar.suspended or bar.close <= 0 for bar in history[-21:]):
            return None
        return (
            history[-1].close / history[-6].close - Decimal("1"),
            history[-1].close / history[-21].close - Decimal("1"),
        )
```

Model 特征看不到现金、持仓、ETF 份额、汇金持有比例、指数接口或其他证券的同期数据。需要
账户感知、自定义调仓频率、汇金数据或指数信号时，应使用 Rule。

## 4. 固定标签和数据切分

框架固定使用 D+1 收盘到 D+2 收盘的前复权收益作为监督学习标签。样本按信号日 D 归入：

- `train_range`：训练集；
- `valid_range`：验证和早停；
- `experiment.yaml` 的日期区间：测试集和正式回测期。

StandardScaler 只在训练集拟合，然后应用于验证集和测试集。训练、验证和测试每个区间都必须
产生至少一个有效样本。一次回测只训练一次，不会在回测过程中逐日重新训练。

## 5. Model 网络工厂接口

`Model` 负责描述一个能够稳定重建的 `torch.nn.Module`。

### `model_id`

返回非空字符串，标识模型结构和含义。网络结构或语义发生不兼容变化时应更新，例如从
`"my_mlp_v1"` 改为 `"my_mlp_v2"`。

### `model_class_name`

通常返回：

```python
return type(self).__name__
```

### `model_parameters`

返回重建相同网络所需的完整参数映射。键和值必须能记录为 JSON：

```python
return {
    "hidden_dim": self.hidden_dim,
    "dropout": self.dropout,
}
```

它应与构造函数和 `MODEL_SETTINGS.model_kwargs` 一致。

### `create(input_dim, seed)`

返回一个新的 `torch.nn.Module`：

```python
def create(self, *, input_dim: int, seed: int) -> object:
    from torch import nn

    del seed
    return nn.Sequential(
        nn.Linear(input_dim, self.hidden_dim),
        nn.ReLU(),
        nn.Linear(self.hidden_dim, 1),
    )
```

- `input_dim` 等于 `feature_names` 的数量；
- 框架调用前已经设置 Python、NumPy 和 PyTorch 随机种子；
- 每次调用必须创建相同结构和相同 `state_dict` 键形状；
- 输入形状是 `[batch, input_dim]`；
- 每个输入样本必须输出一个未限制的实数分数，形状可以是 `[batch]` 或 `[batch, 1]`；
- 最后一层通常不要使用 Softmax、Sigmoid 或 ReLU 限制回归分数；
- 网络必须包含可供 Adam 优化的参数。

## 6. TorchTrainingConfig

| 字段 | 含义 |
|---|---|
| `seed` | 非负随机种子 |
| `max_epochs` | 最大训练轮数 |
| `patience` | 验证集连续多少轮无足够改善后早停 |
| `batch_size` | 小批量大小 |
| `learning_rate` | Adam 学习率 |
| `weight_decay` | Adam 权重衰减，默认 0 |
| `min_delta` | 计为改善所需的最小验证损失下降，默认 0 |

训练设备固定为 CPU，优化器固定为 Adam，损失固定为 MSE。验证集选择最佳状态，训练完成后
在测试期逐日生成分数。

## 7. TopKPortfolio

```python
portfolio=TopKPortfolio(
    top_k=3,
    total_weight="0.90",
    min_score=0.0,
    weighting="score_proportional",
    softmax_temperature=1.0,
)
```

| 字段 | 含义 |
|---|---|
| `top_k` | 每日最多选择的证券数量 |
| `total_weight` | 合格资产合计目标仓位，范围 `(0, 1]` |
| `min_score` | 只有严格大于该值的预测才合格 |
| `weighting` | `equal`、`score_proportional` 或 `softmax` |
| `softmax_temperature` | 仅 Softmax 权重使用，必须大于 0 |

合格证券少于 K 时只配置实际合格资产；没有合格证券时目标全部持币。

三种权重方式：

- `equal`：合格证券等权；
- `score_proportional`：按预测分数超过 `min_score` 的部分分配；
- `softmax`：对预测分数做温度控制的 Softmax 分配。

高级用户可以用 `CustomPortfolio` 包装在 `model.py` 中定义的本地权重函数。自定义函数接收
当天的 `PredictionRecord` 序列并返回“证券 → 权重”映射；框架仍会检查证券范围、有限数值、
非负权重和总仓位上限。

## 8. 运行和输出

在 `experiment.yaml` 中设置 `case: model`，然后通过根目录 `run_backtest.py`、Python API 或
命令行运行该实验。除通用回测结果外，Model 运行还会保存模型 bundle、逐日预测和训练/验证/
测试指标。结果目录由 `qmt_example/configs/system.yaml` 的 `runs_dir` 控制。

完整可编辑示例见 `private_strategy/beginner_example/model.py`。

## 9. 常见问题

### 为什么一个区间没有样本？

检查证券上市日期、日期范围、`required_history_trading_days`，以及 `build_features()` 是否因
停牌、缺失或非法数值持续返回 `None`。

### 为什么网络输出形状不对？

最后一层应为每个 batch 样本输出一个标量，例如 `nn.Linear(hidden_dim, 1)`，不能输出一个
证券横截面矩阵。

### 能否在 Features 中读取汇金或账户持仓？

不能。Model 的特征边界刻意限制为单证券截至 D 日的前复权行情。需要这些信息时使用 Rule。

### 能否直接加载上一次的模型继续回测？

当前公开运行入口每次重新训练一次并保存新的 bundle，不直接复用旧 bundle。
