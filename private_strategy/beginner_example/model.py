# ruff: noqa: RUF002, RUF003
"""初学者 Model 示例：三个日频特征和一个小型 PyTorch MLP。

用户可以做什么：
- 在 ``Features`` 中定义特征名称、最长历史窗口和特征计算；
- 在 ``Model`` 中定义可重建的 PyTorch 网络及网络结构参数；
- 在本文件的 ``MODEL_SETTINGS`` 中设置训练/验证日期、Top-K 组合、训练参数，以及
  ``feature_kwargs``/``model_kwargs``；测试和回测日期统一使用 experiment.yaml 的公共日期。

系统固定完成的部分：
- 特征只能看到信号日 D 及以前的表内前复权日行情，不能看到未来数据；
- 标签固定为 D+1 收盘至 D+2 收盘的前复权收益；
- 样本按信号日归入闭区间 train/valid/test，仅用训练集拟合 StandardScaler；
- 标签兑现日可能跨越切分边界；若要求标签也不跨界，应在区间之间留至少两个交易日；
- 使用 CPU、Adam、MSE、确定性顺序小批次和验证集早停，不在 train+valid 上重新拟合；
- 每个 split 至少要产生一个有效样本；每次运行重新训练一次，当前入口不直接复用旧 bundle；
- 保存验证/测试 MSE、MAE、相关系数、bundle、预测以及特征/切分/快照元数据；
- 回测时每日选择预测分数严格超过阈值的 Top-K；不足 K 只持有合格资产，无合格资产则持币；
- 组合支持等权、分数比例和 Softmax 动态权重，总目标仓位之外的部分保留为现金；
- D 日预测产生的目标最早在 D+1 合法交易日收盘成交。

用户不需要在本文件编写标签、Dataset、训练循环、标准化、预测文件、订单或成交逻辑。
模型运行器输入是每个“日期-证券”的一维特征向量，模型必须为每个输入样本输出一个分数。
高级用户还可用 ``CustomPortfolio`` 包装本文件定义的权重函数；当前 Model 仍固定每日决策，
组合函数只读取当日预测，不读取账户状态。需要自定义调仓频率或账户感知逻辑时使用 Rule。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

from etf_backtest.core.market import MarketBarView
from etf_backtest.strategy.model import (
    DateRange,
    FeatureBuilder,
    ModelSettings,
    TopKPortfolio,
    TorchModelFactory,
    TorchTrainingConfig,
)

# 所有 Model 个性参数只在这里维护，无需在 experiment.yaml 重复填写。
# train/valid 按信号日 D 划分且必须先后、不重叠；测试期直接采用 YAML 的 start/end。
# portfolio 可修改 Top-K、总仓位、最低分数、权重方式和 Softmax 温度，仍无需修改 YAML。
# weighting 可选 equal（等权）、score_proportional（超过阈值的分数比例）或 softmax。
# 高级用户可定义一个接收 PredictionRecord 元组并返回“证券 -> 权重”映射的函数，再用
# CustomPortfolio(name=..., total_weight=..., allocator=...) 替换 TopKPortfolio。
# training 可修改随机种子、最大轮数、早停、batch、学习率、权重衰减和最小改进量。
# feature_kwargs/model_kwargs 会作为关键字参数传给 Features/Model 构造函数；键名必须匹配。
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
        max_epochs=15,
        patience=4,
        batch_size=256,
        learning_rate=0.001,
    ),
    feature_kwargs={},
    model_kwargs={"hidden_dim": 16},
)


class Features(FeatureBuilder):
    """定义单只证券的纯特征函数；可增加构造函数接收 feature_kwargs。"""

    @property
    def feature_names(self) -> tuple[str, ...]:
        # 名称必须非空、唯一，且数量必须和 build_features 的返回值完全一致。
        return ("return_5d", "return_20d", "volume_ratio_5_20")

    @property
    def required_history_trading_days(self) -> int:
        # N 日收益需要 N+1 个收盘观察值；系统只把最后这么多条历史交给特征函数。
        return 21

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> Sequence[Decimal] | None:
        # symbol 和 signal_date 可用于证券特定或日期特定的特征；本示例不需要。
        del symbol, signal_date
        if len(history) < 21:
            return None

        # 系统仅在存在信号日 D 行情时调用；history 是这一只证券自己的历史，可能不足 21 条，
        # 且最多为 required_history_trading_days 条。它严格按日期升序，最后一条就是 D。
        # 这里看不到其他证券、账户、持仓或现金，不能直接构造横截面特征。
        # 常用字段为 symbol/trade_date/open/high/low/close/volume/suspended；没有原始执行价、
        # pre_close 或成交额。可以用 symbol/signal_date 构造证券或日期特定的特征。
        recent = history[-21:]

        # 特征不可可靠计算时返回 None 跳过这个“日期-证券”样本；不要补 0 或前向填充。
        # 特征函数应无内部累计状态，不依赖调用顺序，也不要自行读取数据库、文件或网络。
        if any(bar.suspended or bar.close <= 0 for bar in recent):
            return None
        return_5d = recent[-1].close / recent[-6].close - Decimal("1")
        return_20d = recent[-1].close / recent[0].close - Decimal("1")
        volume_5d = Decimal(sum(bar.volume for bar in recent[-5:])) / Decimal("5")
        volume_20d = Decimal(sum(bar.volume for bar in recent[-20:])) / Decimal("20")
        if volume_20d <= 0:
            return None
        volume_ratio = volume_5d / volume_20d

        # 必须返回有限 Decimal；数量和顺序与非空、唯一的 feature_names 一一对应。
        return (return_5d, return_20d, volume_ratio)


class Model(TorchModelFactory):
    """构造接收二维 [batch, input_dim] 输入并为每个样本输出一个分数的网络。"""

    def __init__(self, hidden_dim: int = 16) -> None:
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim

    @property
    def model_id(self) -> str:
        # 改变网络含义或结构后应更新 ID，例如 beginner_mlp_v2。
        return "beginner_mlp"

    @property
    def model_class_name(self) -> str:
        return type(self).__name__

    @property
    def model_parameters(self) -> Mapping[str, object]:
        # 必须完整记录重建相同网络所需的有限、JSON 兼容参数，供 bundle 做兼容性校验。
        return {"hidden_dim": self.hidden_dim}

    def create(self, *, input_dim: int, seed: int) -> object:
        # PyTorch 在真正训练/加载时才导入；不要在模块顶层强制导入。
        from torch import nn

        # 模型运行器在调用前已经统一设置 Python/NumPy/PyTorch 随机种子。
        del seed

        # batch 可能混合不同日期和证券，不能把 batch 维当作时间维或资产维。系统使用一个
        # 共享模型训练并预测全部 ETF，不会自动为每只 ETF 单独训练。
        # 可以替换为其他 torch.nn.Module，但输出须可展平为 [batch] 或 [batch, 1]，即每个
        # 样本恰好一个原始实数分数；标签可负，末层通常不用 Softmax/Sigmoid/ReLU 限制符号。
        # create 会在训练、校验和预测时重建网络，必须始终产生相同结构和 state_dict 键/形状，
        # 且网络必须包含可供 Adam 优化的参数。
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
