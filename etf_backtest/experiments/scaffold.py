# ruff: noqa: RUF001
"""Non-overwriting templates for private user experiments."""

from __future__ import annotations

import json
from pathlib import Path

from etf_backtest.experiments.config import _safe_component

DEFAULT_PRIVATE_STRATEGY_ROOT = Path(__file__).resolve().parents[2] / "private_strategy"

_EXPERIMENT_FILENAME = "experiment.yaml"
_RULE_FILENAME = "rule.py"
_MODEL_FILENAME = "model.py"

_RULE_TEMPLATE = '''# ruff: noqa: RUF002, RUF003
"""新手 Rule 模板：根据截至 D 日收盘的日频数据返回目标权重。

可用数据：data.signal_date/execution_date/frame_index/symbols/cash/positions，以及
latest、bars、closes、volumes、has_history、close_return、position_quantity、
available_quantity、current_weight、share_on、share_history、latest_huijin_ratio、
latest_combined_huijin_ratio 和 index_bars。MarketBarView 可读
trade_date/open/high/low/close/volume/suspended。
positions 中可读 T+0/T+1、总份额、可用份额和当日买入份额；不提供成本价、盈亏、订单、
交易历史、原始执行价或未来数据。bars 受 lookback 限制，新上市证券可能历史不足。

可以实现打分、过滤、轮动、择时、多资产配置或空仓；RuleSettings 和 parameters 都直接
写在本文件，self.target_weight/self.parameters 读取这些代码内设置。返回资产池内
“证券 -> 目标权重”的映射，总权重不得
超过 1；省略证券的目标为 0，空映射表示目标全部持币。不要生成订单或修改账户，系统会在
D+1 收盘尝试执行并处理交易规则、现金、费用和滑点，交易限制可能造成部分或无法成交。
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from etf_backtest.strategy.rule import RuleMarketData, RuleSettings, UserRule, WeightInput

# ``current_weight(symbol)`` 返回信号日按原始收盘价计算的当前账户权重；
# ``share_on(symbol, date)`` 不向前填充缺失份额，``share_history(symbol)`` 只含 D 及以前；
# 汇金接口只返回 D 之前已经披露的数据，``index_bars(index_code)`` 只返回配置的指数历史。
# 份额、汇金和指数接口只提供给 Rule。需要保持现有目标时返回 NO_REBALANCE；空映射表示
# 目标全部持有现金。不要自行连接数据库、生成订单或修改账户。


class Strategy(UserRule):
    """用户主要编辑 ``RuleSettings`` 和 ``generate_weights``。"""

    # 可修改这里的回看长度、调仓间隔、目标仓位及任意策略参数；无需同步 YAML。
    # 最长 N 日收益至少需要 N+1 条收盘观察值。
    settings = RuleSettings(
        lookback_trading_days=6,
        rebalance_every_trading_days=20,
        target_weight="0.90",
        parameters={
            "momentum_period": 5,
            "minimum_momentum": "0",
        },
    )

    def generate_weights(self, data: RuleMarketData) -> Mapping[str, WeightInput]:
        period = int(self.parameters["momentum_period"])
        threshold = Decimal(str(self.parameters["minimum_momentum"]))
        ranked: list[tuple[Decimal, str]] = []
        for symbol in data.symbols:
            latest = data.latest(symbol)
            if (
                latest is None
                or latest.trade_date != data.signal_date
                or latest.suspended
            ):
                continue
            # close_return(5) = close[D] / close[D-5] - 1；数据不足时为 None。
            momentum = data.close_return(symbol, periods=period)
            if momentum is not None:
                ranked.append((momentum, symbol))
        # 返回空映射即目标全部为现金。
        if not ranked:
            return {}
        best_score, winner = max(ranked, key=lambda item: (item[0], item[1]))
        if best_score <= threshold:
            return {}
        # 也可返回多只 ETF；每项权重在 0 到 1 之间且总和不超过 1。
        return {winner: self.target_weight}
'''

_MODEL_TEMPLATE = '''# ruff: noqa: RUF002, RUF003
"""新手 Model 模板：定义特征和一个每样本输出单分数的 PyTorch 网络。

用户只负责 Features、Model 和本文件的组合参数。系统固定构造 D+1 至 D+2 收益标签，
仅用训练集标准化，使用 CPU/Adam/MSE/验证集早停，每次运行只训练一次并保存 bundle、
预测和组合权重；回测时每日选择超过阈值的 Top-K，D 日目标在 D+1 收盘执行。组合支持
等权、分数比例和 Softmax 动态权重；无合格预测时持币。不要自行编写标签、Dataset、
训练循环、标准化、订单或成交逻辑。需要自定义调仓频率或读取账户状态时使用 Rule。

MODEL_SETTINGS 的 feature_kwargs/model_kwargs 会作为关键字参数传给对应类。Features 每次只看到一只证券
截至 D 日、最多 required_history_trading_days 条的前复权历史，看不到其他证券或账户；
不可计算时返回 None。模型接收可能混合日期和证券的 [batch, input_dim]，使用一个共享模型
训练全部 ETF，必须为每个输入样本恰好输出一个原始实数分数。
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


# 所有 Model 个性参数都在本文件。测试区间直接使用 YAML 的公共 start_date/end_date。
# weighting 可选 equal、score_proportional 或 softmax；不足 K 时只配置合格资产。
# 高级用户也可定义一个接收 PredictionRecord 元组并返回权重映射的函数，再用
# CustomPortfolio(name=..., total_weight=..., allocator=...) 替换 TopKPortfolio。
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


class Features(FeatureBuilder):
    """可增加构造函数，并由 MODEL_SETTINGS.feature_kwargs 传入参数。"""

    @property
    def feature_names(self) -> tuple[str, ...]:
        # 名称必须非空、唯一，并与 build_features 返回值的数量和顺序一致。
        return ("return_1d",)

    @property
    def required_history_trading_days(self) -> int:
        # N 日收益需要 N+1 个收盘观察值。
        return 2

    def build_features(
        self,
        *,
        symbol: str,
        signal_date: date,
        history: Sequence[MarketBarView],
    ) -> Sequence[Decimal] | None:
        # history 仅含一只证券，按日期升序且最后一条为 D；可读取 OHLC/volume/suspended。
        # 特征函数应无内部状态，不依赖调用顺序，也不要自行读取数据库、文件、网络或未来数据。
        del symbol, signal_date
        if len(history) < 2:
            return None
        # 实际策略通常还应检查 suspended、非正价格等；不可用时返回 None，不要补 0。
        # 所有返回值必须是有限 Decimal。
        return (history[-1].close / history[-2].close - Decimal("1"),)


class Model(TorchModelFactory):
    """构造一个可由保存参数和 state_dict 完整重建的网络。"""

    def __init__(self, hidden_dim: int = 16) -> None:
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim

    @property
    def model_id(self) -> str:
        # 改变模型结构或含义后应更新 ID。
        return "private_mlp"

    @property
    def model_class_name(self) -> str:
        return type(self).__name__

    @property
    def model_parameters(self) -> Mapping[str, object]:
        # 完整记录重建网络所需的 JSON 兼容参数。
        return {"hidden_dim": self.hidden_dim}

    def create(self, *, input_dim: int, seed: int) -> object:
        # 延迟导入 PyTorch；模型运行器已经统一设置随机种子。
        from torch import nn

        del seed
        # batch 可能混合日期和证券，不能当作时间维或资产维。可替换为其他 nn.Module，
        # 但必须每次重建相同结构，并为每个样本输出一个分数。
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
'''


def _experiment_template(name: str) -> str:
    quoted_name = json.dumps(name, ensure_ascii=False)
    return f"""# 本文件只填写所有策略共用的回测输入；策略和模型个性参数写在 Python 文件。
# 数据库、费率、滑点、成交量上限、输出目录、SSE 日历、日频、D+1 收盘成交和 ETF
# 规则由系统配置统一管理。
name: {quoted_name}
start_date: 2024-01-01
end_date: 2024-12-31
# 初始资金使用带引号的小数字符串。
initial_cash: "1000000"

# 每次只运行一种策略，填写 rule 或 model；需要比较时分别运行两个实验。
case: rule

# universe 至少填写 symbols 或 pools 之一，也可并用；两者重合按并集合并。
# symbols/pools 各自列表不能重复。可用池：domestic_stock_etf、gold_etf、
# all_supported_etf（前两者并集）。池分类来自冻结的当前 ETF 主表，不是历史逐日成分档案；
# 系统再按上市/退市日期决定回测期有效性，结果属于 RETROSPECTIVE_SNAPSHOT。
universe:
  symbols: [SH.510300, SH.518880, SH.588000]
  pools: []
  # 只用资产池时，将 symbols 改为 []，再把 pools 改成：
  # pools: [domestic_stock_etf, gold_etf]
  # 或 pools: [all_supported_etf]
  # 也可同时保留 symbols 和 pools。
"""


def _experiment_directory(
    name: str,
    private_strategy_root: Path,
) -> Path:
    safe_name = _safe_component(name, "experiment name")
    root = Path(private_strategy_root).resolve()
    target = (root / safe_name).resolve()
    if not target.is_relative_to(root):  # defensive even though name is one component
        raise ValueError("experiment path must stay inside private_strategy")
    return target


def _write_exclusive(target: Path, content: str) -> None:
    """Write one new UTF-8 template without an overwrite race."""

    try:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite existing template: {target}") from None


def scaffold_experiment(
    name: str,
    *,
    private_strategy_root: Path = DEFAULT_PRIVATE_STRATEGY_ROOT,
) -> tuple[Path, Path, Path]:
    """Create one complete experiment without overwriting existing files."""

    directory = _experiment_directory(name, private_strategy_root)
    targets = (
        (directory / _EXPERIMENT_FILENAME, _experiment_template(name)),
        (directory / _RULE_FILENAME, _RULE_TEMPLATE),
        (directory / _MODEL_FILENAME, _MODEL_TEMPLATE),
    )
    existing = tuple(path for path, _ in targets if path.exists())
    if existing:
        raise FileExistsError(f"refusing to overwrite existing template: {existing[0]}")
    directory.mkdir(parents=True, exist_ok=True)
    for target, content in targets:
        _write_exclusive(target, content)
    return targets[0][0], targets[1][0], targets[2][0]


__all__ = [
    "DEFAULT_PRIVATE_STRATEGY_ROOT",
    "scaffold_experiment",
]
