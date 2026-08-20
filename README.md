# QMT 日频 ETF 回测与 MiniQMT PAPER 项目

本项目从 MySQL 只读加载 QMT 日频数据，在本地完成策略计算、模型训练、回测和结果输出；
也可以把同一套 Rule 决策逻辑或经过批准的固定 Model bundle 接入 MiniQMT PAPER 模拟盘，
执行每日信号、下单、撤单、回调消费、账户快照和对账。

回测与 PAPER 信号使用只读策略数据库 `qmt_etf_quant`。PAPER 运行另外使用独立的
`qmt_etf_live_state` 状态数据库保存 deployment、Job、decision、target、订单、成交和快照。
项目不会向 `qmt_etf_quant` 写入数据，但会读写已由用户授权的 Live 状态数据库。

项目提供两种自定义入口：

- **Rule**：编辑 Python 规则，根据行情、持仓、ETF 份额、汇金持有比例或指数数据返回目标权重；
- **Model**：编辑特征、PyTorch 网络、训练参数和 Top-K 组合规则。

Rule 回测和 Rule PAPER 复用同一个用户接口。Model 回测负责训练并生成带完整 metadata 的
`model_bundle.pt`；Model PAPER 只加载固定 bundle 做推理，不会每日训练或自动切换模型。

汇金策略只是随项目提供的一个示例，不是固定运行入口。用户可以创建任意多个独立实验，分别
选择自己的证券池、回测区间以及 Rule 或 Model。

## 1. 环境要求与安装

需要 Python 3.12。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[deep]"
```

`deep` 会安装 Model 所需的 PyTorch。只运行 Rule 时可以使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

运行测试和代码检查时安装开发依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deep,dev]"
```

MiniQMT 的 `xtquant` 由目标 Windows 电脑上的 MiniQMT/QMT 环境提供，不在 PyPI 依赖中。
PAPER 运行前必须确认 MiniQMT 客户端已经登录正确的模拟账户，且 Python 环境能够导入
`xtquant`。

## 2. 配置回测与信号数据库

编辑 [`qmt_example/configs/system.yaml`](qmt_example/configs/system.yaml)：

```yaml
database:
  host: 数据库服务器地址
  port: 3306
  database: qmt_etf_quant
  user: 你的只读用户名
  password_env: QMT_MYSQL_PASSWORD
```

在当前 PowerShell 进程设置密码：

```powershell
$env:QMT_MYSQL_PASSWORD = "你的密码"
```

如果内部环境明确允许把密码写进本地配置，也可以删除 `password_env`，改成：

```yaml
  password: "你的密码"
```

该账号只需要对项目配置的 `qmt_etf_quant` 数据表拥有 `SELECT` 权限。MySQL 提供原始行情、
前复权行情、交易状态、ETF 份额、指数行情和 ETF 主数据；项目不会在这个策略数据库中建表、
更新或删除数据。
公开的汇金数据已经放在 `resources/huijin/huijin_combined.csv`，无需额外下载。

PAPER 状态数据库在 live YAML 的 `state_database` 中单独配置，例如：

```yaml
state_database:
  host: 127.0.0.1
  port: 3306
  database: qmt_etf_live_state
  user: qmt_live_user
  password_env: QMT_LIVE_MYSQL_PASSWORD
```

状态数据库和用户需要由数据库管理员提前创建。该用户需要在 `qmt_etf_live_state` 中建表及
读写数据的权限，但不需要获得策略数据库的写权限。密码只通过环境变量提供：

```powershell
$env:QMT_LIVE_MYSQL_PASSWORD = "Live 状态数据库密码"
```

## 3. 创建自己的实验

### 使用命令创建

```powershell
.\.venv\Scripts\python.exe -m qmt_example new experiment my_strategy
```

生成：

```text
private_strategy/my_strategy/
├── experiment.yaml
├── rule.py
└── model.py
```

脚手架不会覆盖已有实验。

### 不使用命令创建

在 PyCharm、VS Code 或文件管理器中复制：

```text
private_strategy/beginner_example/
```

将副本重命名为自己的策略名称，然后编辑其中三个文件即可。不要直接运行 `rule.py` 或
`model.py`，它们由回测入口根据 `experiment.yaml` 动态加载。

## 4. 设置回测范围和策略类型

编辑自己实验目录中的 `experiment.yaml`：

```yaml
name: my_strategy
start_date: 2024-01-01
end_date: 2024-12-31
initial_cash: "1000000"

# 每次实验只运行一种类型：rule 或 model。
case: rule

universe:
  symbols: [SH.510300, SH.518880, SH.588000]
  pools: []
```

- `case` 只能是 `rule` 或 `model`；
- 证券代码支持 `SH.510300`、`510300.SH` 或 `510300`；
- 资产池支持 `domestic_stock_etf`、`gold_etf`、`all_supported_etf`；
- `symbols` 与 `pools` 可以同时填写，系统取并集；
- Decimal 参数建议写成带引号的文本，避免 YAML 浮点误差。

## 5. 编辑 Rule

当 `case: rule` 时，编辑同目录的 `rule.py`。核心入口是：

```python
class Strategy(UserRule):
    settings = RuleSettings(
        lookback_trading_days=21,
        rebalance_every_trading_days=20,
        target_weight="0.90",
        parameters={"momentum_period": 20},
    )

    def generate_weights(self, data: RuleMarketData):
        return {"SH.510300": self.target_weight}
```

Rule 返回完整的目标权重，不直接创建订单或修改账户。系统会在 D+1 合法交易日收盘尝试执行，
并统一处理停牌、涨跌停、整手、成交量上限、T+0/T+1、现金、费用和滑点。

- `NO_REBALANCE`：本次不创建新目标，保持现状；
- `{}`：目标组合全部持有现金；
- 未返回的证券：目标权重为 0。

全部字段、函数、返回值和示例见 [Rule 策略接口手册](docs/RULE_API.md)。普通入门示例见
[`private_strategy/beginner_example/rule.py`](private_strategy/beginner_example/rule.py)，汇金多 ETF
示例见 [`private_strategy/huijin_multi_example/rule.py`](private_strategy/huijin_multi_example/rule.py)。

## 6. 编辑 Model

当 `case: model` 时，编辑同目录的 `model.py`。文件需要提供：

- `MODEL_SETTINGS`：训练集、验证集、训练参数、组合参数及构造参数；
- `class Features(FeatureBuilder)`：特征名称、历史长度和特征计算；
- `class Model(TorchModelFactory)`：模型身份、重建参数和 PyTorch 网络。

回测测试区间使用 `experiment.yaml` 的起止日期。训练区间和验证区间在 `MODEL_SETTINGS` 中
设置，必须早于回测开始日且互不重叠。框架只用训练集拟合标准化器，一次运行只训练一次，
不会在回测过程中重新训练。

完整接口说明见 [Model 策略接口手册](docs/MODEL_API.md)，可编辑示例见
[`private_strategy/beginner_example/model.py`](private_strategy/beginner_example/model.py)。

## 7. 运行回测

### 命令行运行

```powershell
.\.venv\Scripts\python.exe -m qmt_example run private_strategy/my_strategy/experiment.yaml
```

也可以使用安装后的命令：

```powershell
qmt-etf-backtest run private_strategy/my_strategy/experiment.yaml
```

### 在 IDE 中点击运行

使用 PyCharm 或 VS Code 打开项目根目录，编辑 [`run_backtest.py`](run_backtest.py) 中的路径：

```python
EXPERIMENT_PATH = PROJECT_ROOT / "private_strategy" / "my_strategy" / "experiment.yaml"
```

然后直接运行 `run_backtest.py`。如果使用 `password_env`，需要在 IDE 的运行配置中设置
`QMT_MYSQL_PASSWORD`。IDE 工作目录应为项目根目录。实际运行 Rule 还是 Model，仍由所选
实验的 `case` 决定。

### 可选预检

`validate` 不是运行回测的必要步骤。第一次配置数据库、修改证券池或只想检查连接时可以执行：

```powershell
.\.venv\Scripts\python.exe -m qmt_example validate private_strategy/my_strategy/experiment.yaml
```

它只检查配置、策略文件、MySQL 连接、必需表和显式证券，不执行策略，也不产生正式回测结果。

## 8. 查看结果

每次运行都会在 `qmt_example/results/user_experiments/<run_id>/` 创建一个全新的结果目录，不会
覆盖旧结果。普通成功运行包含：

```text
run.json
daily_nav.csv
daily_positions.csv
orders.csv
trades.csv
metrics.json
final_account.json
cumulative_return.png
drawdown.png
cash.png
```

- `run.json`：运行状态、配置摘要、数据版本和追溯信息；密码会被隐藏；
- `daily_nav.csv`：每日现金、市值和总资产；
- `daily_positions.csv`：每日全部证券的持仓、价格、市值和权重；
- `orders.csv`：全部订单，包括未成交和被拒绝的订单；
- `trades.csv`：实际正式成交；
- `metrics.json`：收益、回撤、夏普、费用、滑点和换手等指标；
- `final_account.json`：最终现金与持仓；
- 三张 PNG：累计收益、回撤和现金曲线。

Model 运行还会保存 `model_bundle.pt` 和 `predictions.csv`。失败运行只保留标记为失败的
`run.json`，不会留下看似完整的净值、指标或图表。

## 9. MiniQMT PAPER 模拟盘

### 9.1 当前能力与验收状态

当前代码已实现 Rule 与 Model PAPER 的生产接入链路，包括：

- MiniQMT 资产、持仓、订单和成交查询；
- xtdata 行情订阅和内部对象映射；
- 固定 Universe、deployment 不可变校验和账户/Job 锁；
- 启动恢复、成交状态同步、订单规划、有限轮询/撤单、最终核对和日终双快照；
- Scheduler 仅包含 `prepare_signal`、`rebalance`、`eod` 三个配置驱动业务 Job；
- Broker 断线由 Engine 主循环停止 Scheduler、重连、对账并恢复 Scheduler；
- Rule source 或固定 Model bundle 通过同一个 `DailyDecisionService` 生成每日决策；
- 手工命令与自动 Scheduler 共用同一组 Jobs。

当前仓库已经完成离线实现和离线测试，但尚未在本开发电脑完成真实 MySQL、MiniQMT、行情、
回调及 PAPER 订单的端到端验收。首次转移到目标电脑时必须先完成 Rule 链路，再验证 Model
链路；不能仅凭离线测试宣称真实 PAPER 已通过。

### 9.2 配置文件

Rule 示例：

```text
qmt_example/configs/live/beginner_example_paper.yaml
```

Model 示例：

```text
qmt_example/configs/live/model_example_paper.yaml
```

示例不包含真实账户、密码和可直接使用的 MiniQMT 路径。运行前至少要配置：

- `deployment.experiment_path` 与 `deployment.case`；
- `deployment.system_path` 和冻结调度锚点；
- `miniqmt.userdata_path` 与唯一正整数 `session_id`；
- PAPER 账户和两个数据库的环境变量；
- Model 情况下由当前代码重新训练生成的 V2 `model_bundle.pt` 路径；
- 风险上限和固定的信号、提交、停止新单、撤单、日终时间。时间必须满足
  `submit_start < stop_new_orders < cancel_open_orders < eod.run_time < signal.run_time`。

Model bundle 与 deployment 是不可变绑定。更换 bundle 必须更换 `deployment_id` 和 live YAML，
不能在运行中的同一个 deployment 内热切换。

### 9.3 PAPER 命令

安装后推荐使用 `qmt-etf-live`。也可以把下列命令中的 `qmt-etf-live` 替换为
`.\.venv\Scripts\python.exe run_paper_trading.py`。

离线检查配置、策略 source 和 Model bundle（不连接账户、不创建 deployment）：

```powershell
qmt-etf-live validate --config qmt_example/configs/live/beginner_example_paper.yaml
```

首次在已存在的 `qmt_etf_live_state` 数据库中创建九张状态表：

```powershell
qmt-etf-live db init --config qmt_example/configs/live/beginner_example_paper.yaml
```

已有 V2 状态库升级时，停止服务后由运维人员手工执行
`etf_backtest/live/persistence/migrations/002_live_recovery_states.sql`；服务不会自动执行该迁移。

常用手工命令：

```powershell
$liveConfig = "qmt_example/configs/live/beginner_example_paper.yaml"
qmt-etf-live reconcile --phase startup --date 2026-08-19 --config $liveConfig
qmt-etf-live signal --date 2026-08-19 --config $liveConfig
qmt-etf-live execute --date 2026-08-20 --config $liveConfig
qmt-etf-live cancel --date 2026-08-20 --config $liveConfig
qmt-etf-live reconcile --phase eod --date 2026-08-20 --config $liveConfig
qmt-etf-live snapshot --date 2026-08-20 --config $liveConfig
qmt-etf-live jobs --date 2026-08-20 --config $liveConfig
qmt-etf-live status --config $liveConfig
```

启动自动日流程：

```powershell
qmt-etf-live service --config $liveConfig
```

`status` 只读取状态库，并分别显示按 `captured_at` 选出的最新 `CURRENT` 与 `EOD` 快照；
它不会实时连接 MiniQMT。`CURRENT` 只在启动恢复、rebalance 最终核对、EOD 三个节点更新。

`service` 不是首次真实验收入口。推荐顺序是：数据库 migration 和锁检查 → Rule 只读启动 →
Rule 手工完整闭环 → Rule 自动完整日流程 → Model bundle 只读加载 → Model 手工完整闭环 →
Model 自动完整日流程。信号或模型推理失败时 Job 会失败，不会自动使用旧 target、备用 bundle、
重新训练或持续追单。

## 10. 数据口径提示

- Rule 和 Model 计算信号、特征时读取等比前复权日行情；
- 成交、估值和账户记账由框架内部使用原始未复权行情；
- D 日收盘形成目标，最早在 D+1 合法交易日收盘尝试成交；
- 当前数据属于统一冻结的回填快照，不应解释为历史每一天当时可获得的严格 PIT 数据。

## 11. 更多文档

- [Rule 策略接口手册](docs/RULE_API.md)
- [Model 策略接口手册](docs/MODEL_API.md)
- [项目整体框架与开发说明](README_DEVELOPER.md)

## 12. 许可证

本项目采用 [MIT License](LICENSE)，版权归 `Str_lab`。在保留许可证和版权声明的前提下，
可以使用、修改、发布和再分发本项目。
