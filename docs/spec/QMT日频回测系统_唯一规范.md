# QMT 日频 ETF 回测系统唯一规范

版本：1.0  
生效日期：2026-08-04  
规范状态：唯一、现行、强制

## 1. 规范地位

本文件是本项目实现、测试、文档和验收的唯一规范来源。不得恢复已删除的多频、多价格模式
或旧示例结构。

发生冲突时，依次服从：用户在当前任务中的明确要求、本规范、代码中已经通过本规范验收的
稳定接口。实施过程不得使用 Git，也不得修改 `D:\量化项目` 中的 SQL、CSV、manifest 或说明文件。

## 2. 产品边界

本项目只实现下列能力：

- MySQL 8 中 QMT ETF 日线数据的只读加载；
- 固定 `bar_interval=1d`；
- 固定原始收盘价 `trade_price_mode=CLOSE`；
- Rule 和 Model 两条策略路径；
- 股票 ETF 为 T+1，`518` 黄金 ETF 为 T+0；
- 真实交易约束、费用、滑点、成交量上限、账户和绩效统计；
- 本地文件结果输出，不向 MySQL 写回任何回测结果；
- 唯一运行入口由 `python -m qmt_example new|validate|run` 三个子命令组成；实际回测统一使用
  `run <experiment.yaml>`，并根据实验中的单个 `case` 运行 Rule 或 Model。

以下能力明确不属于现行版本：

- `1m`、`1h` 或任何盘中 K 线；
- OPEN、HIGH、LOW、VWAP、CUSTOM 成交模式；
- Qlib、MLflow、自动/分布式调参执行器、参数搜索声明或事件级订单簿；
- 通过当前 `dim_etf.current_status` 反推历史证券池；
- 对缺失的交易所日历、历史基金属性或价格限制规则进行猜测；
- MySQL 结果表、MySQL Writer 或跨输入库外键；
- 旧 `example/`、旧 `user_cases/` 和 `example_model` 运行路径。

配置模型必须在解析时拒绝超出边界的值，不能接受后再静默降级。

## 3. 数据源与不可变身份

### 3.1 生产输入

生产数据库固定为统一快照 `qmt_etf_quant`。核心运行只读以下表：

| 表 | 用途 |
| --- | --- |
| `dim_etf` | 证券代码、交易所、名称和上市生命周期 |
| `dim_trading_calendar` | 自然日与开闭市状态 |
| `etf_quote_qmt_unadjusted_daily` | 原始 OHLC、前收、成交量和成交额 |
| `etf_quote_qmt_front_ratio_daily` | 前复权 OHLC 策略视图 |
| `etf_trade_status_daily` | QMT 停复牌状态及 Tushare 涨跌停价格 |
| `etf_share_daily` | Rule 可选读取的 ETF 每日份额 |
| `index_quote_daily` | Rule 可选读取的指数行情 |

`dim_index`、`etf_nav_daily` 等其余表属于同一个数据库快照，但不是回测核心依赖。Rule 可选
读取由 SHA256 冻结的汇金十大持有人 CSV；该 CSV 不在数据库中，不得改变 MarketFrame、
成交或账户语义。

Rule 还可选读取同一快照内的 `index_quote_daily`。当前只加载配置声明的
`000001.SH` `PRICE` 序列，保留源端指数代码，不得套用 ETF 代码归一化，也不得把指数加入
ETF universe、账户或订单。指数行情同样属于 `RETROSPECTIVE_SNAPSHOT`，只按业务日期隔离。

汇金 CSV 只读取 `HuijinEntity`、`Symbol`、`EndDate`、`HolderOfListing`，不得读取
`HoldersShare`。仅接受“中央汇金投资有限责任公司”和“中央汇金资产管理有限责任公司”；
同一公司、证券和报告期的 `HolderOfListing` 先求和，再除以 100 转成 `Decimal` 比例。

输入连接只允许 `SELECT`。数据库密码只从 `QMT_MYSQL_PASSWORD` 环境变量读取，不得写入
YAML、日志、异常消息或结果文件。生产代码不得执行 `INSERT`、`UPDATE`、`DELETE`、DDL、
锁表或创建临时持久对象。

### 3.2 数据集版本

每次运行必须冻结：

- `dataset_version`：数据包 manifest 或经审核导入快照的不可变标识；
- `snapshot_started_at_utc`：一致性数据库快照开始时刻；
- `calendar_source`：默认 `AKSHARE_SINA`；
- 20% 规则资源 manifest 的 SHA256；
- 配置文件的规范化 SHA256。

上述身份必须进入本地运行元数据。不得把“当前数据库内容”当作版本。

### 3.3 快照与 PIT 声明

当前统一数据库包是 2026-08-12 导出的回填研究快照。表内按业务主键保存唯一记录，
没有逐行修订号和可用时间，因此可以用于显式的 snapshot-as-of 研究，但不能宣称为
2020—2025 年逐日可得的严格 PIT 数据。

本版本必须在输出中标记 `data_semantics=RETROSPECTIVE_SNAPSHOT`。若调用方要求严格按历史决策日
可得数据运行，系统应明确拒绝或返回无合格数据错误，不得把回填数据伪装成 PIT 成功。

## 4. Repository 合同

生产 Repository 为：

```text
QmtDailyRepository(
    engine,
    *,
    dataset_version,
    trade_status_table="etf_trade_status_daily",
    calendar_source="AKSHARE_SINA",
)
```

### 4.1 证券代码

外部规范代码为 `SH.510300`、`SZ.159915`。QMT 的 `510300.SH`、`159915.SZ` 和
`dim_etf.etf_code` 的六位代码必须通过一个经过测试的无歧义映射转换。交易所后缀与代码
首位矛盾时必须失败，不能自动改写成另一交易所。

### 4.2 唯一业务键

raw、front、交易状态和份额表直接按数据库主键读取，不进行修订选择或截止时间过滤。
raw/front/状态以 `(etf_code, trade_date)` 配对，份额以 `(etf_code, asof_date)` 识别；
查询结果出现重复业务键、未知证券或违反来源约束时必须失败。

### 4.3 日线构造

- 交易执行、涨跌停判断、账户估值和成交金额使用 raw 原始价格；
- Rule/Model 的研究视图可以使用 front-ratio 前复权价格；
- 停复牌状态来自 `etf_trade_status_daily.qmt_suspend_flag`：`1` 为停牌，`0` 为正常，`-1`
  为当日起复牌；`NULL` 表示 QMT 状态未知，若 raw/front 完整则仍按正常行情处理；成交量和
  成交额来自 raw；
- raw、front 和状态按 `(etf_code, trade_date)` 联接；
- `volume_share` 只有在有限、非负且为整数份额时才可无损转换为整数；
- 任一执行必需价格、`pre_close`、成交量或成交额为 NULL、负数或不满足 OHLC 关系时，
  该记录必须按配置明确拒绝或剔除，并在数据质量报告中计数；不得补零或前向填充；
- 唯一的不可交易估值例外是：同一证券某一开市日 raw/front 同时缺失、状态表存在该业务键且
  `qmt_suspend_flag` 为 `1` 或 `NULL`，并且前后相邻开市日均有完整 raw/front。系统可分别用
  前一日 raw/front 各自的收盘价构造零成交量、零成交额且 `suspended=True` 的单日估值条；
  front 继续只用于策略信号，raw 继续只用于执行和账户估值；该估值条不得用于成交，
  并必须在 provenance 的 `status_only_suspension_carry_keys` 逐条披露；
- 停牌日可以保留价格用于估值，但不得成交；
- 日线时间窗固定为交易所本地 `09:30:00` 至 `15:00:00`，时区为 `Asia/Shanghai`。

`source_bar_key` 必须由 QMT 自然键稳定生成或改为显式自然键类型；Python 进程随机
`hash()`、查询行号和数据库返回顺序都不是合法身份。

### 4.4 日历

日历模式固定为 `SSE_FOR_ALL`：只查询 `dim_trading_calendar.exchange = 'SSE'` 的记录，
并用同一组开闭市日统一驱动支持范围内的沪深 ETF。不得用工作日算法，不得查询、复制或
要求 `SZSE` 日历记录；真实数据包只有 SSE 日历是本合同的正常输入，不限制深市 ETF。

日历版本使用已冻结数据集版本与日历行摘要共同识别。闭市日不得生成 Frame；开市日某只
ETF 缺少行情时不得跳到下一天冒充同一 Frame。仅允许 4.3 所述、可审计的孤立停牌估值条。

### 4.5 上市生命周期与证券池

显式配置的证券必须在目标交易日满足 `list_date <= trade_date`，并且
`delist_date IS NULL OR trade_date <= delist_date`。`dim_etf.current_status` 是当前状态，
不能作为历史成员资格。动态历史证券池只有在提供有效期化属性历史后才可启用；否则应使用
显式证券列表并披露幸存者偏差限制。

## 5. 价格限制与交易制度

基础规则为：

- 普通股票 ETF：10%，T+1；
- `518` 黄金 ETF：10%，T+0；
- 20% ETF：以 `resources/limit_rules/etf_price_limit_20pct.csv` 的有效期记录为准；
- 所有 ETF 默认整手 100 份、最小价格变动 0.001 元。

不得使用 `588` 前缀、基金名称、跟踪指数名称或上市日期单独推断 20% 规则。资源中没有
覆盖的证券一律按 10% 处理，同时在审计元数据记录命中方式。资源 manifest 的 `rule_mode`
固定为 `LATEST_SNAPSHOT_WITH_2020_SEED`：当前成员来自上交所、深交所 2026-08-04 官方名单，
深市初始历史再由 2020-08-24 官方种子名单约束。

`CURRENT_SNAPSHOT_RULE_INFERENCE` 行由官方当前名单、规则条件和上市日期联合推定起点；
`EFFECTIVE_DATED_OFFICIAL_SEED_CONFIRMED_CURRENT` 行同时得到深市种子和当前名单支持；
`SNAPSHOT_DERIVED_REMOVAL_CUTOFF` 只表示下一快照已不再命中，不能冒充交易所发布的真实
退出日。该资源不是完整历史成员档案；需要法律级历史准确性时必须补充交易所有效期名单后
再运行。

每次运行必须校验 CSV SHA256 与 manifest 一致，并拒绝重叠有效期、未知来源、非 20% 比例、
错误交易所或无效日期。

## 6. 策略、时序与成交

### 6.1 统一时序

每个交易日只生成一个日 Frame。固定顺序为：

```text
读取 D 日原始/复权视图
→ 在 D 日收盘时形成目标组合
→ 目标进入 pending
→ 在 D+1 合法交易日收盘价尝试成交
→ 正式 Fill 更新账户
→ 记录 D+1 日终快照
```

禁止用产生信号的 D 日收盘价在同一 Frame 成交。最后一个 Frame 产生但没有下一交易日的
pending 目标不得伪造成交。

### 6.2 Rule

Rule 路径直接注入满足策略协议的具体规则策略。运行不得创建 Workflow 或训练模型。
规则策略只能读取 `MarketBarView`、账户可见状态、已释放持仓和下列日期隔离后的可选数据，
不得接触 raw 执行对象或未来 Frame：

- `share_on(symbol, date)` 返回该证券该日精确的 `total_share` `Decimal`；缺记录返回空，禁止
  向前填充。`share_history(symbol)` 只包含信号日 D 及以前的逐日原始观测。
- `latest_huijin_ratio(symbol, company)` 按公司返回严格满足 `EndDate < D` 的最近一期比例；
  比例是聚合后的 `HolderOfListing / 100` `Decimal`。没有更早报告期或公司名不匹配时返回空。
- `latest_combined_huijin_ratio(symbol)` 选择严格满足 `EndDate < D` 的最新报告期，只合计该
  `EndDate` 实际出现的两家汇金记录，并同时返回 `(EndDate, Decimal比例)`；禁止跨报告期拼接。
- `index_bars(index_code)` 返回配置指数截至 D 的 `PRICE` 日行情，按日期升序并受 Rule 回看
  长度限制；指数只用于信号，不参与持仓和成交。
- `current_weight(symbol)` 返回信号日 D 收盘记账后的实际持仓权重 `Decimal`，统一按该证券
  原始收盘市值除以账户总资产计算；接口对任意多资产 Rule 通用，不包含策略专属字段。
- Rule 可返回 `NO_REBALANCE`，表示该信号日不创建 D+1 pending target、不生成订单并保持
  当前实际份额；它与空权重映射“目标全部持币”含义不同。

这些接口只属于 `RuleMarketData`；Model、FeatureBuilder、MarketBarView 和 MarketFrame 均不得
新增或读取这些字段。

### 6.3 Model

Model 路径必须有一个可实际运行的非 `example_model` 模型和 FeatureBuilder，或经过明确
允许列表验证的外部插件。训练、验证、测试时间段有序且不重叠；训练截止不得晚于首次用于
回测的预测时点。一个运行中不得隐式重新拟合。模型 bundle 必须记录模型类、参数、特征
定义、训练切分、随机种子和数据集版本。

私有 Model 扩展固定经过 `FeatureBuilder → DatasetSplits → DailyTorchWorkflow →
PredictorBundle → DailyModelStrategy` 链路。标准化器只能用训练集拟合；Workflow 在一次运行中
只能训练或加载一次。用户插件是受路径和类名校验的可信本地 Python 代码，不是安全沙箱。
`DailyModelStrategy` 支持从超过阈值的 Top-K 预测生成多资产动态目标权重；组合参数和可选
自定义权重函数只写在 `model.py`，所有输出都必须经过证券、有限权重和总仓位上限校验。

### 6.4 交易链

下列链条必须保留，不能由策略直接修改账户：

```text
TargetPortfolio → Order → TradePriceQuote → Estimate
→ EtfRuleEngine approval → FillResult → Account
```

`EtfRuleEngine` 是批准数量的唯一决策者。必须继续执行：停牌、方向性涨跌停、T+0/T+1
可用数量、整手、成交量阈值、现金约束、手续费、滑点、SELL 先于 BUY，以及多证券买入时
确定性的资金分配。只有正式 `FillResult` 可以改变账户和进入成交输出。

## 7. 配置和唯一入口

新用户命令为：

```powershell
python -m qmt_example new experiment my_strategy
python -m qmt_example validate private_strategy/my_strategy/experiment.yaml
python -m qmt_example run private_strategy/my_strategy/experiment.yaml
```

`new experiment` 必须同时生成不覆盖的 `experiment.yaml`、`rule.py` 和 `model.py`。
`validate` 只是本地预检，不能冒充成功回测；`run` 必须真正创建只读 MySQL 连接、
Repository、策略、Engine 和本地 Writer。一个实验只允许 `case: rule` 或 `case: model`，
需要比较时分别运行两个实验。

CLI 不保留固定策略或内置回归入口；示例与用户策略使用同一实验入口。任何一条 `run` 路径
都不允许只打印 `ready` 或仅验证 YAML。

私有实验 YAML 只保存名称、日期、初始现金、证券池和单个 `case`；Rule 个性参数必须由
`rule.py/RuleSettings` 唯一拥有，Model 个性参数必须由 `model.py/MODEL_SETTINGS` 唯一拥有。
解析后的运行配置至少冻结日期范围、证券池、初始现金、策略类型与参数、数据库连接
非秘密部分、数据集版本、snapshot cutoff、规则资源、费用、滑点、成交量阈值和本地输出目录。
私有实验 YAML 与系统配置必须分离，且不得包含任何数据库字段。正式系统配置应通过
`password_env` 解析密码；解析后的配置、日志和结果必须脱敏。

## 8. 本地结果

结果只写入本次运行的本地目录。目录包含：

- `run.json`，合并非秘密配置、数据版本、证券池和最小复现信息；
- `daily_nav.csv`、`daily_positions.csv`、`orders.csv` 和 `trades.csv`；
- `metrics.json` 和 `final_account.json`；
- 回测全部交易日完成后一次性生成的累计收益率、回撤和现金三张完整区间图；
- Model 运行额外保存 `model_bundle.pt` 和 `predictions.csv`。

写入必须使用临时目录加原子提交；失败运行不得留下伪装成成功的完整目录。运行 ID 唯一，
不得覆盖既有运行。金额与价格保持 Decimal 语义，序列化格式和小数位规则必须稳定。

## 9. 测试数据

核心算法使用小型内存数据覆盖 raw/front、SSE 日历、D+1、T+0/T+1、涨跌停、停牌、
整手、成交量、现金、费用和滑点。项目不维护本地 MySQL fixture；MySQL 方言、账号权限和
真实表可读性在发布前使用真实 SELECT-only 服务器账号验证。

## 10. 验收门槛

### 10.1 自动化

- 配置拒绝所有非 `1d`、非 `CLOSE` 值；
- symbol、Decimal、日期、自然键和空值映射测试；
- calendar source/version、缺失日与双交易所测试；
- raw/front/status 按唯一业务键配对以及停牌/复牌状态测试；
- ETF 份额精确逐日读取、无填充和信号日截断测试；
- 两家汇金公司 `HolderOfListing` 聚合、百分数转 `Decimal` 及严格 `EndDate < D` 测试；
- 上证综指 `PRICE` 日行情、信号日截断和 Rule 可见性测试；
- 同报告期汇金合计与 `NO_REBALANCE` 不产生 pending target/订单测试；
- 规则 CSV schema、SHA256、有效期无重叠和代表代码测试；
- CLOSE 的 D 信号到 D+1 成交测试；
- T+0/T+1、10%/20%、停牌、整手、成交量、现金、费用、滑点测试；
- Rule 和 Model 的小型内存端到端测试；
- 固定输出 Writer 原子提交、失败清理和不覆盖测试；
- Ruff、格式检查、mypy 和精简后的完整 pytest 全部通过；覆盖率只作参考。

### 10.2 真实 MySQL

最终验收必须使用 manifest 校验后的真实 `qmt_etf_quant`，以 SELECT-only 用户执行至少
`SH.510300`、`SH.518880`、`SH.588000` 的 Rule 和 Model 日线/CLOSE 回测。验收前后核对
输入表行数与摘要不变。真实数据测试未收集、跳过、缺凭据或数据库未导入时，状态只能是
BLOCKED/INCOMPLETE，不能记 PASS。

`validate` 使用个人账号执行轻量 SELECT 预检；发布前另外运行一个 Rule 和一个 Model
真实回测。真实入口按 `SSE_FOR_ALL` 只要求 SSE 日历，并必须校验配置证券具有 raw/front
数据。

结果必须明确标记 RETROSPECTIVE_SNAPSHOT，验证存在日终序列、现金不为负、正式成交完整、SELL
先于 BUY，并证明 Model 没有在回测期内偷偷重训。

## 11. 迁移和历史保留

删除旧例子前必须先让新 CLI 的 Rule/Model 和内存端到端测试通过；发布前再执行真实只读
smoke。删除后运行同一套测试和零引用扫描。`main_backtest.7z` 必须原样保留。项目没有 Git，任何计划、
脚本或报告都不得把 Git 状态、分支或提交当作恢复与验收机制。
