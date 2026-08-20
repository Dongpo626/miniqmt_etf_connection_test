# V2 MiniQMT PAPER 目标 Windows 电脑验收手册

本文用于已冻结代码在目标 Windows 电脑上的 Rule/Model PAPER 验收。验收期间不得修改
Strategy、Planner、Broker 接口、状态库数据或交易逻辑，不得人工创建订单、补单、追单或改仓。
Rule 与 Model 必须串行验收，同一账户同一时刻只能运行一个 Engine。

## 1. 验收变量与环境检查

在项目根目录打开 PowerShell，按目标机实际值设置变量：

```powershell
$project = "D:\path\to\qmt-etf-backtest-main"
$python = "$project\.venv\Scripts\python.exe"
$ruleConfig = "$project\qmt_example\configs\live\beginner_example_paper.yaml"
$modelConfig = "$project\qmt_example\configs\live\model_example_paper.yaml"
$migration = "$project\etf_backtest\live\persistence\migrations\002_live_recovery_states.sql"
$dbHost = "127.0.0.1"
$dbPort = 3306
$dbName = "qmt_etf_live_state"
$dbUser = "qmt_live_admin"
$tradeDate = Get-Date -Format "yyyy-MM-dd"
Set-Location $project
```

检查代码、Python、命令行工具和 MiniQMT 运行依赖：

```powershell
git rev-parse HEAD
git status --short
& $python --version
& $python -m pip check
& $python -c "import etf_backtest, sqlalchemy, pymysql, yaml; print('core imports OK')"
& $python -c "import xtquant; print('xtquant import OK')"
& $python -c "from xtquant import xtdata; print('xtdata import OK')"
Get-Command mysql, mysqldump
Test-NetConnection $dbHost -Port $dbPort
& $python -m etf_backtest.live_cli --help
```

通过标准：Python 为 3.12；`pip check` 和导入命令成功；MySQL 端口可达；MiniQMT 客户端已登录
正确 PAPER 账户；`git status --short` 在正式验收提交上为空。任何一项失败都不得启动服务。

验证秘密环境变量已设置但不要打印值：

```powershell
foreach ($name in @("QMT_PAPER_ACCOUNT_ID", "QMT_LIVE_MYSQL_PASSWORD", "QMT_MYSQL_PASSWORD")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Missing environment variable: $name"
    }
}
```

## 2. Model/Torch 前置检查

Model PAPER 的固定 `model_bundle.pt` 使用 PyTorch state_dict，目标机必须安装 `deep` 依赖：

```powershell
& $python -m pip install -e ".[deep,dev]"
& $python -c "import torch; print(torch.__version__); print(torch.load)"
& $python -m pytest `
  tests/unit/strategy/test_daily_torch_runtime.py `
  tests/unit/strategy/test_model_inference_bundle.py `
  --import-mode=importlib -q
```

通过标准：两个文件中的六个测试全部执行且通过，不得出现 `could not import 'torch'`。随后验证
Model 配置和 bundle，只读加载不得训练模型：

```powershell
& $python -m etf_backtest.live_cli validate --config $modelConfig
```

## 3. 状态库备份与 002 人工迁移

先停止所有 `qmt-etf-live service` 进程。确认没有服务持有账户锁后创建可恢复的完整备份：

```powershell
$backupDir = "$project\acceptance_backups"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = "$backupDir\${dbName}_before_002_${stamp}.sql"

& mysqldump `
  --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --single-transaction --routines --triggers --events --no-tablespaces `
  --databases $dbName --add-drop-database --result-file=$backup
if ($LASTEXITCODE -ne 0 -or !(Test-Path $backup) -or (Get-Item $backup).Length -eq 0) {
    throw "State database backup failed"
}
Get-FileHash -Algorithm SHA256 $backup
```

保存备份路径、文件大小和 SHA-256。备份失败时禁止迁移。

记录迁移前定义和数据量：

```powershell
$beforeSql = @"
SHOW COLUMNS FROM live_job_run LIKE 'status';
SHOW COLUMNS FROM live_order_intent LIKE 'status';
SHOW COLUMNS FROM live_broker_order LIKE 'status';
SHOW COLUMNS FROM live_account_snapshot LIKE 'snapshot_type';
SHOW COLUMNS FROM live_position_snapshot LIKE 'snapshot_type';
SELECT 'live_job_run' AS table_name, COUNT(*) AS row_count FROM live_job_run
UNION ALL SELECT 'live_order_intent', COUNT(*) FROM live_order_intent
UNION ALL SELECT 'live_broker_order', COUNT(*) FROM live_broker_order
UNION ALL SELECT 'live_account_snapshot', COUNT(*) FROM live_account_snapshot
UNION ALL SELECT 'live_position_snapshot', COUNT(*) FROM live_position_snapshot;
SELECT
  (SELECT COUNT(*) FROM live_job_run WHERE status='') AS invalid_job_status,
  (SELECT COUNT(*) FROM live_order_intent WHERE status='') AS invalid_intent_status,
  (SELECT COUNT(*) FROM live_broker_order WHERE status='') AS invalid_order_status,
  (SELECT COUNT(*) FROM live_account_snapshot WHERE snapshot_type='') AS invalid_account_type,
  (SELECT COUNT(*) FROM live_position_snapshot WHERE snapshot_type='') AS invalid_position_type;
"@
& mysql --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --database=$dbName --table --execute=$beforeSql
```

人工执行 002；服务不会自动执行或检查该迁移：

```powershell
$sourcePath = $migration.Replace("\", "/")
& mysql --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --database=$dbName --show-warnings --execute="SOURCE $sourcePath"
if ($LASTEXITCODE -ne 0) { throw "002 migration failed; follow recovery section" }
```

002 实际修改且只修改以下五列：

1. `live_job_run.status`
2. `live_order_intent.status`
3. `live_broker_order.status`
4. `live_account_snapshot.snapshot_type`
5. `live_position_snapshot.snapshot_type`

迁移后执行精确核验：

```powershell
$verifySql = @"
SHOW COLUMNS FROM live_job_run LIKE 'status';
SHOW COLUMNS FROM live_order_intent LIKE 'status';
SHOW COLUMNS FROM live_broker_order LIKE 'status';
SHOW COLUMNS FROM live_account_snapshot LIKE 'snapshot_type';
SHOW COLUMNS FROM live_position_snapshot LIKE 'snapshot_type';
SHOW INDEX FROM live_account_snapshot WHERE Key_name='uq_live_account_snapshot_key';
SHOW INDEX FROM live_position_snapshot WHERE Key_name='uq_live_position_snapshot_key';
"@
& mysql --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --database=$dbName --table --execute=$verifySql
```

再次执行迁移前的五表行数查询并与记录值逐表比较，必须完全相同；五个 `invalid_*` 计数在迁移前
必须全部为 0。任何计数不一致都按迁移失败处理并保持服务停止。

期望类型：

```text
live_job_run.status:
enum('RUNNING','SUCCEEDED','FAILED','SKIPPED')

live_order_intent.status:
enum('PLANNED','SUBMITTING','SUBMITTED','SUBMIT_UNKNOWN','COMPLETED','INCOMPLETE','ABANDONED','REJECTED')

live_broker_order.status:
enum('PENDING','PARTIALLY_FILLED','FILLED','CANCELED','REJECTED','UNKNOWN')

两个 snapshot_type:
enum('CURRENT','EOD')
```

账户快照唯一键列顺序应为 `deployment_id, trade_date, snapshot_type`；持仓快照唯一键列顺序应为
`deployment_id, trade_date, snapshot_type, symbol`。

## 4. Rule PAPER 验收

先验证配置，再执行只读启动恢复：

```powershell
& $python -m etf_backtest.live_cli validate --config $ruleConfig
& $python -m etf_backtest.live_cli reconcile --phase startup `
  --date $tradeDate --config $ruleConfig
& $python -m etf_backtest.live_cli status --config $ruleConfig
```

通过标准：账户 ID 匹配；deployment 为 `ACTIVE`；没有未知订单/成交；生成或更新当日
`CURRENT`，不生成第二批意图。启动自动流程并保持前台日志：

```powershell
$ruleLog = "$project\acceptance_rule_${tradeDate}.log"
& $python -m etf_backtest.live_cli service --config $ruleConfig 2>&1 |
  Tee-Object -FilePath $ruleLog
```

按配置时间检查三个且只有三个业务 Job：`rebalance`、`eod`、`prepare_signal`。人工检查命令：

```powershell
& $python -m etf_backtest.live_cli status --config $ruleConfig
& $python -m etf_backtest.live_cli jobs --date $tradeDate --config $ruleConfig
```

Rule 通过标准：没有第二批 intent；没有自动补单/追单；rebalance 只有完成最终订单、成交、资产和
持仓查询核对后才为 `SUCCEEDED`；EOD 后 CURRENT/EOD 均存在；16:30 后信号 Job 产生下一合法
交易日的 decision/target。

## 5. Model PAPER 验收

先用 `Ctrl+C` 正常停止 Rule 服务，确认进程退出和账户锁释放。不要同时运行 Rule 与 Model。
Model deployment ID、bundle path/hash/model_id 必须与冻结配置一致：

```powershell
& $python -m etf_backtest.live_cli validate --config $modelConfig
& $python -m etf_backtest.live_cli reconcile --phase startup `
  --date $tradeDate --config $modelConfig
& $python -m etf_backtest.live_cli status --config $modelConfig

$modelLog = "$project\acceptance_model_${tradeDate}.log"
& $python -m etf_backtest.live_cli service --config $modelConfig 2>&1 |
  Tee-Object -FilePath $modelLog
```

Model 通过标准除 Rule 标准外，还包括：bundle 只加载、不训练；hash/model_id 不匹配时启动失败且
不生成 target/intent；模型推理失败时 Job 为 `FAILED`，不得回退旧 bundle 或旧 target。

## 6. 数据库状态核验

在每个场景后运行：

```powershell
$auditSql = @"
SELECT deployment_id,status,pause_reason,paused_at FROM live_deployment;
SELECT job_type,trade_date,trigger_source,status,error_type,error_message,started_at,finished_at
FROM live_job_run WHERE trade_date='$tradeDate' ORDER BY started_at;
SELECT d.execution_date,i.decision_id,i.intent_id,i.symbol,i.side,i.status,i.reject_reason,i.updated_at
FROM live_order_intent i JOIN live_decision d ON d.decision_id=i.decision_id
WHERE d.execution_date='$tradeDate' ORDER BY i.created_at;
SELECT account_id,broker_order_id,intent_id,status,requested_quantity,filled_quantity,updated_at
FROM live_broker_order ORDER BY updated_at DESC LIMIT 50;
SELECT account_id,broker_trade_id,broker_order_id,intent_id,symbol,side,quantity,price,trade_time
FROM live_broker_trade ORDER BY trade_time DESC LIMIT 50;
SELECT deployment_id,trade_date,snapshot_type,captured_at,total_asset,available_cash
FROM live_account_snapshot ORDER BY captured_at DESC LIMIT 20;
SELECT deployment_id,trade_date,snapshot_type,COUNT(*) AS symbols
FROM live_position_snapshot GROUP BY deployment_id,trade_date,snapshot_type
ORDER BY trade_date DESC,snapshot_type;
"@
& mysql --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --database=$dbName --table --execute=$auditSql
```

同一 decision 不得出现第二批经济订单：`intent_key` 和 `remark_token` 唯一；同一 broker order/trade
不得重复。CURRENT 和 EOD 必须是不同唯一键记录，任一类型的 upsert 不得覆盖另一类型。

## 7. 故障恢复场景

### 7.1 无活动订单的正常重启

1. 启动服务，确认 startup reconcile 成功。
2. `Ctrl+C` 停止，再次启动同一配置。
3. 执行 `status`、`jobs` 和数据库状态核验。

预期：每次生命周期允许新增 startup reconcile 记录；已终态的三个业务 Job 不重复执行；没有新增
intent/broker order。失败标准：重复业务 Job、第二批订单、旧 target 被重新提交或账户锁未释放。

### 7.2 错过新单窗口

在有当日待执行 decision、尚无 intent 的前提下，于 `stop_new_orders` 之后启动服务。

预期：当日 `rebalance=SKIPPED`，`error_message=MISSED_ORDER_WINDOW`；重连、重启和人工 `execute`
均不得再创建终态 Job 或订单。失败标准：状态为 FAILED 并反复重试，或出现新 intent/order。

### 7.3 遗留 PLANNED

不要在真实状态库手工插入或修改 intent。用目标机相同代码运行确定性恢复测试：

```powershell
& $python -m pytest `
  tests/unit/live/test_jobs.py::test_abandoned_stale_intent_skips_without_planning_or_second_batch `
  --import-mode=importlib -q
```

若自然故障留下 PLANNED，保留现场后重启一次。预期：意图变为 `ABANDONED`，
`reject_reason=ABANDONED_STALE_INTENT`；对应 rebalance 为 `SKIPPED` 且同 reason；不出现第二批意图。
失败标准：PLANNED 被提交、重新规划或反复产生 FAILED Job。

### 7.4 已提交活动订单期间进程终止

仅使用策略产生的小额 PAPER 订单。在本地 intent 和 broker order 已存在、订单仍活动且 14:57 前，
记录进程 ID 后终止服务进程；不得手工下单或改仓。立即重启同一配置。

预期：启动对账按 remark/order ID 找回订单；rebalance 从查询阶段继续，到时撤单并最终核对；原
decision 的 intent 数量、broker order ID 和 submit 次数不增加。最终无活动订单后 rebalance 才能
`SUCCEEDED`。查询/撤单失败则 Job 为 FAILED 或 deployment PAUSED，不得自动补单。

### 7.5 MiniQMT 断线

1. 服务正常运行时停止 MiniQMT 客户端或临时断开其网络。
2. 观察服务，随后恢复客户端和网络。
3. 检查日志、startup reconcile、Scheduler 后续 Job 和订单数量。

预期：回调线程只报告异常；Engine 主循环停止 Scheduler、重连 Broker、执行启动对账后恢复
Scheduler。失败重连日志可出现 `MiniQMT recovery attempt failed`；恢复后不得有第二批订单。
失败标准：回调线程直接执行 connect/disconnect、Scheduler 在未对账时继续交易或重复提交。

### 7.6 部分成交后撤单

仅在 PAPER 市场自然形成部分成交时验收，不人工制造订单。等待 14:57 撤单和最终核对。

预期：成交幂等保存；订单终态但未完全成交时 intent 为 `INCOMPLETE`，deployment 为 `PAUSED`，
CURRENT 仍反映最终成功查询事实；不追单、不补齐剩余数量。失败标准：intent 错记 COMPLETED、
自动提交剩余量或无最终查询便将 rebalance 记 SUCCEEDED。

### 7.7 未知订单、UNKNOWN 状态或订单成交不一致

不得为验收向账户手工下单。先运行确定性单测：

```powershell
& $python -m pytest tests/unit/live/test_reconciliation.py --import-mode=importlib -q
```

若真实运行自然出现未知事实，立即保留现场。预期：deployment PAUSED，Job FAILED，状态库保留
原始事实；不得猜测归属、补单或继续 Scheduler。失败标准：未知事实被自动关联、覆盖或忽略。

### 7.8 EOD CURRENT/EOD 分离

15:10 后运行 `status` 和数据库状态核验。

预期：同一批成功查询分别 upsert CURRENT/EOD；两种类型均存在且 captured_at 对应该批事实；
重新运行只更新各自键，不增加无类型记录、不互相覆盖。失败标准：只存在一种类型、类型串写或
`status` 未分别选择最新 captured_at。

### 7.9 跨日恢复

保留前一交易日 target/终态意图，下一交易日启动服务。

预期：不提交旧 target；只有当天 16:30 的新 signal 才能产生后续执行日目标。失败标准：旧
decision 被重新规划、旧意图重新提交或生成第二批订单。

## 8. 验收失败：停止服务并保留现场

优先在前台按 `Ctrl+C` 正常停止。无响应时先记录进程，再强制停止：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "etf_backtest.live_cli service" } |
  Select-Object ProcessId,CreationDate,CommandLine |
  Format-List | Out-File "$project\acceptance_failed_process.txt"

# 仅对上一步确认的具体服务 PID 执行：
Stop-Process -Id <CONFIRMED_SERVICE_PID> -Force
```

不要再次执行 signal/execute/cancel，不要修改状态表。立即保存：

```powershell
& $python -m etf_backtest.live_cli status --config $ruleConfig 2>&1 |
  Tee-Object "$project\acceptance_failed_status.json"
& $python -m etf_backtest.live_cli jobs --date $tradeDate --config $ruleConfig 2>&1 |
  Tee-Object "$project\acceptance_failed_jobs.json"

$failedDump = "$backupDir\${dbName}_failed_${stamp}.sql"
& mysqldump --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --single-transaction --routines --triggers --events --no-tablespaces `
  --databases $dbName --result-file=$failedDump
Get-FileHash -Algorithm SHA256 $failedDump
```

同时保留服务日志、MiniQMT 日志、代码 commit、配置文件 SHA-256、系统时间、账户 ID（可脱敏）和
失败发生时间。未完成原因分析前不得清理 RUNNING/FAILED Job 或重启自动流程。

## 9. 002 迁移失败恢复

MySQL DDL 会隐式提交，五条 ALTER 不是一个可整体回滚的事务。失败后：

1. 保持服务停止，保存 mysql 完整错误输出。
2. 重新执行五条 `SHOW COLUMNS`，确认已经成功到哪一列。
3. 若原因是临时锁、权限或空间问题，修复后可重新执行完整 002；相同 ENUM 定义重复执行不会删除
   数据或已有枚举值，但仍可能获取 metadata lock 或重建表。
4. 若需回到迁移前状态，先另存失败现场 dump；经 DBA 明确批准后，用执行前带
   `--add-drop-database` 的备份恢复。该操作会删除并重建明确指定的 `$dbName`：

```powershell
$backupSource = $backup.Replace("\", "/")
& mysql --host=$dbHost --port=$dbPort --user=$dbUser --password `
  --execute="SOURCE $backupSource"
if ($LASTEXITCODE -ne 0) { throw "Database restore failed; keep service stopped" }
```

恢复后再次执行迁移前 `SHOW COLUMNS`、行数和 SHA-256 记录。数据库未恢复一致前不得启动服务。

## 10. 最终签署条件

- 环境、Torch、配置和完整测试通过；
- 002 备份、迁移、五列 ENUM 与两个唯一键核验通过；
- Rule 和 Model 均完成启动恢复、正常日流程和状态查询；
- 已提交订单重启、断线、错过窗口、部分成交、EOD 和跨日行为符合预期；
- 无人工订单、补单、追单、第二批订单、旧 target 恢复或自动数据库检查；
- 所有失败场景均能安全停止并保留可复现现场。
