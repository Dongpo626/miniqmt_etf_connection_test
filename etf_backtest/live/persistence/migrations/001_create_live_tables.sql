CREATE TABLE live_deployment (
    deployment_id VARCHAR(64) NOT NULL,
    bound_account_id VARCHAR(64) NOT NULL,
    mode VARCHAR(16) NOT NULL,
    experiment_path TEXT NOT NULL,
    experiment_sha256 CHAR(64) NOT NULL,
    strategy_source_sha256 CHAR(64) NOT NULL,
    model_bundle_path TEXT NULL,
    model_bundle_sha256 CHAR(64) NULL,
    model_id VARCHAR(128) NULL,
    schedule_anchor_date DATE NOT NULL,
    universe_json TEXT NOT NULL,
    universe_hash CHAR(64) NOT NULL,
    config_hash CHAR(64) NOT NULL,
    status ENUM('ACTIVE', 'PAUSED', 'RETIRED') NOT NULL,
    pause_reason TEXT NULL,
    paused_at DATETIME(6) NULL,
    created_at DATETIME(6) NOT NULL,
    activated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (deployment_id),
    INDEX ix_live_deployment_bound_account_id (bound_account_id),
    INDEX ix_live_deployment_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_job_run (
    job_run_id VARCHAR(64) NOT NULL,
    deployment_id VARCHAR(64) NOT NULL,
    job_type VARCHAR(64) NOT NULL,
    trade_date DATE NOT NULL,
    trigger_source ENUM('SCHEDULED', 'MANUAL', 'RECOVERY') NOT NULL,
    status ENUM('RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED') NOT NULL,
    started_at DATETIME(6) NOT NULL,
    finished_at DATETIME(6) NULL,
    error_type VARCHAR(128) NULL,
    error_message TEXT NULL,
    PRIMARY KEY (job_run_id),
    INDEX ix_live_job_run_deployment_job_date (deployment_id, job_type, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_decision (
    decision_id VARCHAR(64) NOT NULL,
    deployment_id VARCHAR(64) NOT NULL,
    signal_date DATE NOT NULL,
    execution_date DATE NOT NULL,
    schedule_index INT NOT NULL,
    status ENUM('NOT_SCHEDULED', 'NO_REBALANCE', 'TARGET_CREATED') NOT NULL,
    data_as_of DATE NOT NULL,
    strategy_source_sha256 CHAR(64) NOT NULL,
    model_id VARCHAR(128) NULL,
    config_hash CHAR(64) NOT NULL,
    created_at DATETIME(6) NOT NULL,
    PRIMARY KEY (decision_id),
    CONSTRAINT uq_live_decision_deployment_signal UNIQUE (deployment_id, signal_date),
    CONSTRAINT fk_live_decision_deployment FOREIGN KEY (deployment_id)
        REFERENCES live_deployment (deployment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_target_position (
    decision_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    target_weight DECIMAL(18, 10) NOT NULL,
    CONSTRAINT uq_live_target_decision_symbol UNIQUE (decision_id, symbol),
    CONSTRAINT fk_live_target_decision FOREIGN KEY (decision_id)
        REFERENCES live_decision (decision_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_order_intent (
    intent_id VARCHAR(64) NOT NULL,
    decision_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    side ENUM('BUY', 'SELL') NOT NULL,
    requested_quantity BIGINT NOT NULL,
    valuation_price DECIMAL(24, 8) NOT NULL,
    limit_price DECIMAL(24, 8) NOT NULL,
    intent_key CHAR(64) NOT NULL,
    remark_token VARCHAR(24) NOT NULL,
    status ENUM(
        'PLANNED', 'SUBMITTING', 'SUBMITTED', 'SUBMIT_UNKNOWN',
        'COMPLETED', 'INCOMPLETE', 'ABANDONED', 'REJECTED'
    ) NOT NULL,
    reject_reason TEXT NULL,
    created_at DATETIME(6) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    PRIMARY KEY (intent_id),
    CONSTRAINT uq_live_order_intent_key UNIQUE (intent_key),
    CONSTRAINT uq_live_order_remark_token UNIQUE (remark_token),
    CONSTRAINT fk_live_intent_decision FOREIGN KEY (decision_id)
        REFERENCES live_decision (decision_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_broker_order (
    account_id VARCHAR(64) NOT NULL,
    broker_order_id VARCHAR(128) NOT NULL,
    order_sysid VARCHAR(128) NULL,
    intent_id VARCHAR(64) NOT NULL,
    requested_quantity BIGINT NOT NULL,
    filled_quantity BIGINT NOT NULL,
    average_fill_price DECIMAL(24, 8) NULL,
    status ENUM(
        'PENDING', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'UNKNOWN'
    ) NOT NULL,
    remark_token VARCHAR(24) NOT NULL,
    updated_at DATETIME(6) NOT NULL,
    CONSTRAINT uq_live_broker_order_account_id UNIQUE (account_id, broker_order_id),
    CONSTRAINT fk_live_broker_order_intent FOREIGN KEY (intent_id)
        REFERENCES live_order_intent (intent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_broker_trade (
    account_id VARCHAR(64) NOT NULL,
    broker_trade_id VARCHAR(128) NOT NULL,
    broker_order_id VARCHAR(128) NOT NULL,
    intent_id VARCHAR(64) NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    side ENUM('BUY', 'SELL') NOT NULL,
    quantity BIGINT NOT NULL,
    price DECIMAL(24, 8) NOT NULL,
    trade_time DATETIME(6) NOT NULL,
    CONSTRAINT uq_live_broker_trade_account_id UNIQUE (account_id, broker_trade_id),
    CONSTRAINT fk_live_broker_trade_intent FOREIGN KEY (intent_id)
        REFERENCES live_order_intent (intent_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_account_snapshot (
    deployment_id VARCHAR(64) NOT NULL,
    trade_date DATE NOT NULL,
    snapshot_type ENUM('CURRENT', 'EOD') NOT NULL,
    captured_at DATETIME(6) NOT NULL,
    cash DECIMAL(24, 8) NOT NULL,
    available_cash DECIMAL(24, 8) NOT NULL,
    market_value DECIMAL(24, 8) NOT NULL,
    total_asset DECIMAL(24, 8) NOT NULL,
    frozen_cash DECIMAL(24, 8) NOT NULL,
    CONSTRAINT uq_live_account_snapshot_key UNIQUE (deployment_id, trade_date, snapshot_type),
    CONSTRAINT fk_live_account_snapshot_deployment FOREIGN KEY (deployment_id)
        REFERENCES live_deployment (deployment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE live_position_snapshot (
    deployment_id VARCHAR(64) NOT NULL,
    trade_date DATE NOT NULL,
    snapshot_type ENUM('CURRENT', 'EOD') NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    total_quantity BIGINT NOT NULL,
    available_quantity BIGINT NOT NULL,
    frozen_quantity BIGINT NOT NULL,
    market_value DECIMAL(24, 8) NOT NULL,
    last_price DECIMAL(24, 8) NOT NULL,
    CONSTRAINT uq_live_position_snapshot_key
        UNIQUE (deployment_id, trade_date, snapshot_type, symbol),
    CONSTRAINT fk_live_position_snapshot_deployment FOREIGN KEY (deployment_id)
        REFERENCES live_deployment (deployment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
