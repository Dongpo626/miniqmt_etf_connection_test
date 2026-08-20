-- Manual migration for an existing V2 PAPER state database.
-- Stop the live service before applying this file. The service never runs it automatically.

ALTER TABLE live_job_run
    MODIFY COLUMN status ENUM('RUNNING', 'SUCCEEDED', 'FAILED', 'SKIPPED') NOT NULL;

ALTER TABLE live_order_intent
    MODIFY COLUMN status ENUM(
        'PLANNED', 'SUBMITTING', 'SUBMITTED', 'SUBMIT_UNKNOWN',
        'COMPLETED', 'INCOMPLETE', 'ABANDONED', 'REJECTED'
    ) NOT NULL;

ALTER TABLE live_broker_order
    MODIFY COLUMN status ENUM(
        'PENDING', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'UNKNOWN'
    ) NOT NULL;

ALTER TABLE live_account_snapshot
    MODIFY COLUMN snapshot_type ENUM('CURRENT', 'EOD') NOT NULL;

ALTER TABLE live_position_snapshot
    MODIFY COLUMN snapshot_type ENUM('CURRENT', 'EOD') NOT NULL;
