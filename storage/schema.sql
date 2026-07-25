-- Drop tables if they exist (for clean restarts)
DROP TABLE IF EXISTS routing_decisions CASCADE;
DROP TABLE IF EXISTS fraud_decisions CASCADE;
DROP TABLE IF EXISTS transactions CASCADE;

-- 1. RAW TRANSACTIONS TABLE
-- Stores every incoming transaction before any processing
CREATE TABLE transactions (
    transaction_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    device_id VARCHAR(100),
    session_id VARCHAR(100),
    merchant_category VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    ip_address INET,
    raw_payload JSONB,  -- Full original data
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast time-based queries
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_user_id ON transactions(user_id);

-- 2. FRAUD DECISIONS TABLE
-- Stores ML model predictions and risk scores
CREATE TABLE fraud_decisions (
    decision_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    fraud_score DECIMAL(5, 4) NOT NULL,  -- 0.0000 to 1.0000
    is_fraud BOOLEAN NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    features_used JSONB,  -- Which features contributed to decision
    confidence_level DECIMAL(5, 4),
    rules_triggered TEXT[],  -- Array of rule descriptions
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast transaction lookups
CREATE INDEX idx_fraud_decisions_transaction_id ON fraud_decisions(transaction_id);
CREATE INDEX idx_fraud_decisions_is_fraud ON fraud_decisions(is_fraud);

-- 3. ROUTING DECISIONS TABLE
-- Stores payment gateway routing and final outcomes
CREATE TABLE routing_decisions (
    routing_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    decision_id INTEGER REFERENCES fraud_decisions(decision_id),
    gateway_selected VARCHAR(50),  -- 'Gateway_A', 'Gateway_B', 'DECLINED', etc.
    routing_reason TEXT,
    gateway_response_code VARCHAR(20),
    gateway_latency_ms INTEGER,
    final_status VARCHAR(20) NOT NULL,  -- 'APPROVED', 'DECLINED', 'PENDING', 'ERROR'
    amount_processed DECIMAL(12, 2),
    fees_charged DECIMAL(12, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for analytics queries
CREATE INDEX idx_routing_decisions_transaction_id ON routing_decisions(transaction_id);
CREATE INDEX idx_routing_decisions_gateway ON routing_decisions(gateway_selected);
CREATE INDEX idx_routing_decisions_status ON routing_decisions(final_status);

-- 4. FEATURE STORE METADATA (Optional - for tracking feature drift)
CREATE TABLE feature_store_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(12, 4),
    computed_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_feature_snapshots_transaction_id ON feature_store_snapshots(transaction_id);

-- 5. SYSTEM METRICS TABLE (For monitoring dashboard)
CREATE TABLE system_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12, 4) NOT NULL,
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_name_time ON system_metrics(metric_name, timestamp DESC);

-- VIEWS FOR DASHBOARD QUERIES

-- Real-time fraud rate (last 1 hour)
CREATE OR REPLACE VIEW vw_realtime_fraud_rate AS
SELECT 
    COUNT(*) FILTER (WHERE fd.is_fraud = TRUE) AS fraud_count,
    COUNT(*) AS total_transactions,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE fd.is_fraud = TRUE) / NULLIF(COUNT(*), 0),
        2
    ) AS fraud_percentage,
    DATE_TRUNC('hour', t.timestamp) AS hour_bucket
FROM transactions t
JOIN fraud_decisions fd ON t.transaction_id = fd.transaction_id
WHERE t.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY DATE_TRUNC('hour', t.timestamp);

-- Gateway routing distribution
CREATE OR REPLACE VIEW vw_gateway_distribution AS
SELECT 
    gateway_selected,
    COUNT(*) AS transaction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(gateway_latency_ms) AS avg_latency_ms,
    SUM(amount_processed) AS total_amount
FROM routing_decisions
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY gateway_selected
ORDER BY transaction_count DESC;

-- Top fraud patterns (last 24 hours)
CREATE OR REPLACE VIEW vw_fraud_patterns AS
SELECT 
    fd.rules_triggered,
    COUNT(*) AS occurrence_count,
    AVG(fd.fraud_score) AS avg_fraud_score,
    SUM(t.amount) AS total_amount_flagged
FROM fraud_decisions fd
JOIN transactions t ON fd.transaction_id = t.transaction_id
WHERE fd.is_fraud = TRUE
  AND fd.created_at >= NOW() - INTERVAL '24 hours'
GROUP BY fd.rules_triggered
ORDER BY occurrence_count DESC
LIMIT 10;

-- Insert initial system health metric
INSERT INTO system_metrics (metric_name, metric_value, metric_unit)
VALUES ('system_status', 1.0, 'boolean');

-- Grant permissions (if using non-superuser)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO fraud_user;
