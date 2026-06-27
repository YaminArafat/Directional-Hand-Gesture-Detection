-- Clean up existing assets if running multiple times
DROP TRIGGER IF EXISTS trigger_route_inference ON gesture_inference_logs;
DROP FUNCTION IF EXISTS route_inference_data();
DROP MATERIALIZED VIEW IF EXISTS daily_hardware_latencies;
DROP TABLE IF EXISTS low_confidence_audit;
DROP TABLE IF EXISTS high_confidence_retrain_pool;
DROP TABLE IF EXISTS gesture_inference_logs;

CREATE TABLE gesture_inference_logs (
    id BIGSERIAL,
    device_id VARCHAR(255) NOT NULL,
    predicted_gesture_id INT NOT NULL,
    confidence_score REAL NOT NULL,
    inference_latency_ms REAL NOT NULL,
    logged_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, logged_at)
) PARTITION BY RANGE (logged_at);

CREATE TABLE logs_2026_06 PARTITION OF gesture_inference_logs
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE TABLE logs_2026_07 PARTITION OF gesture_inference_logs
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

-- 2. DEDICATED AUDIT POOL TABLE (For low confidence observations)
CREATE TABLE low_confidence_audit (
    audit_id SERIAL PRIMARY KEY,
    device_id VARCHAR(255),
    predicted_gesture_id INT,
    confidence_score REAL,
    inference_latency_ms REAL,
    flagged_reason TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. AUTOMATED RETRAINING DATA POOL TABLE (For highly accurate structural vectors)
CREATE TABLE high_confidence_retrain_pool (
    pool_id SERIAL PRIMARY KEY,
    device_id VARCHAR(255),
    predicted_gesture_id INT,
    confidence_score REAL,
    is_processed BOOLEAN DEFAULT FALSE,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. PL/pgSQL DATA ROUTING FUNCTION
CREATE OR REPLACE FUNCTION route_inference_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Route to the manual audit table if confidence is lower than 70%
    IF NEW.confidence_score < 0.70 THEN
        INSERT INTO low_confidence_audit (device_id, predicted_gesture_id, confidence_score, inference_latency_ms, flagged_reason)
        VALUES (NEW.device_id, NEW.predicted_gesture_id, NEW.confidence_score, NEW.inference_latency_ms, 'Low confidence prediction threshold breach.');
    
    --Route to retraining dataset pool if confidence is higher than 92%
    ELSIF NEW.confidence_score >= 0.92 THEN
        INSERT INTO high_confidence_retrain_pool (device_id, predicted_gesture_id, confidence_score)
        VALUES (NEW.device_id, NEW.predicted_gesture_id, NEW.confidence_score);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. ATTACH ROUTING TRIGGER TO PARENT LOGS TABLE
CREATE TRIGGER trigger_route_inference
AFTER INSERT ON gesture_inference_logs
FOR EACH ROW
EXECUTE FUNCTION route_inference_data();

-- 6. MATERIALIZED VIEW FOR REAL-TIME METRIC CACHING
CREATE MATERIALIZED VIEW daily_hardware_latencies AS
SELECT 
    device_id,
    DATE(logged_at) AS log_date,
    COUNT(*) AS total_inferences,
    AVG(inference_latency_ms) AS avg_latency_ms,
    AVG(confidence_score) AS avg_confidence
FROM gesture_inference_logs
GROUP BY device_id, DATE(logged_at);

CREATE UNIQUE INDEX idx_daily_hw_latencies ON daily_hardware_latencies (device_id, log_date);