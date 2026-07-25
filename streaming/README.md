# Real-Time Fraud Detection Streaming Pipeline

## Overview

This streaming fraud detection system processes credit card transactions in real-time using:
- **Kafka**: Message broker for transaction streams
- **Redis**: Feature store for behavioral patterns
- **XGBoost**: ML model for fraud scoring
- **SHAP**: Explainability framework for predictions
- **PostgreSQL**: Long-term storage for scored transactions

## Architecture

```
[Transaction Stream] → [Kafka Producer] → [Kafka: transactions.raw]
                                              ↓
                                    [Stream Processor]
                                      ├─ Redis Feature Store
                                      ├─ XGBoost Model
                                      └─ SHAP Explainer
                                              ↓
                                  [Kafka: transactions.scored]
                                              ↓
                                    [Database Consumer] ✨ NEW
                                              ↓
                                      [PostgreSQL]
                                  ├─ transactions table
                                  └─ fraud_decisions table
                                              ↓
                                    [Dashboard (TODO)]
```

## Components

### 1. Kafka Producer (`kafka_producer.py`)
Generates synthetic transactions with **31 features** matching training data:
- Basic: income, customer_age, credit_risk_score
- Behavioral: velocity_6h, velocity_24h, velocity_4w
- Categorical: payment_type, employment_status, housing_status, device_os, source
- Risk indicators: device_fraud_count, foreign_request, email_is_free

**Features:**
- Configurable fraud rate (default 15%)
- Configurable TPS (default 10 transactions/second)
- Realistic fraud vs. legitimate patterns

### 2. Stream Processor (`stream_processor_xgboost.py`)
Processes transactions through a 4-stage pipeline:
1. **Enrichment**: Fetch real-time features from Redis
2. **Preprocessing**: Convert to XGBoost DMatrix with categorical handling
3. **Scoring**: Run XGBoost inference (fraud probability)
4. **Explanation**: Generate SHAP-based explanations

**Model:**
- XGBoost Unchained (artifacts/xgboost_unchained.json)
- 31 input features
- Binary classification (fraud/legitimate)
- Threshold: 0.5

**Explainability:**
- SHAP TreeExplainer
- Top 3 contributing features per prediction
- Natural language explanations

### 3. Feature Store (`feature_store.py`)
Redis-based real-time feature computation:
- Transaction velocity (6h, 24h, 4w windows)
- Device fraud history
- Amount statistics

### 4. Database Consumer (`db_consumer.py`) ✨ **NEW**
Persists scored transactions to PostgreSQL:
- Consumes from `transactions.scored` topic
- Dual-table insertion (transactions + fraud_decisions)
- Stores SHAP explanations in JSONB format
- Error handling with transaction rollback
- Performance tracking (TPS, fraud rate, errors)

**Tables:**
- `transactions` - Raw transaction data + metadata
- `fraud_decisions` - ML predictions + SHAP explanations

### 5. ANFIS Processor (`stream_processor.py`) [DEPRECATED]
Original ANFIS-based processor - **has pandas hanging issue**.
Use `stream_processor_xgboost.py` instead.

## Setup

### Prerequisites
- Python 3.12+ (with venv activated)
- Docker Desktop running
- Packages: xgboost, shap, confluent-kafka, redis, pandas

### Installation

```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies (if not already installed)
pip install xgboost shap confluent-kafka redis pandas

# Verify installation
python streaming/test_pipeline.py
```

### Start Infrastructure

```bash
# Start Kafka, Redis, PostgreSQL, Zookeeper
docker-compose up -d

# Verify services are running
docker ps
```

Expected output:
- fraud-kafka (port 9092)
- fraud-redis (port 6379)
- fraud-postgres (port 5432)
- fraud-zookeeper (port 2181)

## Usage

### Quick Start (3 Terminal Windows) - Full Pipeline ✨ **UPDATED**

**Terminal 1: Start Processor**
```bash
.\.venv\Scripts\activate
python streaming/stream_processor_xgboost.py
```

Wait for:
```
🚀 XGBOOST FRAUD STREAM PROCESSOR STARTED
📥 Consuming from: transactions.raw
📤 Publishing to: transactions.scored
🧠 Model: XGBoost Unchained
💡 Explainability: SHAP TreeExplainer
```

**Terminal 2: Start Database Consumer** ✨ **NEW**
```bash
.\.venv\Scripts\activate
python streaming/db_consumer.py
```

Wait for:
```
🗄️  DATABASE CONSUMER STARTED
📥 Consuming from: transactions.scored
💾 Writing to: PostgreSQL (fraud_detection)
```

**Terminal 3: Start Producer**
```bash
.\.venv\Scripts\activate
python streaming/kafka_producer.py
```

You should see:
```
🚀 Starting transaction stream...
✅ Transaction TXN_XXX → Kafka partition 0
```

**Verification:**
```bash
# Check database records
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "SELECT COUNT(*) FROM fraud_decisions;"
```

### Expected Output (Terminal 1)

```
🔄 Processing TXN_ABC123
   [1/4] Enriching features...
   ✓ Features enriched
   [2/4] Preprocessing...
      → Creating feature array with 31 features...
      → Creating DataFrame for DMatrix...
      → Creating DMatrix...
      → Preprocessing complete! Shape: (1, 31)
   ✓ Preprocessed
   [3/4] Scoring with XGBoost...
   ✓ Scored: 0.8521
   [4/4] Generating SHAP explanation...
   ✓ Explanation generated
🚨 Score: 0.8521 | Amount: $1234.56 | Inference: 12.34ms
   Explanation: Flagged as suspicious due to highly anomalous patterns...
✅ Scored transaction TXN_ABC123 → Kafka
```

### Command-Line Options

**Producer:**
```bash
# Default: 100 transactions at 10 TPS with 15% fraud rate
python streaming/kafka_producer.py

# Modify in code:
simulator.produce_stream(
    num_transactions=500,    # Total transactions
    rate_per_second=20,      # TPS
    fraud_rate=0.20          # 20% fraud
)
```

**Processor:**
```bash
# Skip old messages (default)
python streaming/stream_processor_xgboost.py

# Replay all historical messages
python streaming/stream_processor_xgboost.py --replay-all
```

## Testing

### Unit Tests
```bash
# Test stream processor
python streaming/test_pipeline.py

# Test database consumer ✨ NEW
python streaming/test_db_consumer.py

# Quick automated test ✨ NEW
python streaming/quick_test_task3.py
```

Tests:
1. ✅ Feature Generation (31 features)
2. ✅ Preprocessing & Inference (XGBoost + SHAP)
3. ✅ Module Imports
4. ✅ Database Persistence ✨ NEW
5. ✅ Data Integrity ✨ NEW

### Integration Test (Full Pipeline)
```bash
# Terminal 1: Processor
python streaming/stream_processor_xgboost.py

# Terminal 2: Database Consumer ✨ NEW
python streaming/db_consumer.py

# Terminal 3: Producer
python streaming/kafka_producer.py

# Verify in PostgreSQL
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "SELECT COUNT(*) FROM fraud_decisions;"
```

### Verify Kafka Topics
```bash
# List topics
docker exec -it fraud-kafka kafka-topics --list --bootstrap-server localhost:9092

# Should see:
# transactions.raw
# transactions.scored

# Consume scored transactions
docker exec -it fraud-kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic transactions.scored \
  --from-beginning
```

## Troubleshooting

### Issue 1: "No module named 'xgboost'"
**Solution:** Activate venv and install packages
```bash
.\.venv\Scripts\activate
pip install xgboost shap
```

### Issue 2: "KafkaError: Unknown topic or partition"
**Solution:** Clear Docker volumes and restart
```bash
docker-compose down -v
docker-compose up -d
# Wait 30 seconds for Kafka to initialize
```

### Issue 3: Processor hangs at "[2/4] Preprocessing"
**Solution:** Use XGBoost processor (not ANFIS)
```bash
python streaming/stream_processor_xgboost.py
```

### Issue 4: "feature_names mismatch"
**Solution:** Already fixed! Producer now generates all 31 features.
If still seeing this, verify you're using latest code.

### Issue 5: Docker services not starting
**Solution:**
```bash
# Check Docker is running
docker ps

# Restart Docker Desktop

# Remove old containers
docker-compose down
docker-compose up -d
```

## Performance Metrics

- **Throughput**: ~10-50 TPS (configurable)
- **Latency**: 10-50ms per transaction (depends on SHAP computation)
- **Model**: XGBoost with 31 features
- **Accuracy**: Based on training data (see artifacts/evaluation/)

## Output Schema

Scored transactions in `transactions.scored` topic:

```json
{
  "transaction_id": "TXN_ABC123",
  "user_id": "USER_1234",
  "timestamp": "2026-07-24T10:30:00.123Z",
  "amount": 1234.56,
  
  // All 31 input features
  "income": 0.75,
  "customer_age": 35,
  "velocity_24h": 15.2,
  // ... (28 more features)
  
  // Model outputs
  "fraud_score": 0.8521,
  "is_fraud": true,
  "model_version": "v2.0-xgboost-unchained",
  "inference_time_ms": 12.34,
  
  // Explainability
  "top_features": ["velocity_24h", "credit_risk_score", "income"],
  "shap_values": {
    "velocity_24h": 0.45,
    "credit_risk_score": -0.23,
    "income": -0.18
  },
  "rules_fired": ["Flagged as suspicious due to highly anomalous patterns detected in Velocity 24H, Credit Risk Score, and Income."],
  "confidence_level": 0.8521,
  "explanation_method": "SHAP TreeExplainer",
  
  "processed_at": "2026-07-24T10:30:00.456Z"
}
```

## TODO: Remaining Work

1. ✅ **Database Consumer** (`db_consumer.py`) - COMPLETE ✨
   - Consume from `transactions.scored`
   - Store in PostgreSQL
   - Track performance metrics

2. **Dashboard Updates** (`app.py`)
   - Real-time fraud alerts
   - Transaction timeline
   - SHAP visualizations
   - Performance metrics

3. **Monitoring**
   - MLflow integration
   - Drift detection
   - Model retraining triggers

## Files

### Core Components
- `kafka_producer.py` - Transaction generator (31 features) ✅
- `stream_processor_xgboost.py` - XGBoost + SHAP processor ✅
- `db_consumer.py` - PostgreSQL persistence service ✅ ✨ **NEW**
- `feature_store.py` - Redis integration ✅

### Deprecated
- `stream_processor.py` - ANFIS processor (deprecated, has hanging issue)

### Testing
- `test_pipeline.py` - Stream processor tests ✅
- `test_db_consumer.py` - Database consumer comprehensive tests ✅ ✨ **NEW**
- `quick_test_task3.py` - Quick automated database test ✅ ✨ **NEW**

### Documentation
- `README.md` - This file
- `TASK3_DATABASE_CONSUMER.md` - Database consumer implementation guide ✨ **NEW**
- `SUCCESS.md` - Test results and validation ✅
- `QUICK_TEST_GUIDE.md` - 5-minute walkthrough ✅

## Key Improvements Made

### Before (Broken):
- ❌ Feature mismatch: 15 features sent, 31 expected
- ❌ ANFIS processor hangs at pandas DataFrame creation
- ❌ No explainability
- ❌ Kafka offset issues (processing old messages)

### After (Fixed):
- ✅ Full 31 features with realistic patterns
- ✅ XGBoost processor with categorical handling
- ✅ SHAP explainability (top 3 features + natural language)
- ✅ Configurable offset reset (skip old messages)
- ✅ Comprehensive error handling
- ✅ Detailed logging ([1/4], [2/4], [3/4], [4/4])
- ✅ Unit tests passing
- ✅ PostgreSQL persistence with SHAP storage ✨ **NEW**
- ✅ Database consumer with error handling ✨ **NEW**

## References

- XGBoost Model: `artifacts/xgboost_unchained.json`
- Training Data: `data/Base.csv` (31 features)
- Training Script: `src/run_unchained_xgboost.py`
- Spec: `.kiro/specs/streaming-fraud-pipeline/`

---

**Status**: ✅ Full streaming pipeline working end-to-end (producer → processor → database)
**Next**: Update dashboard for real-time PostgreSQL data visualization
