import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

Base = declarative_base()

# =============================================================================
# ORM Model Definition
# =============================================================================

class PredictionLog(Base):
    """SQLAlchemy ORM model mapping to the SQLite/PostgreSQL prediction_log table."""
    __tablename__ = 'prediction_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    amount = Column(Float, nullable=False)
    fraud_score = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    rules_fired = Column(String, nullable=True) 
    model_version = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)

# =============================================================================
# Database Connection Functions
# =============================================================================

def create_async_engine_from_env() -> AsyncEngine:
    """
    Creates an asynchronous SQLAlchemy engine. 
    Defaults to a local SQLite file named 'fraud_logs.db' in your project root.
    """
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "fraud_logs.db")
    db_url = os.getenv("DB_URL", f"sqlite+aiosqlite:///{db_path}")
    
    engine = create_async_engine(db_url, echo=False)
    return engine

async def init_db(engine: AsyncEngine) -> None:
    """Asynchronously builds the schema if it does not already exist."""
    logger.info("Initializing Database schema...")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema verified/created successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

# =============================================================================
# Database CRUD Operations
# =============================================================================

async def log_prediction(session: AsyncSession, prediction_data: Dict[str, Any]) -> None:
    """Inserts a single prediction record into the database asynchronously."""
    rules_json = json.dumps(prediction_data.get("rules_fired", []))
    
    new_log = PredictionLog(
        transaction_id=prediction_data["transaction_id"],
        amount=prediction_data["amount"],
        fraud_score=prediction_data["fraud_score"],
        is_fraud=prediction_data["is_fraud"],
        rules_fired=rules_json,
        model_version=prediction_data["model_version"],
        latency_ms=prediction_data["latency_ms"]
    )
    
    session.add(new_log)
    
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        logger.warning(f"Duplicate Transaction ID detected: {prediction_data['transaction_id']}. Record not inserted.")
    except Exception as e:
        await session.rollback()
        logger.error(f"Unexpected database error during logging: {e}")

async def get_recent_predictions(session: AsyncSession, limit: int = 1000) -> List[Dict[str, Any]]:
    """Retrieves the most recent predictions for the investigator dashboard."""
    stmt = select(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit)
    result = await session.execute(stmt)
    records = result.scalars().all()
    
    formatted_logs = []
    for record in records:
        formatted_logs.append({
            "id": record.id,
            "transaction_id": record.transaction_id,
            "timestamp": record.timestamp.isoformat(),
            "amount": record.amount,
            "fraud_score": record.fraud_score,
            "is_fraud": record.is_fraud,
            "rules_fired": json.loads(record.rules_fired) if record.rules_fired else [],
            "model_version": record.model_version,
            "latency_ms": record.latency_ms
        })
        
    return formatted_logs