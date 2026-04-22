import os
import time
import json
import joblib
import torch
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import async_sessionmaker

# Import database functions
from src.api.database import create_async_engine_from_env, init_db, log_prediction
from src.preprocessing import cleaning, engineering

# Note: We completely removed PCATranslator because we don't need it anymore!

# 1. Pydantic Models
class TransactionInput(BaseModel):
    transaction_id: str
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int
    device_fraud_count: int
    month: int

class PredictionOutput(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    rules_fired: list[str]
    model_version: str
    latency_ms: float

# 2. Lifespan Manager (Startup/Shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Natively Explainable Fraud Detection API...")
    
    # Database Setup
    try:
        app.state.engine = create_async_engine_from_env()
        app.state.AsyncSessionLocal = async_sessionmaker(app.state.engine, expire_on_commit=False)
        await init_db(app.state.engine)
    except Exception as e:
        print(f"DB Setup Warning: {e}")
        app.state.AsyncSessionLocal = None

    # ML Artifact Loading
    try:
        app.state.model = torch.load("artifacts/anfis_model.pt", map_location=torch.device('cpu'))
        app.state.model.eval()
        
        with open("artifacts/fuzzy_rules.json", "r") as f:
            app.state.rules = json.load(f)
            
        app.state.scaler = joblib.load("artifacts/scaler.pkl")
        app.state.encoder = joblib.load("artifacts/target_encoder.pkl")
        
        # LOAD SELECTOR INSTEAD OF PCA
        app.state.selector = joblib.load("artifacts/selector.pkl") 
        app.state.model_version = "v2.0-native-anfis"
        
        print("All artifacts loaded successfully! (Running in Native XAI Mode)")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load ML artifacts: {e}")
        app.state.model = None

    yield
    print("Shutting down server.")

app = FastAPI(title="Explainable Fraud Detection API", lifespan=lifespan)

# 3. Inference Endpoint
@app.post("/predict", response_model=PredictionOutput)
async def predict(request: TransactionInput, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model unavailable.")

    # THE RULE-BASED GATEKEEPER (Sanity Checks)
    gatekeeper_flags = []
    
    # Adjusted limits to match the NeurIPS BAF dataset distributions
    if request.velocity_6h > 25000:  # BAF normal is ~8000-15000
        gatekeeper_flags.append(f"Impossible 6H Velocity ({request.velocity_6h})")
    if request.velocity_24h > 35000:
        gatekeeper_flags.append("Impossible 24H Velocity")
    if request.zip_count_4w > 5000:
        gatekeeper_flags.append("Physically impossible geographic movement")
    if request.customer_age < 18 or request.customer_age > 120:
        gatekeeper_flags.append(f"Invalid Age ({request.customer_age})")

    # Fast-fail for bots
    if gatekeeper_flags:
        latency = (time.time() - start_time) * 1000
        explanation = f"REJECTED AT GATEWAY: Sanity check failed due to - {', '.join(gatekeeper_flags)}."
        
        if getattr(app.state, "AsyncSessionLocal", None):
            prediction_record = {
                "transaction_id": request.transaction_id,
                "amount": request.intended_balcon_amount,
                "fraud_score": 1.0,
                "is_fraud": True,
                "rules_fired": [explanation],
                "model_version": "v1.0-gatekeeper",
                "latency_ms": latency
            }
            async def background_log(record):
                async with app.state.AsyncSessionLocal() as session:
                    await log_prediction(session, record)
            background_tasks.add_task(background_log, prediction_record)

        return PredictionOutput(
            transaction_id=request.transaction_id,
            fraud_score=1.0,
            is_fraud=True,
            rules_fired=[explanation],
            model_version="v1.0-gatekeeper",
            latency_ms=round(latency, 2)
        )

    # ML Pipeline
    try:
        input_dict = request.model_dump(exclude={"transaction_id"})
        df_raw = pd.DataFrame([input_dict])
        
        # Live Feature Engineering
        df_cleaned, _ = cleaning.handle_sentinels(df_raw, df_raw.copy())
        df_skewed, _ = cleaning.apply_skewness_fix(df_cleaned, df_cleaned.copy())
        df_engineered, _ = engineering.add_domain_ratios(df_skewed, df_skewed.copy())
        
        df_final = app.state.encoder.transform(df_engineered)
        df_final = df_final.replace([np.inf, -np.inf], 0).fillna(0)

        # Scaling, Clipping & FEATURE SELECTION
        scaled_features = app.state.scaler.transform(df_final)
        scaled_features = np.clip(scaled_features, a_min=-5.0, a_max=5.0)
        
        selected_features = app.state.selector.transform(scaled_features)
        
        # ANFIS Neural Inference
        tensor_input = torch.tensor(selected_features, dtype=torch.float32)
        with torch.no_grad():
            fraud_score = app.state.model(tensor_input).item()
            
        # NATIVE XAI TRANSLATION
        is_fraud_status = bool(fraud_score >= 0.5)
        
        # Dynamically extract the top 3 driving features based on their z-score magnitude
        feature_names_out = app.state.selector.get_feature_names_out(app.state.scaler.feature_names_in_)
        feature_magnitudes = pd.Series(np.abs(selected_features[0]), index=feature_names_out)
        top_drivers = feature_magnitudes.sort_values(ascending=False).head(3).index.tolist()
        
        # Format the feature names nicely
        clean_drivers = [f.replace('_', ' ').title() for f in top_drivers]
        if len(clean_drivers) > 1:
            drivers_str = ", ".join(clean_drivers[:-1]) + f", and {clean_drivers[-1]}"
        else:
            drivers_str = clean_drivers[0]

        if is_fraud_status:
            human_explanation = f"Flagged as suspicious due to highly anomalous patterns natively detected in {drivers_str}."
        else:
            human_explanation = f"Cleared as legitimate based on verified, normal behavior natively detected in {drivers_str}."

        latency = (time.time() - start_time) * 1000

        # Async Audit Logging
        if getattr(app.state, "AsyncSessionLocal", None):
            prediction_record = {
                "transaction_id": request.transaction_id,
                "amount": request.intended_balcon_amount,
                "fraud_score": fraud_score,
                "is_fraud": is_fraud_status,
                "rules_fired": [human_explanation], 
                "model_version": app.state.model_version,
                "latency_ms": latency
            }
            async def background_log(record):
                async with app.state.AsyncSessionLocal() as session:
                    await log_prediction(session, record)
            background_tasks.add_task(background_log, prediction_record)
        
        return PredictionOutput(
            transaction_id=request.transaction_id,
            fraud_score=round(fraud_score, 4),
            is_fraud=is_fraud_status,
            rules_fired=[human_explanation], 
            model_version=app.state.model_version,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Inference pipeline failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "online", "artifacts_loaded": app.state.model is not None}