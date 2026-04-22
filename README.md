# GlassBox Fraud Detection System using ANFIS

An end-to-end **Explainable AI (XAI)** fraud detection engine built on the [NeurIPS Bank Account Fraud (BAF)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) dataset. The system pairs a custom **PyTorch ANFIS** (Adaptive Neuro-Fuzzy Inference System) with a production-grade **FastAPI** serving layer and a **Streamlit** investigator dashboard - delivering human-readable explanations for every prediction.

---

## Architecture

```
                  ┌──────────────────────┐
                  │   Streamlit Dashboard │  (app.py)
                  │  KPIs · Live Feed ·   │
                  │  Drift Monitoring     │
                  └──────────┬───────────┘
                             │ HTTP
                  ┌──────────▼───────────┐
                  │    FastAPI Server     │  (src/api/main.py)
                  │  Gatekeeper + ANFIS   │
                  │  Native XAI Engine    │
                  └──────────┬───────────┘
                             │ async
                  ┌──────────▼───────────┐
                  │        SQLite         │  (fraud_logs.db)
                  │  Prediction Audit Log │
                  └──────────────────────┘
```

## Key Features

| Feature | Detail |
|---|---|
| **ANFIS Neural Network** | 5-layer neuro-fuzzy model with Gaussian membership functions, subtractive clustering rule initialization, and sigmoid output |
| **Native Explainability** | Every prediction returns a human-readable explanation derived from feature z-score magnitudes — no post-hoc SHAP required at inference |
| **Rule-Based Gatekeeper** | Velocity, geography, and age sanity checks reject bot traffic before it reaches the ML model |
| **XGBoost Challenger** | Optuna-tuned XGBoost baseline with SHAP summary plots for comparative evaluation |
| **MLflow Tracking** | All training runs, hyperparameters, and model artifacts are versioned and registered |
| **Concept Drift Monitor** | Kolmogorov–Smirnov tests compare live production distributions against the training baseline |
| **Stream Simulator** | Multi-tiered attack simulation (bots, sophisticated fraud, normal traffic) for stress testing |
| **Async Audit Logging** | Every prediction is persisted to SQLite (or PostgreSQL) via SQLAlchemy async sessions |

---

## Project Structure

```
Fraud-Detection-System/
├── app.py                        # Streamlit dashboard (KPIs, live feed, drift tab)
├── stream_simulator.py           # Attack simulation script
├── drift_monitor.py              # K-S statistical drift analysis
├── requirements.txt              # Python dependencies (CUDA 12.1 PyTorch)
├── .env                          # DB_URL, MLFLOW_URI, API_PORT
│
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py        # BAF CSV ingestion & stratified split
│   │   ├── cleaning.py           # Sentinel imputation & signed pseudo-log transforms
│   │   ├── engineering.py        # Domain ratios, target encoding, artifact serialization
│   │   ├── balancing.py          # SMOTE oversampling with validation
│   │   └── pipeline.py          # End-to-end preprocessing orchestrator
│   │
│   ├── model/
│   │   ├── anfis.py              # PyTorch ANFIS architecture (5 layers)
│   │   ├── clustering.py         # Chiu's subtractive clustering & rule generation
│   │   └── train.py              # Training loop with class weighting & MLflow logging
│   │
│   ├── api/
│   │   ├── main.py               # FastAPI /predict and /health endpoints
│   │   └── database.py           # Async SQLAlchemy ORM & CRUD operations
│   │
│   ├── evaluation/
│   │   └── model_evaluation.py   # ANFIS vs XGBoost comparison (ROC, metrics, latency)
│   │
│   ├── run_training.py           # ANFIS training with Optuna hyperparameter search
│   ├── run_unchained_xgboost.py  # XGBoost baseline training with SHAP
│   ├── extract_rules.py          # Subtractive clustering → fuzzy rules JSON
│   └── reports.py                # Feature transformation visualizations
│
├── notebooks/
│   └── data_exploration.ipynb    # Exploratory data analysis
│
├── data/
│   └── Base.csv                  # NeurIPS BAF dataset (~213 MB)
│
└── artifacts/                    # Generated model weights, scalers, plots
    ├── anfis_model.pt
    ├── xgboost_unchained.json
    ├── fuzzy_rules.json
    ├── scaler.pkl / selector.pkl / target_encoder.pkl
    ├── *.parquet                  # Processed train/test splits
    └── evaluation/               # Comparison charts & CSV
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **CUDA 12.1** (optional — CPU inference is supported)
- The [BAF dataset](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) (`Base.csv`) placed in `data/`

### Installation

```bash
git clone https://github.com/ComradeV7/GlassBox-Fraud-ANFIS.git
cd GlassBox-Fraud-ANFIS

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

### Environment Variables

Copy `.env` and fill in your values (defaults work for local development):

```env
DB_URL=sqlite+aiosqlite:///fraud_logs.db   # or a PostgreSQL connection string
MLFLOW_URI=http://localhost:5000
API_PORT=8000
```

---

## Pipeline Execution

Run the scripts **in order** from the project root:

### 1. Data Preprocessing

```bash
python -m src.preprocessing.pipeline
```

Ingests `data/Base.csv`, applies sentinel cleaning, pseudo-log transforms, target encoding, ANOVA feature selection (top 15), and SMOTE balancing. Outputs processed parquet files and fitted preprocessors to `artifacts/`.

### 2. Fuzzy Rule Extraction

```bash
python -m src.extract_rules
```

Runs Chiu's subtractive clustering on the processed training data to discover fuzzy behavioral prototypes. Saves `artifacts/fuzzy_rules.json`.

### 3. ANFIS Training

```bash
python -m src.run_training
```

Performs Optuna Bayesian hyperparameter optimization (learning rate), then trains the ANFIS model on the full dataset with class-weighted BCE loss. Registers the model in MLflow and saves `artifacts/anfis_model.pt`.

### 4. XGBoost Baseline (Optional)

```bash
python -m src.run_unchained_xgboost
```

Trains an Optuna-tuned XGBoost classifier on the raw dataset with native categorical support. Generates SHAP explanations.

### 5. Model Evaluation (Optional)

```bash
python -m src.evaluation.model_evaluation
```

Produces a side-by-side comparison of ANFIS vs. XGBoost (AUC, F1, Precision, Recall, Latency) with ROC curves and metric bar charts.

---

## Serving & Dashboard

### Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Accepts a `TransactionInput` JSON, returns fraud score, decision, and XAI explanation |
| `/health` | GET | Returns API status and artifact loading state |

### Start the Dashboard

```bash
streamlit run app.py
```

The dashboard provides three tabs:

- **Manual Investigation** — Submit a transaction for real-time AI analysis
- **Live Threat Monitor** — Watch predictions stream in from the simulator
- **MLOps & Concept Drift** — Load Evidently/K-S drift reports

### Run the Stream Simulator

```bash
python stream_simulator.py
```

Fires 30 transactions at the API with a mix of normal traffic (~65%), sophisticated fraud (~15%), and bot brute-force attacks (~10%).

### Generate a Drift Report

```bash
python drift_monitor.py
```

Compares the live `fraud_logs.db` distribution against the training baseline using a Kolmogorov–Smirnov test. Saves a drift visualization to `artifacts/evaluation/`.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **ML / DL** | PyTorch, XGBoost, scikit-learn, imbalanced-learn, Optuna, SHAP |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit |
| **Database** | SQLAlchemy (async), SQLite / PostgreSQL |
| **MLOps** | MLflow, Evidently |
| **Data** | Pandas, NumPy, Seaborn, Matplotlib |

---

## License

This project is provided for academic and research purposes.
