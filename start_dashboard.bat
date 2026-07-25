@echo off
echo ============================================================
echo Starting Real-Time Fraud Detection Dashboard
echo ============================================================
echo.

echo [1/4] Checking Docker Desktop...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)
echo     Docker is running!

echo.
echo [2/4] Starting Docker services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start Docker services
    echo.
    pause
    exit /b 1
)
echo     Services started!

echo.
echo [3/4] Waiting for PostgreSQL to initialize (15 seconds)...
timeout /t 15 /nobreak >nul
echo     PostgreSQL ready!

echo.
echo [4/4] Verifying database connection...
docker exec fraud-postgres psql -U fraud_user -d fraud_detection -c "SELECT COUNT(*) FROM fraud_decisions;" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Database connection failed or no data available
    echo You may need to run the streaming pipeline first.
    echo.
    echo To generate data, run in separate terminals:
    echo   Terminal 1: python streaming/stream_processor_xgboost.py
    echo   Terminal 2: python streaming/db_consumer.py
    echo   Terminal 3: python streaming/kafka_producer.py
    echo.
    echo Continue anyway? (Dashboard will show empty state)
    pause
)
echo     Database connection verified!

echo.
echo ============================================================
echo Launching Dashboard...
echo ============================================================
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

call .venv\Scripts\activate
streamlit run app_realtime.py

pause
