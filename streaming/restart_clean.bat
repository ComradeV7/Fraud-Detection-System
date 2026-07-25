@echo off
echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║  FRAUD DETECTION PIPELINE - CLEAN RESTART                 ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

echo [1/3] Clearing old Kafka topics...
docker exec fraud-kafka kafka-topics --delete --topic transactions.raw --bootstrap-server localhost:9092 >nul 2>&1
docker exec fraud-kafka kafka-topics --delete --topic transactions.scored --bootstrap-server localhost:9092 >nul 2>&1
echo       ✅ Topics cleared
echo.

echo [2/3] Waiting 5 seconds for Kafka cleanup...
timeout /t 5 /nobreak >nul
echo       ✅ Ready
echo.

echo [3/3] Starting processor...
echo.
echo ═══════════════════════════════════════════════════════════
echo   Processor will start now. Watch for:
echo   ✅ "SHAP explainer initialized successfully"
echo   ✅ "XGBOOST FRAUD STREAM PROCESSOR STARTED"
echo.
echo   Then open Terminal 2 and run:
echo   python streaming\kafka_producer.py
echo ═══════════════════════════════════════════════════════════
echo.

python streaming\stream_processor_xgboost.py
