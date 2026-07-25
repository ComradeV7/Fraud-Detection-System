@echo off
echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║  NUCLEAR RESET - Clear ALL Kafka Data                    ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo This will:
echo   1. Stop all Docker containers
echo   2. Remove all volumes (clears Kafka data)
echo   3. Restart containers
echo.
echo ⚠️  WARNING: This will delete ALL Kafka messages!
echo.
pause

echo.
echo [1/3] Stopping Docker containers...
docker-compose down -v
echo       ✅ Containers stopped and volumes removed
echo.

echo [2/3] Starting fresh Docker containers...
docker-compose up -d
echo       ✅ Containers started
echo.

echo [3/3] Waiting 30 seconds for Kafka to initialize...
timeout /t 30 /nobreak
echo       ✅ Kafka ready
echo.

echo ╔═══════════════════════════════════════════════════════════╗
echo ║  ✅ RESET COMPLETE!                                        ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo   Terminal 1: python streaming\stream_processor_xgboost.py
echo   Terminal 2: python streaming\kafka_producer.py
echo.
pause
