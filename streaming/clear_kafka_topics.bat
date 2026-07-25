@echo off
echo Clearing Kafka topics...
echo.

echo Deleting transactions.raw topic...
docker exec -it fraud-kafka kafka-topics --delete --topic transactions.raw --bootstrap-server localhost:9092 2>nul
if %errorlevel% == 0 (
    echo   ✅ transactions.raw deleted
) else (
    echo   ⚠️  transactions.raw may not exist (this is OK)
)

echo.
echo Deleting transactions.scored topic...
docker exec -it fraud-kafka kafka-topics --delete --topic transactions.scored --bootstrap-server localhost:9092 2>nul
if %errorlevel% == 0 (
    echo   ✅ transactions.scored deleted
) else (
    echo   ⚠️  transactions.scored may not exist (this is OK)
)

echo.
echo ✅ Kafka topics cleared!
echo.
echo Topics will be auto-created when you restart the processor.
echo.
echo Next steps:
echo   1. Terminal 1: python streaming\stream_processor_xgboost.py
echo   2. Terminal 2: python streaming\kafka_producer.py
