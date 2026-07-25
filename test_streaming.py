"""
Quick test script to verify streaming pipeline works end-to-end
"""
import time
import subprocess
import sys

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    try:
        import confluent_kafka
        import redis
        import psycopg2
        import torch
        import joblib
        import pandas as pd
        import numpy as np
        from src.model.anfis import ANFIS
        print("✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}\n")
        return False

def test_docker_services():
    """Check if Docker services are running"""
    print("🐳 Checking Docker services...")
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=fraud", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        services = result.stdout.strip().split('\n')
        expected = ['fraud-kafka', 'fraud-postgres', 'fraud-redis', 'fraud-zookeeper']
        
        for service in expected:
            if service in services:
                print(f"  ✅ {service}")
            else:
                print(f"  ❌ {service} not running")
        
        if all(s in services for s in expected):
            print("✅ All Docker services running!\n")
            return True
        else:
            print("❌ Some Docker services missing\n")
            return False
    except Exception as e:
        print(f"❌ Docker check failed: {e}\n")
        return False

def test_redis_connection():
    """Test Redis connectivity"""
    print("📦 Testing Redis connection...")
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        print("✅ Redis connected!\n")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}\n")
        return False

def test_postgres_connection():
    """Test PostgreSQL connectivity"""
    print("🐘 Testing PostgreSQL connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='fraud_detection',
            user='fraud_user',
            password='fraud_password'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        print(f"✅ PostgreSQL connected! Transactions in DB: {count}\n")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}\n")
        return False

def test_kafka_connection():
    """Test Kafka connectivity"""
    print("📨 Testing Kafka connection...")
    try:
        from confluent_kafka.admin import AdminClient
        admin_client = AdminClient({'bootstrap.servers': 'localhost:9092'})
        metadata = admin_client.list_topics(timeout=5)
        topics = list(metadata.topics.keys())
        print(f"✅ Kafka connected! Topics: {topics}\n")
        return True
    except Exception as e:
        print(f"❌ Kafka connection failed: {e}\n")
        return False

def test_ml_artifacts():
    """Test ML model loading"""
    print("🤖 Testing ML artifacts...")
    try:
        import torch
        import joblib
        from src.model.anfis import ANFIS
        
        model = torch.load("artifacts/anfis_model.pt", map_location='cpu', weights_only=False)
        scaler = joblib.load("artifacts/scaler.pkl")
        encoder = joblib.load("artifacts/target_encoder.pkl")
        selector = joblib.load("artifacts/selector.pkl")
        
        print("✅ ML artifacts loaded successfully!\n")
        return True
    except Exception as e:
        print(f"❌ ML artifacts failed: {e}\n")
        return False

def main():
    print("\n" + "="*60)
    print("🧪 STREAMING PIPELINE HEALTH CHECK")
    print("="*60 + "\n")
    
    results = {}
    results['imports'] = test_imports()
    results['docker'] = test_docker_services()
    results['redis'] = test_redis_connection()
    results['postgres'] = test_postgres_connection()
    results['kafka'] = test_kafka_connection()
    results['ml_artifacts'] = test_ml_artifacts()
    
    print("="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ System is ready! You can now run:")
        print("   Terminal 1: python streaming/stream_processor.py")
        print("   Terminal 2: python streaming/kafka_producer.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
