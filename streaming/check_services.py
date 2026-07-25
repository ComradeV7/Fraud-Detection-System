"""
Service Health Check Script
Verifies that Docker services (Kafka, Redis, PostgreSQL) are ready
"""
import sys
import time

def check_redis():
    """Check Redis connection"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        print("✅ Redis: CONNECTED (port 6379)")
        return True
    except ImportError:
        print("⚠️  Redis: Module not installed (pip install redis)")
        return False
    except Exception as e:
        print(f"❌ Redis: NOT AVAILABLE - {e}")
        print("   Fix: docker-compose up -d")
        return False

def check_kafka():
    """Check Kafka connection"""
    try:
        from confluent_kafka import Producer
        producer = Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'health-check',
            'socket.timeout.ms': 5000
        })
        
        # Get cluster metadata (will fail if Kafka is down)
        metadata = producer.list_topics(timeout=5)
        
        print("✅ Kafka: CONNECTED (port 9092)")
        print(f"   Topics: {list(metadata.topics.keys())}")
        return True
    except ImportError:
        print("⚠️  Kafka: Module not installed (pip install confluent-kafka)")
        return False
    except Exception as e:
        print(f"❌ Kafka: NOT AVAILABLE - {e}")
        print("   Fix: docker-compose up -d && wait 30 seconds")
        return False

def check_postgres():
    """Check PostgreSQL connection"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='fraud_detection',
            user='fraud_user',
            password='fraud_password',
            connect_timeout=5
        )
        conn.close()
        print("✅ PostgreSQL: CONNECTED (port 5432)")
        return True
    except ImportError:
        print("⚠️  PostgreSQL: Module not installed (pip install psycopg2-binary)")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL: NOT AVAILABLE - {e}")
        print("   Fix: docker-compose up -d")
        return False

def check_docker():
    """Check if Docker is running"""
    import subprocess
    try:
        result = subprocess.run(
            ['docker', 'ps'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Docker: RUNNING")
            
            # Count relevant containers
            lines = result.stdout.strip().split('\n')
            fraud_containers = [l for l in lines if 'fraud-' in l.lower()]
            if fraud_containers:
                print(f"   Active containers: {len(fraud_containers)}")
                for container in fraud_containers:
                    parts = container.split()
                    if len(parts) >= 2:
                        print(f"     - {parts[-1]}")
            return True
        else:
            print("❌ Docker: NOT RUNNING")
            print("   Fix: Start Docker Desktop")
            return False
    except FileNotFoundError:
        print("❌ Docker: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"❌ Docker: ERROR - {e}")
        return False

def check_xgboost_model():
    """Check if XGBoost model file exists"""
    import os
    model_path = "artifacts/xgboost_unchained.json"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ XGBoost Model: FOUND ({size_mb:.2f} MB)")
        return True
    else:
        print(f"❌ XGBoost Model: NOT FOUND")
        print(f"   Expected: {model_path}")
        return False

def check_packages():
    """Check if required Python packages are installed"""
    packages = {
        'xgboost': 'XGBoost',
        'shap': 'SHAP',
        'confluent_kafka': 'Kafka',
        'redis': 'Redis',
        'pandas': 'Pandas'
    }
    
    all_installed = True
    for module_name, display_name in packages.items():
        try:
            __import__(module_name)
            print(f"✅ {display_name}: Installed")
        except ImportError:
            print(f"❌ {display_name}: NOT INSTALLED")
            print(f"   Fix: pip install {module_name.replace('_', '-')}")
            all_installed = False
    
    return all_installed

def main():
    """Run all health checks"""
    print("\n" + "="*60)
    print("🔍 FRAUD DETECTION PIPELINE - SERVICE HEALTH CHECK")
    print("="*60 + "\n")
    
    print("📦 Python Packages:")
    packages_ok = check_packages()
    print()
    
    print("🐳 Docker:")
    docker_ok = check_docker()
    print()
    
    print("🔌 Services:")
    redis_ok = check_redis()
    kafka_ok = check_kafka()
    postgres_ok = check_postgres()
    print()
    
    print("📁 Artifacts:")
    model_ok = check_xgboost_model()
    print()
    
    # Summary
    print("="*60)
    print("📊 HEALTH CHECK SUMMARY")
    print("="*60)
    
    checks = {
        'Python Packages': packages_ok,
        'Docker': docker_ok,
        'Redis': redis_ok,
        'Kafka': kafka_ok,
        'PostgreSQL': postgres_ok,
        'XGBoost Model': model_ok
    }
    
    for check_name, status in checks.items():
        emoji = "✅" if status else "❌"
        status_text = "PASS" if status else "FAIL"
        print(f"{emoji} {check_name}: {status_text}")
    
    print("="*60)
    
    # Overall status
    all_ok = all(checks.values())
    if all_ok:
        print("\n🎉 ALL CHECKS PASSED! Ready to run the pipeline.")
        print("\nNext steps:")
        print("1. Terminal 1: python streaming/stream_processor_xgboost.py")
        print("2. Terminal 2: python streaming/kafka_producer.py")
        return 0
    else:
        print("\n⚠️  SOME CHECKS FAILED. Fix the issues above before running the pipeline.")
        
        # Provide specific guidance
        if not docker_ok:
            print("\n💡 Start Docker Desktop first!")
        elif not kafka_ok or not redis_ok or not postgres_ok:
            print("\n💡 Run: docker-compose up -d")
            print("   Then wait 30 seconds and try again.")
        elif not packages_ok:
            print("\n💡 Activate venv and install packages:")
            print("   .\.venv\Scripts\activate")
            print("   pip install xgboost shap confluent-kafka redis pandas")
        
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Health check interrupted by user")
        sys.exit(1)
