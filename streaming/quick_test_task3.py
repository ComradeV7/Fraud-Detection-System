"""
Quick Test Task 3: Automated Database Consumer Validation
Runs a simplified test without requiring manual terminal coordination
"""
import json
import time
import psycopg2
from datetime import datetime, UTC
from confluent_kafka import Producer, Consumer, KafkaError

class QuickTask3Tester:
    """Automated test for database consumer"""
    
    def __init__(self):
        """Initialize connections"""
        print("📦 Initializing test environment...")
        
        # Kafka Producer (for sending scored transactions)
        self.producer = Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'quick-task3-tester'
        })
        
        # Kafka Consumer (to verify messages were published)
        self.consumer = Consumer({
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'quick-task3-test-group',
            'auto.offset.reset': 'latest'
        })
        self.consumer.subscribe(['transactions.scored'])
        
        # PostgreSQL connection
        try:
            self.conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='fraud_detection',
                user='fraud_user',
                password='fraud_password'
            )
            self.cursor = self.conn.cursor()
            print("✅ Connected to PostgreSQL")
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    
    def check_infrastructure(self):
        """Verify all services are running"""
        print("\n🔍 Checking infrastructure...\n")
        
        checks = {
            'PostgreSQL': False,
            'Kafka Topics': False,
            'Database Schema': False
        }
        
        # Check PostgreSQL
        try:
            self.cursor.execute("SELECT 1")
            checks['PostgreSQL'] = True
            print("✅ PostgreSQL: Running")
        except Exception as e:
            print(f"❌ PostgreSQL: Not responding ({e})")
        
        # Check Kafka topics
        try:
            # Try to poll to verify connection
            msg = self.consumer.poll(timeout=1.0)
            checks['Kafka Topics'] = True
            print("✅ Kafka Topics: Available")
        except Exception as e:
            print(f"❌ Kafka Topics: Error ({e})")
        
        # Check database schema
        try:
            self.cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name IN ('transactions', 'fraud_decisions')
            """)
            count = self.cursor.fetchone()[0]
            if count == 2:
                checks['Database Schema'] = True
                print("✅ Database Schema: Complete")
            else:
                print(f"❌ Database Schema: Missing tables (found {count}/2)")
        except Exception as e:
            print(f"❌ Database Schema: Error ({e})")
        
        all_ready = all(checks.values())
        if not all_ready:
            print("\n⚠️  Some infrastructure components are not ready!")
            return False
        
        print("\n✅ All infrastructure components ready!\n")
        return True
    
    def get_current_record_count(self):
        """Get current count of records in database"""
        self.cursor.execute("SELECT COUNT(*) FROM fraud_decisions")
        return self.cursor.fetchone()[0]
    
    def generate_scored_transaction(self, txn_id: str, is_fraud: bool):
        """Generate a complete scored transaction"""
        return {
            'transaction_id': txn_id,
            'user_id': f"USER_QUICK_{txn_id[-3:]}",
            'intended_balcon_amount': 2500.00 if is_fraud else 125.50,
            'timestamp': datetime.now(UTC).isoformat(),
            'device_id': 'DEV_QUICK_TEST',
            'session_id': 'SESSION_QUICK',
            'merchant_category': 'online',
            'currency': 'USD',
            'ip_address': '10.0.0.1',
            
            # Features (subset for testing)
            'income': 0.4 if is_fraud else 0.85,
            'credit_risk_score': 350 if is_fraud else 720,
            'proposed_credit_limit': 15000 if is_fraud else 3500,
            'has_other_cards': 0 if is_fraud else 1,
            'device_fraud_count': 7 if is_fraud else 0,
            
            # ML results
            'fraud_score': 0.8912 if is_fraud else 0.0834,
            'is_fraud': is_fraud,
            'model_version': 'v2.0-xgboost-unchained-test',
            'inference_time_ms': 2.1,
            
            # SHAP explanation
            'top_features': ['proposed_credit_limit', 'has_other_cards', 'device_fraud_count'],
            'shap_values': {
                'proposed_credit_limit': 0.52 if is_fraud else 0.12,
                'has_other_cards': -0.31 if is_fraud else 0.28,
                'device_fraud_count': 0.41 if is_fraud else 0.0
            },
            'rules_fired': [
                f"{'High risk detected' if is_fraud else 'Low risk confirmed'} based on credit limit and history"
            ],
            'confidence_level': 0.8912 if is_fraud else 0.9166,
            'explanation_method': 'SHAP TreeExplainer',
            'processed_at': datetime.now(UTC).isoformat()
        }
    
    def send_test_transactions(self):
        """Send test transactions to Kafka"""
        print("📤 Sending test transactions to Kafka...\n")
        
        test_txns = [
            ('QUICK_TEST_FRAUD_01', True, 'High-risk fraud'),
            ('QUICK_TEST_LEGIT_01', False, 'Legitimate transaction'),
            ('QUICK_TEST_FRAUD_02', True, 'Another fraud case'),
            ('QUICK_TEST_LEGIT_02', False, 'Another legitimate case'),
            ('QUICK_TEST_FRAUD_03', True, 'Third fraud case'),
        ]
        
        for txn_id, is_fraud, desc in test_txns:
            scored_txn = self.generate_scored_transaction(txn_id, is_fraud)
            
            self.producer.produce(
                topic='transactions.scored',
                key=txn_id.encode('utf-8'),
                value=json.dumps(scored_txn).encode('utf-8')
            )
            
            emoji = "🚨" if is_fraud else "✅"
            print(f"{emoji} Sent {txn_id}: {desc}")
        
        self.producer.flush()
        print(f"\n✅ Sent {len(test_txns)} transactions to Kafka\n")
        return len(test_txns)
    
    def verify_persistence(self, expected_new_records: int):
        """Verify transactions were persisted"""
        print("⏳ Waiting 8 seconds for db_consumer to process...\n")
        time.sleep(8)
        
        print("🔍 Verifying persistence in PostgreSQL...\n")
        
        # Check recent transactions
        self.cursor.execute("""
            SELECT t.transaction_id, t.amount, fd.fraud_score, fd.is_fraud
            FROM transactions t
            JOIN fraud_decisions fd ON t.transaction_id = fd.transaction_id
            WHERE t.transaction_id LIKE 'QUICK_TEST_%'
            ORDER BY t.created_at DESC
        """)
        
        results = self.cursor.fetchall()
        
        if not results:
            print("❌ No test transactions found in database!")
            print("⚠️  Is db_consumer.py running?")
            return False
        
        print(f"✅ Found {len(results)} test transactions in database:\n")
        
        for txn_id, amount, score, is_fraud in results:
            emoji = "🚨" if is_fraud else "✅"
            print(f"{emoji} {txn_id}")
            print(f"   └─ Amount: ${amount:.2f}")
            print(f"   └─ Score: {score:.4f}")
            print(f"   └─ Fraud: {is_fraud}")
        
        return len(results) >= expected_new_records
    
    def get_statistics(self):
        """Get database statistics"""
        print("\n" + "="*60)
        print("📊 DATABASE STATISTICS")
        print("="*60 + "\n")
        
        # Total counts
        self.cursor.execute("SELECT COUNT(*) FROM transactions")
        total_txns = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM fraud_decisions")
        total_decisions = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM fraud_decisions WHERE is_fraud = true")
        total_fraud = self.cursor.fetchone()[0]
        
        # Recent counts (last 5 minutes)
        self.cursor.execute("""
            SELECT COUNT(*) FROM fraud_decisions 
            WHERE created_at >= NOW() - INTERVAL '5 minutes'
        """)
        recent_decisions = self.cursor.fetchone()[0]
        
        print(f"Total Transactions: {total_txns}")
        print(f"Total Fraud Decisions: {total_decisions}")
        print(f"Fraud Count: {total_fraud} ({total_fraud/total_decisions*100:.1f}%)" if total_decisions > 0 else "Fraud Count: 0")
        print(f"Legitimate Count: {total_decisions - total_fraud}")
        print(f"Recent (5 min): {recent_decisions}")
        
        # Average processing time
        self.cursor.execute("""
            SELECT AVG(processing_time_ms), MIN(processing_time_ms), MAX(processing_time_ms)
            FROM fraud_decisions
            WHERE processing_time_ms > 0
        """)
        avg_time, min_time, max_time = self.cursor.fetchone()
        
        if avg_time:
            print(f"\nProcessing Time (ms):")
            print(f"  Average: {avg_time:.2f}")
            print(f"  Min: {min_time:.2f}")
            print(f"  Max: {max_time:.2f}")
    
    def run_test(self):
        """Run complete test"""
        print("\n" + "="*60)
        print("🧪 QUICK TASK 3 TEST - DATABASE CONSUMER")
        print("="*60 + "\n")
        
        # Step 1: Check infrastructure
        if not self.check_infrastructure():
            print("\n❌ Infrastructure check failed. Fix issues and try again.")
            return False
        
        # Step 2: Get initial count
        initial_count = self.get_current_record_count()
        print(f"📊 Current database records: {initial_count}\n")
        
        # Step 3: Send test transactions
        sent_count = self.send_test_transactions()
        
        # Step 4: Verify persistence
        success = self.verify_persistence(sent_count)
        
        # Step 5: Show statistics
        self.get_statistics()
        
        # Final result
        print("\n" + "="*60)
        print("🎯 TEST RESULT")
        print("="*60)
        
        if success:
            print("✅ SUCCESS: Database consumer is working correctly!")
            print("   All test transactions were persisted to PostgreSQL.")
            print("\n💡 Next steps:")
            print("   1. Run full pipeline test (producer → processor → consumer)")
            print("   2. Check dashboard queries")
            print("   3. Test error handling")
        else:
            print("❌ FAILURE: Test transactions not found in database")
            print("\n🔧 Troubleshooting:")
            print("   1. Is db_consumer.py running?")
            print("      → python streaming/db_consumer.py")
            print("   2. Check db_consumer console for errors")
            print("   3. Verify Kafka is processing messages")
        
        print("="*60 + "\n")
        
        return success
    
    def cleanup(self):
        """Close connections"""
        self.consumer.close()
        self.cursor.close()
        self.conn.close()


if __name__ == '__main__':
    print("\n" + "🚨"*30)
    print("IMPORTANT: Start db_consumer.py BEFORE running this test!")
    print("   Terminal 1: python streaming/db_consumer.py")
    print("   Terminal 2: python streaming/quick_test_task3.py")
    print("🚨"*30 + "\n")
    
    response = input("Is db_consumer.py running? (y/n): ").strip().lower()
    
    if response != 'y':
        print("\n⚠️  Please start db_consumer.py first, then run this test again.")
        exit(0)
    
    tester = QuickTask3Tester()
    try:
        success = tester.run_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        tester.cleanup()
