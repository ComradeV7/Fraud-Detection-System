"""
Test Database Consumer: Validate end-to-end persistence
Tests that scored transactions are correctly persisted to PostgreSQL
"""
import json
import time
import psycopg2
from datetime import datetime, UTC
from confluent_kafka import Producer

class DatabaseConsumerTester:
    """Test database persistence with synthetic scored transactions"""
    
    def __init__(self):
        """Initialize Kafka producer and PostgreSQL connection"""
        self.producer = Producer({
            'bootstrap.servers': 'localhost:9092',
            'client.id': 'db-consumer-tester'
        })
        
        self.conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='fraud_detection',
            user='fraud_user',
            password='fraud_password'
        )
        self.cursor = self.conn.cursor()
    
    def generate_scored_transaction(self, txn_id: str, is_fraud: bool = False):
        """Generate a synthetic scored transaction"""
        return {
            # Transaction metadata
            'transaction_id': txn_id,
            'user_id': f"USER_{txn_id[:4]}",
            'intended_balcon_amount': 1234.56 if is_fraud else 99.99,
            'timestamp': datetime.now(UTC).isoformat(),
            'device_id': 'DEV_TEST_001',
            'session_id': 'SESSION_TEST',
            'merchant_category': 'retail',
            'currency': 'USD',
            'ip_address': '192.168.1.100',
            
            # All 31 features (required for complete payload)
            'income': 0.5 if is_fraud else 0.9,
            'name_email_similarity': 0.3 if is_fraud else 0.95,
            'prev_address_months_count': 3 if is_fraud else 24,
            'current_address_months_count': 6 if is_fraud else 36,
            'customer_age': 22 if is_fraud else 45,
            'days_since_request': 1 if is_fraud else 14,
            'payment_type': 'AC' if is_fraud else 'AA',
            'zip_count_4w': 10 if is_fraud else 2,
            'velocity_6h': 20000 if is_fraud else 5,
            'velocity_24h': 40000 if is_fraud else 15,
            'velocity_4w': 150000 if is_fraud else 100,
            'bank_branch_count_8w': 8 if is_fraud else 1,
            'date_of_birth_distinct_emails_4w': 5 if is_fraud else 1,
            'employment_status': 'CD' if is_fraud else 'CA',
            'credit_risk_score': 300 if is_fraud else 750,
            'email_is_free': 1,
            'housing_status': 'BD' if is_fraud else 'BA',
            'phone_home_valid': 0 if is_fraud else 1,
            'phone_mobile_valid': 1,
            'bank_months_count': 3 if is_fraud else 48,
            'has_other_cards': 0 if is_fraud else 1,
            'proposed_credit_limit': 12000 if is_fraud else 3000,
            'foreign_request': 1 if is_fraud else 0,
            'source': 'TELEAPP' if is_fraud else 'INTERNET',
            'session_length_in_minutes': 2 if is_fraud else 15,
            'device_os': 'linux' if is_fraud else 'windows',
            'keep_alive_session': 0 if is_fraud else 1,
            'device_distinct_emails_8w': 12 if is_fraud else 1,
            'device_fraud_count': 5 if is_fraud else 0,
            'month': datetime.now(UTC).month,
            
            # ML prediction results
            'fraud_score': 0.8542 if is_fraud else 0.1234,
            'is_fraud': is_fraud,
            'model_version': 'v2.0-xgboost-unchained',
            'inference_time_ms': 2.5,
            
            # SHAP explainability
            'top_features': ['proposed_credit_limit', 'income', 'credit_risk_score'],
            'shap_values': {
                'proposed_credit_limit': 0.45,
                'income': -0.18,
                'credit_risk_score': -0.23
            },
            'rules_fired': [
                f"{'Flagged as suspicious due to highly anomalous patterns' if is_fraud else 'Cleared as legitimate based on verified normal behavior'} detected in Proposed Credit Limit, Income, and Credit Risk Score."
            ],
            'confidence_level': 0.8542 if is_fraud else 0.8766,
            'explanation_method': 'SHAP TreeExplainer',
            
            # Processing metadata
            'processed_at': datetime.now(UTC).isoformat()
        }
    
    def send_scored_transaction(self, scored_txn: dict):
        """Send scored transaction to Kafka"""
        self.producer.produce(
            topic='transactions.scored',
            key=scored_txn['transaction_id'].encode('utf-8'),
            value=json.dumps(scored_txn).encode('utf-8')
        )
        self.producer.flush()
    
    def verify_in_database(self, txn_id: str) -> dict:
        """Verify transaction was persisted to PostgreSQL"""
        # Check transactions table
        self.cursor.execute("""
            SELECT transaction_id, user_id, amount, timestamp, 
                   merchant_category, currency
            FROM transactions 
            WHERE transaction_id = %s
        """, (txn_id,))
        txn_row = self.cursor.fetchone()
        
        # Check fraud_decisions table
        self.cursor.execute("""
            SELECT fraud_score, is_fraud, model_version, 
                   confidence_level, processing_time_ms
            FROM fraud_decisions 
            WHERE transaction_id = %s
        """, (txn_id,))
        fraud_row = self.cursor.fetchone()
        
        return {
            'transaction_found': txn_row is not None,
            'fraud_decision_found': fraud_row is not None,
            'transaction_data': txn_row,
            'fraud_data': fraud_row
        }
    
    def run_tests(self):
        """Run comprehensive database consumer tests"""
        print("\n" + "="*60)
        print("🧪 DATABASE CONSUMER TESTS")
        print("="*60 + "\n")
        
        test_cases = [
            {'txn_id': 'TEST_FRAUD_001', 'is_fraud': True, 'desc': 'High-risk fraud transaction'},
            {'txn_id': 'TEST_LEGIT_001', 'is_fraud': False, 'desc': 'Legitimate transaction'},
            {'txn_id': 'TEST_FRAUD_002', 'is_fraud': True, 'desc': 'Another fraud case'},
            {'txn_id': 'TEST_LEGIT_002', 'is_fraud': False, 'desc': 'Another legitimate case'},
        ]
        
        print("📤 Step 1: Sending test transactions to Kafka...\n")
        for test in test_cases:
            scored_txn = self.generate_scored_transaction(
                test['txn_id'], 
                test['is_fraud']
            )
            self.send_scored_transaction(scored_txn)
            
            emoji = "🚨" if test['is_fraud'] else "✅"
            print(f"{emoji} Sent {test['txn_id']}: {test['desc']}")
        
        print(f"\n⏳ Waiting 5 seconds for db_consumer to process...\n")
        time.sleep(5)
        
        print("🔍 Step 2: Verifying persistence in PostgreSQL...\n")
        passed = 0
        failed = 0
        
        for test in test_cases:
            result = self.verify_in_database(test['txn_id'])
            
            if result['transaction_found'] and result['fraud_decision_found']:
                print(f"✅ {test['txn_id']}: FOUND in database")
                print(f"   └─ Amount: ${result['transaction_data'][2]:.2f}")
                print(f"   └─ Fraud Score: {result['fraud_data'][0]:.4f}")
                print(f"   └─ Is Fraud: {result['fraud_data'][1]}")
                passed += 1
            else:
                print(f"❌ {test['txn_id']}: NOT FOUND in database")
                print(f"   └─ Transaction table: {'✓' if result['transaction_found'] else '✗'}")
                print(f"   └─ Fraud decisions table: {'✓' if result['fraud_decision_found'] else '✗'}")
                failed += 1
            print()
        
        # Summary statistics
        print("="*60)
        print("📊 DATABASE QUERY RESULTS")
        print("="*60 + "\n")
        
        # Total records
        self.cursor.execute("SELECT COUNT(*) FROM transactions")
        total_txns = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM fraud_decisions")
        total_decisions = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM fraud_decisions WHERE is_fraud = true")
        total_fraud = self.cursor.fetchone()[0]
        
        print(f"Total Transactions: {total_txns}")
        print(f"Total Fraud Decisions: {total_decisions}")
        print(f"Fraud Count: {total_fraud}")
        print(f"Legitimate Count: {total_decisions - total_fraud}")
        
        # Test summary
        print("\n" + "="*60)
        print("🎯 TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {passed}/{len(test_cases)}")
        print(f"❌ Failed: {failed}/{len(test_cases)}")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! Database consumer is working correctly.")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Check that db_consumer.py is running.")
        
        print("="*60 + "\n")
    
    def cleanup(self):
        """Close connections"""
        self.cursor.close()
        self.conn.close()


if __name__ == '__main__':
    print("\n🚨 IMPORTANT: Make sure db_consumer.py is running before starting tests!")
    print("   Start it with: python streaming/db_consumer.py\n")
    
    input("Press Enter to start tests (or Ctrl+C to cancel)...")
    
    tester = DatabaseConsumerTester()
    try:
        tester.run_tests()
    finally:
        tester.cleanup()
