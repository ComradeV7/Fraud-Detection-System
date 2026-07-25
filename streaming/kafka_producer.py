"""
Kafka Producer: Stream Simulator
Generates synthetic fraud transactions and publishes to Kafka
"""
import json
import time
import uuid
import random
from datetime import datetime, UTC
from confluent_kafka import Producer
from typing import Dict, Any

class TransactionStreamSimulator:
    """Generates and streams synthetic transactions to Kafka"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        """Initialize Kafka producer"""
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'fraud-stream-simulator'
        })
        self.topic = 'transactions.raw'
        
    def generate_transaction(self, fraud_probability: float = 0.1) -> Dict[str, Any]:
        """Generate a synthetic transaction with all 31 features from Base.csv"""
        
        is_fraud = random.random() < fraud_probability
        
        # Base transaction metadata (not ML features)
        transaction = {
            'transaction_id': f"TXN_{uuid.uuid4().hex[:12].upper()}",
            'user_id': f"USER_{random.randint(1000, 9999)}",
            'timestamp': datetime.now(UTC).isoformat(),
            'device_id': f"DEV_{random.randint(100, 999)}",
            'session_id': f"SESSION_{uuid.uuid4().hex[:8]}",
            'merchant_category': random.choice(['retail', 'grocery', 'gas', 'restaurant', 'online']),
            'currency': 'USD',
            'ip_address': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }
        
        # ALL 31 FEATURES FROM BASE.CSV (matching XGBoost training data)
        if is_fraud:
            # Fraud pattern: high risk indicators
            transaction['income'] = random.uniform(0.3, 0.6)  # Low income
            transaction['name_email_similarity'] = random.uniform(0.1, 0.4)  # Mismatched
            transaction['prev_address_months_count'] = random.randint(0, 3)  # Recent move
            transaction['current_address_months_count'] = random.randint(0, 6)  # Unstable
            transaction['customer_age'] = random.randint(18, 25)  # Young
            transaction['days_since_request'] = random.uniform(0, 2)  # Immediate
            transaction['intended_balcon_amount'] = random.uniform(500, 5000)  # High amount
            transaction['payment_type'] = random.choice(['AB', 'AC', 'AD'])  # Risky payment
            transaction['zip_count_4w'] = random.randint(5, 15)  # Multiple locations
            transaction['velocity_6h'] = random.uniform(15000, 25000)  # Impossible velocity
            transaction['velocity_24h'] = random.uniform(30000, 50000)
            transaction['velocity_4w'] = random.uniform(100000, 200000)
            transaction['bank_branch_count_8w'] = random.randint(5, 10)  # Multiple banks
            transaction['date_of_birth_distinct_emails_4w'] = random.randint(3, 8)  # Multiple emails
            transaction['employment_status'] = random.choice(['CB', 'CC', 'CD'])  # Unemployed
            transaction['credit_risk_score'] = random.randint(200, 400)  # Poor credit
            transaction['email_is_free'] = 1  # Free email provider
            transaction['housing_status'] = random.choice(['BB', 'BC', 'BD'])  # Unstable housing
            transaction['phone_home_valid'] = 0  # No home phone
            transaction['phone_mobile_valid'] = random.choice([0, 1])  # Sometimes invalid
            transaction['bank_months_count'] = random.randint(0, 6)  # New bank account
            transaction['has_other_cards'] = 0  # No credit history
            transaction['proposed_credit_limit'] = random.uniform(5000, 15000)  # High request
            transaction['foreign_request'] = 1  # Foreign transaction
            transaction['source'] = random.choice(['INTERNET', 'TELEAPP'])  # Remote sources
            transaction['session_length_in_minutes'] = random.uniform(0.5, 3)  # Quick session
            transaction['device_os'] = random.choice(['linux', 'other'])  # Less common OS
            transaction['keep_alive_session'] = 0  # No persistent session
            transaction['device_distinct_emails_8w'] = random.randint(5, 15)  # Device used by many
            transaction['device_fraud_count'] = random.randint(3, 10)  # Known fraud device
            transaction['month'] = datetime.now(UTC).month
        else:
            # Legitimate pattern: normal behavior
            transaction['income'] = random.uniform(0.7, 0.95)  # Good income
            transaction['name_email_similarity'] = random.uniform(0.85, 0.99)  # Matched
            transaction['prev_address_months_count'] = random.randint(12, 120)  # Stable
            transaction['current_address_months_count'] = random.randint(12, 120)  # Stable
            transaction['customer_age'] = random.randint(30, 65)  # Mature
            transaction['days_since_request'] = random.uniform(5, 30)  # Normal timing
            transaction['intended_balcon_amount'] = random.uniform(10, 300)  # Normal amount
            transaction['payment_type'] = random.choice(['AA', 'AB'])  # Common payment
            transaction['zip_count_4w'] = random.randint(1, 3)  # Consistent location
            transaction['velocity_6h'] = random.uniform(2, 8)  # Normal velocity
            transaction['velocity_24h'] = random.uniform(5, 20)
            transaction['velocity_4w'] = random.uniform(50, 200)
            transaction['bank_branch_count_8w'] = random.randint(0, 2)  # Stable banking
            transaction['date_of_birth_distinct_emails_4w'] = random.randint(1, 2)  # Consistent email
            transaction['employment_status'] = random.choice(['CA', 'CB'])  # Employed
            transaction['credit_risk_score'] = random.randint(600, 850)  # Good credit
            transaction['email_is_free'] = random.choice([0, 1])  # Mixed
            transaction['housing_status'] = random.choice(['BA', 'BB'])  # Stable housing
            transaction['phone_home_valid'] = 1  # Valid home phone
            transaction['phone_mobile_valid'] = 1  # Valid mobile
            transaction['bank_months_count'] = random.randint(24, 120)  # Established account
            transaction['has_other_cards'] = 1  # Credit history exists
            transaction['proposed_credit_limit'] = random.uniform(1000, 5000)  # Reasonable request
            transaction['foreign_request'] = 0  # Domestic
            transaction['source'] = random.choice(['INTERNET', 'MANUAL'])  # Common sources
            transaction['session_length_in_minutes'] = random.uniform(5, 30)  # Normal session
            transaction['device_os'] = random.choice(['windows', 'macintosh', 'linux', 'x11'])  # Common OS
            transaction['keep_alive_session'] = 1  # Persistent session
            transaction['device_distinct_emails_8w'] = random.randint(1, 2)  # Personal device
            transaction['device_fraud_count'] = 0  # Clean device
            transaction['month'] = datetime.now(UTC).month
        
        # Add ground truth for validation
        transaction['_ground_truth_is_fraud'] = is_fraud
        
        return transaction
    
    def delivery_report(self, err, msg):
        """Callback for message delivery confirmation"""
        if err is not None:
            print(f'❌ Message delivery failed: {err}')
        else:
            print(f'✅ Transaction {msg.key().decode("utf-8")} → Kafka partition {msg.partition()}')
    
    def produce_stream(self, 
                      num_transactions: int = 100, 
                      rate_per_second: int = 10,
                      fraud_rate: float = 0.1):
        """
        Generate and stream transactions to Kafka
        
        Args:
            num_transactions: Total number of transactions to generate
            rate_per_second: Transactions per second
            fraud_rate: Probability of generating a fraudulent transaction
        """
        print(f"🚀 Starting transaction stream...")
        print(f"   Target: {num_transactions} transactions at {rate_per_second} TPS")
        print(f"   Fraud rate: {fraud_rate * 100}%")
        print(f"   Topic: {self.topic}\n")
        
        interval = 1.0 / rate_per_second
        
        for i in range(num_transactions):
            # Generate transaction
            transaction = self.generate_transaction(fraud_probability=fraud_rate)
            
            # Serialize to JSON
            message = json.dumps(transaction).encode('utf-8')
            key = transaction['transaction_id'].encode('utf-8')
            
            # Publish to Kafka
            self.producer.produce(
                topic=self.topic,
                key=key,
                value=message,
                callback=self.delivery_report
            )
            
            # Poll to handle delivery callbacks
            self.producer.poll(0)
            
            # Rate limiting
            time.sleep(interval)
            
            if (i + 1) % 10 == 0:
                print(f"📊 Progress: {i + 1}/{num_transactions} transactions sent")
        
        # Wait for all messages to be delivered
        print("\n⏳ Flushing remaining messages...")
        self.producer.flush()
        print("✅ Stream complete!")
    
    def close(self):
        """Cleanup producer resources"""
        self.producer.flush()


if __name__ == '__main__':
    # Example usage
    simulator = TransactionStreamSimulator()
    
    try:
        # Stream 100 transactions at 10 TPS with 15% fraud rate
        simulator.produce_stream(
            num_transactions=100,
            rate_per_second=10,
            fraud_rate=0.15
        )
    except KeyboardInterrupt:
        print("\n🛑 Stream interrupted by user")
    finally:
        simulator.close()
