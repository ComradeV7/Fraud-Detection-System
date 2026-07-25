"""
Database Consumer: Persistence Service
Consumes scored transactions from Kafka and persists to PostgreSQL
"""
import sys
import os
import json
import time
import psycopg2
from psycopg2.extras import Json
from datetime import datetime, UTC
from confluent_kafka import Consumer, KafkaError
from typing import Dict, Any

class DatabaseConsumer:
    """
    Consumes scored transactions from Kafka and persists to PostgreSQL
    
    Flow:
    1. Consume from transactions.scored
    2. Insert into transactions table
    3. Insert into fraud_decisions table with explanations
    4. Commit to database
    5. Log persistence status
    """
    
    def __init__(self,
                 kafka_bootstrap: str = 'localhost:9092',
                 db_host: str = 'localhost',
                 db_port: int = 5432,
                 db_name: str = 'fraud_detection',
                 db_user: str = 'fraud_user',
                 db_password: str = 'fraud_password',
                 consumer_group: str = 'db-persistence-group'):
        """Initialize Kafka consumer and PostgreSQL connection"""
        
        print("📦 Initializing Database Consumer...")
        
        # Kafka Consumer
        self.consumer = Consumer({
            'bootstrap.servers': kafka_bootstrap,
            'group.id': consumer_group,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True
        })
        self.consumer.subscribe(['transactions.scored'])
        
        # PostgreSQL Connection
        try:
            self.conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password
            )
            self.cursor = self.conn.cursor()
            print(f"✅ Connected to PostgreSQL: {db_host}:{db_port}/{db_name}")
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
        
        # Performance tracking
        self.persisted_count = 0
        self.fraud_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def persist_transaction(self, scored_transaction: Dict[str, Any]) -> bool:
        """
        Persist scored transaction to PostgreSQL
        
        Args:
            scored_transaction: Scored transaction with fraud prediction and explanation
            
        Returns:
            True if successful, False otherwise
        """
        try:
            txn_id = scored_transaction['transaction_id']
            
            # 1. Insert into transactions table
            self.cursor.execute("""
                INSERT INTO transactions (
                    transaction_id, user_id, amount, timestamp, 
                    device_id, session_id, merchant_category, currency, 
                    ip_address, raw_payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
            """, (
                txn_id,
                scored_transaction.get('user_id', 'unknown'),
                scored_transaction.get('intended_balcon_amount', 0.0),  # Use intended_balcon_amount as amount
                scored_transaction.get('timestamp', datetime.now(UTC).isoformat()),
                scored_transaction.get('device_id'),
                scored_transaction.get('session_id'),
                scored_transaction.get('merchant_category'),
                scored_transaction.get('currency', 'USD'),
                scored_transaction.get('ip_address'),
                Json(scored_transaction)  # Store full payload as JSONB
            ))
            
            # 2. Insert into fraud_decisions table
            self.cursor.execute("""
                INSERT INTO fraud_decisions (
                    transaction_id, fraud_score, is_fraud, model_version,
                    features_used, confidence_level, rules_triggered, processing_time_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                txn_id,
                scored_transaction.get('fraud_score', 0.0),
                scored_transaction.get('is_fraud', False),
                scored_transaction.get('model_version', 'unknown'),
                Json(scored_transaction.get('top_features', [])),  # SHAP top features
                scored_transaction.get('confidence_level', 0.0),
                scored_transaction.get('rules_fired', []),  # SHAP explanations
                scored_transaction.get('inference_time_ms', 0)
            ))
            
            # 3. Commit transaction
            self.conn.commit()
            
            return True
            
        except psycopg2.Error as e:
            print(f"❌ Database error for {txn_id}: {e}")
            self.conn.rollback()
            self.error_count += 1
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error persisting {txn_id}: {e}")
            self.conn.rollback()
            self.error_count += 1
            return False
    
    def process_message(self, msg) -> None:
        """
        Process a single Kafka message
        
        Pipeline:
        1. Parse JSON
        2. Persist to PostgreSQL
        3. Log status
        """
        try:
            # Parse scored transaction
            scored_transaction = json.loads(msg.value().decode('utf-8'))
            txn_id = scored_transaction.get('transaction_id', 'UNKNOWN')
            
            # Persist to database
            success = self.persist_transaction(scored_transaction)
            
            if success:
                self.persisted_count += 1
                
                # Track fraud count
                if scored_transaction.get('is_fraud', False):
                    self.fraud_count += 1
                
                # Log status
                fraud_emoji = "🚨" if scored_transaction.get('is_fraud') else "✅"
                score = scored_transaction.get('fraud_score', 0.0)
                amount = scored_transaction.get('intended_balcon_amount', 0.0)
                
                print(f"{fraud_emoji} Persisted {txn_id} | Score: {score:.4f} | Amount: ${amount:.2f}")
                
                # Print stats every 10 transactions
                if self.persisted_count % 10 == 0:
                    self.print_stats()
            
        except json.JSONDecodeError as e:
            print(f"❌ Malformed JSON message: {e}")
            self.error_count += 1
        except Exception as e:
            print(f"❌ Unexpected error processing message: {e}")
            self.error_count += 1
    
    def run(self):
        """
        Main processing loop
        """
        print("\n" + "="*60)
        print("🗄️  DATABASE CONSUMER STARTED")
        print("="*60)
        print(f"📥 Consuming from: transactions.scored")
        print(f"💾 Writing to: PostgreSQL (fraud_detection)")
        print(f"📊 Tables: transactions, fraud_decisions")
        print("="*60 + "\n")
        
        try:
            while True:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f'❌ Kafka error: {msg.error()}')
                    continue
                
                # Process the message
                self.process_message(msg)
                
        except KeyboardInterrupt:
            print("\n🛑 Shutting down consumer...")
        finally:
            self.cleanup()
    
    def print_stats(self):
        """Print persistence statistics"""
        elapsed = time.time() - self.start_time
        tps = self.persisted_count / elapsed if elapsed > 0 else 0
        fraud_rate = (self.fraud_count / self.persisted_count * 100) if self.persisted_count > 0 else 0
        
        print("\n" + "─"*60)
        print(f"📊 STATS: Persisted {self.persisted_count} | "
              f"Fraud: {self.fraud_count} ({fraud_rate:.1f}%) | "
              f"Errors: {self.error_count} | "
              f"TPS: {tps:.2f}")
        print("─"*60 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Closing connections...")
        
        # Commit any pending transactions
        try:
            self.conn.commit()
        except:
            pass
        
        # Close database
        self.cursor.close()
        self.conn.close()
        
        # Close Kafka consumer
        self.consumer.close()
        
        print("✅ Cleanup complete")
        self.print_stats()


if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Database Consumer for Fraud Detection')
    parser.add_argument('--kafka-bootstrap', default='localhost:9092',
                        help='Kafka bootstrap servers (default: localhost:9092)')
    parser.add_argument('--db-host', default='localhost',
                        help='PostgreSQL host (default: localhost)')
    parser.add_argument('--db-port', type=int, default=5432,
                        help='PostgreSQL port (default: 5432)')
    parser.add_argument('--db-name', default='fraud_detection',
                        help='Database name (default: fraud_detection)')
    parser.add_argument('--db-user', default='fraud_user',
                        help='Database user (default: fraud_user)')
    parser.add_argument('--db-password', default='fraud_password',
                        help='Database password (default: fraud_password)')
    
    args = parser.parse_args()
    
    # Start consumer
    consumer = DatabaseConsumer(
        kafka_bootstrap=args.kafka_bootstrap,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    consumer.run()
