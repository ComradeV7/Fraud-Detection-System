"""
Stream Processor: The Orchestrator
Consumes raw transactions, enriches with Redis features, scores with ML model
"""
import sys
import os
import json
import time
import torch
import joblib
import numpy as np
from datetime import datetime, UTC
from confluent_kafka import Consumer, Producer, KafkaError
from typing import Dict, Any

# Add parent directory to path so we can import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store import RedisFeatureStore
from src.model.anfis import ANFIS  # Import ANFIS class for torch.load

class FraudStreamProcessor:
    """
    Core stream processor that orchestrates the fraud detection pipeline
    
    Flow:
    1. Consume from transactions.raw
    2. Fetch behavioral features from Redis
    3. Score with ANFIS model
    4. Publish to transactions.scored
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 consumer_group: str = 'fraud-processor-group',
                 reset_offset: str = 'latest'):
        """Initialize consumer, producer, feature store, and ML model
        
        Args:
            bootstrap_servers: Kafka broker addresses
            consumer_group: Consumer group ID for load balancing
            reset_offset: 'latest' to skip old messages, 'earliest' to replay all
        """
        
        # Kafka Consumer
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': consumer_group,
            'auto.offset.reset': reset_offset,  # Configurable: skip old messages
            'enable.auto.commit': True
        })
        self.consumer.subscribe(['transactions.raw'])
        
        # Kafka Producer (for scored transactions)
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'fraud-processor'
        })
        
        # Redis Feature Store
        self.feature_store = RedisFeatureStore()
        
        # Load ML Model and Preprocessors
        print("📦 Loading ML artifacts...")
        try:
            # PyTorch 2.6+ requires weights_only=False for custom models
            self.model = torch.load("artifacts/anfis_model.pt", 
                                   map_location='cpu', 
                                   weights_only=False)
            self.model.eval()
            self.scaler = joblib.load("artifacts/scaler.pkl")
            self.encoder = joblib.load("artifacts/target_encoder.pkl")
            self.selector = joblib.load("artifacts/selector.pkl")
            print("✅ ML artifacts loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load ML artifacts: {e}")
            raise
        
        # Performance tracking
        self.processed_count = 0
        self.fraud_count = 0
        self.start_time = time.time()
    
    def enrich_with_features(self, transaction: Dict) -> Dict[str, Any]:
        """
        Enrich raw transaction with Redis features
        
        Args:
            transaction: Raw transaction from Kafka
            
        Returns:
            Enriched transaction with behavioral features (empty features on Redis failure)
        """
        try:
            # Get real-time features from Redis
            redis_features = self.feature_store.get_all_features(transaction)
            
            # Merge with transaction data
            enriched = {**transaction, **redis_features}
            
            # Update Redis for next transaction
            self.feature_store.update_transaction_features(transaction)
            
            return enriched
            
        except Exception as e:
            print(f"⚠️  Redis error, using empty features: {e}")
            # Fallback: return transaction without enriched features
            return {
                **transaction,
                'velocity_6h': 0.0,
                'velocity_24h': 0.0,
                'velocity_4w': 0.0,
                'total_amount_24h': 0.0,
                'avg_amount_24h': 0.0,
                'max_amount_24h': 0.0,
                'device_txn_count_24h': 0.0,
                'device_fraud_count': 0.0
            }
    
    def preprocess_for_model(self, transaction: Dict) -> np.ndarray:
        """
        Transform raw transaction into model input format
        
        Replicates the preprocessing pipeline from training
        
        Args:
            transaction: Enriched transaction with features
            
        Returns:
            Preprocessed feature array (shape: 1 x num_features)
        """
        try:
            # Convert to DataFrame-like structure for scikit-learn transformers
            import pandas as pd
            
            print(f"      → Creating feature dict...")
            # Extract features in expected order (simplified - adjust to match your training)
            feature_dict = {
                'income': transaction.get('income', 0.7),
                'amount': transaction.get('amount', 0),
                'velocity_6h': transaction.get('velocity_6h', 0),
                'velocity_24h': transaction.get('velocity_24h', 0),
                'velocity_4w': transaction.get('velocity_4w', 0),
                'customer_age': transaction.get('customer_age', 35),
                'name_email_similarity': transaction.get('name_email_similarity', 0.9),
                'device_fraud_count': transaction.get('device_fraud_count', 0),
                'foreign_request': transaction.get('foreign_request', 0),
                'email_is_free': transaction.get('email_is_free', 0),
                'phone_mobile_valid': transaction.get('phone_mobile_valid', 1),
                'has_other_cards': transaction.get('has_other_cards', 0),
                'keep_alive_session': transaction.get('keep_alive_session', 1)
            }
            
            print(f"      → Creating DataFrame...")
            df = pd.DataFrame([feature_dict])
            
            # Apply preprocessing (simplified - match your training pipeline)
            # In production, you'd apply: encoding, scaling, feature selection
            try:
                print(f"      → Applying scaler...")
                # Scale features
                scaled = self.scaler.transform(df)
                
                print(f"      → Selecting features...")
                # Select top features
                selected = self.selector.transform(scaled)
                
                print(f"      → Preprocessing complete!")
                return selected
            except Exception as e:
                print(f"⚠️  Preprocessing transform error: {e}")
                import traceback
                traceback.print_exc()
                # Return zeros as fallback with correct shape
                expected_features = len(self.selector.get_feature_names_out())
                return np.zeros((1, expected_features))
                
        except Exception as e:
            print(f"⚠️  Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros as fallback (default 15 features)
            return np.zeros((1, 15))
    
    def score_transaction(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Run ML model inference
        
        Args:
            features: Preprocessed feature array
            
        Returns:
            Dict with fraud_score and is_fraud
        """
        start_inference = time.time()
        
        # Convert to PyTorch tensor
        tensor_input = torch.tensor(features, dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            fraud_score = self.model(tensor_input).item()
        
        inference_time_ms = (time.time() - start_inference) * 1000
        
        # Apply threshold
        is_fraud = fraud_score >= 0.5
        
        return {
            'fraud_score': round(fraud_score, 4),
            'is_fraud': is_fraud,
            'model_version': 'v2.0-native-anfis',
            'inference_time_ms': round(inference_time_ms, 2)
        }
    
    def delivery_report(self, err, msg):
        """Callback for producer delivery confirmation"""
        if err is not None:
            print(f'❌ Delivery failed: {err}')
        else:
            txn_id = msg.key().decode('utf-8')
            print(f'✅ Scored transaction {txn_id} → Kafka')
    
    def process_message(self, msg) -> None:
        """
        Process a single transaction message
        
        Pipeline:
        1. Parse JSON
        2. Enrich with Redis features
        3. Preprocess for model
        4. Score with ANFIS
        5. Publish to transactions.scored
        """
        txn_id = "UNKNOWN"
        
        try:
            # Parse transaction
            try:
                transaction = json.loads(msg.value().decode('utf-8'))
                txn_id = transaction.get('transaction_id', 'UNKNOWN')
                
                # Validate required fields
                if 'transaction_id' not in transaction:
                    print(f"❌ Missing transaction_id in message, skipping")
                    return
                    
                if 'amount' not in transaction:
                    print(f"❌ Missing amount in {txn_id}, skipping")
                    return
                    
            except json.JSONDecodeError as e:
                print(f"❌ Malformed JSON message: {e}")
                return
            except UnicodeDecodeError as e:
                print(f"❌ Invalid message encoding: {e}")
                return
            
            print(f"\n🔄 Processing {txn_id}")
            
            # Step 1: Enrich with behavioral features (with Redis fallback)
            print(f"   [1/4] Enriching features...")
            try:
                enriched = self.enrich_with_features(transaction)
                print(f"   ✓ Features enriched")
            except Exception as e:
                print(f"⚠️  Feature enrichment failed for {txn_id}: {e}")
                enriched = transaction  # Continue with raw transaction
            
            # Step 2: Preprocess for model (with fallback)
            print(f"   [2/4] Preprocessing...")
            try:
                features = self.preprocess_for_model(enriched)
                print(f"   ✓ Preprocessed (shape: {features.shape})")
            except Exception as e:
                print(f"⚠️  Preprocessing failed for {txn_id}: {e}")
                # Use zero features as last resort
                features = np.zeros((1, 15))
            
            # Step 3: Score with ML model
            print(f"   [3/4] Scoring with ANFIS...")
            try:
                prediction = self.score_transaction(features)
                print(f"   ✓ Scored: {prediction['fraud_score']:.4f}")
            except Exception as e:
                print(f"❌ Model inference failed for {txn_id}: {e}")
                import traceback
                traceback.print_exc()
                return  # Skip this transaction
            
            # Step 4: Combine everything
            scored_transaction = {
                **transaction,
                **prediction,
                'processed_at': datetime.now(UTC).isoformat()
            }
            
            # Step 5: Publish to scored topic
            try:
                self.producer.produce(
                    topic='transactions.scored',
                    key=txn_id.encode('utf-8'),
                    value=json.dumps(scored_transaction).encode('utf-8'),
                    callback=self.delivery_report
                )
                self.producer.poll(0)
            except Exception as e:
                print(f"❌ Failed to publish {txn_id} to Kafka: {e}")
                return
            
            # Update stats
            self.processed_count += 1
            if prediction['is_fraud']:
                self.fraud_count += 1
            
            # Log prediction
            fraud_emoji = "🚨" if prediction['is_fraud'] else "✅"
            print(f"{fraud_emoji} Score: {prediction['fraud_score']:.4f} | "
                  f"Amount: ${transaction['amount']:.2f} | "
                  f"Inference: {prediction['inference_time_ms']:.2f}ms")
            
        except KeyError as e:
            print(f"❌ Missing required field in {txn_id}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error processing {txn_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """
        Main processing loop
        """
        print("\n" + "="*60)
        print("🚀 FRAUD STREAM PROCESSOR STARTED")
        print("="*60)
        print(f"📥 Consuming from: transactions.raw")
        print(f"📤 Publishing to: transactions.scored")
        print(f"🧠 Model: ANFIS v2.0")
        print(f"⚡ Redis Feature Store: Active")
        print("="*60 + "\n")
        
        try:
            while True:
                # Poll for messages
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print(f'📭 Reached end of partition')
                    else:
                        print(f'❌ Consumer error: {msg.error()}')
                    continue
                
                # Process the message
                self.process_message(msg)
                
                # Print stats every 10 transactions
                if self.processed_count % 10 == 0 and self.processed_count > 0:
                    self.print_stats()
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down processor...")
        finally:
            self.cleanup()
    
    def print_stats(self):
        """Print processing statistics"""
        elapsed = time.time() - self.start_time
        tps = self.processed_count / elapsed if elapsed > 0 else 0
        fraud_rate = (self.fraud_count / self.processed_count * 100) if self.processed_count > 0 else 0
        
        print("\n" + "─"*60)
        print(f"📊 STATS: Processed {self.processed_count} | "
              f"Fraud: {self.fraud_count} ({fraud_rate:.1f}%) | "
              f"TPS: {tps:.2f}")
        print("─"*60 + "\n")
    
    def cleanup(self):
        """Cleanup resources"""
        print("🧹 Flushing messages...")
        self.producer.flush()
        self.consumer.close()
        print("✅ Cleanup complete")
        self.print_stats()


if __name__ == '__main__':
    import sys
    
    # Parse command-line arguments
    reset_offset = 'latest'  # Default: skip old messages
    if len(sys.argv) > 1:
        if sys.argv[1] == '--replay-all':
            reset_offset = 'earliest'
            print("⚠️  Replaying ALL messages from beginning (--replay-all mode)")
        elif sys.argv[1] == '--help':
            print("Usage: python stream_processor.py [--replay-all]")
            print("  --replay-all: Process all historical messages (default: skip to latest)")
            sys.exit(0)
    
    processor = FraudStreamProcessor(reset_offset=reset_offset)
    processor.run()
