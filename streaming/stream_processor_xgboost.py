"""
Stream Processor with XGBoost + SHAP: The Explainable Orchestrator
Consumes raw transactions, enriches with Redis features, scores with XGBoost, explains with SHAP
"""
import sys
import os
import json
import time
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from confluent_kafka import Consumer, Producer, KafkaError
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.feature_store import RedisFeatureStore

class FraudStreamProcessorXGB:
    """
    XGBoost-based stream processor with SHAP explainability
    
    Flow:
    1. Consume from transactions.raw
    2. Fetch behavioral features from Redis
    3. Score with XGBoost model
    4. Explain with SHAP values
    5. Publish to transactions.scored
    """
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 consumer_group: str = 'fraud-processor-xgb-group',
                 reset_offset: str = 'latest'):
        """Initialize consumer, producer, feature store, and XGBoost model"""
        
        print("📦 Initializing XGBoost Stream Processor...")
        
        # Kafka Consumer
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': consumer_group,
            'auto.offset.reset': reset_offset,
            'enable.auto.commit': True
        })
        self.consumer.subscribe(['transactions.raw'])
        
        # Kafka Producer
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'fraud-processor-xgb'
        })
        
        # Redis Feature Store
        self.feature_store = RedisFeatureStore()
        
        # Load XGBoost Model
        print("📦 Loading XGBoost model...")
        try:
            self.model = xgb.Booster()
            self.model.load_model("artifacts/xgboost_unchained.json")
            print("✅ XGBoost model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load XGBoost model: {e}")
            raise
        
        # ALL 31 FEATURES FROM BASE.CSV (matching XGBoost model training)
        self.feature_names = [
            'income', 'name_email_similarity', 'prev_address_months_count',
            'current_address_months_count', 'customer_age', 'days_since_request',
            'intended_balcon_amount', 'payment_type', 'zip_count_4w',
            'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w',
            'date_of_birth_distinct_emails_4w', 'employment_status', 'credit_risk_score',
            'email_is_free', 'housing_status', 'phone_home_valid', 'phone_mobile_valid',
            'bank_months_count', 'has_other_cards', 'proposed_credit_limit',
            'foreign_request', 'source', 'session_length_in_minutes', 'device_os',
            'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count', 'month'
        ]
        
        # Initialize SHAP explainer
        print(f"📦 Initializing SHAP explainer with {len(self.feature_names)} features...")
        try:
            # Load original training data sample (not the feature-selected version)
            X_train_raw = pd.read_csv("data/Base.csv", nrows=100)
            X_train_sample = X_train_raw[self.feature_names].copy()
            
            # Handle categorical columns - convert to category codes (numeric)
            categorical_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
            for col in categorical_cols:
                if col in X_train_sample.columns:
                    # Convert to category then to codes (numeric representation)
                    X_train_sample[col] = pd.Categorical(X_train_sample[col]).codes
            
            # Ensure all columns are numeric for SHAP
            X_train_sample = X_train_sample.astype('float64')
            
            self.explainer = shap.TreeExplainer(self.model)
            print(f"✅ SHAP explainer initialized successfully")
        except Exception as e:
            print(f"⚠️  SHAP explainer initialization failed: {e}")
            print(f"   SHAP explanations will use fallback mode")
            self.explainer = None
        
        # Performance tracking
        self.processed_count = 0
        self.fraud_count = 0
        self.start_time = time.time()
    
    def enrich_with_features(self, transaction: Dict) -> Dict[str, Any]:
        """
        Enrich raw transaction with Redis features
        """
        try:
            redis_features = self.feature_store.get_all_features(transaction)
            enriched = {**transaction, **redis_features}
            self.feature_store.update_transaction_features(transaction)
            return enriched
        except Exception as e:
            print(f"⚠️  Redis error, using empty features: {e}")
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
    
    def preprocess_for_model(self, transaction: Dict) -> xgb.DMatrix:
        """
        Transform raw transaction into XGBoost DMatrix with all 31 features
        
        Creates DMatrix directly without pandas to avoid hanging issues
        """
        try:
            print(f"      → Creating feature array with 31 features...")
            
            # Define default values for each feature
            feature_defaults = {
                'income': 0.7,
                'name_email_similarity': 0.9,
                'prev_address_months_count': 24.0,
                'current_address_months_count': 24.0,
                'customer_age': 35.0,
                'days_since_request': 7.0,
                'intended_balcon_amount': 100.0,
                'payment_type': 'AA',
                'zip_count_4w': 1.0,
                'velocity_6h': 5.0,
                'velocity_24h': 12.0,
                'velocity_4w': 100.0,
                'bank_branch_count_8w': 1.0,
                'date_of_birth_distinct_emails_4w': 1.0,
                'employment_status': 'CA',
                'credit_risk_score': 650.0,
                'email_is_free': 0,
                'housing_status': 'BA',
                'phone_home_valid': 1,
                'phone_mobile_valid': 1,
                'bank_months_count': 24.0,
                'has_other_cards': 1,
                'proposed_credit_limit': 2000.0,
                'foreign_request': 0,
                'source': 'INTERNET',
                'session_length_in_minutes': 10.0,
                'device_os': 'windows',
                'keep_alive_session': 1,
                'device_distinct_emails_8w': 1.0,
                'device_fraud_count': 0.0,
                'month': datetime.now(UTC).month
            }
            
            # Create feature vector (with categorical handling)
            feature_data = {}
            for feature_name in self.feature_names:
                value = transaction.get(feature_name, feature_defaults.get(feature_name, 0))
                
                # Categorical features need to remain as their original type
                if feature_name in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
                    feature_data[feature_name] = str(value)
                else:
                    feature_data[feature_name] = float(value)
            
            print(f"      → Creating DataFrame for DMatrix...")
            # Create DataFrame (necessary for categorical features in XGBoost)
            df = pd.DataFrame([feature_data])
            
            # Set categorical dtypes for XGBoost
            for col in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
                df[col] = df[col].astype('category')
            
            print(f"      → Creating DMatrix...")
            dmatrix = xgb.DMatrix(df, enable_categorical=True)
            
            print(f"      → Preprocessing complete! Shape: {df.shape}")
            return dmatrix
            
        except Exception as e:
            print(f"⚠️  Preprocessing error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create zero-filled DMatrix with correct structure
            feature_data = {fname: 0.0 for fname in self.feature_names}
            for col in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
                feature_data[col] = 'AA'
            
            df = pd.DataFrame([feature_data])
            for col in ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']:
                df[col] = df[col].astype('category')
            
            return xgb.DMatrix(df, enable_categorical=True)
    
    def score_transaction(self, dmatrix: xgb.DMatrix) -> Dict[str, Any]:
        """
        Run XGBoost model inference
        
        Returns:
            Dict with fraud_score and is_fraud
        """
        start_inference = time.time()
        
        # Run inference
        fraud_score = self.model.predict(dmatrix)[0]
        
        inference_time_ms = (time.time() - start_inference) * 1000
        
        # Apply threshold
        is_fraud = fraud_score >= 0.5
        
        return {
            'fraud_score': round(float(fraud_score), 4),
            'is_fraud': bool(is_fraud),
            'model_version': 'v2.0-xgboost-unchained',
            'inference_time_ms': round(inference_time_ms, 2)
        }
    
    def generate_explanation(self, dmatrix: xgb.DMatrix, fraud_score: float, is_fraud: bool) -> Dict[str, Any]:
        """
        Generate SHAP-based explanation for prediction
        
        Returns:
            {
                'top_features': ['feature1', 'feature2', 'feature3'],
                'shap_values': {'feature1': 0.25, 'feature2': -0.15, ...},
                'rules_fired': ['Explanation text'],
                'confidence_level': 0.85
            }
        """
        try:
            if self.explainer is None:
                # Fallback explanation without SHAP
                return {
                    'top_features': ['velocity_24h', 'credit_risk_score', 'income'],
                    'shap_values': {},
                    'rules_fired': [f"{'Flagged as suspicious' if is_fraud else 'Cleared as legitimate'} (SHAP unavailable)"],
                    'confidence_level': round(fraud_score if is_fraud else 1 - fraud_score, 4)
                }
            
            # Convert DMatrix back to DataFrame for SHAP
            # Extract the data from DMatrix (workaround since get_data() is deprecated)
            import pandas as pd
            
            # Get slice from DMatrix (XGBoost internal method)
            df_for_shap = pd.DataFrame(
                dmatrix.get_data().toarray() if hasattr(dmatrix.get_data(), 'toarray') else dmatrix.get_data(),
                columns=self.feature_names
            )
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(df_for_shap)
            
            # Get top 3 contributing features
            feature_contributions = {
                self.feature_names[i]: float(shap_values[0][i])
                for i in range(len(self.feature_names))
            }
            
            # Sort by absolute contribution
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            
            top_features = [f[0] for f in sorted_features]
            top_values = {f[0]: f[1] for f in sorted_features}
            
            # Format feature names for human readability
            clean_features = [f.replace('_', ' ').title() for f in top_features]
            drivers_str = ", ".join(clean_features[:-1]) + f", and {clean_features[-1]}" if len(clean_features) > 1 else clean_features[0]
            
            # Generate natural language explanation
            if is_fraud:
                explanation = f"Flagged as suspicious due to highly anomalous patterns detected in {drivers_str}."
            else:
                explanation = f"Cleared as legitimate based on verified normal behavior in {drivers_str}."
            
            return {
                'top_features': top_features,
                'shap_values': top_values,
                'rules_fired': [explanation],
                'confidence_level': round(fraud_score if is_fraud else 1 - fraud_score, 4),
                'explanation_method': 'SHAP TreeExplainer'
            }
            
        except Exception as e:
            print(f"⚠️  SHAP explanation error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback explanation
            return {
                'top_features': ['velocity_24h', 'credit_risk_score', 'income'],
                'shap_values': {},
                'rules_fired': [f"{'Flagged as suspicious' if is_fraud else 'Cleared as legitimate'} (SHAP computation failed)"],
                'confidence_level': round(fraud_score if is_fraud else 1 - fraud_score, 4)
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
        3. Preprocess for XGBoost
        4. Score with XGBoost
        5. Explain with SHAP
        6. Publish to transactions.scored
        """
        txn_id = "UNKNOWN"
        
        try:
            # Parse transaction
            try:
                transaction = json.loads(msg.value().decode('utf-8'))
                txn_id = transaction.get('transaction_id', 'UNKNOWN')
                
                # Validate transaction_id
                if 'transaction_id' not in transaction:
                    print(f"❌ Missing transaction_id in message, skipping")
                    return
                
                # Basic validation - check if message has some required features
                # (No need to check 'amount' since the feature is 'intended_balcon_amount')
                if 'intended_balcon_amount' not in transaction and 'income' not in transaction:
                    print(f"⚠️  {txn_id}: Old message format detected. Skipping...")
                    return
                
                # Check for all 31 required features (for debugging)
                missing_features = [f for f in self.feature_names if f not in transaction]
                if missing_features and len(missing_features) <= 5:
                    # Show which features are missing (if few)
                    print(f"⚠️  {txn_id}: Missing {len(missing_features)} features: {missing_features[:5]}")
                    print(f"   This is likely an old message from before the fix. Skipping...")
                    return
                elif missing_features:
                    # Too many missing features
                    print(f"⚠️  {txn_id}: Missing {len(missing_features)} features (old schema). Skipping...")
                    return
                    
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"❌ Malformed message: {e}")
                return
            
            print(f"\n🔄 Processing {txn_id}")
            
            # Step 1: Enrich with features
            print(f"   [1/4] Enriching features...")
            enriched = self.enrich_with_features(transaction)
            print(f"   ✓ Features enriched")
            
            # Step 2: Preprocess (XGBoost DMatrix)
            print(f"   [2/4] Preprocessing...")
            dmatrix = self.preprocess_for_model(enriched)
            print(f"   ✓ Preprocessed")
            
            # Step 3: Score with XGBoost
            print(f"   [3/4] Scoring with XGBoost...")
            prediction = self.score_transaction(dmatrix)
            print(f"   ✓ Scored: {prediction['fraud_score']:.4f}")
            
            # Step 4: Generate SHAP explanation
            print(f"   [4/4] Generating SHAP explanation...")
            explanation = self.generate_explanation(
                dmatrix,
                prediction['fraud_score'],
                prediction['is_fraud']
            )
            print(f"   ✓ Explanation generated")
            
            # Combine everything
            scored_transaction = {
                **transaction,
                **prediction,
                **explanation,
                'processed_at': datetime.now(UTC).isoformat()
            }
            
            # Publish to scored topic
            self.producer.produce(
                topic='transactions.scored',
                key=txn_id.encode('utf-8'),
                value=json.dumps(scored_transaction).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
            
            # Update stats
            self.processed_count += 1
            if prediction['is_fraud']:
                self.fraud_count += 1
            
            # Log prediction
            fraud_emoji = "🚨" if prediction['is_fraud'] else "✅"
            amount = transaction.get('intended_balcon_amount', 0)
            print(f"{fraud_emoji} Score: {prediction['fraud_score']:.4f} | "
                  f"Amount: ${amount:.2f} | "
                  f"Inference: {prediction['inference_time_ms']:.2f}ms")
            print(f"   Explanation: {explanation['rules_fired'][0][:80]}...")
            
        except Exception as e:
            print(f"❌ Unexpected error processing {txn_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main processing loop"""
        print("\n" + "="*60)
        print("🚀 XGBOOST FRAUD STREAM PROCESSOR STARTED")
        print("="*60)
        print(f"📥 Consuming from: transactions.raw")
        print(f"📤 Publishing to: transactions.scored")
        print(f"🧠 Model: XGBoost Unchained")
        print(f"💡 Explainability: SHAP TreeExplainer")
        print(f"⚡ Redis Feature Store: Active")
        print("="*60 + "\n")
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f'❌ Consumer error: {msg.error()}')
                    continue
                
                self.process_message(msg)
                
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
    
    reset_offset = 'latest'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--replay-all':
            reset_offset = 'earliest'
            print("⚠️  Replaying ALL messages from beginning (--replay-all mode)")
        elif sys.argv[1] == '--help':
            print("Usage: python stream_processor_xgboost.py [--replay-all]")
            print("  --replay-all: Process all historical messages (default: skip to latest)")
            sys.exit(0)
    
    processor = FraudStreamProcessorXGB(reset_offset=reset_offset)
    processor.run()
