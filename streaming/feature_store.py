"""
Redis Feature Store: Real-Time Feature Engineering
Maintains sliding window aggregations for instant ML feature lookups
"""
import redis
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class RedisFeatureStore:
    """
    Manages real-time behavioral features using Redis data structures
    
    Features tracked:
    - Transaction velocity (count in time windows)
    - Amount aggregations (sum, avg in time windows)
    - Device usage patterns
    - Geographic diversity
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        """Initialize Redis connection"""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False  # Work with bytes for sorted sets
        )
        
        # Feature time windows (in seconds)
        self.WINDOW_6H = 6 * 60 * 60
        self.WINDOW_24H = 24 * 60 * 60
        self.WINDOW_4W = 28 * 24 * 60 * 60
        
        # Key prefixes
        self.PREFIX_USER_TXN = "user:txn:"
        self.PREFIX_DEVICE_TXN = "device:txn:"
        self.PREFIX_USER_AMOUNT = "user:amount:"
        self.PREFIX_USER_ZIP = "user:zip:"
        
    def _get_current_timestamp(self) -> float:
        """Get current Unix timestamp"""
        return time.time()
    
    def update_transaction_features(self, transaction: Dict) -> None:
        """
        Update all feature counters for a new transaction
        
        Args:
            transaction: Dict containing user_id, device_id, amount, timestamp, etc.
        """
        user_id = transaction['user_id']
        device_id = transaction.get('device_id', 'unknown')
        amount = float(transaction.get('intended_balcon_amount', transaction.get('amount', 0)))
        current_time = self._get_current_timestamp()
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis_client.pipeline()
        
        # 1. Track transaction timestamps for velocity calculation
        user_txn_key = f"{self.PREFIX_USER_TXN}{user_id}"
        pipe.zadd(user_txn_key, {transaction['transaction_id']: current_time})
        pipe.expire(user_txn_key, self.WINDOW_4W)  # Auto-expire old data
        
        # 2. Track device usage
        device_txn_key = f"{self.PREFIX_DEVICE_TXN}{device_id}"
        pipe.zadd(device_txn_key, {transaction['transaction_id']: current_time})
        pipe.expire(device_txn_key, self.WINDOW_4W)
        
        # 3. Track amounts for sum/avg calculations
        user_amount_key = f"{self.PREFIX_USER_AMOUNT}{user_id}"
        pipe.zadd(user_amount_key, {f"{transaction['transaction_id']}:{amount}": current_time})
        pipe.expire(user_amount_key, self.WINDOW_4W)
        
        # 4. Track geographic diversity (zip codes - simulated)
        if 'zip_code' in transaction:
            user_zip_key = f"{self.PREFIX_USER_ZIP}{user_id}"
            pipe.sadd(user_zip_key, transaction['zip_code'])
            pipe.expire(user_zip_key, self.WINDOW_4W)
        
        # Execute all commands atomically
        pipe.execute()
    
    def get_velocity_features(self, user_id: str) -> Dict[str, float]:
        """
        Calculate transaction velocity for different time windows
        
        Returns:
            Dict with keys: velocity_6h, velocity_24h, velocity_4w
        """
        user_txn_key = f"{self.PREFIX_USER_TXN}{user_id}"
        current_time = self._get_current_timestamp()
        
        # Calculate cutoff timestamps
        cutoff_6h = current_time - self.WINDOW_6H
        cutoff_24h = current_time - self.WINDOW_24H
        cutoff_4w = current_time - self.WINDOW_4W
        
        # Count transactions in each window using Redis ZCOUNT
        velocity_6h = self.redis_client.zcount(user_txn_key, cutoff_6h, current_time)
        velocity_24h = self.redis_client.zcount(user_txn_key, cutoff_24h, current_time)
        velocity_4w = self.redis_client.zcount(user_txn_key, cutoff_4w, current_time)
        
        return {
            'velocity_6h': float(velocity_6h),
            'velocity_24h': float(velocity_24h),
            'velocity_4w': float(velocity_4w)
        }
    
    def get_amount_features(self, user_id: str) -> Dict[str, float]:
        """
        Calculate amount aggregations for the user
        
        Returns:
            Dict with keys: total_amount_24h, avg_amount_24h, max_amount_24h
        """
        user_amount_key = f"{self.PREFIX_USER_AMOUNT}{user_id}"
        current_time = self._get_current_timestamp()
        cutoff_24h = current_time - self.WINDOW_24H
        
        # Get all amounts in the last 24 hours
        amounts_raw = self.redis_client.zrangebyscore(
            user_amount_key, 
            cutoff_24h, 
            current_time,
            withscores=False
        )
        
        if not amounts_raw:
            return {
                'total_amount_24h': 0.0,
                'avg_amount_24h': 0.0,
                'max_amount_24h': 0.0
            }
        
        # Parse amounts from "txn_id:amount" format
        amounts = []
        for entry in amounts_raw:
            try:
                amount_str = entry.decode('utf-8').split(':')[1]
                amounts.append(float(amount_str))
            except (IndexError, ValueError):
                continue
        
        return {
            'total_amount_24h': sum(amounts),
            'avg_amount_24h': sum(amounts) / len(amounts) if amounts else 0.0,
            'max_amount_24h': max(amounts) if amounts else 0.0
        }
    
    def get_device_features(self, device_id: str) -> Dict[str, float]:
        """
        Calculate device usage patterns
        
        Returns:
            Dict with keys: device_txn_count_24h, device_fraud_count (simulated)
        """
        device_txn_key = f"{self.PREFIX_DEVICE_TXN}{device_id}"
        current_time = self._get_current_timestamp()
        cutoff_24h = current_time - self.WINDOW_24H
        
        device_txn_count = self.redis_client.zcount(device_txn_key, cutoff_24h, current_time)
        
        return {
            'device_txn_count_24h': float(device_txn_count),
            'device_fraud_count': 0.0  # Would be populated by fraud decision feedback loop
        }
    
    def get_all_features(self, transaction: Dict) -> Dict[str, float]:
        """
        Get all engineered features for a transaction in one call
        
        Args:
            transaction: Raw transaction dict
            
        Returns:
            Dict of all computed features
        """
        user_id = transaction['user_id']
        device_id = transaction.get('device_id', 'unknown')
        
        # Gather features from different sources
        velocity = self.get_velocity_features(user_id)
        amounts = self.get_amount_features(user_id)
        device = self.get_device_features(device_id)
        
        # Combine all features
        features = {
            **velocity,
            **amounts,
            **device
        }
        
        return features
    
    def cleanup_old_data(self, user_id: str) -> None:
        """
        Remove expired data for a user (maintenance task)
        Redis EXPIRE handles this automatically, but this provides manual cleanup
        """
        current_time = self._get_current_timestamp()
        cutoff = current_time - self.WINDOW_4W
        
        # Remove old transactions
        user_txn_key = f"{self.PREFIX_USER_TXN}{user_id}"
        self.redis_client.zremrangebyscore(user_txn_key, 0, cutoff)
        
        # Remove old amounts
        user_amount_key = f"{self.PREFIX_USER_AMOUNT}{user_id}"
        self.redis_client.zremrangebyscore(user_amount_key, 0, cutoff)
    
    def get_stats(self) -> Dict[str, int]:
        """Get Redis feature store statistics"""
        user_keys = len(self.redis_client.keys(f"{self.PREFIX_USER_TXN}*"))
        device_keys = len(self.redis_client.keys(f"{self.PREFIX_DEVICE_TXN}*"))
        
        return {
            'tracked_users': user_keys,
            'tracked_devices': device_keys,
            'total_memory_mb': round(self.redis_client.info('memory')['used_memory'] / 1024 / 1024, 2)
        }


# Example usage and testing
if __name__ == '__main__':
    feature_store = RedisFeatureStore()
    
    # Simulate transaction
    test_transaction = {
        'transaction_id': 'TXN_TEST_001',
        'user_id': 'USER_1234',
        'device_id': 'DEV_567',
        'amount': 125.50,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Update features
    print("📝 Updating features for transaction...")
    feature_store.update_transaction_features(test_transaction)
    
    # Retrieve features
    print("\n📊 Retrieving computed features...")
    features = feature_store.get_all_features(test_transaction)
    
    for feature_name, value in features.items():
        print(f"   {feature_name}: {value}")
    
    # Show stats
    print("\n📈 Feature Store Stats:")
    stats = feature_store.get_stats()
    for stat_name, value in stats.items():
        print(f"   {stat_name}: {value}")
