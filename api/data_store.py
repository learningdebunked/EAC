"""
Data Store for Real-Time Analytics

Stores all transactions and feedback for the analytics dashboard
"""
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class TransactionStore:
    """Store transactions for analytics"""
    
    def __init__(self, csv_path: str = "live_transactions.csv"):
        self.csv_path = csv_path
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create CSV file if it doesn't exist"""
        if not Path(self.csv_path).exists():
            df = pd.DataFrame(columns=[
                'timestamp',
                'user_id',
                'transaction_id',
                'policy_used',
                'num_recommendations',
                'accepted_count',
                'declined_count',
                'total_savings',
                'total_nutrition_improvement',
                'acceptance_rate',
                'latency_ms',
                'protected_group',
                'income_group',
                'snap_eligible',
                'fairness_check'
            ])
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Created new transaction store: {self.csv_path}")
    
    def add_transaction(self, transaction_data: Dict[str, Any]):
        """Add a new transaction"""
        try:
            # Read existing data
            df = pd.read_csv(self.csv_path)
            
            # Add new transaction
            new_row = pd.DataFrame([transaction_data])
            df = pd.concat([df, new_row], ignore_index=True)
            
            # Save
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Added transaction: {transaction_data['transaction_id']}")
            
        except Exception as e:
            logger.error(f"Error adding transaction: {e}")
    
    def update_transaction(self, transaction_id: str, updates: Dict[str, Any]):
        """Update an existing transaction (e.g., when user accepts/declines)"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Find and update transaction
            mask = df['transaction_id'] == transaction_id
            for key, value in updates.items():
                df.loc[mask, key] = value
            
            # Recalculate acceptance rate
            df.loc[mask, 'acceptance_rate'] = (
                df.loc[mask, 'accepted_count'] / df.loc[mask, 'num_recommendations']
            )
            
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Updated transaction: {transaction_id}")
            
        except Exception as e:
            logger.error(f"Error updating transaction: {e}")
    
    def get_recent_transactions(self, n: int = 100) -> pd.DataFrame:
        """Get recent transactions"""
        try:
            df = pd.read_csv(self.csv_path)
            return df.tail(n)
        except Exception as e:
            logger.error(f"Error reading transactions: {e}")
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics"""
        try:
            df = pd.read_csv(self.csv_path)
            
            if len(df) == 0:
                return {
                    'total_transactions': 0,
                    'avg_acceptance_rate': 0,
                    'avg_savings': 0,
                    'avg_nutrition': 0,
                    'avg_latency': 0
                }
            
            return {
                'total_transactions': len(df),
                'avg_acceptance_rate': df['acceptance_rate'].mean() * 100,
                'avg_savings': df['total_savings'].mean(),
                'avg_nutrition': df['total_nutrition_improvement'].mean(),
                'avg_latency': df['latency_ms'].mean()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Global instance
transaction_store = TransactionStore()
