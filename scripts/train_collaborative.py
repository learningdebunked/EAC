"""
Train Collaborative Filtering Model

Trains on user-product interaction data
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from models.collaborative import CollaborativeFilter


def load_transaction_data(data_path: str) -> pd.DataFrame:
    """
    Load transaction data from Instacart
    
    Expected files:
    - orders.csv
    - order_products.csv
    """
    print("Loading transaction data...")
    
    orders = pd.read_csv(f"{data_path}/orders.csv")
    order_products = pd.read_csv(f"{data_path}/order_products.csv")
    
    # Merge
    transactions = order_products.merge(orders[['order_id', 'user_id']], on='order_id')
    
    # Aggregate: count how many times each user bought each product
    transactions = transactions.groupby(['user_id', 'product_id']).size().reset_index(name='quantity')
    
    print(f"  Users: {transactions['user_id'].nunique():,}")
    print(f"  Products: {transactions['product_id'].nunique():,}")
    print(f"  Interactions: {len(transactions):,}")
    
    return transactions


def generate_synthetic_transactions(n_users: int = 1000, n_products: int = 500) -> pd.DataFrame:
    """Generate synthetic transaction data"""
    print(f"Generating synthetic transactions...")
    print(f"  Users: {n_users}")
    print(f"  Products: {n_products}")
    
    np.random.seed(42)
    
    # Generate sparse interactions
    n_interactions = n_users * 20  # Each user buys ~20 products on average
    
    transactions = pd.DataFrame({
        'user_id': np.random.randint(0, n_users, n_interactions),
        'product_id': np.random.randint(0, n_products, n_interactions),
        'quantity': np.random.randint(1, 5, n_interactions)
    })
    
    # Aggregate
    transactions = transactions.groupby(['user_id', 'product_id'])['quantity'].sum().reset_index()
    
    print(f"  Interactions: {len(transactions):,}")
    print(f"  Sparsity: {1 - len(transactions)/(n_users*n_products):.2%}")
    
    return transactions


def main():
    """Train collaborative filtering model"""
    print("="*60)
    print("TRAINING COLLABORATIVE FILTERING MODEL")
    print("="*60)
    
    # Try to load real data
    instacart_path = "data/instacart"
    
    if Path(f"{instacart_path}/orders.csv").exists():
        print("\n✓ Found Instacart data")
        transactions = load_transaction_data(instacart_path)
        # Sample for demo (full dataset is large)
        transactions = transactions.sample(min(50000, len(transactions)))
    else:
        print(f"\nℹ Transaction data not found at {instacart_path}")
        print("  Using synthetic data for demo")
        transactions = generate_synthetic_transactions(n_users=1000, n_products=500)
    
    # Train model
    print("\n" + "="*60)
    print("Training matrix factorization...")
    print("="*60)
    
    model = CollaborativeFilter(n_factors=50, method="nmf")
    stats = model.train(transactions)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Users: {stats['n_users']:,}")
    print(f"Products: {stats['n_items']:,}")
    print(f"Sparsity: {stats['sparsity']:.2%}")
    
    # Test recommendations
    print("\n" + "="*60)
    print("Testing recommendations...")
    print("="*60)
    
    # Sample users
    sample_users = transactions['user_id'].unique()[:5]
    
    for user_id in sample_users:
        # Get recommendations
        recs = model.recommend_products(str(user_id), top_k=5)
        
        if recs:
            print(f"\nUser {user_id}:")
            print("  Recommended products:")
            for i, (product_id, score) in enumerate(recs, 1):
                print(f"    {i}. Product {product_id} (score: {score:.3f})")
    
    # Save model
    model_path = "models/collaborative.pkl"
    Path("models").mkdir(exist_ok=True)
    model.save(model_path)
    
    print("\n" + "="*60)
    print(f"✓ Model saved to: {model_path}")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Use for personalized product recommendations")
    print("2. Find similar users for targeting")
    print("3. Improve substitution suggestions")


if __name__ == "__main__":
    main()
