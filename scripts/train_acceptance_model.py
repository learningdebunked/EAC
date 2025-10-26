"""
Train XGBoost Acceptance Model

Trains on Instacart substitution data to predict P(user accepts recommendation)
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from models.acceptance import XGBoostAcceptanceModel


def load_instacart_data(data_path: str) -> pd.DataFrame:
    """
    Load and prepare Instacart data for training
    
    Expected files in data_path:
    - orders.csv: order_id, user_id, order_number, order_dow, order_hour_of_day
    - order_products.csv: order_id, product_id, add_to_cart_order, reordered
    - products.csv: product_id, product_name, aisle_id, department_id
    
    Returns:
        DataFrame with substitution events
    """
    print("Loading Instacart data...")
    
    # Load data
    orders = pd.read_csv(f"{data_path}/orders.csv")
    order_products = pd.read_csv(f"{data_path}/order_products.csv")
    products = pd.read_csv(f"{data_path}/products.csv")
    
    print(f"  Orders: {len(orders):,}")
    print(f"  Order products: {len(order_products):,}")
    print(f"  Products: {len(products):,}")
    
    # Merge data
    data = order_products.merge(orders, on='order_id')
    data = data.merge(products, on='product_id')
    
    # Create substitution events
    # Use 'reordered' flag as proxy for acceptance
    # If reordered=1, user accepted the product in a previous order
    substitutions = []
    
    for user_id in data['user_id'].unique()[:10000]:  # Sample for demo
        user_orders = data[data['user_id'] == user_id].sort_values('order_number')
        
        if len(user_orders) < 2:
            continue
        
        # For each product in order N, check if reordered in order N+1
        for i in range(len(user_orders) - 1):
            current_product = user_orders.iloc[i]
            next_orders = user_orders[user_orders['order_number'] > current_product['order_number']]
            
            # Check if product was reordered
            accepted = current_product['product_id'] in next_orders['product_id'].values
            
            substitutions.append({
                'user_id': user_id,
                'order_id': current_product['order_id'],
                'product_id': current_product['product_id'],
                'original_product_id': current_product['product_id'],
                'suggested_product_id': current_product['product_id'],
                'accepted': int(accepted),
                'category_match': 1,  # Same product
                'brand_match': 1,
                'price_original': np.random.uniform(2, 15),  # Synthetic prices
                'price_suggested': np.random.uniform(2, 15),
                'nutrition_improvement': np.random.uniform(-5, 10),
                'user_past_acceptance_rate': 0.5,
                'user_price_sensitivity': 0.5,
                'user_order_count': len(user_orders),
                'sdoh_food_insecurity': np.random.uniform(0, 1),
                'sdoh_financial_constraint': np.random.uniform(0, 1),
                'sdoh_mobility_limitation': np.random.uniform(0, 1)
            })
    
    df = pd.DataFrame(substitutions)
    print(f"\nGenerated {len(df):,} substitution events")
    print(f"Acceptance rate: {df['accepted'].mean():.1%}")
    
    return df


def generate_synthetic_training_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic training data for demo purposes
    
    Use this if you don't have Instacart data yet
    """
    print(f"Generating {n_samples:,} synthetic training samples...")
    
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        # Generate features
        savings = np.random.uniform(-2, 5)
        nutrition_improvement = np.random.uniform(-5, 10)
        category_match = np.random.choice([0, 1], p=[0.3, 0.7])
        brand_match = np.random.choice([0, 1], p=[0.5, 0.5])
        price_sensitivity = np.random.uniform(0, 1)
        food_insecurity = np.random.uniform(0, 1)
        
        # Compute acceptance probability (ground truth)
        p_accept = 0.6  # Base rate
        p_accept += savings * 0.1  # Savings effect
        p_accept += nutrition_improvement * 0.01  # Nutrition effect
        p_accept += category_match * 0.1  # Category match bonus
        p_accept += brand_match * 0.05  # Brand match bonus
        p_accept -= (1 - price_sensitivity) * savings * 0.05  # Price sensitivity
        p_accept = np.clip(p_accept, 0, 1)
        
        # Sample acceptance
        accepted = np.random.random() < p_accept
        
        data.append({
            'user_id': f'user_{i % 1000}',
            'order_id': f'order_{i}',
            'product_id': f'prod_{i % 500}',
            'original_product_id': f'prod_{i % 500}',
            'suggested_product_id': f'prod_{(i+1) % 500}',
            'accepted': int(accepted),
            'category_match': category_match,
            'brand_match': brand_match,
            'price_original': np.random.uniform(2, 15),
            'price_suggested': np.random.uniform(2, 15),
            'nutrition_improvement': nutrition_improvement,
            'user_past_acceptance_rate': np.random.uniform(0.3, 0.8),
            'user_price_sensitivity': price_sensitivity,
            'user_order_count': np.random.randint(1, 50),
            'sdoh_food_insecurity': food_insecurity,
            'sdoh_financial_constraint': np.random.uniform(0, 1),
            'sdoh_mobility_limitation': np.random.uniform(0, 1)
        })
    
    df = pd.DataFrame(data)
    print(f"Acceptance rate: {df['accepted'].mean():.1%}")
    
    return df


def main():
    """Train XGBoost acceptance model"""
    print("="*60)
    print("TRAINING XGBOOST ACCEPTANCE MODEL")
    print("="*60)
    
    # Try to load real Instacart data
    instacart_path = "data/instacart"
    
    if Path(instacart_path).exists():
        print("\n✓ Found Instacart data")
        data = load_instacart_data(instacart_path)
    else:
        print(f"\nℹ Instacart data not found at {instacart_path}")
        print("  Using synthetic data for demo")
        print("  Download Instacart dataset from: https://www.kaggle.com/c/instacart-market-basket-analysis")
        data = generate_synthetic_training_data(n_samples=10000)
    
    # Initialize model
    print("\n" + "="*60)
    print("Training XGBoost model...")
    print("="*60)
    
    model = XGBoostAcceptanceModel()
    
    # Train
    results = model.train(data, validation_split=0.2)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"AUC: {results['auc']:.4f}")
    print(f"Log Loss: {results['log_loss']:.4f}")
    
    print("\nTop 10 Feature Importances:")
    for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10], 1):
        print(f"  {i}. {feature}: {importance:.4f}")
    
    # Save model
    model_path = "models/acceptance_model.pkl"
    Path("models").mkdir(exist_ok=True)
    model.save(model_path)
    
    print("\n" + "="*60)
    print(f"✓ Model saved to: {model_path}")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Run simulation with trained model:")
    print("   python examples/run_simulation.py")
    print("2. Expected improvements:")
    print("   • Acceptance rate: 5% → 60-70%")
    print("   • More realistic user behavior")
    print("   • Better treatment effect estimates")


if __name__ == "__main__":
    main()
