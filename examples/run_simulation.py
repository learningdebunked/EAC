"""
Example: Run counterfactual simulation
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from simulation import SimulationEngine, SimulationAnalyzer
from config import EACConfig


def generate_synthetic_transactions(n_users: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic transaction data for simulation
    
    In production, load from Instacart/dunnhumby datasets
    """
    np.random.seed(42)
    
    transactions = []
    
    for i in range(n_users):
        # Generate user attributes
        income = np.random.choice([25000, 35000, 50000, 75000, 100000])
        race = np.random.choice(['white', 'black', 'hispanic', 'asian', 'other'])
        has_snap = income < 40000 and np.random.random() > 0.5
        
        # Generate cart
        n_items = np.random.randint(3, 8)
        cart = []
        for j in range(n_items):
            cart.append({
                'product_id': f'prod_{np.random.randint(1, 100)}',
                'quantity': np.random.randint(1, 3),
                'price': np.random.uniform(2.0, 15.0)
            })
        
        # Payment methods
        payment_methods = ['CREDIT_CARD']
        if has_snap:
            payment_methods.insert(0, 'SNAP_EBT')
        
        transaction = {
            'user_id': f'user_{i}',
            'transaction_id': f'txn_{i}',
            'cart': cart,
            'zip_code': f'941{np.random.randint(0, 99):02d}',
            'census_tract': f'0607501{np.random.randint(1000, 9999)}',
            'payment_methods': payment_methods,
            'income': income,
            'race': race
        }
        
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)


def main():
    """Run simulation example"""
    print("="*60)
    print("EAC COUNTERFACTUAL SIMULATION")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic transaction data...")
    transactions = generate_synthetic_transactions(n_users=100)  # Small for demo
    print(f"   Generated {len(transactions)} transactions")
    
    # Initialize simulation engine
    print("\n2. Initializing simulation engine...")
    config = EACConfig()
    
    # Try to use advanced models if available
    use_advanced = True
    try:
        from models.acceptance import XGBoostAcceptanceModel
        print("   ✓ Advanced ML models detected")
    except ImportError:
        use_advanced = False
        print("   ℹ Using baseline models (install advanced models for better results)")
    
    engine = SimulationEngine(config, use_advanced_models=use_advanced)
    
    # Run simulation
    print("\n3. Running counterfactual simulation...")
    print("   (This compares EAC vs. baseline for each transaction)")
    results = engine.run_simulation(
        transactions=transactions,
        n_replications=10,  # Small for demo (use 100-1000 in production)
        random_seed=42
    )
    
    print(f"\n   Simulation complete: {len(results)} observations")
    
    # Analyze results
    print("\n4. Analyzing results...")
    analyzer = SimulationAnalyzer()
    analysis = analyzer.analyze(results)
    
    # Generate report
    print("\n" + "="*60)
    report = analyzer.generate_report(analysis)
    print(report)
    
    # Save results
    output_file = "simulation_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    summary = analysis['summary_statistics']
    print(f"\n✓ Average savings: ${-summary['mean_delta_spend']:.2f} per transaction")
    print(f"✓ Nutrition improvement: +{summary['mean_delta_nutrition']:.1f} HEI points")
    print(f"✓ Acceptance rate: {summary['mean_acceptance_rate']:.1%}")
    print(f"✓ Average latency: {summary['mean_latency_ms']:.1f}ms")
    
    # Hypothesis test results
    tests = analysis['hypothesis_tests']
    print("\nHypothesis Tests:")
    for test_name, test_result in tests.items():
        symbol = "✓" if test_result['result'] == 'PASS' else "✗"
        print(f"  {symbol} {test_name}: {test_result['result']}")
    
    # Fairness
    fairness = analysis['fairness_analysis']
    print(f"\nFairness Check: {fairness['equalized_uplift']['result']}")
    print(f"  Max disparity: ${fairness['equalized_uplift']['max_disparity']:.2f}")
    
    print("\n" + "="*60)
    print("Simulation complete! Check simulation_results.csv for details.")
    print("="*60)
    
    # Show info about advanced models
    if not use_advanced:
        print("\n" + "="*60)
        print("💡 TIP: Train Advanced ML Models for Better Results")
        print("="*60)
        print("\nTo improve simulation accuracy:")
        print("1. Download real datasets (Instacart, dunnhumby)")
        print("2. Train XGBoost acceptance model:")
        print("   python scripts/train_acceptance_model.py")
        print("3. Train product embeddings:")
        print("   python scripts/train_embeddings.py")
        print("4. Re-run simulation with trained models")
        print("\nExpected improvements:")
        print("  • Acceptance rate: 5% → 60-70%")
        print("  • Savings: $0.31 → $1-2 per transaction")
        print("  • Nutrition: 0 → +5-10 HEI points")
        print("\nSee ADVANCED_MODELS.md for details.")
        print("="*60)


if __name__ == "__main__":
    main()
