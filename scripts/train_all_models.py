"""
Train All Advanced ML Models

Runs all training scripts in sequence
"""
import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_script(script_name: str):
    """Run a training script"""
    print("\n" + "="*70)
    print(f"RUNNING: {script_name}")
    print("="*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, f"scripts/{script_name}"],
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        return False
    
    print(f"\n✓ {script_name} completed successfully")
    return True


def main():
    """Train all models"""
    print("="*70)
    print("TRAINING ALL ADVANCED ML MODELS")
    print("="*70)
    print("\nThis will train:")
    print("1. XGBoost Acceptance Model (~5-10 minutes)")
    print("2. Product Embeddings (~10-30 minutes)")
    print("3. Collaborative Filter (~2-5 minutes)")
    print("\nTotal time: ~20-45 minutes")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    # Train models
    scripts = [
        "train_acceptance_model.py",
        "train_embeddings.py",
        "train_collaborative.py"
    ]
    
    results = {}
    for script in scripts:
        results[script] = run_script(script)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    for script, success in results.items():
        status = "✓" if success else "❌"
        print(f"{status} {script}")
    
    if all(results.values()):
        print("\n" + "="*70)
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        print("\nTrained models saved to:")
        print("  • models/acceptance_model.pkl")
        print("  • models/embeddings.pt")
        print("  • models/collaborative.pkl")
        print("\nNext steps:")
        print("1. Run simulation with trained models:")
        print("   python examples/run_simulation.py")
        print("2. Expected improvements:")
        print("   • Acceptance rate: 5% → 60-70%")
        print("   • Savings: $0.31 → $1-2 per transaction")
        print("   • Nutrition: 0 → +5-10 HEI points")
    else:
        print("\n⚠️  Some models failed to train")
        print("Check the output above for errors")


if __name__ == "__main__":
    main()
