# Training Guide for Advanced ML Models

## Quick Start

### Option 1: Train All Models at Once (Recommended)

```bash
python scripts/train_all_models.py
```

This will train all 3 models in sequence (~20-45 minutes total).

### Option 2: Train Individual Models

```bash
# Train XGBoost acceptance model (~5-10 min)
python scripts/train_acceptance_model.py

# Train product embeddings (~10-30 min)
python scripts/train_embeddings.py

# Train collaborative filter (~2-5 min)
python scripts/train_collaborative.py
```

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Datasets (Optional but Recommended)

The scripts work with **synthetic data** by default, but for best results, download real datasets:

#### **Instacart Dataset** (Recommended)
- **Source**: https://www.kaggle.com/c/instacart-market-basket-analysis
- **Size**: ~3M orders, 200K users, 50K products
- **Files needed**:
  - `orders.csv`
  - `order_products.csv`
  - `products.csv`
- **Place in**: `data/instacart/`

#### **dunnhumby Dataset** (Alternative)
- **Source**: https://www.dunnhumby.com/source-files/
- **Size**: 2,500 households, 92K transactions
- **Files needed**:
  - `transactions.csv`
  - `households.csv`
  - `products.csv`
- **Place in**: `data/dunnhumby/`

---

## Training Details

### 1. XGBoost Acceptance Model

**Purpose**: Predicts P(user accepts recommendation)

**Training Data**:
- **With Instacart**: Uses reorder patterns as acceptance proxy
- **Synthetic**: Generates 10,000 samples with realistic acceptance patterns

**Features** (14 dimensions):
- Product similarity (category_match, brand_match, price_delta)
- Nutrition improvement
- User history (past_acceptance_rate, order_count)
- SDOH signals (food_insecurity, financial_constraint)

**Expected Performance**:
- AUC: 0.85-0.90 (real data) / 0.75-0.80 (synthetic)
- Training time: 5-10 minutes
- Model size: ~5 MB

**Output**: `models/acceptance_model.pkl`

**Usage**:
```python
from models.acceptance import XGBoostAcceptanceModel

model = XGBoostAcceptanceModel()
model.load('models/acceptance_model.pkl')

proba = model.predict_proba(recommendation, user_features)
# Returns: 0.75 (75% chance of acceptance)
```

---

### 2. Product Embeddings (BERT)

**Purpose**: Semantic product representations for similarity matching

**Training Data**:
- **With Instacart**: Product names, categories, co-purchase patterns
- **Synthetic**: 1,000 products across 10 categories

**Architecture**:
- Base model: `bert-base-uncased` (110M parameters)
- Fine-tuned on product similarity task
- Output: 768-dim embeddings

**Training Process**:
1. Generate 5,000 product pairs (positive/negative)
2. Fine-tune BERT for 3 epochs
3. Test on similarity task

**Expected Performance**:
- Similarity accuracy: 80-85%
- Training time: 10-30 minutes (GPU recommended)
- Model size: ~400 MB

**Output**: `models/embeddings.pt`

**Usage**:
```python
from models.embeddings import ProductTransformer

model = ProductTransformer()
model.load('models/embeddings.pt')

similarity = model.compute_similarity(product1, product2)
# Returns: 0.85 (85% similar)
```

---

### 3. Collaborative Filter

**Purpose**: User-based and item-based recommendations

**Training Data**:
- **With Instacart**: User-product purchase histories
- **Synthetic**: 1,000 users × 500 products with sparse interactions

**Method**: Non-negative Matrix Factorization (NMF)
- Factors: 50 latent dimensions
- Optimization: Coordinate descent

**Expected Performance**:
- Sparsity: 95-99% (typical for retail)
- Training time: 2-5 minutes
- Model size: ~10 MB

**Output**: `models/collaborative.pkl`

**Usage**:
```python
from models.collaborative import CollaborativeFilter

model = CollaborativeFilter()
model.load('models/collaborative.pkl')

recommendations = model.recommend_products('user_123', top_k=10)
# Returns: [(product_id, score), ...]
```

---

## Verification

After training, verify models are working:

```bash
# Run simulation with trained models
python examples/run_simulation.py
```

**Expected output**:
```
✓ Advanced ML models detected
...
Acceptance rate: 65%
Savings: $1.50 per transaction
```

Compare to baseline (without trained models):
```
ℹ Using baseline models
...
Acceptance rate: 5%
Savings: $0.31 per transaction
```

**Improvement**: 10-15x better results!

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformers'"

**Solution**:
```bash
pip install transformers torch
```

### Issue: "CUDA out of memory" (when training embeddings)

**Solution**: Reduce batch size
```python
# In train_embeddings.py, line 87:
model.fine_tune(pairs, labels, epochs=3, batch_size=16)  # Reduce from 32 to 16
```

Or train on CPU (slower but works):
```python
# In models/embeddings.py, add:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

### Issue: Training is very slow

**Solutions**:
1. Use GPU if available (10-20x faster for embeddings)
2. Reduce dataset size:
   ```python
   # In train_acceptance_model.py, line 42:
   for user_id in data['user_id'].unique()[:1000]:  # Reduce from 10000
   ```
3. Use synthetic data (faster but less accurate)

### Issue: "FileNotFoundError: data/instacart/orders.csv"

**Solution**: Either:
1. Download Instacart dataset and place in `data/instacart/`
2. Or let it use synthetic data (automatic fallback)

---

## Advanced Options

### Custom Hyperparameters

Edit the training scripts to customize:

**XGBoost** (`train_acceptance_model.py`):
```python
model = XGBoostAcceptanceModel()
# Modify in models/acceptance.py __init__:
n_estimators=200,  # More trees = better accuracy, slower
max_depth=6,       # Deeper = more complex patterns
learning_rate=0.1  # Lower = more careful learning
```

**Embeddings** (`train_embeddings.py`):
```python
model = ProductTransformer(model_name="bert-base-uncased")
# Try different models:
# - "bert-large-uncased" (better but slower)
# - "distilbert-base-uncased" (faster but less accurate)
# - "roberta-base" (alternative architecture)
```

**Collaborative Filter** (`train_collaborative.py`):
```python
model = CollaborativeFilter(n_factors=50, method="nmf")
# Adjust factors:
n_factors=100  # More factors = capture more patterns
# Try neural method:
method="neural"  # Slower but can capture non-linear patterns
```

### AutoML Optimization

Use Optuna to find best hyperparameters:

```python
from models.automl import OptunaOptimizer

optimizer = OptunaOptimizer(n_trials=100, timeout=3600)
best_params = optimizer.optimize_xgboost(X_train, y_train, X_val, y_val)
# Returns: {'n_estimators': 200, 'max_depth': 6, ...}
```

---

## Expected Improvements

| Metric | Baseline | With Trained Models | Improvement |
|--------|----------|---------------------|-------------|
| **Acceptance Rate** | 5% | 60-70% | **12-14x** |
| **Savings/Transaction** | $0.31 | $1-2 | **3-6x** |
| **Nutrition Improvement** | 0 HEI | +5-10 HEI | **∞** |
| **Realism** | Heuristic | Data-driven | **Much better** |

---

## Production Deployment

After training, deploy models:

1. **Copy models to production**:
   ```bash
   scp models/*.pkl models/*.pt production-server:/app/models/
   ```

2. **Load in API**:
   ```python
   # In api/main.py
   from models.acceptance import XGBoostAcceptanceModel
   
   acceptance_model = XGBoostAcceptanceModel()
   acceptance_model.load('models/acceptance_model.pkl')
   ```

3. **Monitor performance**:
   ```python
   from models.monitoring import EvidentlyMonitor
   
   monitor = EvidentlyMonitor()
   monitor.set_reference(reference_data, predictions, targets)
   drift_report = monitor.detect_data_drift(current_data)
   ```

---

## Retraining Schedule

Retrain models periodically to adapt to changing patterns:

- **Acceptance Model**: Monthly (user preferences change)
- **Embeddings**: Quarterly (new products added)
- **Collaborative Filter**: Weekly (purchase patterns evolve)

Automate with cron:
```bash
# Retrain acceptance model monthly
0 0 1 * * cd /app && python scripts/train_acceptance_model.py

# Retrain collaborative filter weekly
0 0 * * 0 cd /app && python scripts/train_collaborative.py
```

---

## Support

- **Documentation**: See ADVANCED_MODELS.md
- **Issues**: https://github.com/learningdebunked/EAC/issues
- **Examples**: See `examples/` directory

---

**Ready to train?** Run: `python scripts/train_all_models.py` 🚀
