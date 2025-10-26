"""
Advanced ML Models for EAC

Production-ready models trained on real datasets:
- Instacart (3M orders, 200K users)
- dunnhumby (2,500 households)
- UCI Retail (500K transactions)
- RetailRocket (2M events)
"""

from models.acceptance import XGBoostAcceptanceModel
from models.embeddings import ProductTransformer
from models.collaborative import CollaborativeFilter
from models.causal import CausalMLEstimator
from models.automl import OptunaOptimizer
from models.monitoring import EvidentlyMonitor

__all__ = [
    "XGBoostAcceptanceModel",
    "ProductTransformer",
    "CollaborativeFilter",
    "CausalMLEstimator",
    "OptunaOptimizer",
    "EvidentlyMonitor"
]
