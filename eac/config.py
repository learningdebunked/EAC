"""
Configuration for EAC Agent
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EACConfig:
    """Configuration for Equity-Aware Checkout Agent"""
    
    # Agent settings
    agent_name: str = "EAC-Agent-v1"
    max_latency_ms: int = 100
    safe_mode: bool = True
    
    # Privacy settings
    differential_privacy_epsilon: float = 0.1
    differential_privacy_delta: float = 1e-6
    census_tract_min_population: int = 1200
    
    # Fairness constraints
    equalized_uplift_threshold: float = 0.05
    price_burden_ratio_threshold: float = 0.30
    safety_harm_rate_threshold: float = 0.01
    
    # Need state learning
    need_state_confidence_threshold: float = 0.7
    need_state_model_path: Optional[str] = None
    
    # Contextual bandit settings
    bandit_algorithm: str = "linucb"  # or "thompson_sampling"
    bandit_alpha: float = 1.0  # Exploration parameter
    bandit_context_dim: int = 128
    
    # Policy settings
    enabled_policies: List[str] = field(default_factory=lambda: [
        "snap_wic_substitution",
        "low_glycemic_alternative",
        "otc_coverage",
        "mobility_delivery",
        "safety_nudge"
    ])
    
    # Reward weights (multi-objective)
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "acceptance": 1.0,
        "cost_savings": 0.1,  # $1 saved = +0.1 reward
        "nutrition_improvement": 0.05,  # +1 HEI = +0.05 reward
        "fairness_violation": -2.0  # Heavy penalty
    })
    
    # Data paths
    sdoh_data_path: str = "data/sdoh/"
    product_data_path: str = "data/products/"
    transaction_data_path: str = "data/transactions/"
    
    # Model paths
    models_dir: str = "models/"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/eac_agent.log"
    
    # Monitoring
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    
    def __post_init__(self):
        """Validate configuration"""
        assert 0 < self.differential_privacy_epsilon <= 1.0
        assert 0 < self.differential_privacy_delta < 1.0
        assert 0 < self.equalized_uplift_threshold < 1.0
        assert 0 < self.price_burden_ratio_threshold < 1.0
        assert 0 < self.safety_harm_rate_threshold < 1.0
        assert self.max_latency_ms > 0
        assert self.bandit_algorithm in ["linucb", "thompson_sampling"]
