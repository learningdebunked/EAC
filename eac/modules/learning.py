"""
Learning Module - Online learning and adaptation
"""
import logging
from typing import Dict, Any
import numpy as np
from collections import deque

from eac.config import EACConfig


class LearningModule:
    """
    Learning Module: Updates agent from user feedback
    
    Responsibilities:
    1. Update contextual bandit weights
    2. Track policy performance
    3. Detect concept drift
    4. Trigger model retraining
    """
    
    def __init__(self, config: EACConfig):
        self.config = config
        self.logger = logging.getLogger("EACAgent.Learning")
        
        # Track recent performance for drift detection
        self.recent_rewards = deque(maxlen=1000)
        self.recent_acceptance_rates = deque(maxlen=1000)
        
        # Performance tracking
        self.total_updates = 0
        self.total_reward = 0.0
        
        self.logger.info("Learning Module initialized")
    
    def update(
        self, 
        context: Dict[str, Any], 
        policy: str, 
        reward: float,
        user_feedback: Dict[str, Any]
    ):
        """
        Update from user feedback
        
        Args:
            context: Context features
            policy: Policy that was executed
            reward: Computed reward
            user_feedback: User's response
        """
        # Track metrics
        self.recent_rewards.append(reward)
        
        acceptance_rate = 0.0
        if user_feedback.get('total_recommendations', 0) > 0:
            acceptance_rate = (
                user_feedback.get('accepted_count', 0) / 
                user_feedback['total_recommendations']
            )
        self.recent_acceptance_rates.append(acceptance_rate)
        
        self.total_updates += 1
        self.total_reward += reward
        
        # Check for concept drift
        if self.total_updates % 100 == 0:
            drift_detected = self._detect_drift()
            if drift_detected:
                self.logger.warning("Concept drift detected! Consider model retraining.")
        
        self.logger.debug(
            f"Learning update #{self.total_updates}: "
            f"reward={reward:.3f}, acceptance={acceptance_rate:.2f}"
        )
    
    def _detect_drift(self) -> bool:
        """
        Detect concept drift using recent performance
        
        Returns:
            True if drift detected
        """
        if len(self.recent_rewards) < 100:
            return False
        
        # Compare recent vs. historical performance
        recent_avg_reward = np.mean(list(self.recent_rewards)[-100:])
        historical_avg_reward = np.mean(list(self.recent_rewards)[:-100])
        
        # Drift if recent performance drops by >20%
        if recent_avg_reward < historical_avg_reward * 0.8:
            self.logger.warning(
                f"Performance drop detected: "
                f"recent={recent_avg_reward:.3f}, "
                f"historical={historical_avg_reward:.3f}"
            )
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_updates': self.total_updates,
            'total_reward': self.total_reward,
            'avg_reward': self.total_reward / self.total_updates if self.total_updates > 0 else 0,
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0,
            'recent_avg_acceptance': np.mean(self.recent_acceptance_rates) if self.recent_acceptance_rates else 0
        }
