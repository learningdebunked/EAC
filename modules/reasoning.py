"""
Reasoning Module - Infers needs and selects policies
"""
import logging
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from config import EACConfig


class NeedStateModel(nn.Module):
    """
    Multi-task neural network for need state inference
    
    Tasks:
    1. Food insecurity prediction
    2. Transportation constraint detection
    3. Chronic condition proxy
    4. Financial constraint analysis
    5. Mobility limitation assessment
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Task-specific heads
        head_dim = hidden_dim // 4
        self.food_insecurity_head = nn.Linear(head_dim, 3)  # Low/Med/High
        self.transport_head = nn.Linear(head_dim, 2)  # Has/No transit
        self.chronic_condition_head = nn.Linear(head_dim, 3)  # Risk scores
        self.financial_head = nn.Linear(head_dim, 1)  # Budget sensitivity
        self.mobility_head = nn.Linear(head_dim, 1)  # Mobility score
        
        # Uncertainty estimation (MC Dropout enabled)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through all task heads"""
        shared_features = self.shared(x)
        
        return {
            'food_insecurity': torch.softmax(self.food_insecurity_head(shared_features), dim=-1),
            'transport_constraint': torch.softmax(self.transport_head(shared_features), dim=-1),
            'chronic_condition': torch.sigmoid(self.chronic_condition_head(shared_features)),
            'financial_constraint': torch.sigmoid(self.financial_head(shared_features)),
            'mobility_limitation': torch.sigmoid(self.mobility_head(shared_features))
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, Any]:
        """
        Predict with uncertainty using MC Dropout
        
        Returns:
            Predictions with mean and std for uncertainty quantification
        """
        # Set to eval mode first to avoid BatchNorm issues with single samples
        self.eval()
        
        predictions = []
        
        # Enable dropout manually for MC Dropout while keeping BatchNorm in eval mode
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()
        
        # Compute mean and std
        result = {}
        for key in predictions[0].keys():
            values = torch.stack([p[key] for p in predictions])
            result[key] = {
                'mean': values.mean(dim=0),
                'std': values.std(dim=0),
                'confidence': 1.0 - values.std(dim=0).mean().item()
            }
        
        return result


class ContextualBandit:
    """
    Contextual Bandit for policy selection
    
    Implements LinUCB algorithm with fairness constraints
    """
    
    def __init__(self, n_actions: int, context_dim: int, alpha: float = 1.0):
        """
        Initialize contextual bandit
        
        Args:
            n_actions: Number of policies
            context_dim: Dimension of context vector
            alpha: Exploration parameter (higher = more exploration)
        """
        self.n_actions = n_actions
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Initialize A and b for each action (LinUCB)
        self.A = [np.eye(context_dim) for _ in range(n_actions)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_actions)]
        
        # Track policy performance
        self.policy_counts = np.zeros(n_actions)
        self.policy_rewards = np.zeros(n_actions)
    
    def select_action(self, context: np.ndarray, valid_actions: Optional[list] = None) -> int:
        """
        Select action using LinUCB
        
        Args:
            context: Context vector (features)
            valid_actions: List of valid action indices (None = all valid)
            
        Returns:
            Selected action index
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))
        
        context = context.reshape(-1, 1)
        ucb_values = []
        
        for action in valid_actions:
            A_inv = np.linalg.inv(self.A[action])
            theta = A_inv @ self.b[action]
            
            # UCB = expected reward + exploration bonus
            expected_reward = theta.T @ context
            confidence_bonus = self.alpha * np.sqrt(context.T @ A_inv @ context)
            ucb = expected_reward + confidence_bonus
            
            ucb_values.append((action, ucb[0, 0]))
        
        # Select action with highest UCB
        best_action = max(ucb_values, key=lambda x: x[1])[0]
        return best_action
    
    def update(self, action: int, context: np.ndarray, reward: float):
        """
        Update bandit parameters
        
        Args:
            action: Action taken
            context: Context vector
            reward: Observed reward
        """
        context = context.reshape(-1, 1)
        
        # Update A and b (LinUCB update)
        self.A[action] += context @ context.T
        self.b[action] += reward * context
        
        # Track statistics
        self.policy_counts[action] += 1
        self.policy_rewards[action] += reward
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy performance statistics"""
        avg_rewards = np.divide(
            self.policy_rewards, 
            self.policy_counts,
            out=np.zeros_like(self.policy_rewards),
            where=self.policy_counts > 0
        )
        
        return {
            'counts': self.policy_counts.tolist(),
            'total_rewards': self.policy_rewards.tolist(),
            'avg_rewards': avg_rewards.tolist()
        }


class ReasoningModule:
    """
    Reasoning Module: Thinks about needs and selects policies
    
    Responsibilities:
    1. Infer need states from context (multi-task NN)
    2. Quantify uncertainty (MC Dropout)
    3. Select policy (contextual bandit)
    4. Ensure confidence thresholds
    """
    
    def __init__(self, config: EACConfig):
        self.config = config
        self.logger = logging.getLogger("EACAgent.Reasoning")
        
        # Initialize need state model
        self.need_state_model = NeedStateModel(
            input_dim=config.bandit_context_dim,
            hidden_dim=256
        )
        
        # Load pretrained model if available
        if config.need_state_model_path:
            try:
                self.need_state_model.load_state_dict(
                    torch.load(config.need_state_model_path)
                )
                self.logger.info(f"Loaded need state model from {config.need_state_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}, using random initialization")
        
        self.need_state_model.eval()
        
        # Initialize contextual bandit
        self.bandit = ContextualBandit(
            n_actions=len(config.enabled_policies),
            context_dim=config.bandit_context_dim,
            alpha=config.bandit_alpha
        )
        
        # Policy mapping
        self.policy_names = config.enabled_policies
        self.policy_to_idx = {name: idx for idx, name in enumerate(self.policy_names)}
        
        self.logger.info(f"Reasoning Module initialized with {len(self.policy_names)} policies")
    
    def infer_needs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer need states from context
        
        Args:
            context: Context from perception module
            
        Returns:
            Need states with confidence scores
        """
        # Get feature vector
        features = context.get('features')
        if features is None:
            self.logger.error("No features in context")
            return {'confidence': 0.0}
        
        # Convert to tensor
        x = torch.from_numpy(features).float().unsqueeze(0)
        
        # Predict with uncertainty
        predictions = self.need_state_model.predict_with_uncertainty(x, n_samples=10)
        
        # Extract need states
        need_states = {
            'food_insecurity': self._extract_risk_level(predictions['food_insecurity']),
            'transport_constraint': self._extract_binary(predictions['transport_constraint']),
            'chronic_condition': self._extract_risk_scores(predictions['chronic_condition']),
            'financial_constraint': predictions['financial_constraint']['mean'].item(),
            'mobility_limitation': predictions['mobility_limitation']['mean'].item(),
        }
        
        # Compute overall confidence
        confidences = [pred['confidence'] for pred in predictions.values()]
        overall_confidence = np.mean(confidences)
        
        need_states['confidence'] = overall_confidence
        need_states['predictions'] = predictions  # Store full predictions
        
        self.logger.debug(f"Need states inferred: confidence={overall_confidence:.3f}")
        
        return need_states
    
    def select_policy(self, context: Dict[str, Any], need_states: Dict[str, Any]) -> str:
        """
        Select policy using contextual bandit
        
        Args:
            context: Context from perception
            need_states: Inferred need states
            
        Returns:
            Selected policy name
        """
        # Get feature vector
        features = context.get('features')
        
        # Determine valid policies based on need states and constraints
        valid_policies = self._get_valid_policies(context, need_states)
        valid_indices = [self.policy_to_idx[p] for p in valid_policies]
        
        if not valid_indices:
            self.logger.warning("No valid policies, using safe default")
            return "safe_default"
        
        # Select action using bandit
        action_idx = self.bandit.select_action(features, valid_actions=valid_indices)
        selected_policy = self.policy_names[action_idx]
        
        self.logger.debug(f"Policy selected: {selected_policy} (UCB-based)")
        
        return selected_policy
    
    def _get_valid_policies(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> list:
        """
        Determine which policies are valid for this context
        
        Returns:
            List of valid policy names
        """
        valid = []
        
        # SNAP/WIC: Valid if user has SNAP/EBT or high food insecurity
        if ('SNAP_EBT' in context.get('payment_methods', []) or 
            need_states.get('food_insecurity', {}).get('level') in ['medium', 'high']):
            if 'snap_wic_substitution' in self.policy_names:
                valid.append('snap_wic_substitution')
        
        # Low-glycemic: Valid if chronic condition risk or nutrition concerned
        if (need_states.get('chronic_condition', {}).get('diabetes', 0) > 0.5 or
            context.get('constraints', {}).get('nutrition_concerned', False)):
            if 'low_glycemic_alternative' in self.policy_names:
                valid.append('low_glycemic_alternative')
        
        # OTC coverage: Valid if has FSA/HSA
        if any(pm in ['FSA', 'HSA'] for pm in context.get('payment_methods', [])):
            if 'otc_coverage' in self.policy_names:
                valid.append('otc_coverage')
        
        # Mobility delivery: Valid if mobility limited
        if (need_states.get('mobility_limitation', 0) > 0.6 or
            context.get('constraints', {}).get('mobility_limited', False)):
            if 'mobility_delivery' in self.policy_names:
                valid.append('mobility_delivery')
        
        # Safety nudge: Always valid
        if 'safety_nudge' in self.policy_names:
            valid.append('safety_nudge')
        
        return valid if valid else self.policy_names  # Fallback to all
    
    def update_bandit(self, policy: str, features: np.ndarray, reward: float):
        """
        Update bandit from feedback
        
        Args:
            policy: Policy that was executed
            features: Context features
            reward: Observed reward
        """
        if policy in self.policy_to_idx:
            action_idx = self.policy_to_idx[policy]
            self.bandit.update(action_idx, features, reward)
            self.logger.debug(f"Bandit updated: policy={policy}, reward={reward:.3f}")
    
    def get_policy_stats(self) -> Dict[str, Any]:
        """Get policy performance statistics"""
        stats = self.bandit.get_policy_stats()
        stats['policy_names'] = self.policy_names
        return stats
    
    @staticmethod
    def _extract_risk_level(prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract risk level from softmax prediction"""
        probs = prediction['mean'].squeeze().numpy()
        level_names = ['low', 'medium', 'high']
        level_idx = np.argmax(probs)
        
        return {
            'level': level_names[level_idx],
            'probabilities': {name: float(prob) for name, prob in zip(level_names, probs)},
            'confidence': prediction['confidence']
        }
    
    @staticmethod
    def _extract_binary(prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract binary prediction"""
        probs = prediction['mean'].squeeze().numpy()
        has_constraint = probs[1] > 0.5
        
        return {
            'has_constraint': bool(has_constraint),
            'probability': float(probs[1]),
            'confidence': prediction['confidence']
        }
    
    @staticmethod
    def _extract_risk_scores(prediction: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Extract risk scores for chronic conditions"""
        scores = prediction['mean'].squeeze().numpy()
        condition_names = ['diabetes', 'cvd', 'other']
        
        return {
            name: float(score) 
            for name, score in zip(condition_names, scores)
        }
