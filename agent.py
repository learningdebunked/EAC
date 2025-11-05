"""
Core EAC Agent - Implements Observe-Think-Act-Learn cycle
"""
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from config import EACConfig
from modules.perception import PerceptionModule
from modules.reasoning import ReasoningModule
from modules.action import ActionModule
from modules.learning import LearningModule
from modules.guardrails import GuardrailSystem
from utils.monitoring import AgentMonitor


@dataclass
class CheckoutEvent:
    """Represents a checkout event"""
    user_id: str
    cart: List[Dict[str, Any]]
    delivery_address: Dict[str, str]
    payment_methods: List[str]
    timestamp: float
    consent: Dict[str, bool]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Agent's response to checkout event"""
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    policy_used: str
    confidence: float
    fairness_check: str
    explanation: str
    metadata: Dict[str, Any]


class EACAgent:
    """
    Equity-Aware Checkout AI Agent
    
    Implements the Observe-Think-Act-Learn cycle:
    1. OBSERVE: Perceive checkout context (cart, SDOH, constraints)
    2. THINK: Infer needs and select policy (contextual bandit)
    3. ACT: Execute policy and generate recommendations
    4. LEARN: Update from user feedback (online learning)
    """
    
    def __init__(self, config: Optional[EACConfig] = None):
        """
        Initialize EAC Agent
        
        Args:
            config: Agent configuration
        """
        self.config = config or EACConfig()
        self.logger = self._setup_logging()
        
        # Initialize modules
        self.logger.info("Initializing EAC Agent modules...")
        self.perception = PerceptionModule(self.config)
        self.reasoning = ReasoningModule(self.config)
        self.action = ActionModule(self.config)
        self.learning = LearningModule(self.config)
        self.guardrails = GuardrailSystem(self.config)
        
        # Monitoring
        self.monitor = AgentMonitor(self.config)
        
        self.logger.info(f"EAC Agent initialized: {self.config.agent_name}")
    
    def process_checkout(self, event: CheckoutEvent) -> AgentResponse:
        """
        Main agent loop: Observe → Think → Act → Learn
        
        Args:
            event: Checkout event to process
            
        Returns:
            AgentResponse with recommendations
        """
        start_time = time.time()
        
        try:
            # Check consent
            if not event.consent.get('personalization', False):
                return self._safe_default_response(
                    "User has not consented to personalization",
                    time.time() - start_time
                )
            
            # 1. OBSERVE: Perceive environment
            self.logger.debug(f"OBSERVE: Processing checkout for user {event.user_id}")
            context = self.perception.observe(event)
            
            # Check latency budget
            if (time.time() - start_time) * 1000 > self.config.max_latency_ms * 0.5:
                self.logger.warning("Latency budget exceeded in perception, using safe default")
                return self._safe_default_response(
                    "Latency budget exceeded",
                    time.time() - start_time
                )
            
            # 2. THINK: Reason about needs and select policy
            self.logger.debug("THINK: Inferring need states and selecting policy")
            need_states = self.reasoning.infer_needs(context)
            
            # Check confidence
            if need_states.get('confidence', 0) < self.config.need_state_confidence_threshold:
                self.logger.warning(f"Low confidence in need states: {need_states.get('confidence')}")
                if self.config.safe_mode:
                    return self._safe_default_response(
                        "Low confidence in need state prediction",
                        time.time() - start_time
                    )
            
            # Select policy using contextual bandit
            policy = self.reasoning.select_policy(context, need_states)
            
            # 3. CHECK GUARDRAILS
            self.logger.debug("GUARDRAILS: Checking fairness and safety constraints")
            guardrail_result = self.guardrails.check(policy, context, need_states)
            
            if not guardrail_result['passed']:
                self.logger.warning(f"Guardrail violation: {guardrail_result['reason']}")
                return self._safe_default_response(
                    f"Guardrail violation: {guardrail_result['reason']}",
                    time.time() - start_time
                )
            
            # 4. ACT: Execute policy and generate recommendations
            self.logger.debug(f"ACT: Executing policy '{policy}'")
            recommendations = self.action.execute_policy(policy, context, need_states)
            
            # Final latency check
            latency_ms = (time.time() - start_time) * 1000
            if latency_ms > self.config.max_latency_ms:
                self.logger.error(f"Latency SLA violated: {latency_ms:.2f}ms > {self.config.max_latency_ms}ms")
                self.monitor.record_latency_violation(latency_ms)
                return self._safe_default_response(
                    "Latency SLA violated",
                    time.time() - start_time
                )
            
            # Build response
            response = AgentResponse(
                recommendations=recommendations,
                latency_ms=latency_ms,
                policy_used=policy,
                confidence=need_states.get('confidence', 0.0),
                fairness_check="passed",
                explanation=self._generate_explanation(policy, recommendations),
                metadata={
                    'need_states': need_states,
                    'context_features': context.get('features', {}),
                    'guardrail_details': guardrail_result
                }
            )
            
            # Record metrics
            self.monitor.record_decision(response)
            
            self.logger.info(
                f"Checkout processed: user={event.user_id}, "
                f"policy={policy}, recs={len(recommendations)}, "
                f"latency={latency_ms:.2f}ms"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing checkout: {e}", exc_info=True)
            self.monitor.record_error(str(e))
            return self._safe_default_response(
                f"Error: {str(e)}",
                time.time() - start_time
            )
    
    def update_from_feedback(
        self, 
        event: CheckoutEvent, 
        response: AgentResponse, 
        user_feedback: Dict[str, Any]
    ):
        """
        5. LEARN: Update agent from user feedback
        
        Args:
            event: Original checkout event
            response: Agent's response
            user_feedback: User's response (accepted/rejected recommendations)
        """
        try:
            # Compute reward
            reward = self._compute_reward(response, user_feedback)
            
            # Update learning module
            self.learning.update(
                context=response.metadata.get('context_features', {}),
                policy=response.policy_used,
                reward=reward,
                user_feedback=user_feedback
            )
            
            self.logger.info(
                f"Learning update: user={event.user_id}, "
                f"policy={response.policy_used}, reward={reward:.3f}"
            )
            
            # Record metrics
            self.monitor.record_feedback(user_feedback, reward)
            
        except Exception as e:
            self.logger.error(f"Error updating from feedback: {e}", exc_info=True)
    
    def _compute_reward(
        self, 
        response: AgentResponse, 
        user_feedback: Dict[str, Any]
    ) -> float:
        """
        Compute multi-objective reward using Nash Equilibrium
        
        Paper Section IV.C, Equations 6-8:
        Three-player game (Users, Business, Equity) with alternating optimization
        
        Balances:
        - User utility: satisfaction, savings, nutrition
        - Business utility: revenue, retention
        - Equity utility: coverage, fairness
        """
        # Check if Nash equilibrium mode is enabled
        if hasattr(self.config, 'use_nash_equilibrium') and self.config.use_nash_equilibrium:
            return self._compute_nash_equilibrium_reward(response, user_feedback)
        else:
            # Fallback to weighted sum (legacy mode)
            return self._compute_weighted_sum_reward(response, user_feedback)
    
    def _compute_nash_equilibrium_reward(
        self,
        response: AgentResponse,
        user_feedback: Dict[str, Any]
    ) -> float:
        """
        Nash Equilibrium multi-objective reward
        
        Implements alternating gradient updates from paper:
        θ^{k+1}_U = θ^k_U + η∇U(θ_U, θ^k_B, θ^k_E)
        θ^{k+1}_B = θ^k_B + η∇B(θ^{k+1}_U, θ_B, θ^k_E)
        θ^{k+1}_E = θ^k_E + η∇E(θ^{k+1}_U, θ^{k+1}_B, θ_E)
        """
        # Initialize player utilities
        if not hasattr(self, '_nash_params'):
            self._nash_params = {
                'theta_U': np.zeros(3),  # User utility parameters
                'theta_B': np.zeros(3),  # Business utility parameters
                'theta_E': np.zeros(2),  # Equity utility parameters
                'learning_rates': {'U': 0.01, 'B': 0.01, 'E': 0.01}
            }
        
        # Extract metrics from feedback
        acceptance_rate = (
            user_feedback.get('accepted_count', 0) / 
            user_feedback.get('total_recommendations', 1)
        )
        savings = user_feedback.get('total_savings', 0.0)
        nutrition_gain = user_feedback.get('nutrition_improvement', 0.0)
        revenue_impact = user_feedback.get('revenue_impact', 0.0)
        retention_score = user_feedback.get('retention_score', 0.5)
        fairness_score = 1.0 - float(user_feedback.get('fairness_violation', False))
        
        # Compute individual utilities
        U_user = (
            self._nash_params['theta_U'][0] * acceptance_rate +
            self._nash_params['theta_U'][1] * (savings / 20.0) +  # Normalize
            self._nash_params['theta_U'][2] * (nutrition_gain / 20.0)
        )
        
        U_business = (
            self._nash_params['theta_B'][0] * (revenue_impact / 100.0) +
            self._nash_params['theta_B'][1] * retention_score +
            self._nash_params['theta_B'][2] * acceptance_rate  # Engagement
        )
        
        U_equity = (
            self._nash_params['theta_E'][0] * fairness_score +
            self._nash_params['theta_E'][1] * (savings / 20.0)  # Benefit to vulnerable
        )
        
        # Compute gradients (simplified - in production use autograd)
        grad_U = np.array([acceptance_rate, savings / 20.0, nutrition_gain / 20.0])
        grad_B = np.array([revenue_impact / 100.0, retention_score, acceptance_rate])
        grad_E = np.array([fairness_score, savings / 20.0])
        
        # Alternating gradient updates (paper Equations 6-8)
        eta = self._nash_params['learning_rates']
        
        # Update user parameters
        self._nash_params['theta_U'] = self._project_simplex(
            self._nash_params['theta_U'] + eta['U'] * grad_U
        )
        
        # Update business parameters (conditioned on updated user)
        self._nash_params['theta_B'] = self._project_simplex(
            self._nash_params['theta_B'] + eta['B'] * grad_B
        )
        
        # Update equity parameters (conditioned on updated user & business)
        self._nash_params['theta_E'] = self._project_simplex(
            self._nash_params['theta_E'] + eta['E'] * grad_E
        )
        
        # Compute equilibrium reward (weighted combination)
        # Weights from paper: balance stakeholder interests
        alpha_U, alpha_B, alpha_E = 0.5, 0.3, 0.2  # User-centric weighting
        
        nash_reward = alpha_U * U_user + alpha_B * U_business + alpha_E * U_equity
        
        return nash_reward
    
    def _compute_weighted_sum_reward(
        self,
        response: AgentResponse,
        user_feedback: Dict[str, Any]
    ) -> float:
        """
        Legacy weighted sum reward (fallback)
        """
        reward = 0.0
        weights = self.config.reward_weights
        
        # User satisfaction (acceptance rate)
        if user_feedback.get('total_recommendations', 0) > 0:
            acceptance_rate = (
                user_feedback.get('accepted_count', 0) / 
                user_feedback['total_recommendations']
            )
            reward += acceptance_rate * weights['acceptance']
        
        # Cost savings
        savings = user_feedback.get('total_savings', 0.0)
        reward += savings * weights['cost_savings']
        
        # Nutrition improvement
        nutrition_gain = user_feedback.get('nutrition_improvement', 0.0)
        reward += nutrition_gain * weights['nutrition_improvement']
        
        # Fairness penalty
        if user_feedback.get('fairness_violation', False):
            reward += weights['fairness_violation']
        
        return reward
    
    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """
        Project vector onto probability simplex
        
        Ensures parameters sum to 1 and are non-negative
        Used for Nash equilibrium parameter updates
        """
        # Sort in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        w = np.maximum(v - theta, 0)
        return w / (w.sum() + 1e-10)  # Normalize
    
    def _safe_default_response(self, reason: str, elapsed_time: float) -> AgentResponse:
        """Return safe default response (no recommendations)"""
        return AgentResponse(
            recommendations=[],
            latency_ms=elapsed_time * 1000,
            policy_used="safe_default",
            confidence=0.0,
            fairness_check="skipped",
            explanation=f"No recommendations: {reason}",
            metadata={'reason': reason}
        )
    
    def _generate_explanation(
        self, 
        policy: str, 
        recommendations: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable explanation"""
        if not recommendations:
            return "No recommendations at this time."
        
        policy_explanations = {
            'snap_wic_substitution': "Based on your payment method, we found SNAP-eligible alternatives",
            'low_glycemic_alternative': "We found healthier alternatives that may help manage blood sugar",
            'otc_coverage': "These items may be covered by your FSA/HSA",
            'mobility_delivery': "We adjusted delivery times to match your schedule",
            'safety_nudge': "We found ways to improve nutrition and save money"
        }
        
        base_explanation = policy_explanations.get(
            policy, 
            "We found personalized recommendations for you"
        )
        
        return f"{base_explanation}. {len(recommendations)} suggestion(s) available."
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("EACAgent")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_file:
            # Create log directory if it doesn't exist
            from pathlib import Path
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return self.monitor.get_stats()
