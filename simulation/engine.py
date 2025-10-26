"""
Counterfactual Simulation Engine

Simulates EAC agent performance on historical transaction data
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from agent import EACAgent, CheckoutEvent
from config import EACConfig
from simulation.models import OutcomeModels


class SimulationEngine:
    """
    Counterfactual Simulation Engine
    
    Compares:
    - Control: Baseline (no personalization)
    - Treatment: EAC system with policies
    
    For each transaction, simulates both arms and computes treatment effect
    """
    
    def __init__(self, config: Optional[EACConfig] = None, use_advanced_models: bool = False):
        self.config = config or EACConfig()
        self.logger = logging.getLogger("EAC.Simulation")
        
        # Initialize agent
        self.agent = EACAgent(self.config)
        
        # Initialize outcome models
        self.outcome_models = OutcomeModels(use_advanced_models=use_advanced_models)
        
        self.logger.info(f"Simulation Engine initialized (advanced_models={use_advanced_models})")
    
    def run_simulation(
        self,
        transactions: pd.DataFrame,
        n_replications: int = 100,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Run counterfactual simulation
        
        Args:
            transactions: DataFrame with transaction data
            n_replications: Number of simulation replications
            random_seed: Random seed for reproducibility
            
        Returns:
            DataFrame with simulation results
        """
        self.logger.info(
            f"Starting simulation: {len(transactions)} transactions, "
            f"{n_replications} replications"
        )
        
        results = []
        
        for replication in range(n_replications):
            # Set seed for reproducibility
            np.random.seed(random_seed + replication)
            
            self.logger.info(f"Replication {replication + 1}/{n_replications}")
            
            for idx, transaction in tqdm(transactions.iterrows(), total=len(transactions)):
                # Run counterfactual for this transaction
                result = self._run_counterfactual(transaction, replication)
                results.append(result)
        
        results_df = pd.DataFrame(results)
        
        self.logger.info(f"Simulation complete: {len(results)} observations")
        
        return results_df
    
    def _run_counterfactual(
        self,
        transaction: pd.Series,
        replication: int
    ) -> Dict[str, Any]:
        """
        Run counterfactual for a single transaction
        
        Simulates both control and treatment arms
        """
        # CONTROL ARM: Baseline (no personalization)
        control_outcome = self._simulate_control(transaction)
        
        # TREATMENT ARM: EAC system
        treatment_outcome = self._simulate_treatment(transaction)
        
        # Compute treatment effect
        treatment_effect = {
            'user_id': transaction.get('user_id'),
            'transaction_id': transaction.get('transaction_id'),
            'replication': replication,
            
            # Control outcomes
            'control_spend': control_outcome['spend'],
            'control_nutrition': control_outcome['nutrition'],
            'control_satisfaction': control_outcome['satisfaction'],
            
            # Treatment outcomes
            'treatment_spend': treatment_outcome['spend'],
            'treatment_nutrition': treatment_outcome['nutrition'],
            'treatment_satisfaction': treatment_outcome['satisfaction'],
            'treatment_recommendations': treatment_outcome['n_recommendations'],
            'treatment_accepted': treatment_outcome['n_accepted'],
            
            # Treatment effects (delta)
            'delta_spend': treatment_outcome['spend'] - control_outcome['spend'],
            'delta_nutrition': treatment_outcome['nutrition'] - control_outcome['nutrition'],
            'delta_satisfaction': treatment_outcome['satisfaction'] - control_outcome['satisfaction'],
            
            # Acceptance rate
            'acceptance_rate': (
                treatment_outcome['n_accepted'] / treatment_outcome['n_recommendations']
                if treatment_outcome['n_recommendations'] > 0 else 0
            ),
            
            # Protected attributes (for fairness analysis)
            'protected_group': transaction.get('race', 'unknown'),
            'income_group': 'low' if transaction.get('income', 50000) < 50000 else 'high',
            
            # Metadata
            'policy_used': treatment_outcome.get('policy_used'),
            'latency_ms': treatment_outcome.get('latency_ms', 0)
        }
        
        return treatment_effect
    
    def _simulate_control(self, transaction: pd.Series) -> Dict[str, Any]:
        """
        Simulate control arm (no personalization)
        
        Returns baseline outcomes
        """
        # Original cart
        cart = transaction.get('cart', [])
        
        # Compute baseline outcomes
        total_spend = sum(item.get('price', 0) * item.get('quantity', 1) for item in cart)
        nutrition_score = self.outcome_models.compute_nutrition_score(cart)
        satisfaction = 50  # Neutral baseline
        
        return {
            'spend': total_spend,
            'nutrition': nutrition_score,
            'satisfaction': satisfaction
        }
    
    def _simulate_treatment(self, transaction: pd.Series) -> Dict[str, Any]:
        """
        Simulate treatment arm (EAC system)
        
        Applies agent and simulates user response
        """
        # Create checkout event
        event = self._transaction_to_event(transaction)
        
        # Get agent recommendations
        response = self.agent.process_checkout(event)
        
        # Simulate user acceptance
        accepted_recs = self.outcome_models.simulate_acceptance(
            response.recommendations,
            transaction
        )
        
        # Apply accepted recommendations to cart
        modified_cart = self._apply_recommendations(
            transaction.get('cart', []),
            accepted_recs
        )
        
        # Compute treatment outcomes
        total_spend = sum(item.get('price', 0) * item.get('quantity', 1) for item in modified_cart)
        nutrition_score = self.outcome_models.compute_nutrition_score(modified_cart)
        satisfaction = self.outcome_models.compute_satisfaction(
            response.recommendations,
            accepted_recs,
            total_spend - sum(item.get('price', 0) * item.get('quantity', 1) for item in transaction.get('cart', []))
        )
        
        return {
            'spend': total_spend,
            'nutrition': nutrition_score,
            'satisfaction': satisfaction,
            'n_recommendations': len(response.recommendations),
            'n_accepted': len(accepted_recs),
            'policy_used': response.policy_used,
            'latency_ms': response.latency_ms
        }
    
    def _transaction_to_event(self, transaction: pd.Series) -> CheckoutEvent:
        """Convert transaction data to CheckoutEvent"""
        return CheckoutEvent(
            user_id=transaction.get('user_id', 'sim_user'),
            cart=transaction.get('cart', []),
            delivery_address={
                'zip_code': transaction.get('zip_code', '94102'),
                'census_tract': transaction.get('census_tract')
            },
            payment_methods=transaction.get('payment_methods', ['CREDIT_CARD']),
            timestamp=time.time(),
            consent={
                'personalization': True,
                'sdoh_signals': True
            }
        )
    
    def _apply_recommendations(
        self,
        original_cart: List[Dict],
        accepted_recommendations: List[Dict]
    ) -> List[Dict]:
        """
        Apply accepted recommendations to cart
        
        Returns modified cart
        """
        modified_cart = original_cart.copy()
        
        for rec in accepted_recommendations:
            # Find and replace original product
            for i, item in enumerate(modified_cart):
                if item.get('product_id') == rec.get('original_product_id'):
                    # Replace with suggested product
                    modified_cart[i] = {
                        'product_id': rec.get('suggested_product_id'),
                        'name': rec.get('suggested_product_name'),
                        'price': item['price'] - rec.get('savings', 0),
                        'quantity': item['quantity']
                    }
                    break
        
        return modified_cart
