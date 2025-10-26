"""
Outcome Models for Simulation

Models user behavior and outcomes:
1. Acceptance Model: P(user accepts recommendation)
2. Spend Impact Model: Change in out-of-pocket spend
3. Nutrition Impact Model: Change in nutritional quality
4. Satisfaction Model: Customer satisfaction proxy
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class OutcomeModels:
    """
    Outcome models for simulation
    
    These models predict how users respond to recommendations
    Can use advanced ML models if available
    """
    
    def __init__(self, use_advanced_models: bool = False):
        self.logger = logging.getLogger("EAC.Simulation.Models")
        
        # Model parameters (fallback for simple model)
        self.acceptance_base_rate = 0.6  # 60% base acceptance
        self.price_sensitivity = 0.5  # How much price affects acceptance
        self.nutrition_value = 0.3  # How much nutrition affects acceptance
        
        # Advanced models (optional)
        self.use_advanced_models = use_advanced_models
        self.xgboost_acceptance = None
        self.product_embeddings = None
        self.collaborative_filter = None
        
        if use_advanced_models:
            self._load_advanced_models()
        
        self.logger.info(f"Outcome Models initialized (advanced={use_advanced_models})")
    
    def _load_advanced_models(self):
        """Load advanced ML models if available"""
        try:
            from models.acceptance import XGBoostAcceptanceModel
            from models.embeddings import ProductTransformer
            from models.collaborative import CollaborativeFilter
            
            # Try to load trained models
            try:
                self.xgboost_acceptance = XGBoostAcceptanceModel()
                self.xgboost_acceptance.load('models/acceptance_model.pkl')
                self.logger.info("Loaded XGBoost acceptance model")
            except:
                self.logger.warning("XGBoost model not found, using base model")
                self.xgboost_acceptance = None
            
            try:
                self.product_embeddings = ProductTransformer()
                self.product_embeddings.load('models/embeddings.pt')
                self.logger.info("Loaded product embeddings")
            except:
                self.logger.warning("Product embeddings not found, using base model")
                self.product_embeddings = None
            
            try:
                self.collaborative_filter = CollaborativeFilter()
                self.collaborative_filter.load('models/collaborative.pkl')
                self.logger.info("Loaded collaborative filter")
            except:
                self.logger.warning("Collaborative filter not found, using base model")
                self.collaborative_filter = None
                
        except ImportError as e:
            self.logger.warning(f"Advanced models not available: {e}")
            self.use_advanced_models = False
    
    def simulate_acceptance(
        self,
        recommendations: List[Dict[str, Any]],
        transaction: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Simulate which recommendations user accepts
        
        Args:
            recommendations: List of recommendations from agent
            transaction: Transaction data
            
        Returns:
            List of accepted recommendations
        """
        accepted = []
        
        for rec in recommendations:
            # Compute acceptance probability
            p_accept = self._compute_acceptance_probability(rec, transaction)
            
            # Simulate acceptance (Bernoulli trial)
            if np.random.random() < p_accept:
                accepted.append(rec)
        
        return accepted
    
    def _compute_acceptance_probability(
        self,
        recommendation: Dict[str, Any],
        transaction: pd.Series
    ) -> float:
        """
        Compute probability user accepts recommendation
        
        Uses XGBoost model if available, otherwise falls back to heuristic
        
        Factors:
        - Savings (higher = more likely to accept)
        - Nutrition improvement (higher = more likely)
        - Confidence (higher = more likely)
        - User price sensitivity
        - Product similarity (if embeddings available)
        """
        # Use XGBoost if available
        if self.xgboost_acceptance is not None:
            try:
                user_features = {
                    'past_acceptance_rate': transaction.get('past_acceptance_rate', 0.5),
                    'price_sensitivity': transaction.get('price_sensitivity', 0.5),
                    'order_count': transaction.get('order_count', 1),
                    'food_insecurity': transaction.get('sdoh_food_insecurity', 0),
                    'financial_constraint': transaction.get('sdoh_financial_constraint', 0),
                    'mobility_limitation': transaction.get('sdoh_mobility_limitation', 0),
                    'health_risk': transaction.get('sdoh_health_risk', 0)
                }
                
                # Add product similarity if embeddings available
                if self.product_embeddings is not None:
                    try:
                        similarity = self.product_embeddings.compute_similarity(
                            {'name': recommendation.get('original_product_name', ''),
                             'category': recommendation.get('category', '')},
                            {'name': recommendation.get('suggested_product_name', ''),
                             'category': recommendation.get('category', '')}
                        )
                        recommendation['category_match'] = similarity > 0.7
                        recommendation['brand_match'] = similarity > 0.8
                    except:
                        pass
                
                p_accept = self.xgboost_acceptance.predict_proba(recommendation, user_features)
                return float(p_accept)
            except Exception as e:
                self.logger.warning(f"XGBoost prediction failed: {e}, using fallback")
        
        # Fallback to heuristic model
        # Base acceptance rate
        p_accept = self.acceptance_base_rate
        
        # Savings effect (positive savings increases acceptance)
        savings = recommendation.get('savings', 0)
        if savings > 0:
            p_accept += savings * self.price_sensitivity * 0.1  # $1 saved = +5% acceptance
        elif savings < 0:
            p_accept += savings * self.price_sensitivity * 0.2  # Price increase hurts more
        
        # Nutrition effect
        nutrition_gain = recommendation.get('nutrition_improvement', 0)
        if nutrition_gain > 0:
            p_accept += nutrition_gain * self.nutrition_value * 0.01  # +1 HEI = +0.3% acceptance
        
        # Confidence effect
        confidence = recommendation.get('confidence', 0.5)
        p_accept *= confidence  # Scale by confidence
        
        # User-specific factors
        income = transaction.get('income', 50000)
        if income < 30000:
            # Low-income users more price-sensitive
            p_accept += savings * 0.15
        
        # Clip to [0, 1]
        return np.clip(p_accept, 0, 1)
    
    def compute_nutrition_score(self, cart: List[Dict[str, Any]]) -> float:
        """
        Compute nutrition score for cart (HEI-like)
        
        Simplified Healthy Eating Index:
        - Higher fiber = better
        - Lower sugar = better
        - Lower sodium = better
        - More produce = better
        
        Returns score 0-100
        """
        if not cart:
            return 50  # Neutral
        
        score = 50  # Start at neutral
        
        for item in cart:
            nutrition = item.get('nutrition', {})
            
            # Fiber (higher is better)
            fiber = nutrition.get('fiber_g', 0)
            score += fiber * 2
            
            # Sugar (lower is better)
            sugar = nutrition.get('sugar_g', 0)
            score -= sugar * 0.5
            
            # Sodium (lower is better)
            sodium = nutrition.get('sodium_mg', 0)
            score -= sodium * 0.01
            
            # Category bonuses
            category = item.get('category', '')
            if category in ['fruits', 'vegetables']:
                score += 10
            elif category in ['whole_grains']:
                score += 5
        
        return np.clip(score, 0, 100)
    
    def compute_satisfaction(
        self,
        recommendations: List[Dict[str, Any]],
        accepted_recommendations: List[Dict[str, Any]],
        spend_delta: float
    ) -> float:
        """
        Compute customer satisfaction (NPS-like score 0-100)
        
        Factors:
        - Acceptance rate (higher = more satisfied)
        - Savings (higher = more satisfied)
        - Friction (too many recs = less satisfied)
        """
        base_satisfaction = 50  # Neutral
        
        if not recommendations:
            return base_satisfaction
        
        # Acceptance rate effect
        acceptance_rate = len(accepted_recommendations) / len(recommendations)
        base_satisfaction += acceptance_rate * 30  # Up to +30 points
        
        # Savings effect
        if spend_delta < 0:  # Negative delta = savings
            base_satisfaction += min(abs(spend_delta) * 0.5, 20)  # Up to +20 points
        
        # Friction penalty (too many recommendations)
        if len(recommendations) > 5:
            base_satisfaction -= (len(recommendations) - 5) * 2
        
        return np.clip(base_satisfaction, 0, 100)
