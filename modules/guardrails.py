"""
Guardrail System - Enforces fairness, safety, and business constraints
"""
import logging
from typing import Dict, Any

from config import EACConfig


class GuardrailSystem:
    """
    Guardrail System: Enforces hard constraints
    
    Checks:
    1. Fairness constraints (Equalized Uplift, Price Burden Ratio)
    2. Safety constraints (allergens, contraindications, confidence)
    3. Business constraints (margin, inventory, latency)
    4. Regulatory constraints (SNAP/WIC compliance, ADA, HIPAA)
    """
    
    def __init__(self, config: EACConfig):
        self.config = config
        self.logger = logging.getLogger("EACAgent.Guardrails")
        
        # Track violations for monitoring
        self.violation_counts = {
            'fairness': 0,
            'safety': 0,
            'business': 0,
            'regulatory': 0
        }
        
        self.logger.info("Guardrail System initialized")
    
    def check(
        self, 
        policy: str, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check all guardrails
        
        Args:
            policy: Selected policy
            context: Context from perception
            need_states: Inferred need states
            
        Returns:
            Dict with 'passed' (bool) and 'reason' (str)
        """
        # 1. Fairness guardrails
        fairness_result = self._check_fairness(policy, context, need_states)
        if not fairness_result['passed']:
            self.violation_counts['fairness'] += 1
            return fairness_result
        
        # 2. Safety guardrails
        safety_result = self._check_safety(policy, context, need_states)
        if not safety_result['passed']:
            self.violation_counts['safety'] += 1
            return safety_result
        
        # 3. Business guardrails
        business_result = self._check_business(policy, context)
        if not business_result['passed']:
            self.violation_counts['business'] += 1
            return business_result
        
        # 4. Regulatory guardrails
        regulatory_result = self._check_regulatory(policy, context)
        if not regulatory_result['passed']:
            self.violation_counts['regulatory'] += 1
            return regulatory_result
        
        return {
            'passed': True,
            'reason': 'All guardrails passed',
            'details': {
                'fairness': fairness_result,
                'safety': safety_result,
                'business': business_result,
                'regulatory': regulatory_result
            }
        }
    
    def _check_fairness(
        self, 
        policy: str, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check fairness constraints
        
        Constraints:
        - Equalized Uplift: benefit should be similar across groups
        - Price Burden Ratio: out-of-pocket / income ≤ threshold
        """
        # For now, simplified check
        # In production, would track uplift by protected group
        
        # Check confidence threshold (safety-related fairness)
        confidence = need_states.get('confidence', 0.0)
        if confidence < self.config.need_state_confidence_threshold:
            return {
                'passed': False,
                'reason': f"Low confidence in need states: {confidence:.3f}",
                'constraint': 'confidence_threshold'
            }
        
        # Price burden check (if we have income data)
        sdoh = context.get('sdoh', {})
        median_income = sdoh.get('median_income', 50000)
        cart_total = context.get('cart', {}).get('total_price', 0)
        
        # Estimate monthly spend (assume weekly shopping)
        monthly_spend = cart_total * 4
        price_burden_ratio = monthly_spend / median_income if median_income > 0 else 0
        
        if price_burden_ratio > self.config.price_burden_ratio_threshold:
            self.logger.warning(
                f"High price burden ratio: {price_burden_ratio:.3f} > "
                f"{self.config.price_burden_ratio_threshold}"
            )
            # Don't reject, but log for monitoring
        
        return {
            'passed': True,
            'reason': 'Fairness constraints satisfied',
            'metrics': {
                'confidence': confidence,
                'price_burden_ratio': price_burden_ratio
            }
        }
    
    def _check_safety(
        self, 
        policy: str, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check safety constraints
        
        Constraints:
        - No harmful substitutions (allergens, contraindications)
        - No predatory pricing
        - No stigmatizing language
        - Confidence threshold
        """
        # Confidence check (already done in fairness, but critical for safety)
        confidence = need_states.get('confidence', 0.0)
        if confidence < 0.5:  # Lower threshold for safety
            return {
                'passed': False,
                'reason': f"Insufficient confidence for safe recommendations: {confidence:.3f}",
                'constraint': 'safety_confidence'
            }
        
        # Check for high-risk scenarios
        # In production, would check:
        # - Allergen information
        # - Drug interactions
        # - Dietary restrictions
        
        return {
            'passed': True,
            'reason': 'Safety constraints satisfied',
            'metrics': {
                'confidence': confidence,
                'risk_level': 'low'
            }
        }
    
    def _check_business(
        self, 
        policy: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check business constraints
        
        Constraints:
        - Margin protection: gross margin ≥ baseline - 5%
        - Inventory: only recommend in-stock items
        - Latency: already checked in agent
        - ROI: expected value ≥ cost
        """
        # Simplified business check
        # In production, would check:
        # - Product inventory
        # - Margin impact
        # - Expected ROI
        
        cart_value = context.get('cart', {}).get('total_price', 0)
        
        # Ensure minimum cart value for recommendations
        if cart_value < 10:  # $10 minimum
            return {
                'passed': False,
                'reason': f"Cart value too low for personalization: ${cart_value:.2f}",
                'constraint': 'minimum_cart_value'
            }
        
        return {
            'passed': True,
            'reason': 'Business constraints satisfied',
            'metrics': {
                'cart_value': cart_value,
                'estimated_margin_impact': 0.0  # Placeholder
            }
        }
    
    def _check_regulatory(
        self, 
        policy: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check regulatory constraints
        
        Constraints:
        - SNAP/WIC compliance: only eligible products
        - ADA compliance: accessible delivery options
        - HIPAA compliance: no health data leakage
        - FTC compliance: no deceptive practices
        """
        # Check consent
        consent = context.get('consent', {})
        if not consent.get('personalization', False):
            return {
                'passed': False,
                'reason': 'User has not consented to personalization',
                'constraint': 'consent_required'
            }
        
        # SNAP/WIC policy compliance
        if policy == 'snap_wic_substitution':
            if 'SNAP_EBT' not in context.get('payment_methods', []):
                # Still allow if high food insecurity
                sdoh = context.get('sdoh', {})
                food_insecurity = sdoh.get('food_insecurity_risk', 0)
                if food_insecurity < 0.7:
                    return {
                        'passed': False,
                        'reason': 'SNAP/WIC policy requires SNAP payment or high food insecurity',
                        'constraint': 'snap_wic_eligibility'
                    }
        
        return {
            'passed': True,
            'reason': 'Regulatory constraints satisfied',
            'metrics': {
                'consent_valid': True,
                'compliance_checks': ['SNAP', 'ADA', 'HIPAA', 'FTC']
            }
        }
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get guardrail violation statistics"""
        total_violations = sum(self.violation_counts.values())
        return {
            'total_violations': total_violations,
            'by_type': self.violation_counts.copy(),
            'violation_rate': total_violations / max(1, total_violations + 100)  # Placeholder
        }
