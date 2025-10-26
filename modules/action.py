"""
Action Module - Executes policies and generates recommendations
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from config import EACConfig
from data.products import ProductDataLoader


@dataclass
class Recommendation:
    """A single recommendation"""
    original_product_id: str
    suggested_product_id: str
    original_product_name: str
    suggested_product_name: str
    reason: str
    savings: float
    nutrition_improvement: float
    confidence: float
    metadata: Dict[str, Any]


class PolicyExecutor:
    """Base class for policy executors"""
    
    def __init__(self, config: EACConfig, product_loader: ProductDataLoader):
        self.config = config
        self.product_loader = product_loader
        self.logger = logging.getLogger(f"EACAgent.Policy.{self.__class__.__name__}")
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Execute policy and generate recommendations"""
        raise NotImplementedError


class SNAPWICSubstitutionPolicy(PolicyExecutor):
    """Policy 1: SNAP/WIC-compatible substitutions"""
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Find SNAP/WIC-eligible alternatives"""
        recommendations = []
        cart_items = context['cart']['items']
        
        for item in cart_items:
            # Skip if already SNAP-eligible
            if item.get('snap_eligible', False):
                continue
            
            # Find SNAP-eligible alternative
            alternative = self.product_loader.find_snap_alternative(
                product_id=item['product_id'],
                category=item['category'],
                max_price_delta=item['price'] * 0.1  # Within 10%
            )
            
            if alternative:
                savings = item['price'] - alternative['price']
                nutrition_gain = self._compute_nutrition_gain(
                    item.get('nutrition', {}),
                    alternative.get('nutrition', {})
                )
                
                rec = Recommendation(
                    original_product_id=item['product_id'],
                    suggested_product_id=alternative['product_id'],
                    original_product_name=item.get('name', 'Unknown'),
                    suggested_product_name=alternative.get('name', 'Unknown'),
                    reason=f"SNAP-eligible alternative saves ${savings:.2f}",
                    savings=savings,
                    nutrition_improvement=nutrition_gain,
                    confidence=0.9,
                    metadata={
                        'policy': 'snap_wic_substitution',
                        'snap_eligible': True,
                        'price_delta_pct': (savings / item['price']) * 100
                    }
                )
                recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} SNAP/WIC recommendations")
        return recommendations
    
    @staticmethod
    def _compute_nutrition_gain(original: Dict, alternative: Dict) -> float:
        """Compute nutrition improvement (simplified HEI delta)"""
        # Simplified: compare key nutrients
        gain = 0.0
        
        # Fiber (higher is better)
        orig_fiber = original.get('fiber_g', 0)
        alt_fiber = alternative.get('fiber_g', 0)
        gain += (alt_fiber - orig_fiber) * 2
        
        # Sugar (lower is better)
        orig_sugar = original.get('sugar_g', 0)
        alt_sugar = alternative.get('sugar_g', 0)
        gain += (orig_sugar - alt_sugar) * 1
        
        # Sodium (lower is better)
        orig_sodium = original.get('sodium_mg', 0)
        alt_sodium = alternative.get('sodium_mg', 0)
        gain += (orig_sodium - alt_sodium) * 0.01
        
        return gain


class LowGlycemicAlternativePolicy(PolicyExecutor):
    """Policy 2: Low-glycemic alternatives"""
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Find low-glycemic alternatives"""
        recommendations = []
        cart_items = context['cart']['items']
        
        for item in cart_items:
            gi = item.get('glycemic_index')
            if gi is None or gi < 70:  # Already low-GI
                continue
            
            # Find low-GI alternative
            alternative = self.product_loader.find_low_gi_alternative(
                product_id=item['product_id'],
                category=item['category'],
                max_gi=55,  # Low GI threshold
                max_price_delta_pct=15
            )
            
            if alternative:
                price_delta = alternative['price'] - item['price']
                gi_improvement = gi - alternative.get('glycemic_index', 55)
                
                rec = Recommendation(
                    original_product_id=item['product_id'],
                    suggested_product_id=alternative['product_id'],
                    original_product_name=item.get('name', 'Unknown'),
                    suggested_product_name=alternative.get('name', 'Unknown'),
                    reason="Better for blood sugar management",
                    savings=-price_delta if price_delta < 0 else 0,
                    nutrition_improvement=gi_improvement * 0.5,  # Convert to HEI-like score
                    confidence=0.85,
                    metadata={
                        'policy': 'low_glycemic_alternative',
                        'original_gi': gi,
                        'alternative_gi': alternative.get('glycemic_index'),
                        'gi_improvement': gi_improvement
                    }
                )
                recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} low-GI recommendations")
        return recommendations


class OTCCoveragePolicy(PolicyExecutor):
    """Policy 3: Plan-aware OTC coverage"""
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Find FSA/HSA-eligible OTC items"""
        recommendations = []
        cart_items = context['cart']['items']
        payment_methods = context.get('payment_methods', [])
        
        has_fsa_hsa = any(pm in ['FSA', 'HSA'] for pm in payment_methods)
        if not has_fsa_hsa:
            return recommendations
        
        for item in cart_items:
            # Check if OTC item
            if item.get('category') not in ['otc_medication', 'health_supplies']:
                continue
            
            # Check FSA/HSA eligibility
            if self.product_loader.is_fsa_hsa_eligible(item['product_id']):
                # Calculate savings (assume FSA/HSA covers 100%)
                savings = item['price'] * item['quantity']
                
                rec = Recommendation(
                    original_product_id=item['product_id'],
                    suggested_product_id=item['product_id'],  # Same product
                    original_product_name=item.get('name', 'Unknown'),
                    suggested_product_name=item.get('name', 'Unknown'),
                    reason=f"Covered by your FSA/HSA - save ${savings:.2f}",
                    savings=savings,
                    nutrition_improvement=0,
                    confidence=0.95,
                    metadata={
                        'policy': 'otc_coverage',
                        'fsa_hsa_eligible': True,
                        'coverage_amount': savings
                    }
                )
                recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} OTC coverage recommendations")
        return recommendations


class MobilityDeliveryPolicy(PolicyExecutor):
    """Policy 4: Mobility-aligned delivery windows"""
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Suggest accessible delivery windows"""
        recommendations = []
        
        # Get SDOH data for transit info
        sdoh = context.get('sdoh', {})
        transit_score = sdoh.get('transit_score', 0.5)
        
        # If low transit access, suggest delivery windows aligned with transit
        if transit_score < 0.4:
            # This would integrate with transit schedule API
            # For now, return a recommendation about delivery timing
            rec = Recommendation(
                original_product_id='delivery_window',
                suggested_product_id='delivery_window_transit_aligned',
                original_product_name='Standard Delivery',
                suggested_product_name='Transit-Aligned Delivery',
                reason="Delivery time matches local bus schedule",
                savings=0,
                nutrition_improvement=0,
                confidence=0.8,
                metadata={
                    'policy': 'mobility_delivery',
                    'transit_score': transit_score,
                    'suggested_windows': ['2pm-4pm', '6pm-8pm'],  # Example
                    'accessibility_score': 0.95
                }
            )
            recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} delivery recommendations")
        return recommendations


class SafetyNudgePolicy(PolicyExecutor):
    """Policy 5: Safety-first product nudges"""
    
    def execute(
        self, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Recommendation]:
        """Suggest nutritional improvements and cost savings"""
        recommendations = []
        cart_items = context['cart']['items']
        
        # Nutritional improvement suggestions
        for item in cart_items:
            # Check for bulk/unit pricing opportunities
            bulk_alternative = self.product_loader.find_bulk_alternative(
                product_id=item['product_id'],
                min_savings_pct=10
            )
            
            if bulk_alternative:
                unit_savings = (
                    item['price'] / item['quantity'] - 
                    bulk_alternative['price'] / bulk_alternative['quantity']
                ) * item['quantity']
                
                rec = Recommendation(
                    original_product_id=item['product_id'],
                    suggested_product_id=bulk_alternative['product_id'],
                    original_product_name=item.get('name', 'Unknown'),
                    suggested_product_name=bulk_alternative.get('name', 'Unknown'),
                    reason=f"Save ${unit_savings:.2f} by buying larger size",
                    savings=unit_savings,
                    nutrition_improvement=0,
                    confidence=0.9,
                    metadata={
                        'policy': 'safety_nudge',
                        'nudge_type': 'bulk_pricing',
                        'unit_price_original': item['price'] / item['quantity'],
                        'unit_price_alternative': bulk_alternative['price'] / bulk_alternative['quantity']
                    }
                )
                recommendations.append(rec)
        
        # Check for missing nutritional categories
        categories = set(item['category'] for item in cart_items)
        if 'fruits' not in categories and 'vegetables' not in categories:
            # Suggest adding fruits/vegetables
            suggested_product = self.product_loader.get_affordable_produce()
            if suggested_product:
                rec = Recommendation(
                    original_product_id='none',
                    suggested_product_id=suggested_product['product_id'],
                    original_product_name='',
                    suggested_product_name=suggested_product.get('name', 'Fresh Produce'),
                    reason="Add fruits or vegetables for better nutrition",
                    savings=0,
                    nutrition_improvement=10,  # Significant HEI boost
                    confidence=0.75,
                    metadata={
                        'policy': 'safety_nudge',
                        'nudge_type': 'nutritional_gap',
                        'missing_category': 'produce'
                    }
                )
                recommendations.append(rec)
        
        self.logger.info(f"Generated {len(recommendations)} safety nudge recommendations")
        return recommendations


class ActionModule:
    """
    Action Module: Executes policies and generates recommendations
    
    Responsibilities:
    1. Execute selected policy
    2. Generate recommendations with explanations
    3. Rank recommendations by impact
    4. Limit to top N recommendations
    """
    
    def __init__(self, config: EACConfig):
        self.config = config
        self.logger = logging.getLogger("EACAgent.Action")
        
        # Load product data
        self.product_loader = ProductDataLoader(config.product_data_path)
        
        # Initialize policy executors
        self.policies = {
            'snap_wic_substitution': SNAPWICSubstitutionPolicy(config, self.product_loader),
            'low_glycemic_alternative': LowGlycemicAlternativePolicy(config, self.product_loader),
            'otc_coverage': OTCCoveragePolicy(config, self.product_loader),
            'mobility_delivery': MobilityDeliveryPolicy(config, self.product_loader),
            'safety_nudge': SafetyNudgePolicy(config, self.product_loader)
        }
        
        self.logger.info(f"Action Module initialized with {len(self.policies)} policies")
    
    def execute_policy(
        self, 
        policy_name: str, 
        context: Dict[str, Any], 
        need_states: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute policy and generate recommendations
        
        Args:
            policy_name: Name of policy to execute
            context: Context from perception
            need_states: Inferred need states
            
        Returns:
            List of recommendations
        """
        if policy_name == 'safe_default':
            return []
        
        if policy_name not in self.policies:
            self.logger.warning(f"Unknown policy: {policy_name}")
            return []
        
        # Execute policy
        policy = self.policies[policy_name]
        recommendations = policy.execute(context, need_states)
        
        # Rank by impact (savings + nutrition)
        recommendations = self._rank_recommendations(recommendations)
        
        # Limit to top N
        max_recommendations = 5
        recommendations = recommendations[:max_recommendations]
        
        # Convert to dict format
        return [self._recommendation_to_dict(rec) for rec in recommendations]
    
    def _rank_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """
        Rank recommendations by impact
        
        Impact = savings + nutrition_improvement + confidence
        """
        def impact_score(rec: Recommendation) -> float:
            return (
                rec.savings * 1.0 +
                rec.nutrition_improvement * 0.5 +
                rec.confidence * 0.3
            )
        
        return sorted(recommendations, key=impact_score, reverse=True)
    
    @staticmethod
    def _recommendation_to_dict(rec: Recommendation) -> Dict[str, Any]:
        """Convert Recommendation to dict"""
        return {
            'original_product_id': rec.original_product_id,
            'suggested_product_id': rec.suggested_product_id,
            'original_product_name': rec.original_product_name,
            'suggested_product_name': rec.suggested_product_name,
            'reason': rec.reason,
            'savings': rec.savings,
            'nutrition_improvement': rec.nutrition_improvement,
            'confidence': rec.confidence,
            'metadata': rec.metadata
        }
