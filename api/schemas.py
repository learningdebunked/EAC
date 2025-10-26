"""
Pydantic schemas for API requests and responses
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CartItem(BaseModel):
    """Cart item"""
    product_id: str
    quantity: int = Field(ge=1)
    price: float = Field(ge=0)


class DeliveryAddress(BaseModel):
    """Delivery address"""
    zip_code: str
    census_tract: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None


class Consent(BaseModel):
    """User consent"""
    personalization: bool = False
    sdoh_signals: bool = False


class CheckoutRequest(BaseModel):
    """Request for checkout decision"""
    user_id: str
    cart: List[CartItem]
    delivery_address: DeliveryAddress
    payment_methods: List[str]
    consent: Consent
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "cart": [
                    {"product_id": "prod_001", "quantity": 1, "price": 4.99},
                    {"product_id": "prod_002", "quantity": 2, "price": 3.49}
                ],
                "delivery_address": {
                    "zip_code": "94102",
                    "census_tract": "06075017902"
                },
                "payment_methods": ["SNAP_EBT", "CREDIT_CARD"],
                "consent": {
                    "personalization": True,
                    "sdoh_signals": True
                }
            }
        }


class Recommendation(BaseModel):
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


class CheckoutResponse(BaseModel):
    """Response with recommendations"""
    recommendations: List[Recommendation]
    latency_ms: float
    policy_used: str
    confidence: float
    fairness_check: str
    explanation: str
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    """User feedback on recommendations"""
    user_id: str
    transaction_id: str
    total_recommendations: int
    accepted_count: int
    total_savings: float = 0.0
    nutrition_improvement: float = 0.0
    fairness_violation: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "transaction_id": "txn_67890",
                "total_recommendations": 3,
                "accepted_count": 2,
                "total_savings": 1.50,
                "nutrition_improvement": 8.0,
                "fairness_violation": False
            }
        }


class FeedbackResponse(BaseModel):
    """Response to feedback"""
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    agent_stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Agent statistics response"""
    decisions: Dict[str, Any]
    latency: Dict[str, Any]
    policies: Dict[str, Any]
    feedback: Dict[str, Any]
