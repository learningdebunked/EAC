"""
Additional API endpoints for frontend interactions
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CartUpdateRequest(BaseModel):
    """Request to update cart"""
    user_id: str
    action: str  # 'add' or 'remove'
    product_id: str
    product_name: str
    price: float
    category: str


class CartUpdateResponse(BaseModel):
    """Response after cart update"""
    success: bool
    cart_total: float
    item_count: int
    message: str


class ProductRecommendationRequest(BaseModel):
    """Request for product recommendations"""
    user_id: str
    current_cart: List[Dict[str, Any]]
    user_context: Dict[str, Any]


class ProductRecommendationResponse(BaseModel):
    """Response with product recommendations"""
    recommendations: List[Dict[str, Any]]
    reason: str


class UserProfileUpdateRequest(BaseModel):
    """Request to update user profile"""
    user_id: str
    profile: Dict[str, Any]


class UserProfileUpdateResponse(BaseModel):
    """Response after profile update"""
    success: bool
    message: str
    updated_profile: Dict[str, Any]


@router.post("/cart/update", response_model=CartUpdateResponse)
async def update_cart(request: CartUpdateRequest):
    """
    Update shopping cart
    
    This simulates cart operations and could trigger
    real-time recommendations from the EAC Agent
    """
    # In production, this would update a database
    # For now, we just acknowledge the action
    
    return CartUpdateResponse(
        success=True,
        cart_total=0.0,  # Frontend calculates this
        item_count=0,    # Frontend tracks this
        message=f"Product {request.action}ed successfully"
    )


@router.post("/products/recommend", response_model=ProductRecommendationResponse)
async def recommend_products(request: ProductRecommendationRequest):
    """
    Get product recommendations based on current cart
    
    This could use the EAC Agent to suggest products
    before checkout (proactive recommendations)
    """
    # This could call the EAC Agent's recommendation engine
    # For now, return empty recommendations
    
    return ProductRecommendationResponse(
        recommendations=[],
        reason="No additional recommendations at this time"
    )


@router.post("/user/profile", response_model=UserProfileUpdateResponse)
async def update_user_profile(request: UserProfileUpdateRequest):
    """
    Update user profile
    
    This updates SDOH and demographic information
    that the EAC Agent uses for personalization
    """
    # In production, this would update a database
    # The EAC Agent would then use this for future recommendations
    
    return UserProfileUpdateResponse(
        success=True,
        message="Profile updated successfully",
        updated_profile=request.profile
    )
