"""
FastAPI application for EAC Agent
"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import EACAgent, CheckoutEvent, AgentResponse
from config import EACConfig
from api.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    StatsResponse,
    Recommendation
)
from api.endpoints import router as additional_router
from api.data_store import transaction_store
import torch
import numpy as np
from datetime import datetime


def convert_to_python(obj):
    """Convert torch tensors and numpy arrays to Python types"""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return float(obj.item()) if obj.size == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj

# Global agent instance
agent: EACAgent = None
app_start_time: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global agent, app_start_time
    
    # Startup
    logging.info("Starting EAC Agent API...")
    config = EACConfig()
    agent = EACAgent(config)
    app_start_time = time.time()
    logging.info("EAC Agent API started successfully")
    
    yield
    
    # Shutdown
    logging.info("Shutting down EAC Agent API...")


# Create FastAPI app
app = FastAPI(
    title="Equity-Aware Checkout API",
    description="AI Agent for fair, privacy-preserving checkout personalization",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include additional endpoints
app.include_router(additional_router, prefix="/api/v1", tags=["Frontend"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Equity-Aware Checkout API",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    uptime = time.time() - app_start_time
    stats = agent.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
        agent_stats=stats
    )


@app.post("/api/v1/checkout/decide", response_model=CheckoutResponse, tags=["Checkout"])
async def checkout_decide(request: CheckoutRequest):
    """
    Get personalized recommendations for checkout
    
    This endpoint:
    1. Observes the checkout context (cart, SDOH, payment methods)
    2. Infers user needs
    3. Selects and executes appropriate policy
    4. Returns recommendations with explanations
    
    All processing happens in ≤100ms with fairness guarantees.
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Convert request to CheckoutEvent
        event = CheckoutEvent(
            user_id=request.user_id,
            cart=[item.dict() for item in request.cart],
            delivery_address=request.delivery_address.dict(),
            payment_methods=request.payment_methods,
            timestamp=time.time(),
            consent=request.consent.dict(),
            metadata=request.metadata
        )
        
        # Process checkout
        response: AgentResponse = agent.process_checkout(event)
        
        # Convert tensors to Python types
        clean_recommendations = [convert_to_python(rec) for rec in response.recommendations]
        clean_metadata = convert_to_python(response.metadata)
        
        # Store transaction for analytics
        transaction_id = f"txn_{int(time.time() * 1000)}"
        transaction_store.add_transaction({
            'timestamp': datetime.now().isoformat(),
            'user_id': request.user_id,
            'transaction_id': transaction_id,
            'policy_used': response.policy_used,
            'num_recommendations': len(clean_recommendations),
            'accepted_count': 0,  # Will be updated by feedback
            'declined_count': 0,
            'total_savings': sum(rec.get('savings', 0) for rec in clean_recommendations),
            'total_nutrition_improvement': sum(rec.get('nutrition_improvement', 0) for rec in clean_recommendations),
            'acceptance_rate': 0.0,
            'latency_ms': float(response.latency_ms),
            'protected_group': request.metadata.get('race', 'unknown') if request.metadata else 'unknown',
            'income_group': 'low' if request.metadata and request.metadata.get('income', 50000) < 30000 else 'medium',
            'snap_eligible': 'SNAP_EBT' in request.payment_methods,
            'fairness_check': response.fairness_check
        })
        
        # Convert to API response
        api_response = CheckoutResponse(
            recommendations=[
                Recommendation(**rec) for rec in clean_recommendations
            ],
            latency_ms=float(response.latency_ms),
            policy_used=response.policy_used,
            confidence=float(response.confidence),
            fairness_check=response.fairness_check,
            explanation=response.explanation,
            metadata={**clean_metadata, 'transaction_id': transaction_id}
        )
        
        return api_response
        
    except Exception as e:
        logging.error(f"Error processing checkout: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing checkout: {str(e)}"
        )


@app.post("/api/v1/checkout/feedback", response_model=FeedbackResponse, tags=["Checkout"])
async def checkout_feedback(request: FeedbackRequest):
    """
    Submit user feedback on recommendations
    
    This endpoint allows the agent to learn from user responses:
    - Which recommendations were accepted/rejected
    - Actual savings and nutrition improvements
    - Any fairness violations
    
    The agent uses this feedback to improve future recommendations.
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        # Note: In production, you'd retrieve the original event and response
        # For now, we just update the learning module directly
        
        user_feedback = request.dict()
        
        # Compute reward
        reward = 0.0
        if request.total_recommendations > 0:
            acceptance_rate = request.accepted_count / request.total_recommendations
            reward += acceptance_rate * 1.0
        reward += request.total_savings * 0.1
        reward += request.nutrition_improvement * 0.05
        if request.fairness_violation:
            reward -= 2.0
        
        # Update learning module
        # (In production, would pass full context and policy)
        agent.learning.update(
            context={},
            policy="unknown",
            reward=reward,
            user_feedback=user_feedback
        )
        
        # Update transaction store for analytics
        transaction_store.update_transaction(
            transaction_id=request.transaction_id,
            updates={
                'accepted_count': request.accepted_count,
                'declined_count': request.total_recommendations - request.accepted_count,
                'total_savings': request.total_savings,
                'total_nutrition_improvement': request.nutrition_improvement
            }
        )
        
        return FeedbackResponse(
            status="success",
            message=f"Feedback recorded: {request.accepted_count}/{request.total_recommendations} accepted"
        )
        
    except Exception as e:
        logging.error(f"Error processing feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing feedback: {str(e)}"
        )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats():
    """
    Get agent statistics
    
    Returns performance metrics:
    - Total decisions made
    - Latency percentiles
    - Policy usage
    - Feedback metrics
    """
    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent not initialized"
        )
    
    try:
        stats = agent.get_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        logging.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
