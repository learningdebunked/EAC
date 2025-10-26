"""
FastAPI application for EAC Agent
"""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from eac.agent import EACAgent, CheckoutEvent, AgentResponse
from eac.config import EACConfig
from api.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    StatsResponse,
    Recommendation
)

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
        
        # Convert to API response
        return CheckoutResponse(
            recommendations=[
                Recommendation(**rec) for rec in response.recommendations
            ],
            latency_ms=response.latency_ms,
            policy_used=response.policy_used,
            confidence=response.confidence,
            fairness_check=response.fairness_check,
            explanation=response.explanation,
            metadata=response.metadata
        )
        
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
