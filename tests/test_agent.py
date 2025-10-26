"""
Tests for EAC Agent
"""
import pytest
import time
from eac.agent import EACAgent, CheckoutEvent
from eac.config import EACConfig


@pytest.fixture
def agent():
    """Create agent for testing"""
    config = EACConfig()
    return EACAgent(config)


@pytest.fixture
def sample_event():
    """Create sample checkout event"""
    return CheckoutEvent(
        user_id="test_user",
        cart=[
            {"product_id": "prod_001", "quantity": 1, "price": 4.99},
            {"product_id": "prod_002", "quantity": 1, "price": 2.49},
        ],
        delivery_address={
            "zip_code": "94102",
            "census_tract": "06075017902"
        },
        payment_methods=["SNAP_EBT"],
        timestamp=time.time(),
        consent={
            "personalization": True,
            "sdoh_signals": True
        }
    )


def test_agent_initialization(agent):
    """Test agent initializes correctly"""
    assert agent is not None
    assert agent.config is not None
    assert agent.perception is not None
    assert agent.reasoning is not None
    assert agent.action is not None
    assert agent.learning is not None
    assert agent.guardrails is not None


def test_process_checkout(agent, sample_event):
    """Test basic checkout processing"""
    response = agent.process_checkout(sample_event)
    
    assert response is not None
    assert response.latency_ms > 0
    assert response.latency_ms <= agent.config.max_latency_ms
    assert response.policy_used is not None
    assert response.fairness_check in ["passed", "skipped"]


def test_latency_sla(agent, sample_event):
    """Test latency SLA is enforced"""
    response = agent.process_checkout(sample_event)
    assert response.latency_ms <= agent.config.max_latency_ms


def test_consent_required(agent, sample_event):
    """Test that consent is required"""
    sample_event.consent['personalization'] = False
    response = agent.process_checkout(sample_event)
    
    assert len(response.recommendations) == 0
    assert response.policy_used == "safe_default"


def test_feedback_learning(agent, sample_event):
    """Test learning from feedback"""
    response = agent.process_checkout(sample_event)
    
    user_feedback = {
        "total_recommendations": len(response.recommendations),
        "accepted_count": 1,
        "total_savings": 1.0,
        "nutrition_improvement": 5.0,
        "fairness_violation": False
    }
    
    # Should not raise exception
    agent.update_from_feedback(sample_event, response, user_feedback)
    
    stats = agent.get_stats()
    assert stats['feedback']['total'] == 1


def test_multiple_checkouts(agent):
    """Test processing multiple checkouts"""
    for i in range(5):
        event = CheckoutEvent(
            user_id=f"user_{i}",
            cart=[{"product_id": f"prod_{i}", "quantity": 1, "price": 5.0}],
            delivery_address={"zip_code": "94102"},
            payment_methods=["CREDIT_CARD"],
            timestamp=time.time(),
            consent={"personalization": True, "sdoh_signals": True}
        )
        
        response = agent.process_checkout(event)
        assert response is not None
    
    stats = agent.get_stats()
    assert stats['decisions']['total'] == 5


def test_policy_selection(agent, sample_event):
    """Test that appropriate policies are selected"""
    # SNAP payment should trigger SNAP/WIC policy
    sample_event.payment_methods = ["SNAP_EBT"]
    response = agent.process_checkout(sample_event)
    
    # Should select a valid policy (not safe_default if everything works)
    assert response.policy_used in agent.config.enabled_policies or response.policy_used == "safe_default"


def test_error_handling(agent):
    """Test error handling with invalid input"""
    # Create event with missing required fields
    event = CheckoutEvent(
        user_id="test",
        cart=[],  # Empty cart
        delivery_address={},
        payment_methods=[],
        timestamp=time.time(),
        consent={"personalization": True, "sdoh_signals": True}
    )
    
    # Should not crash
    response = agent.process_checkout(event)
    assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
