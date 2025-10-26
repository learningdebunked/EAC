"""
Basic usage example of EAC Agent
"""
import time
from eac.agent import EACAgent, CheckoutEvent
from eac.config import EACConfig


def main():
    """Demonstrate basic EAC Agent usage"""
    
    # Initialize agent with default config
    print("Initializing EAC Agent...")
    config = EACConfig()
    agent = EACAgent(config)
    
    # Create a sample checkout event
    print("\n" + "="*60)
    print("SCENARIO: Low-income household with SNAP/EBT")
    print("="*60)
    
    event = CheckoutEvent(
        user_id="user_12345",
        cart=[
            {"product_id": "prod_001", "quantity": 1, "price": 4.99},  # Chips
            {"product_id": "prod_002", "quantity": 1, "price": 2.49},  # Soda
            {"product_id": "prod_003", "quantity": 2, "price": 3.99},  # White bread
            {"product_id": "prod_004", "quantity": 1, "price": 5.99},  # Sugary cereal
        ],
        delivery_address={
            "zip_code": "94102",
            "census_tract": "06075017902"
        },
        payment_methods=["SNAP_EBT", "CREDIT_CARD"],
        timestamp=time.time(),
        consent={
            "personalization": True,
            "sdoh_signals": True
        }
    )
    
    # Process checkout
    print("\nProcessing checkout...")
    response = agent.process_checkout(event)
    
    # Display results
    print(f"\n{'='*60}")
    print("AGENT RESPONSE")
    print(f"{'='*60}")
    print(f"Policy Used: {response.policy_used}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Fairness Check: {response.fairness_check}")
    print(f"\n{response.explanation}")
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS ({len(response.recommendations)})")
    print(f"{'='*60}")
    
    for i, rec in enumerate(response.recommendations, 1):
        print(f"\n{i}. {rec['reason']}")
        print(f"   Original: {rec['original_product_name']}")
        print(f"   Suggested: {rec['suggested_product_name']}")
        print(f"   Savings: ${rec['savings']:.2f}")
        print(f"   Nutrition Improvement: +{rec['nutrition_improvement']:.1f} points")
        print(f"   Confidence: {rec['confidence']:.0%}")
    
    # Simulate user feedback
    print(f"\n{'='*60}")
    print("USER FEEDBACK SIMULATION")
    print(f"{'='*60}")
    
    # User accepts 2 out of 3 recommendations
    user_feedback = {
        "total_recommendations": len(response.recommendations),
        "accepted_count": min(2, len(response.recommendations)),
        "total_savings": sum(rec['savings'] for rec in response.recommendations[:2]),
        "nutrition_improvement": sum(rec['nutrition_improvement'] for rec in response.recommendations[:2]),
        "fairness_violation": False
    }
    
    print(f"User accepted: {user_feedback['accepted_count']}/{user_feedback['total_recommendations']}")
    print(f"Total savings: ${user_feedback['total_savings']:.2f}")
    print(f"Nutrition improvement: +{user_feedback['nutrition_improvement']:.1f} HEI points")
    
    # Update agent from feedback
    agent.update_from_feedback(event, response, user_feedback)
    
    # Display agent statistics
    print(f"\n{'='*60}")
    print("AGENT STATISTICS")
    print(f"{'='*60}")
    
    stats = agent.get_stats()
    print(f"Total Decisions: {stats['decisions']['total']}")
    print(f"Average Latency: {stats['latency']['p50_ms']:.2f}ms")
    print(f"Error Rate: {stats['decisions']['error_rate']:.2f}%")
    
    if stats['feedback']['total'] > 0:
        print(f"Average Acceptance Rate: {stats['feedback']['avg_acceptance_rate']:.0%}")
        print(f"Total Savings: ${stats['feedback']['total_savings']:.2f}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
