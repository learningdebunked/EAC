"""
Example API client for EAC Agent
"""
import requests
import json


def main():
    """Demonstrate API usage"""
    
    # API base URL
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("EAC API CLIENT EXAMPLE")
    print("="*60)
    
    # 1. Health check
    print("\n1. Checking API health...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"   ✓ API is healthy")
        print(f"   Uptime: {health['uptime_seconds']:.0f}s")
        print(f"   Decisions made: {health['agent_stats']['decisions']['total']}")
    else:
        print(f"   ✗ API health check failed: {response.status_code}")
        return
    
    # 2. Make checkout decision request
    print("\n2. Requesting checkout recommendations...")
    
    checkout_request = {
        "user_id": "demo_user_001",
        "cart": [
            {"product_id": "prod_001", "quantity": 1, "price": 4.99},
            {"product_id": "prod_002", "quantity": 1, "price": 2.49},
            {"product_id": "prod_003", "quantity": 2, "price": 3.99}
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
    
    response = requests.post(
        f"{base_url}/api/v1/checkout/decide",
        json=checkout_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Received {len(result['recommendations'])} recommendations")
        print(f"   Policy used: {result['policy_used']}")
        print(f"   Latency: {result['latency_ms']:.2f}ms")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"\n   {result['explanation']}")
        
        # Display recommendations
        print("\n   Recommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"\n   {i}. {rec['reason']}")
            print(f"      Original: {rec['original_product_name']}")
            print(f"      Suggested: {rec['suggested_product_name']}")
            print(f"      Savings: ${rec['savings']:.2f}")
            print(f"      Nutrition: +{rec['nutrition_improvement']:.1f}")
        
        # 3. Submit feedback
        print("\n3. Submitting user feedback...")
        
        feedback_request = {
            "user_id": "demo_user_001",
            "transaction_id": "demo_txn_001",
            "total_recommendations": len(result['recommendations']),
            "accepted_count": min(2, len(result['recommendations'])),
            "total_savings": sum(rec['savings'] for rec in result['recommendations'][:2]),
            "nutrition_improvement": sum(rec['nutrition_improvement'] for rec in result['recommendations'][:2]),
            "fairness_violation": False
        }
        
        response = requests.post(
            f"{base_url}/api/v1/checkout/feedback",
            json=feedback_request
        )
        
        if response.status_code == 200:
            feedback_result = response.json()
            print(f"   ✓ {feedback_result['message']}")
        else:
            print(f"   ✗ Feedback submission failed: {response.status_code}")
    
    else:
        print(f"   ✗ Checkout request failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return
    
    # 4. Get statistics
    print("\n4. Fetching agent statistics...")
    response = requests.get(f"{base_url}/api/v1/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✓ Statistics retrieved")
        print(f"   Total decisions: {stats['decisions']['total']}")
        print(f"   Average latency: {stats['latency']['p50_ms']:.2f}ms")
        if stats['feedback']['total'] > 0:
            print(f"   Average acceptance: {stats['feedback']['avg_acceptance_rate']:.1%}")
    else:
        print(f"   ✗ Stats request failed: {response.status_code}")
    
    print("\n" + "="*60)
    print("API demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
