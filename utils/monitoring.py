"""
Monitoring utilities for EAC Agent
"""
import time
from typing import Dict, Any
from collections import defaultdict, deque


class AgentMonitor:
    """
    Monitor agent performance and metrics
    
    Tracks:
    - Latency (p50, p95, p99)
    - Decision counts
    - Policy usage
    - Errors
    - Fairness metrics
    """
    
    def __init__(self, config):
        self.config = config
        
        # Metrics storage
        self.latencies = deque(maxlen=10000)
        self.decision_count = 0
        self.error_count = 0
        self.latency_violations = 0
        
        self.policy_counts = defaultdict(int)
        self.policy_confidences = defaultdict(list)
        
        self.feedback_count = 0
        self.total_acceptance_rate = 0.0
        self.total_savings = 0.0
        
        self.start_time = time.time()
    
    def record_decision(self, response):
        """Record a decision"""
        self.decision_count += 1
        self.latencies.append(response.latency_ms)
        
        self.policy_counts[response.policy_used] += 1
        self.policy_confidences[response.policy_used].append(response.confidence)
    
    def record_latency_violation(self, latency_ms: float):
        """Record a latency SLA violation"""
        self.latency_violations += 1
    
    def record_error(self, error: str):
        """Record an error"""
        self.error_count += 1
    
    def record_feedback(self, user_feedback: Dict[str, Any], reward: float):
        """Record user feedback"""
        self.feedback_count += 1
        
        if user_feedback.get('total_recommendations', 0) > 0:
            acceptance_rate = (
                user_feedback.get('accepted_count', 0) / 
                user_feedback['total_recommendations']
            )
            self.total_acceptance_rate += acceptance_rate
        
        self.total_savings += user_feedback.get('total_savings', 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics"""
        uptime = time.time() - self.start_time
        
        # Latency percentiles
        if self.latencies:
            import numpy as np
            latencies_array = np.array(self.latencies)
            p50 = np.percentile(latencies_array, 50)
            p95 = np.percentile(latencies_array, 95)
            p99 = np.percentile(latencies_array, 99)
        else:
            p50 = p95 = p99 = 0
        
        # Policy stats
        policy_stats = {}
        for policy, count in self.policy_counts.items():
            confidences = self.policy_confidences[policy]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            policy_stats[policy] = {
                'count': count,
                'percentage': (count / self.decision_count * 100) if self.decision_count > 0 else 0,
                'avg_confidence': avg_confidence
            }
        
        return {
            'uptime_seconds': uptime,
            'decisions': {
                'total': self.decision_count,
                'rate_per_second': self.decision_count / uptime if uptime > 0 else 0,
                'errors': self.error_count,
                'error_rate': (self.error_count / self.decision_count * 100) if self.decision_count > 0 else 0
            },
            'latency': {
                'p50_ms': p50,
                'p95_ms': p95,
                'p99_ms': p99,
                'violations': self.latency_violations,
                'violation_rate': (self.latency_violations / self.decision_count * 100) if self.decision_count > 0 else 0
            },
            'policies': policy_stats,
            'feedback': {
                'total': self.feedback_count,
                'avg_acceptance_rate': (self.total_acceptance_rate / self.feedback_count) if self.feedback_count > 0 else 0,
                'total_savings': self.total_savings,
                'avg_savings_per_feedback': (self.total_savings / self.feedback_count) if self.feedback_count > 0 else 0
            }
        }
