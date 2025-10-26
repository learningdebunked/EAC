# Equity-Aware Checkout (EAC) - Architecture Summary

## System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   Checkout   │  │   Cart View  │  │   Delivery   │                  │
│  │   Interface  │  │              │  │   Selection  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTPS/REST API
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY LAYER                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Server (≤100ms latency SLA)                             │  │
│  │  • Authentication (OAuth2)                                        │  │
│  │  • Rate Limiting                                                  │  │
│  │  • Request Validation                                             │  │
│  │  • Circuit Breaker                                                │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRIVACY & CONSENT LAYER                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
│  │ Consent Check  │  │ PII Protection │  │ Explainability Engine  │   │
│  │ (Opt-in/Out)   │  │ (Anonymizer)   │  │ (Why this rec?)        │   │
│  └────────────────┘  └────────────────┘  └────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                ▼                                 ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│   BEHAVIORAL FEATURES        │    │   SDOH DATA INTEGRATION      │
│   (Non-PII, Consented)       │    │   (Census Tract Level)       │
│                              │    │                              │
│  • Cart composition          │    │  • CDC SVI (vulnerability)   │
│  • Browsing patterns         │    │  • ADI (deprivation)         │
│  • Purchase timing           │    │  • USDA Food Access          │
│  • Payment methods           │    │  • Transit availability      │
│  • Delivery preferences      │    │  • Heat/hazard risk          │
│  • Substitution history      │    │  • Census demographics       │
└──────────────┬───────────────┘    └──────────────┬───────────────┘
               │                                   │
               └────────────────┬──────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Feature Store (Feast)                                           │  │
│  │  • Behavioral embeddings                                         │  │
│  │  • SDOH composite scores                                         │  │
│  │  • Product context vectors                                       │  │
│  │  • Cached features (Redis)                                       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEED STATE LEARNING ENGINE                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Multi-Task Deep Neural Network (PyTorch)                        │  │
│  │                                                                   │  │
│  │  Input Layer (512 dims)                                           │  │
│  │       ↓                                                           │  │
│  │  Shared Layers (Dense → BatchNorm → ReLU → Dropout)              │  │
│  │       ↓                                                           │  │
│  │  Task-Specific Heads:                                             │  │
│  │    ├─ Food Insecurity [Low/Med/High]                             │  │
│  │    ├─ Transportation Constraint [Yes/No]                          │  │
│  │    ├─ Chronic Condition Proxy [Risk Scores]                      │  │
│  │    ├─ Financial Constraint [Budget Sensitivity]                  │  │
│  │    └─ Mobility Limitation [Mobility Score]                       │  │
│  │                                                                   │  │
│  │  Uncertainty Quantification:                                      │  │
│  │    • Monte Carlo Dropout                                          │  │
│  │    • Confidence thresholds (reject if < 0.7)                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   CONSTRAINED POLICY ENGINE                              │
│                       (Policy-as-Code)                                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Policy 1: SNAP/WIC Substitutions                              │    │
│  │  Trigger: SNAP/EBT payment OR high food insecurity             │    │
│  │  Action: Suggest eligible alternatives with savings            │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Policy 2: Low-Glycemic Alternatives                           │    │
│  │  Trigger: Diabetes risk OR high-GI items                       │    │
│  │  Action: Suggest low-GI swaps (gentle nudge)                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Policy 3: Plan-Aware OTC Coverage                             │    │
│  │  Trigger: OTC items + FSA/HSA/insurance                        │    │
│  │  Action: Show coverage, auto-apply FSA/HSA                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Policy 4: Mobility-Aligned Delivery                           │    │
│  │  Trigger: Mobility limitation OR low transit access            │    │
│  │  Action: Align delivery with transit, avoid heat/hazards       │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Policy 5: Safety-First Nudges                                 │    │
│  │  Trigger: Cart improvement opportunities                       │    │
│  │  Action: Nutritional/cost improvements, preventive health      │    │
│  └────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              GUARDRAILED CONTEXTUAL BANDIT ENGINE                        │
│                        (≤100ms Decision)                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  LinUCB Algorithm (Vowpal Wabbit)                                │  │
│  │                                                                   │  │
│  │  1. Receive context (user, cart, SDOH, need states)              │  │
│  │  2. Compute UCB for each policy                                  │  │
│  │  3. Apply guardrails ────────────────────┐                       │  │
│  │  4. Select best constrained policy       │                       │  │
│  │  5. Execute policy actions               │                       │  │
│  │  6. Observe reward                       │                       │  │
│  │  7. Update policy weights                │                       │  │
│  │                                          │                       │  │
│  │  Exploration: ε-greedy (20% → 5%)        │                       │  │
│  └──────────────────────────────────────────┼───────────────────────┘  │
│                                             │                           │
│  ┌──────────────────────────────────────────▼───────────────────────┐  │
│  │  GUARDRAIL SYSTEM (Hard Constraints)                             │  │
│  │                                                                   │  │
│  │  ✓ Fairness: Equalized Uplift, Price Burden Ratio               │  │
│  │  ✓ Safety: No harmful recs, confidence threshold                │  │
│  │  ✓ Business: Margin ≥ baseline-5%, latency ≤100ms               │  │
│  │  ✓ Regulatory: SNAP/WIC compliance, ADA, HIPAA, FTC             │  │
│  │                                                                   │  │
│  │  If violation detected → Reject action → Fallback to safe default│  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  MULTI-OBJECTIVE OPTIMIZATION LAYER                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Objective Function (Pareto Optimization)                        │  │
│  │                                                                   │  │
│  │  Minimize:                        Maximize:                      │  │
│  │    • Out-of-pocket spend            • Attach rate                │  │
│  │    • Adverse health proxies         • Completion rate            │  │
│  │    • Safety Harm Rate               • Customer satisfaction      │  │
│  │                                                                   │  │
│  │  Subject to:                                                      │  │
│  │    • Equalized Uplift across groups                              │  │
│  │    • Price Burden Ratio ≤ threshold                              │  │
│  │    • Latency ≤100ms                                              │  │
│  │    • Margin ≥ baseline - 5%                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FAIRNESS & SAFETY EVALUATION                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  New Equity Metrics for Retail                                   │  │
│  │                                                                   │  │
│  │  1. Equalized Uplift (EU)                                        │  │
│  │     Δ(outcome | treatment) equal across protected groups         │  │
│  │     Target: |EU_A - EU_B| < 0.05                                 │  │
│  │                                                                   │  │
│  │  2. Price Burden Ratio (PBR)                                     │  │
│  │     Out-of-pocket spend / household income                       │  │
│  │     Target: PBR_low_income < 0.30                                │  │
│  │                                                                   │  │
│  │  3. Safety Harm Rate (SHR)                                       │  │
│  │     % recommendations causing adverse outcomes                   │  │
│  │     Target: SHR < 0.01 (1%)                                      │  │
│  │                                                                   │  │
│  │  Continuous Monitoring:                                           │  │
│  │    • Demographic parity checks                                   │  │
│  │    • Disparate impact analysis                                   │  │
│  │    • Intersectional fairness (race × income × health)            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ONLINE LEARNING & ROLLOUT                             │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Stepped-Wedge Deployment                                        │  │
│  │                                                                   │  │
│  │  Phase 1: Pilot (5% users, 4 weeks)                              │  │
│  │    → Intensive monitoring, daily fairness audits                 │  │
│  │                                                                   │  │
│  │  Phase 2: Expansion (25% users, 8 weeks)                         │  │
│  │    → Automated fairness checks, A/B/n testing                    │  │
│  │                                                                   │  │
│  │  Phase 3: Scale (75% users, 8 weeks)                             │  │
│  │    → Continuous learning, policy templates released              │  │
│  │                                                                   │  │
│  │  Phase 4: Full Rollout (100% users)                              │  │
│  │    → Open source release, evaluation checklists published        │  │
│  │                                                                   │  │
│  │  Continuous Learning Pipeline:                                    │  │
│  │    • Real-time feedback integration (Kafka)                      │  │
│  │    • Weekly model retraining (Airflow)                           │  │
│  │    • Drift detection (Evidently AI)                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MONITORING & OBSERVABILITY                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Real-Time Dashboards (Grafana)                                  │  │
│  │    • Latency monitoring (p50, p95, p99)                          │  │
│  │    • Fairness metrics (EU, PBR, SHR)                             │  │
│  │    • Business KPIs (conversion, AOV, NPS)                        │  │
│  │    • Safety alerts (harmful recommendations)                     │  │
│  │    • Model performance (accuracy, drift)                         │  │
│  │                                                                   │  │
│  │  Logging & Alerting (ELK + PagerDuty)                            │  │
│  │    • All requests logged (audit trail)                           │  │
│  │    • Fairness violation alerts                                   │  │
│  │    • Latency SLA breach alerts                                   │  │
│  │    • Safety incident escalation                                  │  │
│  │                                                                   │  │
│  │  Experiment Tracking (MLflow)                                     │  │
│  │    • Model versions, hyperparameters                             │  │
│  │    • A/B test results                                            │  │
│  │    • Policy performance comparison                               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
User Request
    │
    ▼
[1] API Gateway (FastAPI)
    │
    ├─→ [2] Privacy Check (Consent validation)
    │        │
    │        ├─→ Consented? ──No──→ Return default recommendations
    │        │
    │        └─→ Yes
    │             │
    ▼             ▼
[3] Feature Extraction (Parallel)
    │
    ├─→ Behavioral Features (cart, browsing, timing)
    │
    └─→ SDOH Signals (geocode → census tract → join indices)
         │
         ▼
[4] Feature Store (Feast)
    │
    ├─→ Cache hit? ──Yes──→ Return cached features
    │
    └─→ No ──→ Compute & cache
         │
         ▼
[5] Need State Learning (Multi-task NN)
    │
    ├─→ Food insecurity score
    ├─→ Transportation constraint
    ├─→ Chronic condition proxy
    ├─→ Financial constraint
    └─→ Mobility limitation
         │
         ▼
[6] Uncertainty Quantification
    │
    ├─→ Confidence > 0.7? ──No──→ Fallback to safe default
    │
    └─→ Yes
         │
         ▼
[7] Policy Selection (Contextual Bandit)
    │
    ├─→ Compute UCB for each policy
    ├─→ Apply guardrails (fairness, safety, business)
    └─→ Select best constrained policy
         │
         ▼
[8] Policy Execution
    │
    ├─→ Policy 1: SNAP/WIC substitutions
    ├─→ Policy 2: Low-glycemic alternatives
    ├─→ Policy 3: OTC coverage
    ├─→ Policy 4: Delivery windows
    └─→ Policy 5: Product nudges
         │
         ▼
[9] Multi-Objective Optimization
    │
    ├─→ Optimize for: cost, health, satisfaction
    └─→ Subject to: fairness, safety, business constraints
         │
         ▼
[10] Guardrail Validation
    │
    ├─→ All guardrails passed? ──No──→ Reject & fallback
    │
    └─→ Yes
         │
         ▼
[11] Response Generation
    │
    ├─→ Format recommendations
    ├─→ Add explainability ("Why this rec?")
    └─→ Log for monitoring
         │
         ▼
[12] Return to User (≤100ms total)
    │
    ▼
[13] Feedback Collection
    │
    ├─→ User accepts/rejects recommendation
    ├─→ Track outcomes (conversion, satisfaction)
    └─→ Update bandit weights (online learning)
```

---

## Technology Stack

### Core Infrastructure
```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                          │
│  • Python 3.10+ (primary language)                          │
│  • FastAPI (REST API framework)                             │
│  • Pydantic (data validation)                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ML/AI Layer                                                 │
│  • PyTorch 2.0+ (deep learning)                             │
│  • Vowpal Wabbit (contextual bandit)                        │
│  • Scikit-learn (preprocessing, metrics)                    │
│  • Fairlearn (fairness metrics)                             │
│  • AIF360 (bias mitigation)                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Data Layer                                                  │
│  • PostgreSQL 14+ (relational data)                         │
│  • Redis 7+ (caching, feature store)                        │
│  • Apache Kafka (event streaming)                           │
│  • Feast (feature store)                                    │
│  • BigQuery/Snowflake (data warehouse)                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Orchestration & Deployment                                  │
│  • Docker (containerization)                                │
│  • Kubernetes (orchestration)                               │
│  • Apache Airflow (workflow orchestration)                  │
│  • GitHub Actions (CI/CD)                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Monitoring & Observability                                  │
│  • Prometheus (metrics collection)                          │
│  • Grafana (dashboards)                                     │
│  • ELK Stack (logging: Elasticsearch, Logstash, Kibana)    │
│  • MLflow (experiment tracking, model registry)             │
│  • Evidently AI (ML monitoring, drift detection)            │
│  • PagerDuty (alerting)                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Privacy by Design
- **Census tract aggregation**: No street-level addresses stored
- **Opt-in consent**: Granular, revocable consent for each data type
- **PII protection**: No personal identifiers in SDOH signals
- **Explainability**: Every recommendation comes with a "why"

### 2. Fairness by Design
- **Guardrails**: Hard constraints on fairness metrics
- **Continuous monitoring**: Real-time fairness audits
- **Equalized Uplift**: Benefits distributed equally across groups
- **Price Burden Ratio**: Affordability monitored for low-income users

### 3. Safety by Design
- **Uncertainty quantification**: Reject low-confidence predictions
- **No harmful recommendations**: Allergen checks, contraindications
- **Safety Harm Rate**: Track and minimize adverse outcomes
- **Human oversight**: Manual review during pilot phase

### 4. Performance by Design
- **≤100ms latency SLA**: Optimized for real-time decisions
- **Caching**: Pre-computed features, SDOH lookups
- **Circuit breaker**: Fallback to safe defaults if latency exceeded
- **Horizontal scaling**: Kubernetes auto-scaling

### 5. Equity by Design
- **Policy-as-Code**: Formal fairness constraints compiled into policies
- **Zero-touch**: No extra user input required
- **Multi-objective**: Balance equity, safety, and business goals
- **Stepped-wedge rollout**: Causal inference with fairness audits

---

## Success Metrics Summary

| Category | Metric | Target | Current |
|----------|--------|--------|---------|
| **Fairness** | Equalized Uplift | \|EU_A - EU_B\| < 0.05 | TBD |
| **Fairness** | Price Burden Ratio | PBR < 0.30 | TBD |
| **Safety** | Safety Harm Rate | SHR < 1% | TBD |
| **Business** | Conversion Rate | +5-10% | TBD |
| **Business** | Average Order Value | +10-15% | TBD |
| **Technical** | Latency (p99) | ≤100ms | TBD |
| **Technical** | Uptime | 99.9% | TBD |
| **Impact** | Out-of-Pocket Savings | $50-100/mo | TBD |
| **Impact** | Nutritional Improvement | +20% fruits/veggies | TBD |

---

## Next Steps

1. **Review architecture** with stakeholders
2. **Set up development environment** (Week 1)
3. **Integrate SDOH datasets** (Week 2)
4. **Build need state learning model** (Weeks 7-8)
5. **Implement constrained policies** (Weeks 11-14)
6. **Deploy guardrailed bandit** (Weeks 15-18)
7. **Launch pilot** (Week 28)
8. **Full rollout** (Week 52)

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the complete 52-week plan.
