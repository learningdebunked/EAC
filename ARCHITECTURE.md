# Equity-Aware Checkout (EAC) - System Architecture

## Executive Summary

The Equity-Aware Checkout (EAC) is an AI agentic framework providing **zero-touch personalization** at checkout by integrating privacy-preserving Social Determinants of Health (SDOH) signals. The system operates under strict latency constraints (≤100ms) while optimizing for fairness, safety, and business metrics.

---

## Core Contributions

1. **Policy-as-Code for Equity**: Formal fairness/safety constraints compiled into checkout policies
2. **Zero-Touch Signals**: Personalization without extra user input; opt-in, revocable, explainable
3. **Multi-Objective Optimizer**: Minimizes cost/risk while protecting revenue/latency SLAs
4. **New Equity Metrics**: Equalized Uplift, Price Burden Ratio, Safety Harm Rate
5. **Stepped-Wedge Deployment**: Causal inference at scale with fairness audits

---

## High-Level Architecture

```
User Checkout → Privacy Layer → SDOH Data Integration → Need State Learning
     ↓
Constrained Policies (5 types) → Guardrailed Bandit (≤100ms) → Multi-Objective Optimizer
     ↓
Fairness Evaluation → Stepped-Wedge Rollout → Monitoring & Feedback
```

---

## 1. Data Integration Layer

### A) SDOH, Equity & Environment (Census Tract/ZIP3)

| Dataset | Purpose | Source |
|---------|---------|--------|
| **CDC Social Vulnerability Index (SVI)** | Tract-level social risk indicators | ATSDR/CDC SVI 2022 |
| **Area Deprivation Index (ADI)** | Neighborhood disadvantage scores | Neighborhood Atlas |
| **USDA Food Access Research Atlas** | Food desert metrics, grocery distance | USDA Economic Research Service |
| **SNAP Retailer Locator** | SNAP-authorized retailer locations | USDA Food & Nutrition Service |
| **EPA EJScreen (CEJST)** | Environmental burdens, disadvantaged flags | US EPA Data + Screening Tools |
| **NOAA NWS HeatRisk** | Daily heat risk for safe delivery | NOAA NCEP Weather Prediction |
| **FEMA National Risk Index** | Hazard risk for resilience routing | FEMA Hazards |
| **National Transit Map** | Transit availability for mobility | Bureau of Transportation Statistics |
| **U.S. Census ACS API** | Demographics, income, broadband | Census.gov API |

### B) Product & Nutrition Knowledge

| Dataset | Purpose | Source |
|---------|---------|--------|
| **USDA FoodData Central** | Nutrients, ingredients, glycemic index | FoodData Central API |
| **Open Food Facts** | UPC-level labels, allergens | Open Food Facts + AWS Open Data |
| **WIC Authorized Product Lists** | SNAP/WIC eligibility by UPC/PLU | State APIs (e.g., Data.ca.gov) |

### C) Retail Baskets, Sessions & RecSys

| Dataset | Purpose | Source |
|---------|---------|--------|
| **Instacart Market Basket** | Multi-table grocery orders | Kaggle |
| **UCI Online Retail I/II** | Transactional logs with cancellations | UCI ML Repository |
| **dunnhumby Complete Journey** | Household spend, promotions | dunnhumby (T&Cs) |
| **RetailRocket Events** | Session views/carts/purchases | Kaggle |

### D) (Optional) CGM/Diabetes Research

| Dataset | Purpose | Source |
|---------|---------|--------|
| **OhioT1DM** | CGM + insulin + meals | UNC Charlotte (controlled access) |
| **OpenAPS Data Commons** | Community CGM/loop data | openaps.org |
| **UVA/Padova Simulators** | Synthetic glucose trajectories | tegvirginia.com + GitHub |

---

## 2. Privacy & Consent Layer

- **Opt-in/Opt-out Management**: User controls for SDOH signal usage
- **Consent Tracking**: Revocable, auditable consent logs
- **Explainability Engine**: "Why this recommendation?" transparency
- **PII Protection**: No personal identifiers in SDOH signals (census tract aggregates only)

---

## 3. Feature Engineering & Signal Processing

### Behavioral Features (Non-PII, Consented)
- Cart composition (categories, price points)
- Browsing patterns (time-on-site, searches)
- Purchase timing (time of day, day of week)
- Substitution history (past swaps, acceptance)
- Delivery preferences (windows, frequency)
- Payment methods (SNAP/EBT, FSA/HSA usage)

### SDOH Signal Aggregation
- Geocode delivery address → census tract (privacy-preserving)
- Join SDOH indices: SVI, ADI, food access, transit
- Compute composite risk scores: food insecurity, mobility, health
- Environmental context: heat risk, hazard risk, pollution

---

## 4. Need State Learning Engine

### Multi-Task Deep Neural Network

**Architecture:**
```
Input: [Behavioral Features] + [SDOH Signals] + [Product Context]
  ↓
Shared Layers:
  Dense(512) → BatchNorm → ReLU → Dropout(0.3)
  Dense(256) → BatchNorm → ReLU → Dropout(0.3)
  Dense(128) → BatchNorm → ReLU
  ↓
Task-Specific Heads:
  ├─ Food Insecurity Predictor → [Low/Medium/High Risk]
  ├─ Transportation Constraint Detector → [Has Transit/No Transit]
  ├─ Chronic Condition Proxy → [Diabetes/CVD Risk Scores]
  ├─ Financial Constraint Analyzer → [Budget Sensitivity]
  └─ Mobility Limitation Assessor → [Mobility Score]
```

**Uncertainty Quantification:**
- Monte Carlo Dropout for epistemic uncertainty
- Temperature scaling for calibration
- Confidence thresholds (reject if uncertainty > 0.3)
- Conformal prediction for coverage guarantees

---

## 5. Constrained Policy Engine (Policy-as-Code)

### Policy 1: SNAP/WIC-Compatible Substitutions
**Trigger:** SNAP/EBT payment OR high food insecurity score  
**Actions:**
- Identify non-eligible items in cart
- Query WIC Authorized Product List for alternatives
- Match by category, nutrition, brand preference
- Rank by price, nutritional value, availability
- Present top 3 swaps with savings

**Constraints:**
- Maintain/improve nutritional value
- Price ≤ original (or show SNAP savings)
- Respect dietary restrictions
- Explainability: "SNAP-eligible alternative saves $X"

### Policy 2: Low-Glycemic Alternatives
**Trigger:** Diabetes risk OR high-glycemic items in cart  
**Actions:**
- Score cart items by glycemic index (USDA FoodData)
- Identify high-GI items (GI > 70)
- Find low-GI alternatives (GI < 55)
- Gentle nudge: "Lower-sugar option available"

**Constraints:**
- Safety-first: no shaming/stigma
- Price delta ≤ 15%
- Maintain meal satisfaction
- Explainability: "Better for blood sugar"

### Policy 3: Plan-Aware OTC Coverage
**Trigger:** OTC items + FSA/HSA/insurance  
**Actions:**
- Check FSA/HSA eligibility
- Query insurance formulary
- Calculate out-of-pocket with/without coverage
- Auto-apply FSA/HSA if eligible

**Constraints:**
- Privacy: consented insurance data only
- Show clear savings
- Explainability: "Covered by your plan"

### Policy 4: Mobility-Aligned Delivery Windows
**Trigger:** Mobility limitation OR low transit availability  
**Actions:**
- Query transit schedules (National Transit Map)
- Align delivery windows with transit
- Prioritize accessible delivery (ground floor)
- Integrate heat risk (NOAA) and hazard risk (FEMA)

**Constraints:**
- Offer ≥3 accessible windows
- No price penalty
- Heat safety: avoid extreme heat times
- Explainability: "Matches bus schedule"

### Policy 5: Safety-First Product Nudges
**Trigger:** Cart improvement opportunities  
**Actions:**
- Nutritional improvement (fruits/veggies)
- Cost-effective alternatives (bulk pricing)
- Preventive health items (vitamins, first aid)
- Complementary items (equity-aware)

**Constraints:**
- Never pushy/judgmental
- Improve cart value (nutrition, cost, convenience)
- Respect budget constraints
- Explainability: "Save $X by buying larger size"

---

## 6. Guardrailed Contextual Bandit Engine (≤100ms)

### Algorithm: LinUCB with Safety Constraints

**Decision Process:**
1. Receive context (user, cart, SDOH)
2. Compute upper confidence bounds for each policy
3. Apply guardrails (fairness, safety, business)
4. Select policy with highest constrained UCB
5. Execute policy actions
6. Observe reward (conversion, satisfaction, fairness)
7. Update policy weights

**Exploration Strategy:**
- ε-greedy with decay (20% → 5%)
- Thompson Sampling for uncertainty-aware exploration
- Forced exploration for underrepresented groups

**Latency Optimization:**
- Pre-computed policy embeddings (cached)
- Fast linear algebra (BLAS/LAPACK)
- Async SDOH lookups (cached by census tract)
- Circuit breaker: fallback if >100ms

### Guardrail System (Hard Constraints)

**1. Fairness Guardrails:**
- Equalized Uplift: Δ(conversion | policy) ≈ equal across groups
- Price Burden Ratio: out-of-pocket / income ≤ threshold
- Demographic Parity: P(policy | group) ≈ equal

**2. Safety Guardrails:**
- No harmful substitutions (allergens, contraindications)
- No predatory pricing
- No stigmatizing language
- Confidence threshold: reject if uncertainty > 0.3

**3. Business Guardrails:**
- Margin protection: ≥ baseline - 5%
- Inventory constraints: in-stock only
- Latency SLA: ≤100ms
- ROI threshold: expected value ≥ cost

**4. Regulatory Guardrails:**
- SNAP/WIC compliance
- ADA compliance
- HIPAA compliance
- FTC compliance

---

## 7. Multi-Objective Optimization

### Objective Function (Pareto Optimization)

**Minimize:**
- Out-of-pocket spend (primary equity metric)
- Post-purchase adverse proxies (health, financial stress)
- Safety Harm Rate (% harmful recommendations)

**Maximize:**
- Attach rate (additional items purchased)
- Completion rate (checkout conversion)
- Customer satisfaction (NPS, repeat purchase)

**Subject to Constraints:**
- Equalized Uplift across protected groups
- Price Burden Ratio ≤ threshold
- Latency ≤100ms
- Margin ≥ baseline - 5%

**Solution Method:**
- Weighted scalarization with dynamic weights
- Lagrangian relaxation for constraints
- Gradient-based optimization (Adam)

---

## 8. Fairness & Safety Evaluation

### New Equity Metrics for Retail

**1. Equalized Uplift (EU)**
- Definition: Δ(outcome | treatment) equal across groups
- Formula: |EU_groupA - EU_groupB| ≤ ε
- Measures: Fairness in benefit distribution

**2. Price Burden Ratio (PBR)**
- Definition: Out-of-pocket spend / household income
- Formula: PBR = Σ(spend_i) / income
- Measures: Affordability impact across income levels

**3. Safety Harm Rate (SHR)**
- Definition: % recommendations causing adverse outcomes
- Formula: SHR = (harmful_recs / total_recs) × 100
- Measures: Safety of personalization system

### Evaluation Framework
- Demographic parity checks
- Disparate impact analysis
- Intersectional fairness (race × income × health)
- Longitudinal monitoring (drift detection)

---

## 9. Stepped-Wedge Rollout

### Deployment Strategy

**Phase 1: Pilot (Weeks 1-4)**
- 5% of users, single geography
- Intensive monitoring, manual review
- Fairness audits daily

**Phase 2: Expansion (Weeks 5-12)**
- 25% of users, multiple geographies
- Automated fairness checks
- A/B/n testing framework

**Phase 3: Scale (Weeks 13-24)**
- 75% of users, nationwide
- Continuous learning enabled
- Policy template release

**Phase 4: Full Rollout (Week 25+)**
- 100% of users
- Open-source policy templates
- Evaluation checklists published

### Causal Inference
- Randomized controlled trials (RCTs) within cohorts
- Difference-in-differences for rollout waves
- Synthetic control for geographic expansion
- Propensity score matching for observational data

---

## 10. Monitoring & Continuous Learning

### Real-Time Metrics Dashboard
- Latency monitoring (≤100ms SLA)
- Fairness metrics (EU, PBR, SHR)
- Business KPIs (conversion, AOV, margin)
- Safety alerts (harmful recommendations)

### Continuous Learning Pipeline
- Real-time feedback integration
- Model retraining triggers (weekly)
- Drift detection (data, concept, fairness)
- Policy performance tracking
- A/B test results analysis

### Alerting System
- Fairness violation alerts
- Safety incident escalation
- Latency SLA breaches
- Business metric anomalies

---

## 11. Technology Stack

### Core Framework
- **Language:** Python 3.10+
- **ML Framework:** PyTorch 2.0+
- **Bandit Library:** Vowpal Wabbit / custom LinUCB
- **API Framework:** FastAPI
- **Database:** PostgreSQL (user data), Redis (caching)
- **Message Queue:** Apache Kafka (event streaming)

### Data Processing
- **ETL:** Apache Airflow
- **Feature Store:** Feast
- **Data Warehouse:** BigQuery / Snowflake

### Deployment
- **Containerization:** Docker
- **Orchestration:** Kubernetes
- **CI/CD:** GitHub Actions
- **Monitoring:** Prometheus + Grafana
- **Logging:** ELK Stack (Elasticsearch, Logstash, Kibana)

### ML Ops
- **Experiment Tracking:** MLflow
- **Model Registry:** MLflow Model Registry
- **Feature Monitoring:** Evidently AI
- **Fairness Auditing:** Fairlearn, AIF360

---

## 12. API Design

### Checkout Decision API

**Endpoint:** `POST /api/v1/checkout/decide`

**Request:**
```json
{
  "user_id": "hashed_user_id",
  "cart": [
    {"product_id": "12345", "quantity": 2, "price": 4.99},
    {"product_id": "67890", "quantity": 1, "price": 12.99}
  ],
  "delivery_address": {
    "zip_code": "94102",
    "census_tract": "06075017902"
  },
  "payment_methods": ["SNAP_EBT", "CREDIT_CARD"],
  "consent": {
    "sdoh_signals": true,
    "personalization": true
  }
}
```

**Response (≤100ms):**
```json
{
  "recommendations": [
    {
      "policy": "snap_wic_substitution",
      "action": "substitute",
      "original_product_id": "12345",
      "suggested_product_id": "12346",
      "reason": "SNAP-eligible alternative saves $1.50",
      "savings": 1.50,
      "confidence": 0.92
    },
    {
      "policy": "low_glycemic_alternative",
      "action": "nudge",
      "product_id": "67890",
      "suggested_product_id": "67891",
      "reason": "Lower-sugar option available",
      "confidence": 0.85
    }
  ],
  "delivery_windows": [
    {
      "start": "2025-10-26T14:00:00Z",
      "end": "2025-10-26T16:00:00Z",
      "reason": "Matches local bus schedule",
      "accessibility_score": 0.95
    }
  ],
  "latency_ms": 87,
  "fairness_check": "passed"
}
```

---

## 13. Security & Privacy

### Data Protection
- **Encryption:** TLS 1.3 in transit, AES-256 at rest
- **Anonymization:** Census tract aggregation (no street addresses stored)
- **Access Control:** RBAC with least privilege
- **Audit Logging:** All data access logged and monitored

### Compliance
- **GDPR:** Right to erasure, data portability
- **CCPA:** Opt-out mechanisms, data disclosure
- **HIPAA:** No PHI storage (proxies only)
- **PCI DSS:** Payment data handled by certified processors

### Consent Management
- **Granular Consent:** Separate opt-ins for SDOH, personalization, data sharing
- **Revocable:** Users can withdraw consent anytime
- **Transparent:** Clear explanations of data usage
- **Auditable:** Consent history tracked

---

## 14. Evaluation Checklist

### Pre-Deployment
- [ ] Fairness metrics (EU, PBR, SHR) meet thresholds
- [ ] Safety testing (no harmful recommendations in test set)
- [ ] Latency benchmarking (p99 ≤ 100ms)
- [ ] Privacy audit (no PII leakage)
- [ ] Regulatory compliance review
- [ ] A/B test design approved
- [ ] Rollback plan documented

### Post-Deployment (Weekly)
- [ ] Fairness metrics monitored across protected groups
- [ ] Safety incidents reviewed and addressed
- [ ] Latency SLA compliance checked
- [ ] Business metrics (conversion, AOV) tracked
- [ ] User feedback analyzed
- [ ] Model drift detected and retrained if needed
- [ ] Policy performance compared across cohorts

---

## 15. Open Source Release Plan

### Policy Templates (JSON/YAML)
- SNAP/WIC substitution rules
- Low-glycemic alternative logic
- Mobility-aligned delivery constraints
- Safety-first nudge guidelines

### Evaluation Checklists
- Fairness audit procedures
- Safety testing protocols
- Latency benchmarking tools
- Regulatory compliance guides

### Reference Implementation
- Simplified bandit algorithm (Python)
- SDOH data integration examples
- Fairness metric calculators
- Stepped-wedge rollout framework

---

## Conclusion

The Equity-Aware Checkout (EAC) framework represents a novel approach to personalization that centers equity, safety, and privacy. By integrating SDOH signals, constrained policies, and guardrailed bandits, EAC demonstrates that it's possible to improve business outcomes while reducing disparities and protecting vulnerable populations.

**Key Innovations:**
1. Zero-touch personalization with opt-in SDOH signals
2. Policy-as-code for formal fairness constraints
3. New equity metrics (Equalized Uplift, Price Burden Ratio, Safety Harm Rate)
4. Real-time decisions under 100ms latency
5. Stepped-wedge deployment with causal inference

**Next Steps:**
1. Set up development environment
2. Integrate SDOH datasets
3. Implement need state learning model
4. Build constrained policy engine
5. Deploy guardrailed bandit
6. Launch pilot with fairness audits
