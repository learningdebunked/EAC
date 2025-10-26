# Equity-Aware Checkout (EAC)

> An AI agentic framework for zero-touch personalization that integrates privacy-preserving Social Determinants of Health (SDOH) signals to reduce cost, risk, and time-to-access for essential goods.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🎯 Overview

The Equity-Aware Checkout (EAC) system learns need states from consented, non-PII behavioral features and public-level SDOH indices, then applies constrained policies at checkout:

- **SNAP/WIC-compatible substitutions**
- **Low-glycemic alternatives**
- **Plan-aware OTC coverage**
- **Delivery windows aligned to mobility limits**
- **Safety-first product nudges**

We formalize fairness and safety objectives alongside business metrics and deploy a **guardrailed contextual bandit** for real-time decisions under **≤100ms latency**.

**Validation Approach:** Since real production deployment is not feasible, we validate the hypothesis through **high-fidelity simulations** using real-world datasets (Instacart, dunnhumby, UCI Retail, RetailRocket) enriched with SDOH data. This enables rigorous counterfactual analysis, reproducibility, and ethical testing without risk to real users.

---

## 🌟 Core Contributions

1. **Policy-as-Code for Equity**: Formal fairness/safety constraints compiled into checkout policies
2. **Zero-Touch Signals**: Personalization without extra user input; opt-in, revocable, explainable
3. **Multi-Objective Optimizer**: Minimizes cost/risk while protecting revenue/latency SLAs
4. **New Equity Metrics**: Equalized Uplift, Price Burden Ratio, Safety Harm Rate
5. **Simulation-Based Validation**: Counterfactual analysis with 100K synthetic users, 1000 replications
6. **Formal Theoretical Guarantees**: Differential privacy (ε ≤ 0.1), convergence proofs, PAC-learning
7. **Reproducible Research**: Full simulation code, synthetic datasets, Docker containers

---

## 📊 Key Features

### Privacy-Preserving SDOH Integration
- Census tract-level aggregation (no street addresses)
- **Differential privacy guarantees**: (ε ≤ 0.1, δ ≤ 10^-6)
- Opt-in consent management
- Explainable recommendations
- GDPR/CCPA/HIPAA compliant

### Real-Time Decision Engine
- Guardrailed contextual bandit (LinUCB)
- **Convergence guarantee**: O(d√(T log T)) regret bound
- Multi-objective optimization with Nash equilibrium
- **≤100ms latency SLA** with complexity proofs
- Continuous learning pipeline

### Fairness & Safety
- **Equalized Uplift**: Equal benefit across protected groups (|EU_A - EU_B| ≤ 0.05)
- **Price Burden Ratio**: Affordability monitoring (PBR ≤ 0.30)
- **Safety Harm Rate**: Harmful recommendation tracking (SHR ≤ 0.01)
- **PAC-learning guarantees** for need state prediction
- Automated fairness audits with formal verification

### Constrained Policies
- SNAP/WIC eligibility checking
- Glycemic index-aware substitutions
- FSA/HSA coverage optimization
- Transit-aligned delivery windows
- Nutritional improvement nudges

---

## 🏗️ Architecture

```
User Checkout → Privacy Layer → SDOH Data Integration → Need State Learning
     ↓
Constrained Policies (5 types) → Guardrailed Bandit (≤100ms) → Multi-Objective Optimizer
     ↓
Fairness Evaluation → Stepped-Wedge Rollout → Monitoring & Feedback
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## 🤖 The AI Agent's Role

### What is the AI Agent?

The **AI Agent** is the autonomous decision-making system at the heart of EAC that operates during checkout:

```
┌─────────────────────────────────────────────────────────────┐
│                    USER AT CHECKOUT                          │
│  Cart: [Bread, Milk, Chips, Soda]                          │
│  Payment: SNAP/EBT + Credit Card                            │
│  Location: High food insecurity area                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI AGENT (EAC System)                       │
│                                                              │
│  1. OBSERVE: Parse cart + SDOH signals                      │
│     → Detect: SNAP/EBT, high food insecurity, budget items  │
│                                                              │
│  2. THINK: Infer needs + Select policy                      │
│     → Need states: Food insecurity HIGH, Nutrition risk MED │
│     → Policy: SNAP/WIC substitution (selected by bandit)    │
│     → Guardrails: ✓ Fairness OK, Safety OK                  │
│                                                              │
│  3. ACT: Execute policy + Generate recommendations          │
│     → Chips → Whole grain crackers (SNAP, saves $0.50)     │
│     → Soda → 100% juice (SNAP, saves $0.50, +3 nutrition)  │
│                                                              │
│  4. LEARN: Update from user response                        │
│     → User accepts 2/2 recommendations                      │
│     → Reward: +1.5 (savings + nutrition + acceptance)       │
│     → Update: Increase SNAP/WIC policy weight               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              CHECKOUT RECOMMENDATIONS                        │
│                                                              │
│  💡 "SNAP-eligible alternative saves $0.50"                 │
│     Chips → Whole Grain Crackers [Accept] [Reject]         │
│                                                              │
│  💡 "Better nutrition, saves $0.50"                         │
│     Soda → 100% Juice [Accept] [Reject]                    │
│                                                              │
│  Total Savings: $1.00 | Nutrition: +3 HEI points           │
└─────────────────────────────────────────────────────────────┘
```

### Why "Agentic"?

The system exhibits key **agent properties**:

1. **Autonomy**: Makes decisions without human intervention in real-time (≤100ms)
2. **Reactivity**: Responds immediately to checkout events and context changes
3. **Proactivity**: Anticipates needs from SDOH signals before problems arise
4. **Social Ability**: Interacts with users through explainable recommendations
5. **Learning**: Continuously improves from user feedback (online learning)

### Agent Decision Cycle

```python
class EACAgent:
    """AI Agent for Equity-Aware Checkout"""
    
    def process_checkout(self, checkout_event):
        # 1. OBSERVE: Perceive environment
        context = self.perception.observe(checkout_event)
        # → cart, SDOH signals, payment methods, constraints
        
        # 2. THINK: Reason about needs and select policy
        need_states = self.reasoning.infer_needs(context)
        # → food_insecurity: 0.85, financial_constraint: 0.92
        
        policy = self.reasoning.select_policy(context, need_states)
        # → 'snap_wic_substitution' (selected by contextual bandit)
        
        # Check guardrails
        if not self.reasoning.check_guardrails(policy, context):
            return self.safe_default()
        
        # 3. ACT: Execute policy and generate recommendations
        recommendations = self.action.execute_policy(policy, context)
        # → [substitute chips, substitute soda, ...]
        
        # 4. LEARN: Update from user response
        user_response = self.wait_for_user_response(recommendations)
        reward = self.compute_reward(user_response, context)
        self.learning.update(context, policy, reward)
        
        return recommendations
```

### Multi-Objective Optimization

The agent balances competing objectives:

- **User Satisfaction**: Maximize acceptance rate
- **Cost Reduction**: Minimize out-of-pocket spend
- **Health Improvement**: Maximize nutritional quality
- **Fairness**: Ensure equalized uplift across groups
- **Business Viability**: Maintain retailer margins

### Example Scenario

**User**: Low-income household with SNAP/EBT  
**Cart**: Sugary cereal ($4), white bread ($3), soda ($2), chips ($4) = $13 total

**Agent Actions**:
1. **Observes**: SNAP payment + high food insecurity (SDOH)
2. **Thinks**: Food insecurity HIGH (0.92), Nutrition risk HIGH (0.78)
3. **Selects**: SNAP/WIC substitution policy (UCB = 0.88)
4. **Acts**: 
   - Sugary cereal → Whole grain cereal (SNAP, -$0.50, +8 HEI)
   - White bread → Whole wheat bread (SNAP, $0, +5 HEI)
   - Soda → 100% juice (SNAP, -$0.30, +3 HEI)
   - Chips → Whole grain crackers (SNAP, -$0.20, +4 HEI)
5. **Learns**: User accepts 3/4 → Reward +1.5 → Strengthen SNAP policy

**Result**: User saves $1.00, improves nutrition by +17 HEI points, agent learns effective strategy

---

## 📚 Documentation

- **[Architecture](ARCHITECTURE.md)**: Detailed system architecture with formal theory
  - Section 15: Formal Theoretical Framework (6 theorems with proofs)
  - Section 16: Simulation-Based Validation Framework (counterfactual analysis)
- **[Architecture Summary](ARCHITECTURE_SUMMARY.md)**: Visual diagrams and quick reference
- **[Project Roadmap](PROJECT_ROADMAP.md)**: 52-week implementation plan
- **[Simulation Guide](docs/SIMULATION.md)**: How to run simulations (coming soon)
- **[API Documentation](docs/API.md)**: REST API reference (coming soon)
- **[Policy Templates](docs/POLICIES.md)**: Policy-as-code examples (coming soon)
- **[Evaluation Checklist](docs/EVALUATION.md)**: Fairness audit procedures (coming soon)

---

## 🎨 User Interface

### React Frontend

The EAC system includes a modern, responsive React frontend for user interaction:

![Frontend - Shopping Cart](docs/screenshots/frontend-cart.png)
*Shopping cart with user profile customization and checkout button*

![Frontend - Recommendations](docs/screenshots/frontend-recommendations.png)
*AI-powered recommendations with savings, nutrition improvements, and accept/decline options*

![Frontend - Impact Dashboard](docs/screenshots/frontend-impact.png)
*Real-time impact visualization showing savings, nutrition gains, and processing time*

**Features:**
- 🛒 Interactive shopping cart
- 👤 User profile customization (income, SNAP eligibility, SDOH factors)
- ✨ Real-time recommendations from EAC Agent
- 📊 Impact visualization
- 📱 Mobile-responsive design

**Access**: http://localhost:3000 (when running)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose

### Installation

```bash
# Clone the repository
git clone https://github.com/learningdebunked/EAC.git
cd EAC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Start services
docker-compose up -d

# Run database migrations
python scripts/migrate.py

# Start the API server
uvicorn app.main:app --reload
```

### Running Your First Simulation

```python
from eac.simulation import run_counterfactual_simulation
from eac.data import load_instacart_data, enrich_with_sdoh
from eac.system import EACSystem

# Load and prepare data
transactions = load_instacart_data(n_users=1000)
transactions = enrich_with_sdoh(transactions)

# Initialize EAC system
eac_system = EACSystem(
    policies=['snap_wic', 'low_glycemic', 'otc_coverage'],
    fairness_constraints={'equalized_uplift': 0.05, 'price_burden_ratio': 0.30}
)

# Run simulation
results = run_counterfactual_simulation(
    transactions=transactions,
    eac_system=eac_system,
    n_replications=100
)

# Analyze results
from eac.analysis import analyze_simulation_results
summary = analyze_simulation_results(results)
print(summary)
```

### Running Production API (Optional)

```python
import requests

response = requests.post("http://localhost:8000/api/v1/checkout/decide", json={
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
        "sdoh_signals": True,
        "personalization": True
    }
})

print(response.json())
```

---

## 📊 Datasets

### SDOH & Environment
- CDC Social Vulnerability Index (SVI)
- Area Deprivation Index (ADI)
- USDA Food Access Research Atlas
- SNAP Retailer Locator
- EPA EJScreen (CEJST)
- NOAA NWS HeatRisk
- FEMA National Risk Index
- National Transit Map (USDOT BTS)
- U.S. Census ACS API

### Product & Nutrition
- USDA FoodData Central
- Open Food Facts
- WIC Authorized Product Lists

### Retail Baskets
- Instacart Market Basket Analysis
- UCI Online Retail I/II
- dunnhumby Complete Journey
- RetailRocket e-commerce events

See [ARCHITECTURE.md](ARCHITECTURE.md) for data integration details.

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run fairness tests
pytest tests/fairness

# Run load tests (latency benchmarking)
locust -f tests/load/locustfile.py
```

---

## 📈 Metrics & Monitoring

### Fairness Metrics
- **Equalized Uplift**: |EU_groupA - EU_groupB| < 0.05
- **Price Burden Ratio**: PBR_low_income < 0.30
- **Safety Harm Rate**: SHR < 0.01 (1%)

### Business Metrics
- Conversion Rate: +5-10% target
- Average Order Value: +10-15% target
- Customer Satisfaction: NPS > 50
- Repeat Purchase Rate: +15-20% target

### Technical Metrics
- Latency: p99 ≤ 100ms
- Uptime: 99.9% SLA
- Model Accuracy: AUC > 0.85
- API Throughput: >1000 req/sec

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black .
flake8 .
mypy .

# Run tests
pytest
```

---

## 📊 Monitoring & Analytics Dashboard

### Real-Time System Observability

The EAC system includes a **complete real-time analytics pipeline** that tracks every user interaction from frontend to backend:

```
User Interaction → API → EAC Agent → Transaction Store → Analytics Dashboard
                                            ↓
                                    Auto-refresh (5 sec)
```

### Analytics Dashboard Features

**Access**: http://localhost:8501 (when running)

![Analytics Dashboard Overview](docs/screenshots/dashboard-overview.png)
*Real-time analytics dashboard showing key metrics, charts, and fairness analysis*

#### 📈 **Key Metrics (Real-Time)**
- **Acceptance Rate**: % of recommendations users accept
- **Average Savings**: Actual savings per transaction
- **Nutrition Improvement**: HEI points gained
- **System Latency**: Processing time (SLA: ≤100ms)

#### 📊 **Interactive Charts**

![Dashboard - Performance Charts](docs/screenshots/dashboard-charts.png)
*Interactive charts showing acceptance rates, savings distribution, and nutrition impact by policy*

- **Acceptance by Policy**: Compare SNAP/WIC, Low Glycemic, Budget Optimizer
- **Savings Distribution**: Histogram of user savings
- **Nutrition Impact**: Box plots by policy
- **Latency Trends**: Time series performance monitoring
- **Fairness Analysis**: Savings and acceptance by demographic group

#### ⚖️ **Fairness Monitoring**

![Dashboard - Fairness Analysis](docs/screenshots/dashboard-fairness.png)
*Real-time fairness monitoring showing disparity across demographic groups with automated alerts*

- **Real-time disparity tracking** across protected groups
- **Automated alerts** if max disparity > $3
- **Equalized Uplift verification**
- **Visual fairness dashboard**

#### 📋 **Transaction Table**

![Dashboard - Transaction Table](docs/screenshots/dashboard-transactions.png)
*Detailed transaction table with sorting, filtering, and CSV export capabilities*

- Recent transactions with full details
- Sortable and filterable
- Export to CSV
- Drill-down capabilities

### Quick Start

```bash
# Terminal 1: Start API
source .venv/bin/activate
uvicorn api.main:app --reload

# Terminal 2: Start Frontend
cd frontend/react-app
npm run dev

# Terminal 3: Start Analytics Dashboard
streamlit run frontend/streamlit_dashboard.py
```

**Access Points:**
- Frontend: http://localhost:3000
- Analytics: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Data Flow

Every user interaction is tracked:

1. **Checkout**: Transaction created with unique ID
2. **Recommendations**: Policy selection, potential savings recorded
3. **Accept/Decline**: User feedback updates transaction
4. **Dashboard**: Auto-refreshes every 5 seconds with new data

### Monitoring Capabilities

#### **System Health**
- ✅ Latency monitoring (P50, P95, P99)
- ✅ Error rate tracking
- ✅ Throughput metrics
- ✅ SLA compliance (≤100ms)

#### **Business Metrics**
- ✅ Acceptance rate trends
- ✅ Revenue impact
- ✅ User engagement
- ✅ Policy performance comparison

#### **Fairness Metrics**
- ✅ Equalized Uplift by group
- ✅ Price Burden Ratio
- ✅ Disparity alerts
- ✅ Demographic breakdowns

#### **ML Model Performance**
- ✅ Prediction accuracy
- ✅ Calibration metrics
- ✅ Drift detection
- ✅ A/B test results

### Production Deployment

For production, replace CSV storage with:

**PostgreSQL** for transaction storage:
```python
# In api/data_store.py
import psycopg2
conn = psycopg2.connect(DATABASE_URL)
```

**Redis** for real-time streaming:
```python
import redis
r = redis.Redis()
r.publish('transactions', json.dumps(data))
```

**Grafana + Prometheus** for monitoring:
- System metrics
- Custom business metrics
- Alerting rules
- SLA dashboards

### Documentation

- **REAL_TIME_ANALYTICS.md**: Complete analytics guide
- **FRONTEND_BACKEND_FLOW.md**: API integration details
- **FRONTEND_GUIDE.md**: Frontend usage guide

### Example Analytics Session

```bash
# 1. Start all services
./scripts/start_all_frontends.sh

# 2. Use frontend to create transactions
# Open http://localhost:3000
# Click checkout, accept/decline recommendations

# 3. Watch live data
tail -f live_transactions.csv

# 4. View analytics
# Open http://localhost:8501
# Click "🔄 Refresh Data"
# See real-time metrics update
```

### Benefits

- **Real-time insights**: See impact immediately
- **Data-driven decisions**: Which policies work best
- **Fairness monitoring**: Continuous equity tracking
- **System health**: Performance and reliability metrics
- **Stakeholder visibility**: Live dashboard for demos
- **Continuous learning**: ML models improve from feedback

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SDOH Data Providers**: CDC, USDA, EPA, NOAA, FEMA, Census Bureau
- **Product Data**: USDA FoodData Central, Open Food Facts
- **Retail Datasets**: Instacart, UCI, dunnhumby, RetailRocket
- **Fairness Libraries**: Fairlearn, AIF360
- **ML Frameworks**: PyTorch, Vowpal Wabbit, MLflow

---

## 📞 Contact

- **Project Lead**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub Issues**: [github.com/learningdebunked/EAC/issues](https://github.com/learningdebunked/EAC/issues)
- **Discussions**: [github.com/learningdebunked/EAC/discussions](https://github.com/learningdebunked/EAC/discussions)

---

## 🗺️ Roadmap

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the complete 52-week implementation plan.

**Current Status**: Architecture + Formal Theory + Simulation Framework Complete

**Next Milestones**:
- **Months 1-4**: Data preparation, outcome model training
- **Months 5-8**: Simulation runs (1000 replications, 100K users)
- **Months 9-12**: Analysis, paper writing, FAccT submission
- **Year 2**: FAccT publication + NeurIPS submission (theoretical contributions)
- **Year 3**: NeurIPS publication + Nature/Science submission (large-scale simulation)
- **Year 4**: Nature publication + industry outreach + policy recommendations

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{eac2025,
  title={Equity-Aware Checkout: An AI Agentic Framework for Fair Personalization},
  author={[Your Name]},
  year={2025},
  url={https://github.com/learningdebunked/EAC}
}
```

---

## ⚠️ Disclaimer

This system is designed to improve equity and reduce disparities, but it is not a substitute for human judgment or professional advice. Always consult with domain experts (nutritionists, healthcare providers, financial advisors) for critical decisions.

---

**Built with ❤️ for equitable access to essential goods**
