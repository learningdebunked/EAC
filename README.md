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

---

## 🌟 Core Contributions

1. **Policy-as-Code for Equity**: Formal fairness/safety constraints compiled into checkout policies
2. **Zero-Touch Signals**: Personalization without extra user input; opt-in, revocable, explainable
3. **Multi-Objective Optimizer**: Minimizes cost/risk while protecting revenue/latency SLAs
4. **New Equity Metrics**: Equalized Uplift, Price Burden Ratio, Safety Harm Rate
5. **Stepped-Wedge Deployment**: Causal inference at scale with fairness audits

---

## 📊 Key Features

### Privacy-Preserving SDOH Integration
- Census tract-level aggregation (no street addresses)
- Opt-in consent management
- Explainable recommendations
- GDPR/CCPA/HIPAA compliant

### Real-Time Decision Engine
- Guardrailed contextual bandit (LinUCB)
- Multi-objective optimization
- ≤100ms latency SLA
- Continuous learning pipeline

### Fairness & Safety
- **Equalized Uplift**: Equal benefit across protected groups
- **Price Burden Ratio**: Affordability monitoring
- **Safety Harm Rate**: Harmful recommendation tracking
- Automated fairness audits

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

## 📚 Documentation

- **[Architecture](ARCHITECTURE.md)**: Detailed system architecture and design
- **[Project Roadmap](PROJECT_ROADMAP.md)**: 52-week implementation plan
- **[API Documentation](docs/API.md)**: REST API reference (coming soon)
- **[Policy Templates](docs/POLICIES.md)**: Policy-as-code examples (coming soon)
- **[Evaluation Checklist](docs/EVALUATION.md)**: Fairness audit procedures (coming soon)

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

### Running Your First Recommendation

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

**Current Status**: Phase 1 - Foundation (Week 1)

**Next Milestones**:
- Week 4: Privacy & Consent Layer complete
- Week 10: Need State Learning Model trained
- Week 18: Guardrailed Bandit deployed
- Week 30: Pilot launch (5% of users)
- Week 52: Full rollout & open source release

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
