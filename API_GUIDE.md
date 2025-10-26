# EAC API Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Option 1: Using uvicorn directly
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Using Python
python -m api.main
```

The API will be available at: `http://localhost:8000`

### 3. View API Documentation

Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
GET /health
```

Returns API health status and agent statistics.

**Example:**
```bash
curl http://localhost:8000/health
```

### Checkout Decision

```bash
POST /api/v1/checkout/decide
```

Get personalized recommendations for a checkout.

**Request Body:**
```json
{
  "user_id": "user_12345",
  "cart": [
    {"product_id": "prod_001", "quantity": 1, "price": 4.99},
    {"product_id": "prod_002", "quantity": 2, "price": 3.49}
  ],
  "delivery_address": {
    "zip_code": "94102",
    "census_tract": "06075017902"
  },
  "payment_methods": ["SNAP_EBT", "CREDIT_CARD"],
  "consent": {
    "personalization": true,
    "sdoh_signals": true
  }
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "original_product_id": "prod_001",
      "suggested_product_id": "prod_001_alt",
      "original_product_name": "Chips",
      "suggested_product_name": "Whole Grain Crackers",
      "reason": "SNAP-eligible alternative saves $0.50",
      "savings": 0.50,
      "nutrition_improvement": 5.0,
      "confidence": 0.9,
      "metadata": {}
    }
  ],
  "latency_ms": 45.2,
  "policy_used": "snap_wic_substitution",
  "confidence": 0.85,
  "fairness_check": "passed",
  "explanation": "Based on your payment method, we found SNAP-eligible alternatives. 1 suggestion(s) available.",
  "metadata": {}
}
```

### Submit Feedback

```bash
POST /api/v1/checkout/feedback
```

Submit user feedback on recommendations.

**Request Body:**
```json
{
  "user_id": "user_12345",
  "transaction_id": "txn_67890",
  "total_recommendations": 3,
  "accepted_count": 2,
  "total_savings": 1.50,
  "nutrition_improvement": 8.0,
  "fairness_violation": false
}
```

### Get Statistics

```bash
GET /api/v1/stats
```

Get agent performance statistics.

## Using the Python Client

```python
from examples.api_client import main

# Run the example client
main()
```

Or use requests directly:

```python
import requests

# Make a checkout request
response = requests.post(
    "http://localhost:8000/api/v1/checkout/decide",
    json={
        "user_id": "test_user",
        "cart": [{"product_id": "prod_001", "quantity": 1, "price": 5.99}],
        "delivery_address": {"zip_code": "94102"},
        "payment_methods": ["CREDIT_CARD"],
        "consent": {"personalization": True, "sdoh_signals": True}
    }
)

recommendations = response.json()
print(recommendations)
```

## Configuration

The API uses the default `EACConfig` settings. To customize:

1. Create a custom config file:

```python
# config/custom_config.py
from eac.config import EACConfig

config = EACConfig(
    max_latency_ms=100,
    equalized_uplift_threshold=0.05,
    # ... other settings
)
```

2. Modify `api/main.py` to use your config:

```python
config = EACConfig()  # Replace with your custom config
agent = EACAgent(config)
```

## Production Deployment

### Using Docker

```bash
# Build image
docker build -t eac-api .

# Run container
docker run -p 8000:8000 eac-api
```

### Using Gunicorn

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Environment Variables

```bash
export EAC_LOG_LEVEL=INFO
export EAC_MAX_LATENCY_MS=100
export EAC_ENABLE_PROMETHEUS=true
```

## Monitoring

The API exposes metrics for monitoring:

- **Latency**: p50, p95, p99
- **Throughput**: Requests per second
- **Errors**: Error rate
- **Policy Usage**: Distribution of policies used
- **Fairness**: Acceptance rates by group

Access metrics at: `http://localhost:9090/metrics` (if Prometheus enabled)

## Testing

Run API tests:

```bash
pytest tests/test_api.py -v
```

## Security

For production deployment:

1. **Enable HTTPS**: Use reverse proxy (nginx, Caddy)
2. **Add Authentication**: Implement API key or OAuth2
3. **Rate Limiting**: Prevent abuse
4. **CORS**: Configure allowed origins
5. **Input Validation**: Already handled by Pydantic

## Troubleshooting

### API won't start

- Check port 8000 is not in use: `lsof -i :8000`
- Verify dependencies: `pip install -r requirements.txt`
- Check logs for errors

### Slow responses

- Check latency metrics: `GET /api/v1/stats`
- Verify p99 latency < 100ms
- Consider scaling horizontally

### No recommendations returned

- Verify user consent is granted
- Check agent logs for guardrail violations
- Ensure cart has valid items

## Support

- **Documentation**: See README.md and ARCHITECTURE.md
- **Issues**: https://github.com/learningdebunked/EAC/issues
- **Examples**: See `examples/` directory
