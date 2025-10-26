# Frontend ↔ Backend ↔ EAC Agent Flow

Complete interaction flow showing how every frontend action calls the backend, which internally uses the EAC Agent.

---

## 🔄 **Complete Flow Architecture**

```
┌─────────────────┐
│  React Frontend │
│   (Port 3000)   │
└────────┬────────┘
         │
         │ HTTP Requests
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│  (Port 8000)    │
└────────┬────────┘
         │
         │ Internal Calls
         │
         ▼
┌─────────────────┐
│   EAC Agent     │
│  + ML Models    │
└─────────────────┘
```

---

## 📋 **All Frontend → Backend Interactions**

### **1. User Profile Updates**

**Frontend Action:** User changes income, SNAP eligibility, etc.

**API Call:**
```javascript
POST /api/v1/user/profile
{
  "user_id": "demo_user",
  "profile": {
    "income": 35000,
    "snap_eligible": true,
    "food_insecurity": 0.7,
    "diabetes_risk": 0.4,
    "household_size": 3
  }
}
```

**Backend:** Stores profile for EAC Agent to use in future recommendations

**EAC Agent:** Uses profile data for SDOH-aware personalization

---

### **2. Checkout (Get Recommendations)**

**Frontend Action:** User clicks "Proceed to Checkout"

**API Call:**
```javascript
POST /api/v1/checkout/decide
{
  "user_id": "demo_user",
  "cart": [
    {"product_id": "1", "quantity": 1, "price": 4.99},
    {"product_id": "2", "quantity": 1, "price": 2.49}
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

**Backend:** Calls EAC Agent's `process_checkout()` method

**EAC Agent:**
1. ✅ Loads SDOH data for census tract
2. ✅ Selects appropriate policy (SNAP/WIC, Low Glycemic, etc.)
3. ✅ Generates recommendations using ML models
4. ✅ Runs guardrails (fairness, safety, business)
5. ✅ Returns recommendations with explanations

**Response:**
```javascript
{
  "recommendations": [
    {
      "original_product_name": "Whole Milk",
      "suggested_product_name": "2% Milk",
      "savings": 1.50,
      "nutrition_improvement": 10,
      "reason": "Lower fat, SNAP eligible",
      "confidence": 0.85
    }
  ],
  "latency_ms": 2.3,
  "policy_used": "snap_wic_substitution",
  "fairness_check": "PASS"
}
```

---

### **3. Accept Recommendation**

**Frontend Action:** User clicks "Accept" on a recommendation

**API Call:**
```javascript
POST /api/v1/checkout/feedback
{
  "user_id": "demo_user",
  "transaction_id": "txn_1234567890",
  "total_recommendations": 3,
  "accepted_count": 1,
  "total_savings": 1.50,
  "nutrition_improvement": 10,
  "fairness_violation": false
}
```

**Backend:** Calls EAC Agent's learning module

**EAC Agent:**
1. ✅ Updates acceptance model with positive feedback
2. ✅ Adjusts policy weights
3. ✅ Stores for future model retraining
4. ✅ Updates user preference profile

---

### **4. Decline Recommendation**

**Frontend Action:** User clicks "Decline" on a recommendation

**API Call:**
```javascript
POST /api/v1/checkout/feedback
{
  "user_id": "demo_user",
  "transaction_id": "txn_1234567890",
  "total_recommendations": 3,
  "accepted_count": 0,
  "total_savings": 0,
  "nutrition_improvement": 0,
  "fairness_violation": false
}
```

**Backend:** Calls EAC Agent's learning module

**EAC Agent:**
1. ✅ Updates acceptance model with negative feedback
2. ✅ Learns user preferences
3. ✅ Adjusts future recommendations

---

## 🎯 **What the EAC Agent Does Internally**

### **On Checkout:**

```python
# Backend calls:
response = agent.process_checkout(event)

# Agent internally:
1. Load SDOH data (census tract, food access, etc.)
2. Infer user needs (SNAP eligible? Diabetes risk?)
3. Select policy (SNAP/WIC, Low Glycemic, Budget Optimizer)
4. Generate recommendations:
   - Use XGBoost acceptance model
   - Use product embeddings for similarity
   - Use collaborative filter for personalization
5. Run guardrails:
   - Fairness check (no discrimination)
   - Safety check (no harmful swaps)
   - Business check (inventory, margins)
6. Return recommendations with explanations
```

### **On Feedback:**

```python
# Backend calls:
agent.learning_module.update(feedback)

# Agent internally:
1. Compute reward (savings + nutrition + acceptance)
2. Update policy weights (Thompson Sampling)
3. Store for model retraining
4. Update user preference profile
5. Detect drift (is model performance degrading?)
```

---

## 📊 **Data Flow**

```
Frontend State → API Request → Backend → EAC Agent → ML Models
                                                    ↓
                                              Recommendations
                                                    ↓
                                              Guardrails
                                                    ↓
                                              API Response → Frontend Display
                                                    ↓
                                              User Action (Accept/Decline)
                                                    ↓
                                              Feedback API → Learning Module
                                                    ↓
                                              Model Updates
```

---

## 🔍 **How to Verify Everything is Connected**

### **1. Check API Logs**

Start the API and watch for requests:
```bash
uvicorn api.main:app --reload
```

You should see:
```
INFO: POST /api/v1/user/profile - 200 OK          # Profile update
INFO: POST /api/v1/checkout/decide - 200 OK       # Checkout
INFO: POST /api/v1/checkout/feedback - 200 OK     # Accept/Decline
```

### **2. Check Browser Console**

Open DevTools (F12) and watch for:
```
✓ Profile updated on backend: income 35000
✓ Feedback sent: Accepted 2% Milk (1 gal)
✓ Feedback sent: Declined Whole Wheat Bread
```

### **3. Check EAC Agent Logs**

In the API terminal, you'll see EAC Agent logs:
```
INFO - EACAgent - Checkout processed: user=demo_user, policy=snap_wic_substitution, recs=3
INFO - EACAgent.Learning - Feedback received: accepted=1, reward=15.5
```

---

## 🎮 **Try It Now**

1. **Start Backend:**
   ```bash
   source .venv/bin/activate
   uvicorn api.main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd frontend/react-app
   npm run dev
   ```

3. **Open Browser:** http://localhost:3000

4. **Test Each Interaction:**
   - ✅ Change income slider → See API log
   - ✅ Click "Proceed to Checkout" → See EAC Agent process
   - ✅ Click "Accept" → See feedback log
   - ✅ Click "Decline" → See learning update

---

## 📈 **Benefits of This Architecture**

### **1. Real Simulation**
- Frontend mimics real user behavior
- Backend processes real transactions
- EAC Agent makes real decisions
- ML models learn from real feedback

### **2. Data Collection**
- Every interaction is logged
- Can analyze user behavior
- Can measure acceptance rates
- Can track savings and nutrition impact

### **3. Model Improvement**
- Feedback trains acceptance model
- Policy weights adjust automatically
- Drift detection catches issues
- Continuous learning

### **4. Production Ready**
- Same architecture as production
- API can handle multiple frontends
- Scalable backend
- Real-time processing

---

## 🚀 **Next Steps**

### **1. Add Real Datasets**
Replace synthetic data with:
- Instacart transactions
- USDA nutrition data
- CDC SDOH data

### **2. Deploy to Production**
- Frontend → Vercel/Netlify
- Backend → Railway/Render
- Database → PostgreSQL
- Monitoring → Datadog/Sentry

### **3. Run User Studies**
- Deploy to test users
- Collect real acceptance data
- Retrain models on real data
- Measure actual impact

---

## 📊 **Current Status**

| Component | Status | Connected |
|-----------|--------|-----------|
| **Frontend** | ✅ Running | ✅ Calls API |
| **Backend API** | ✅ Running | ✅ Calls Agent |
| **EAC Agent** | ✅ Active | ✅ Uses ML Models |
| **ML Models** | ✅ Trained | ✅ Making Predictions |
| **Learning Module** | ✅ Active | ✅ Learning from Feedback |

---

**You now have a complete, end-to-end simulation where:**
- ✅ Frontend makes real API calls
- ✅ Backend runs real EAC Agent
- ✅ Agent uses real ML models
- ✅ System learns from user feedback

**This is production-ready architecture!** 🎉
