# Real-Time Analytics Integration

Complete guide to the real-time data flow from frontend → backend → analytics dashboard.

---

## 🔄 **Complete Data Flow**

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION                         │
│              (React Frontend - Port 3000)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 1. User clicks "Checkout"
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  API ENDPOINT                               │
│         POST /api/v1/checkout/decide                        │
│              (FastAPI - Port 8000)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 2. Process with EAC Agent
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    EAC AGENT                                │
│         • Load SDOH data                                    │
│         • Select policy                                     │
│         • Generate recommendations (ML models)              │
│         • Run guardrails                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 3. Store transaction
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSACTION STORE                              │
│           (live_transactions.csv)                           │
│   • transaction_id                                          │
│   • policy_used                                             │
│   • recommendations                                         │
│   • latency_ms                                              │
│   • timestamp                                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 4. Return recommendations
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FRONTEND DISPLAY                               │
│         Show recommendations to user                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 5. User accepts/declines
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FEEDBACK ENDPOINT                              │
│        POST /api/v1/checkout/feedback                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 6. Update transaction
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         UPDATE TRANSACTION STORE                            │
│   • accepted_count                                          │
│   • total_savings                                           │
│   • acceptance_rate                                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ 7. Auto-refresh (every 5 sec)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           ANALYTICS DASHBOARD                               │
│        (Streamlit - Port 8501)                              │
│   • Real-time metrics                                       │
│   • Live charts                                             │
│   • Fairness analysis                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 **Data Schema**

### **live_transactions.csv**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | When transaction occurred |
| `user_id` | string | User identifier |
| `transaction_id` | string | Unique transaction ID |
| `policy_used` | string | Which policy was selected |
| `num_recommendations` | int | Total recommendations shown |
| `accepted_count` | int | How many user accepted |
| `declined_count` | int | How many user declined |
| `total_savings` | float | Total potential savings |
| `total_nutrition_improvement` | float | Total HEI improvement |
| `acceptance_rate` | float | accepted_count / num_recommendations |
| `latency_ms` | float | Processing time |
| `protected_group` | string | Demographic group |
| `income_group` | string | Income bracket |
| `snap_eligible` | boolean | SNAP eligibility |
| `fairness_check` | string | PASS/REVIEW/FAIL |

---

## 🚀 **How to Run Everything**

### **Terminal 1: Start Backend**
```bash
cd /Users/kapilsindhu/Documents/OpenSourcProjects/EAC
source .venv/bin/activate
uvicorn api.main:app --reload
```

### **Terminal 2: Start Frontend**
```bash
cd frontend/react-app
npm run dev
```

### **Terminal 3: Start Analytics Dashboard**
```bash
streamlit run frontend/streamlit_dashboard.py
```

### **Access Points:**
- **Frontend**: http://localhost:3000
- **Analytics**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🎯 **Test the Complete Flow**

1. **Open Frontend** (http://localhost:3000)
2. **Open Analytics Dashboard** (http://localhost:8501) in another tab
3. **In Frontend:**
   - Adjust user profile
   - Click "Proceed to Checkout"
   - Accept or decline recommendations
4. **In Analytics Dashboard:**
   - Click "🔄 Refresh Data" button
   - See your transaction appear!
   - Watch metrics update in real-time

---

## 📈 **What Gets Tracked**

### **On Checkout:**
- ✅ Transaction created with unique ID
- ✅ Policy selection recorded
- ✅ Number of recommendations
- ✅ Potential savings calculated
- ✅ Latency measured
- ✅ User demographics captured
- ✅ Fairness check result

### **On Accept/Decline:**
- ✅ Acceptance count updated
- ✅ Actual savings recorded
- ✅ Nutrition improvement tracked
- ✅ Acceptance rate recalculated
- ✅ Learning module updated

### **In Dashboard:**
- ✅ Auto-refreshes every 5 seconds
- ✅ Shows all transactions
- ✅ Aggregates metrics
- ✅ Displays charts
- ✅ Fairness analysis
- ✅ Export to CSV

---

## 🔍 **Verify It's Working**

### **1. Check Transaction Store**
```bash
# After using the frontend, check the file
cat live_transactions.csv
```

You should see your transactions!

### **2. Check API Logs**
In the API terminal, you'll see:
```
INFO: POST /api/v1/checkout/decide - 200 OK
INFO: Added transaction: txn_1234567890
INFO: POST /api/v1/checkout/feedback - 200 OK
INFO: Updated transaction: txn_1234567890
```

### **3. Check Dashboard**
- Open http://localhost:8501
- Click "🔄 Refresh Data"
- See your transaction in the table
- Watch metrics update

---

## 📊 **Dashboard Features**

### **Real-Time Metrics:**
- Acceptance Rate (updates as users accept/decline)
- Average Savings (from actual transactions)
- Nutrition Improvement (real data)
- System Latency (measured)

### **Charts:**
- **Acceptance by Policy**: Which policies work best
- **Savings Distribution**: How much users save
- **Nutrition Impact**: Health improvements
- **Latency Trends**: System performance
- **Fairness Analysis**: Equity across groups

### **Data Table:**
- Recent transactions
- Sortable and filterable
- Export to CSV
- Shows all details

---

## 🎮 **Advanced Usage**

### **1. Multiple Users**
Open multiple browser tabs and simulate different users:
- Each gets unique transaction_id
- All tracked separately
- Aggregated in dashboard

### **2. A/B Testing**
Compare different policies:
- Run transactions with different profiles
- Dashboard shows policy comparison
- Identify best-performing policies

### **3. Fairness Monitoring**
Track equity across demographics:
- Dashboard shows disparity by group
- Alerts if max disparity > $3
- Real-time fairness checks

### **4. Performance Monitoring**
Track system health:
- Latency trends over time
- P99 latency
- Alert if > 5ms SLA

---

## 🔧 **Customization**

### **Change Refresh Rate**
In `streamlit_dashboard.py` line 57:
```python
@st.cache_data(ttl=5)  # Change to 10 for 10 seconds
```

### **Add Custom Metrics**
In `data_store.py`, add columns:
```python
'custom_metric': value
```

### **Filter by Time**
In dashboard sidebar, select time range:
- Last Hour
- Last 24 Hours
- Last 7 Days
- Last 30 Days

---

## 📦 **Production Deployment**

### **Replace CSV with Database**

Update `data_store.py` to use PostgreSQL:
```python
import psycopg2

class TransactionStore:
    def __init__(self, db_url):
        self.conn = psycopg2.connect(db_url)
    
    def add_transaction(self, data):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO transactions (...) VALUES (...)",
            data
        )
        self.conn.commit()
```

### **Add Redis for Real-Time**
```python
import redis

r = redis.Redis()
r.publish('transactions', json.dumps(transaction_data))
```

### **Stream to Dashboard**
```python
# In Streamlit
import streamlit as st

for message in r.subscribe('transactions'):
    st.rerun()
```

---

## 🎯 **Benefits**

### **1. Real-Time Insights**
- See impact immediately
- No batch processing delay
- Live system monitoring

### **2. Data-Driven Decisions**
- Which policies work best
- User acceptance patterns
- Fairness issues

### **3. Continuous Learning**
- ML models learn from real feedback
- Policy weights adjust automatically
- System improves over time

### **4. Stakeholder Visibility**
- Live dashboard for demos
- Real metrics, not synthetic
- Transparent system behavior

---

## 🐛 **Troubleshooting**

### **Dashboard shows no data:**
```bash
# Check if file exists
ls -la live_transactions.csv

# Check if API is storing data
tail -f live_transactions.csv
```

### **Transactions not updating:**
```bash
# Check API logs
# Look for "Added transaction" and "Updated transaction"
```

### **Dashboard not refreshing:**
- Click "🔄 Refresh Data" button manually
- Check ttl parameter in @st.cache_data
- Clear Streamlit cache: Press 'C' in dashboard

---

## 📊 **Example Session**

```
1. Start all services
2. Open frontend → Checkout → Accept 2/3 recommendations
3. Check live_transactions.csv:
   - 1 new row
   - acceptance_rate = 0.67
   - total_savings = $3.50
4. Open dashboard → Refresh
   - See transaction in table
   - Acceptance rate: 67%
   - Savings: $3.50
5. Do 10 more transactions
6. Dashboard shows:
   - Avg acceptance: 55%
   - Avg savings: $2.80
   - Charts update
   - Fairness check: PASS
```

---

## 🎉 **You Now Have:**

✅ **Complete real-time pipeline**  
✅ **Frontend → Backend → Analytics**  
✅ **Live data tracking**  
✅ **Auto-refreshing dashboard**  
✅ **Production-ready architecture**  

**This is a complete, working simulation with real-time analytics!** 🚀
