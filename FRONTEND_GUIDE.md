# Frontend Guide - EAC Agent

Complete guide to all three frontend interfaces for the EAC Agent system.

---

## 🎨 **Three Frontend Options**

### **1. React App** - Production-ready, full-featured
### **2. Streamlit Dashboard** - Analytics and monitoring
### **3. Simple HTML Demo** - No build tools, instant demo

---

## 🚀 **Option 1: React Frontend**

### **Features**
- ✅ Interactive shopping cart
- ✅ Real-time recommendations
- ✅ User profile customization
- ✅ Impact visualization
- ✅ Modern, responsive UI
- ✅ Production-ready

### **Quick Start**

```bash
# Navigate to React app
cd frontend/react-app

# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:3000

### **Production Build**

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

### **Deployment**

**Netlify:**
```bash
npm run build
netlify deploy --prod --dir=dist
```

**Vercel:**
```bash
vercel deploy --prod
```

**Docker:**
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "run", "preview"]
```

### **Environment Variables**

Create `.env` file:
```
VITE_API_URL=http://localhost:8000
```

### **Customization**

**Change colors** (`tailwind.config.js`):
```javascript
theme: {
  extend: {
    colors: {
      primary: '#10b981',    // Green
      secondary: '#3b82f6',  // Blue
    }
  }
}
```

**Add products** (`src/App.jsx`):
```javascript
const [cart, setCart] = useState([
  { id: 1, name: 'Product', price: 4.99, category: 'Category', hei: 45 },
  // Add more...
])
```

---

## 📊 **Option 2: Streamlit Dashboard**

### **Features**
- ✅ Real-time analytics
- ✅ Performance metrics
- ✅ Fairness analysis
- ✅ Interactive charts
- ✅ Data export
- ✅ Zero configuration

### **Quick Start**

```bash
# Install Streamlit
pip install streamlit plotly

# Run dashboard
streamlit run frontend/streamlit_dashboard.py
```

Open http://localhost:8501

### **Features Overview**

**Key Metrics:**
- Acceptance rate
- Average savings
- Nutrition improvement
- System latency

**Charts:**
- Acceptance by policy
- Savings distribution
- Nutrition impact
- Latency over time
- Fairness by demographic

**Data Table:**
- Recent transactions
- Filterable and sortable
- Export to CSV

### **Customization**

**Change data source** (line 52):
```python
@st.cache_data
def load_simulation_data():
    # Load from your database
    df = pd.read_sql("SELECT * FROM transactions", conn)
    return df
```

**Add custom metrics**:
```python
col5 = st.columns(1)
with col5:
    custom_metric = df['your_column'].mean()
    st.metric("Custom Metric", f"{custom_metric:.2f}")
```

### **Deployment**

**Streamlit Cloud:**
```bash
# Push to GitHub
git push origin main

# Deploy at streamlit.io/cloud
# Connect your repo
```

**Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_dashboard.py"]
```

---

## 🌐 **Option 3: Simple HTML Demo**

### **Features**
- ✅ No build tools needed
- ✅ Single file
- ✅ Works offline
- ✅ Instant demo
- ✅ Easy to customize
- ✅ Perfect for presentations

### **Quick Start**

```bash
# Just open in browser
open frontend/simple-demo.html

# Or with a simple server
python -m http.server 8080
# Open http://localhost:8080/frontend/simple-demo.html
```

### **Customization**

**Change products** (line 125):
```javascript
const cart = [
    { id: 1, name: 'Your Product', price: 4.99, category: 'Category' },
    // Add more...
];
```

**Change recommendations** (line 133):
```javascript
const recommendations = [
    {
        id: 'r1',
        original: 'Original Product',
        suggested: 'Suggested Product',
        savings: 1.50,
        nutrition: 10,
        reason: 'Your reason here'
    },
    // Add more...
];
```

**Change colors** (CSS section):
```css
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.btn-primary {
    background: #10b981;  /* Your color */
}
```

### **Deployment**

**GitHub Pages:**
```bash
# Push to GitHub
git add frontend/simple-demo.html
git commit -m "Add demo"
git push origin main

# Enable GitHub Pages in repo settings
# Set source to main branch
```

**Netlify Drop:**
- Go to https://app.netlify.com/drop
- Drag and drop `simple-demo.html`
- Get instant URL

---

## 🔌 **API Integration**

All frontends connect to the FastAPI backend.

### **Start the API**

```bash
# From project root
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at http://localhost:8000

### **API Endpoints**

**Checkout:**
```
POST /api/v1/checkout/decide
```

**Request:**
```json
{
  "cart": [
    {"id": 1, "name": "Milk", "price": 4.99}
  ],
  "user": {
    "income": 35000,
    "snap_eligible": true
  },
  "policy": "snap_wic_substitution"
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "original_product": "Whole Milk",
      "suggested_product": "2% Milk",
      "savings": 1.50,
      "nutrition_improvement": 10,
      "reason": "Lower fat, SNAP eligible"
    }
  ]
}
```

### **CORS Configuration**

Already configured in `api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

For production, restrict origins:
```python
allow_origins=[
    "https://your-frontend.com",
    "https://your-dashboard.com"
]
```

---

## 🎯 **Use Cases**

### **React App**
- ✅ Production deployment
- ✅ User-facing application
- ✅ A/B testing
- ✅ Real user research
- ✅ Pilot programs

### **Streamlit Dashboard**
- ✅ Internal analytics
- ✅ Performance monitoring
- ✅ Stakeholder demos
- ✅ Research analysis
- ✅ Model evaluation

### **Simple HTML Demo**
- ✅ Quick demos
- ✅ Presentations
- ✅ Proof of concept
- ✅ Offline demos
- ✅ Email attachments

---

## 🚀 **Full Stack Deployment**

### **Option A: Single Server**

```bash
# Terminal 1: API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: React
cd frontend/react-app && npm run dev

# Terminal 3: Streamlit
streamlit run frontend/streamlit_dashboard.py
```

### **Option B: Docker Compose**

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn api.main:app --host 0.0.0.0
  
  frontend:
    build: ./frontend/react-app
    ports:
      - "3000:3000"
    depends_on:
      - api
  
  dashboard:
    build: ./frontend
    ports:
      - "8501:8501"
    command: streamlit run streamlit_dashboard.py
    depends_on:
      - api
```

Run:
```bash
docker-compose up
```

### **Option C: Cloud Deployment**

**API** → Railway/Render/Fly.io  
**React** → Vercel/Netlify  
**Streamlit** → Streamlit Cloud  

---

## 📱 **Mobile Responsive**

All frontends are mobile-responsive:

- **React**: TailwindCSS responsive utilities
- **Streamlit**: Built-in responsive layout
- **HTML**: CSS media queries

Test on mobile:
```bash
# Get your local IP
ipconfig getifaddr en0  # Mac
# or
hostname -I  # Linux

# Access from phone
http://YOUR_IP:3000  # React
http://YOUR_IP:8501  # Streamlit
```

---

## 🎨 **Screenshots**

### React App
![React Cart](docs/screenshots/react-cart.png)
![React Recommendations](docs/screenshots/react-recs.png)
![React Impact](docs/screenshots/react-impact.png)

### Streamlit Dashboard
![Dashboard Overview](docs/screenshots/streamlit-overview.png)
![Dashboard Charts](docs/screenshots/streamlit-charts.png)

### HTML Demo
![HTML Demo](docs/screenshots/html-demo.png)

---

## 🐛 **Troubleshooting**

### **React: Port already in use**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### **Streamlit: Module not found**
```bash
pip install streamlit plotly pandas numpy
```

### **API: Connection refused**
```bash
# Check API is running
curl http://localhost:8000/health

# Start API if not running
uvicorn api.main:app --reload
```

### **CORS errors**
Check API logs and ensure CORS middleware is configured.

---

## 📚 **Additional Resources**

- **React Docs**: https://react.dev
- **Streamlit Docs**: https://docs.streamlit.io
- **TailwindCSS**: https://tailwindcss.com
- **FastAPI**: https://fastapi.tiangolo.com

---

## 🤝 **Contributing**

Want to improve the frontends?

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 **License**

MIT License - See LICENSE file

---

**Ready to launch?** Pick your frontend and start building! 🚀
