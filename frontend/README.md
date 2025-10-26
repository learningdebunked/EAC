# EAC Frontends

Three complete frontend interfaces for the EAC Agent system.

## 🎯 Quick Start

### **Option 1: React App** (Recommended for production)
```bash
cd react-app
npm install
npm run dev
```
Open http://localhost:3000

### **Option 2: Streamlit Dashboard** (Analytics)
```bash
pip install streamlit plotly
streamlit run streamlit_dashboard.py
```
Open http://localhost:8501

### **Option 3: Simple HTML Demo** (No build tools)
```bash
open simple-demo.html
```

## 📁 Structure

```
frontend/
├── react-app/              # React + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.jsx         # Main app
│   │   ├── components/     # React components
│   │   └── index.css       # Styles
│   ├── package.json
│   └── README.md
├── streamlit_dashboard.py  # Analytics dashboard
├── simple-demo.html        # Single-file demo
└── README.md               # This file
```

## 🚀 Features

### React App
- ✅ Interactive shopping cart
- ✅ Real-time recommendations
- ✅ User profile editor
- ✅ Impact visualization
- ✅ Production-ready

### Streamlit Dashboard
- ✅ Performance metrics
- ✅ Fairness analysis
- ✅ Interactive charts
- ✅ Data export
- ✅ Real-time monitoring

### HTML Demo
- ✅ No dependencies
- ✅ Works offline
- ✅ Single file
- ✅ Easy to customize
- ✅ Perfect for demos

## 📖 Documentation

See [FRONTEND_GUIDE.md](../FRONTEND_GUIDE.md) for complete documentation.

## 🔌 API Connection

All frontends connect to the FastAPI backend:

```bash
# Start API first
uvicorn api.main:app --reload
```

API runs at http://localhost:8000

## 🎨 Screenshots

### React App
Beautiful, modern UI with interactive components.

### Streamlit Dashboard
Real-time analytics and monitoring.

### HTML Demo
Simple, clean interface for quick demos.

## 🚀 Start Everything

```bash
# From project root
./scripts/start_all_frontends.sh
```

This starts:
- React app (port 3000)
- Streamlit dashboard (port 8501)
- Opens HTML demo
- Requires API running on port 8000

## 📱 Mobile Support

All frontends are mobile-responsive and work on:
- 📱 iOS
- 🤖 Android
- 💻 Desktop
- 📟 Tablet

## 🐛 Troubleshooting

**Port already in use:**
```bash
lsof -ti:3000 | xargs kill -9  # React
lsof -ti:8501 | xargs kill -9  # Streamlit
```

**Dependencies missing:**
```bash
cd react-app && npm install     # React
pip install streamlit plotly    # Streamlit
```

**API not responding:**
```bash
# Check API health
curl http://localhost:8000/health

# Start API
uvicorn api.main:app --reload
```

## 🎯 Use Cases

| Frontend | Best For |
|----------|----------|
| **React** | Production, user testing, pilots |
| **Streamlit** | Analytics, monitoring, research |
| **HTML** | Demos, presentations, POCs |

## 🤝 Contributing

Improvements welcome! See [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📄 License

MIT
