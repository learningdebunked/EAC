#!/bin/bash
# Start all EAC frontends and API

echo "🚀 Starting EAC Full Stack..."
echo ""

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "⚠️  API not running. Starting API..."
    echo "Run in a separate terminal: uvicorn api.main:app --reload"
    echo ""
fi

# Start React frontend
echo "1️⃣  Starting React Frontend (port 3000)..."
cd frontend/react-app
if [ ! -d "node_modules" ]; then
    echo "   Installing dependencies..."
    npm install
fi
npm run dev &
REACT_PID=$!

# Wait a bit
sleep 2

# Start Streamlit dashboard
echo ""
echo "2️⃣  Starting Streamlit Dashboard (port 8501)..."
cd ../..
streamlit run frontend/streamlit_dashboard.py &
STREAMLIT_PID=$!

# Open simple demo
echo ""
echo "3️⃣  Opening Simple HTML Demo..."
sleep 2
open frontend/simple-demo.html || xdg-open frontend/simple-demo.html

echo ""
echo "✅ All frontends started!"
echo ""
echo "📍 Access points:"
echo "   React App:      http://localhost:3000"
echo "   Streamlit:      http://localhost:8501"
echo "   HTML Demo:      file://$(pwd)/frontend/simple-demo.html"
echo "   API:            http://localhost:8000"
echo "   API Docs:       http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "kill $REACT_PID $STREAMLIT_PID; exit" INT
wait
