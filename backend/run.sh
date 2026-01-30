#!/bin/bash
# run.sh - Complete startup script for development

echo "🚀 Starting Customer Churn Prediction System..."

# Check if running from correct directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

echo ""
echo "========== BACKEND SETUP =========="
echo "1. Checking Python environment..."

# Navigate to backend
cd backend

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
source venv/Scripts/activate || source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "📊 Training models..."
    python train.py
    echo "✓ Models trained"
else
    echo "✓ Models found"
fi

echo ""
echo "========== FRONTEND SETUP =========="
cd ../frontend

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

echo "✓ Node.js found: $(node -v)"

# Install npm dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
    echo "✓ Dependencies installed"
else
    echo "✓ node_modules found"
fi

echo ""
echo "========== READY TO START =========="
echo ""
echo "Open 3 terminals and run:"
echo ""
echo "Terminal 1 (API Server):"
echo "  cd backend"
echo "  . venv/Scripts/activate"
echo "  uvicorn api.main:app --reload"
echo ""
echo "Terminal 2 (Next.js Frontend):"
echo "  cd frontend"
echo "  npm run dev"
echo ""
echo "Terminal 3 (Alternative - Streamlit):"
echo "  cd backend"
echo "  . venv/Scripts/activate"
echo "  streamlit run ../app/streamlit_app.py"
echo ""
echo "Then visit:"
echo "  Next.js:  http://localhost:3000"
echo "  Streamlit: http://localhost:8501"
echo "  API:      http://localhost:8000/docs"
echo ""
