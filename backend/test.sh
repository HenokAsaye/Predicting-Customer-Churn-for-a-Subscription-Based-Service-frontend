#!/bin/bash
# test.sh - Test the complete system

echo "🧪 Testing Customer Churn Prediction System..."
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to test API endpoints
test_api() {
    echo "=== Testing API Endpoints ==="
    echo ""
    
    API_URL="http://127.0.0.1:8000"
    
    # Test health check
    echo "1. Testing health check..."
    response=$(curl -s "$API_URL/health")
    if echo "$response" | grep -q "healthy"; then
        echo -e "${GREEN}✓ Health check passed${NC}"
    else
        echo -e "${RED}✗ Health check failed${NC}"
        echo "Response: $response"
        return 1
    fi
    echo ""
    
    # Test prediction endpoint
    echo "2. Testing prediction endpoint..."
    response=$(curl -s -X POST "$API_URL/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
      }')
    
    if echo "$response" | grep -q "prediction"; then
        echo -e "${GREEN}✓ Prediction endpoint works${NC}"
        echo "Sample response: $response" | head -c 200
        echo "..."
    else
        echo -e "${RED}✗ Prediction endpoint failed${NC}"
        echo "Response: $response"
        return 1
    fi
    echo ""
    
    # Test model info endpoint
    echo "3. Testing model info endpoint..."
    response=$(curl -s "$API_URL/model/info")
    if echo "$response" | grep -q "model"; then
        echo -e "${GREEN}✓ Model info endpoint works${NC}"
    else
        echo -e "${RED}✗ Model info endpoint failed${NC}"
        echo "Response: $response"
        return 1
    fi
    echo ""
}

# Function to test frontend
test_frontend() {
    echo "=== Testing Frontend ==="
    echo ""
    
    FRONTEND_URL="http://localhost:3000"
    
    echo "1. Checking if Next.js is running..."
    if curl -s "$FRONTEND_URL" | grep -q "html"; then
        echo -e "${GREEN}✓ Frontend is accessible${NC}"
    else
        echo -e "${YELLOW}⚠ Frontend not running or not responding${NC}"
        echo "Make sure to run: cd frontend && npm run dev"
    fi
    echo ""
}

# Function to check Python environment
check_python() {
    echo "=== Checking Python Environment ==="
    echo ""
    
    cd backend
    
    if [ ! -d "venv" ]; then
        echo -e "${RED}✗ Virtual environment not found${NC}"
        return 1
    fi
    
    source venv/Scripts/activate || source venv/bin/activate
    
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version | cut -d' ' -f2)"
    
    # Check critical packages
    echo ""
    echo "Checking required packages..."
    
    packages=("pandas" "scikit-learn" "xgboost" "fastapi" "uvicorn" "streamlit" "numpy")
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            echo -e "${GREEN}✓ $package${NC}"
        else
            echo -e "${RED}✗ $package${NC}"
        fi
    done
    
    echo ""
    cd ..
}

# Function to check models
check_models() {
    echo "=== Checking Model Files ==="
    echo ""
    
    if [ -f "backend/models/best_model.pkl" ]; then
        echo -e "${GREEN}✓ Best model found${NC}"
    else
        echo -e "${RED}✗ Best model not found${NC}"
    fi
    
    if [ -f "backend/models/preprocessor.pkl" ]; then
        echo -e "${GREEN}✓ Preprocessor found${NC}"
    else
        echo -e "${RED}✗ Preprocessor not found${NC}"
    fi
    
    if [ -f "backend/models/metrics.pkl" ]; then
        echo -e "${GREEN}✓ Metrics found${NC}"
    else
        echo -e "${RED}✗ Metrics not found${NC}"
    fi
    
    echo ""
}

# Main execution
echo ""
echo "========== SYSTEM TEST REPORT =========="
echo ""

check_python
check_models

echo "Make sure the API server is running in another terminal:"
echo "  cd backend && uvicorn api.main:app --reload"
echo ""
echo "Press Enter to test API endpoints..."
read

test_api

echo "Make sure Next.js is running in another terminal:"
echo "  cd frontend && npm run dev"
echo ""
echo "Press Enter to test frontend..."
read

test_frontend

echo ""
echo "========== TEST COMPLETE =========="
echo ""
