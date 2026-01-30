# 🚀 Customer Churn Prediction System - Complete Setup Guide

## Project Structure

```
.
├── backend/                          # All ML and API code
│   ├── api/
│   │   ├── main.py                  # FastAPI application
│   │   └── schemas.py               # Request/Response schemas
│   ├── src/
│   │   ├── data_loader.py           # Data loading
│   │   ├── preprocessing.py         # Data preprocessing
│   │   ├── model_training.py        # Model training
│   │   └── evaluation.py            # Model evaluation
│   ├── notebooks/
│   │   └── 01_exploratory_data_analysis.ipynb
│   ├── models/                       # Saved models (auto-generated)
│   ├── data/                         # Dataset (auto-generated)
│   ├── reports/                      # Generated reports
│   ├── train.py                      # Training script
│   ├── config.py                     # Configuration
│   ├── requirements.txt              # Python dependencies
│   └── .env                          # Environment variables
│
├── frontend/                         # Next.js React frontend
│   ├── src/
│   │   ├── components/               # React components
│   │   ├── pages/                    # Next.js pages
│   │   ├── services/                 # API services
│   │   └── styles/                   # CSS modules
│   ├── package.json
│   ├── next.config.js
│   └── README.md
│
├── app/                              # Alternative: Streamlit frontend
│   └── streamlit_app.py
│
└── README.md                         # This file
```

## Quick Start (All-in-One)

### 1. Backend Setup

#### Terminal 1: Start Training

```bash
cd backend

# Create virtual environment (first time only)
python -m venv venv
. venv/Scripts/activate  # Windows Git Bash

# Install dependencies
pip install -r requirements.txt

# Train models
python train.py
```

Expected output:
```
✓ Models trained and saved
✓ Best model: XGBoost
✓ Preprocessor saved
```

#### Terminal 2: Start FastAPI Server

```bash
cd backend
. venv/Scripts/activate

# Start API server on http://127.0.0.1:8000
uvicorn api.main:app --reload
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 2. Frontend Setup

Choose either **Option A (Recommended)** or **Option B**:

#### Option A: Next.js Frontend (Modern React)

```bash
# Terminal 3: Start Next.js
cd frontend

# Install dependencies (first time)
npm install

# Start development server on http://localhost:3000
npm run dev
```

Visit: **http://localhost:3000**

#### Option B: Streamlit Frontend (Quick & Simple)

```bash
# Terminal 3: Start Streamlit
cd app

# With same venv as backend:
. ../backend/venv/Scripts/activate

# Start app on http://localhost:8501
streamlit run streamlit_app.py
```

Visit: **http://localhost:8501**

## Production Deployment

### Option 1: Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Endpoints:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
```

### Option 2: Heroku

```bash
# Deploy backend only
git push heroku main
```

## API Endpoints

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Single Prediction
```bash
curl -X POST http://127.0.0.1:8000/predict \
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
  }'
```

### Batch Predictions
```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"gender": "Female", ...},
      {"gender": "Male", ...}
    ]
  }'
```

### Model Info
```bash
curl http://127.0.0.1:8000/model/info
```

### Feature Importance
```bash
curl http://127.0.0.1:8000/model/features
```

## Features

### Backend (Python/FastAPI)
- ✅ Machine Learning pipeline (3 models: Logistic Regression, Random Forest, XGBoost)
- ✅ Advanced preprocessing (SMOTE for class imbalance, feature engineering)
- ✅ REST API with batch processing
- ✅ Model serialization and persistence
- ✅ Comprehensive evaluation metrics
- ✅ Risk factor identification

### Frontend (Next.js)
- ✅ Real-time prediction interface
- ✅ Interactive customer input form
- ✅ Risk visualization
- ✅ API connection monitoring
- ✅ Responsive design
- ✅ Error handling

### Alternative Frontend (Streamlit)
- ✅ Multi-page interface (Predict, Performance, About)
- ✅ Live model performance dashboard
- ✅ Risk factor visualization
- ✅ Feature importance charts
- ✅ Gauge charts for probability display

## Configuration

### Backend Configuration (backend/config.py)

```python
# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CROSS_VAL_FOLDS = 5

# Feature engineering
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES = [...19 features...]

# Model parameters
LOGISTIC_REGRESSION_PARAMS = {...}
RANDOM_FOREST_PARAMS = {...}
XGBOOST_PARAMS = {...}
```

### Frontend Configuration (frontend/.env.local)

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Troubleshooting

### Issue: Port Already in Use

**Port 8000 (API)**
```bash
lsof -i :8000
kill -9 <PID>
```

**Port 3000 (Next.js)**
```bash
npm run dev -- -p 3001
```

**Port 8501 (Streamlit)**
```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### Issue: API Connection Failed

```bash
# Check if API is running
curl http://127.0.0.1:8000/health

# Restart API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Issue: Models Not Found

```bash
# Retrain models
cd backend
python train.py
```

### Issue: Dependencies Not Installing

```bash
# Clear and reinstall
pip cache purge
pip install -r requirements.txt --force-reinstall
```

## Data Pipeline

```
Raw Data → Cleaning → Feature Engineering → Preprocessing → Model Training → Evaluation → Serialization → API → Frontend
```

### Data Features (24 total)

**Input Features (19)**:
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Services: PhoneService, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Account: tenure, Contract, PaperlessBilling, PaymentMethod
- Billing: MonthlyCharges, TotalCharges

**Engineered Features (5)**:
- TenureGroup: Binned tenure into age groups
- AvgMonthlySpend: MonthlyCharges normalized
- ChargeIncrease: Change rate analysis
- HasFamily: Partner OR Dependents indicator
- SecureContract: Long-term contract indicator

**Target Feature (1)**:
- Churn: Yes/No

## Model Performance

Each model is evaluated with:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation (StratifiedKFold, k=5)
- Class imbalance handling (SMOTE)

## Development Workflow

1. **Make changes** to backend code
2. **Restart API** (auto-reloading enabled with `--reload`)
3. **Frontend updates** automatically (Next.js hot reload)
4. **Test predictions** through UI or curl

## Testing

```bash
# Run backend tests
cd backend
pytest tests/

# API test
python -m pytest tests/test_api.py -v

# Preprocessing test
python -m pytest tests/test_preprocessing.py -v
```

## Next Steps

1. ✅ Backend setup and training
2. ✅ API server running
3. ✅ Frontend connected to API
4. ✅ Make predictions
5. 🎯 Deploy to production
6. 🎯 Monitor performance
7. 🎯 Retrain models periodically

## Support

### Common Commands

```bash
# Kill all Python processes
taskkill /F /IM python.exe

# Restart everything
pkill -f uvicorn  # Stop API
pkill -f streamlit  # Stop Streamlit
npm stop  # Stop Next.js

# Rebuild everything
python train.py
uvicorn api.main:app --reload

# View model info
python -c "import joblib; print(joblib.load('models/best_model.pkl'))"
```

## Project Statistics

- **Python Files**: 10+
- **Models**: 3 (Logistic Regression, Random Forest, XGBoost)
- **Frontend Pages**: 2 (Next.js) or 3 (Streamlit)
- **API Endpoints**: 6
- **Data Features**: 24
- **Dependencies**: 40+
- **Docker Services**: 3 (API, Streamlit, Trainer)

## License

MIT

## Contact

For issues or questions, refer to the individual README files:
- Backend: [backend/README.md](backend/README.md) (if exists)
- Frontend: [frontend/README.md](frontend/README.md)
- Streamlit: [app/streamlit_app.py](app/streamlit_app.py)
