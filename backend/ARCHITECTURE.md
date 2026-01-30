# System Architecture Diagram

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER BROWSER                                   │
│                       (http://localhost:3000)                            │
└─────────────────────────┬──────────────────────────────────────────────┘
                          │
                          │ HTTP/HTTPS
                          │ REST API
                          │ JSON
                          │
┌─────────────────────────▼──────────────────────────────────────────────┐
│                    NEXT.JS FRONTEND                                     │
│            (Port 3000 - React 19 + TypeScript)                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         Dashboard                                │  │
│  │  • Input Tab (Customer Form)                                    │  │
│  │  • Prediction Tab (Results)                                     │  │
│  │  • Batch Tab (CSV Processing)                                   │  │
│  │  • Metrics Tab (Model Performance)                              │  │
│  │  • Explanation Tab (Risk Factors)                               │  │
│  └──────────────────┬───────────────────────────────────────────┬──┘  │
│                     │                                           │       │
│  ┌──────────────────▼───────────────────────────────────────────▼──┐  │
│  │              API Service Layer (api.ts)                          │  │
│  │                                                                   │  │
│  │  • checkHealth()          → GET /health                         │  │
│  │  • getModelInfo()         → GET /model/info                     │  │
│  │  • getFeatureImportance() → GET /model/feature-importance       │  │
│  │  • predict()              → POST /predict                       │  │
│  │  • predictBatch()         → POST /predict/batch                 │  │
│  │                                                                   │  │
│  │  Features:                                                      │  │
│  │  • Async/await operations                                       │  │
│  │  • Full TypeScript typing                                       │  │
│  │  • Error handling                                               │  │
│  │  • Response mapping                                             │  │
│  └──────────────────┬──────────────────────────────────────────────┘  │
│                     │                                                   │
│  ┌──────────────────▼──────────────────────────────────────────────┐  │
│  │                     Components                                  │  │
│  │                                                                  │  │
│  │  • CustomerForm      - Telco service data input                │  │
│  │  • BatchPrediction   - CSV processing                          │  │
│  │  • PredictionResult  - Display single prediction               │  │
│  │  • FeatureImportance - Show importance chart                   │  │
│  │  • ModelMetrics      - Display performance metrics             │  │
│  │  • Shadcn/UI         - Radix UI + Tailwind CSS                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Libraries:                                                         │
│  • React 19, Next.js 16                                            │
│  • TypeScript 5                                                     │
│  • Recharts (charting)                                             │
│  • Shadcn/ui (components)                                          │
│  • Tailwind CSS 4 (styling)                                        │
└─────────────────────────┬──────────────────────────────────────────┘
                          │
                          │ HTTP/HTTPS REST API
                          │ POST /predict
                          │ POST /predict/batch
                          │ GET /health, /model/info, etc.
                          │
┌─────────────────────────▼──────────────────────────────────────────────┐
│                    FASTAPI BACKEND                                      │
│            (Port 8000 - Python 3.8+)                                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    API Routes                                    │  │
│  │                                                                   │  │
│  │  GET /              - Welcome                                    │  │
│  │  GET /health        - Health check                              │  │
│  │  POST /predict      - Single prediction                         │  │
│  │  POST /predict/batch - Batch predictions                        │  │
│  │  GET /model/info    - Model metadata                            │  │
│  │  GET /model/feature-importance - Feature importance            │  │
│  │                                                                   │  │
│  │  GET /docs          - Swagger UI                                │  │
│  │  GET /redoc         - ReDoc                                     │  │
│  └──────────────────┬──────────────────────────────────────────────┘  │
│                     │                                                   │
│  ┌──────────────────▼──────────────────────────────────────────────┐  │
│  │          Request Processing Pipeline                            │  │
│  │                                                                   │  │
│  │  1. Receive JSON request                                        │  │
│  │  2. Validate with Pydantic schema                               │  │
│  │  3. Convert SeniorCitizen field                                 │  │
│  │  4. Clean data (missing values, types)                          │  │
│  │  5. Engineer features (derive new features)                     │  │
│  │  6. Preprocess (scaling, encoding)                              │  │
│  │  7. Run ML model                                                │  │
│  │  8. Identify risk factors                                       │  │
│  │  9. Calculate confidence                                        │  │
│  │  10. Return JSON response                                       │  │
│  └──────────────────┬──────────────────────────────────────────────┘  │
│                     │                                                   │
│  ┌──────────────────▼──────────────────────────────────────────────┐  │
│  │                  ML Pipeline                                    │  │
│  │                                                                   │  │
│  │  Input: Customer data (19 features)                             │  │
│  │     ↓                                                            │  │
│  │  Preprocessing:                                                 │  │
│  │  • Data cleaning                                                │  │
│  │  • Feature engineering                                          │  │
│  │  • Scaling/Normalization                                        │  │
│  │     ↓                                                            │  │
│  │  ML Model: [Best trained model from train.py]                   │  │
│  │  • Loaded from: models/best_model.joblib                        │  │
│  │  • Type: RandomForest, XGBoost, etc.                            │  │
│  │  • Training data: Telco customer churn dataset                  │  │
│  │     ↓                                                            │  │
│  │  Prediction:                                                    │  │
│  │  • Churn probability (0-1)                                      │  │
│  │  • Binary prediction (Yes/No)                                   │  │
│  │  • Confidence (High/Medium/Low)                                 │  │
│  │     ↓                                                            │  │
│  │  Risk Analysis:                                                 │  │
│  │  • Identify risk factors based on data                          │  │
│  │  • Generate factors: tenure, contract, payment, etc.            │  │
│  │     ↓                                                            │  │
│  │  Output: Structured prediction response                         │  │
│  └──────────────────┬──────────────────────────────────────────────┘  │
│                     │                                                   │
│  ┌──────────────────▼──────────────────────────────────────────────┐  │
│  │              Models & Data Files                                │  │
│  │                                                                   │  │
│  │  • models/best_model.joblib      - Trained ML model             │  │
│  │  • models/preprocessor.joblib    - Feature preprocessor         │  │
│  │  • models/model_metrics.json     - Model performance            │  │
│  │  • models/feature_importance.json - Feature importance          │  │
│  │  • data/raw/telco_*.csv          - Training dataset             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Technology Stack:                                                      │
│  • FastAPI (async web framework)                                       │
│  • Pydantic (data validation)                                          │
│  • scikit-learn/XGBoost (ML models)                                    │
│  • pandas/numpy (data processing)                                      │
│  • joblib (model serialization)                                        │
│  • CORS enabled                                                         │
│  • Logging with loguru                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow Sequence Diagram

### Single Prediction Flow

```
Frontend                      API Service              Backend
   │                              │                        │
   │─1. User enters data──────────►│                        │
   │                              │                        │
   │                        2. Validate data              │
   │                              │                        │
   │                         3. API Call                  │
   │                              │─POST /predict─────────►│
   │                              │                        │
   │                              │              4. Receive Request
   │                              │                    │
   │                              │              5. Validate (Pydantic)
   │                              │                    │
   │                              │              6. Preprocess Data
   │                              │                    │
   │                              │              7. Run ML Model
   │                              │                    │
   │                              │              8. Analyze Risk
   │                              │                    │
   │                         9. Response              │
   │                              │◄─JSON response────┤
   │                              │                    │
   │◄──10. Handle response────────┤
   │                              │
   │──11. Display results────────►│
   │                              │
```

### Component Hierarchy

```
App (page.tsx)
├── Header
│   ├── Logo/Title
│   ├── Connection Status
│   ├── Reset Button
│   └── Predict Button
│
├── Tabs
│   ├── Input Tab
│   │   └── CustomerForm
│   │       ├── Demographics Section
│   │       ├── Account Info Section
│   │       ├── Phone & Internet Section
│   │       └── Internet Add-ons Section
│   │
│   ├── Prediction Tab
│   │   ├── PredictionResult
│   │   │   ├── Churn indicator (Yes/No)
│   │   │   ├── Probability bar
│   │   │   ├── Confidence level
│   │   │   └── Risk factors list
│   │   │
│   │   └── FeatureImportance
│   │       └── Bar chart
│   │
│   ├── Batch Tab
│   │   └── BatchPrediction
│   │       ├── CSV Format Guide
│   │       ├── CSV Input Textarea
│   │       ├── Process Button
│   │       └── Results Table
│   │
│   ├── Metrics Tab
│   │   ├── ModelMetrics
│   │   │   ├── Accuracy
│   │   │   ├── Precision
│   │   │   ├── Recall
│   │   │   └── F1 Score
│   │   │
│   │   └── About Model Card
│   │       └── Explanations
│   │
│   └── Explanation Tab
│       ├── Risk Factors Card
│       │   └── Factor list
│       │
│       └── Recommendations Card
│           └── Retention strategies
│
└── Footer
    └── Copyright & branding
```

## State Management

```
App State (React Hooks)
├── customerData (Partial<CustomerData>)
│   │ Contains all form inputs
│   └── Updated by CustomerForm.onChange
│
├── prediction (PredictionResponse | null)
│   │ Contains prediction result from API
│   └── Set by runPrediction()
│
├── isLoading (boolean)
│   │ Shows/hides loading indicators
│   └── Set during async API calls
│
├── activeTab (string)
│   │ Tracks which tab is visible
│   └── Updated by setActiveTab()
│
├── modelMetrics (ModelMetrics | null)
│   │ Model performance data
│   └── Loaded on component mount
│
├── featureImportance (FeatureImportanceItem[])
│   │ Feature importance data
│   └── Loaded on component mount
│
├── apiError (string | null)
│   │ Error messages to display
│   └── Set when API fails
│
├── apiHealthy (boolean)
│   │ API connection status
│   └── Set by health check
│
└── [More state in child components]
```

## API Response Examples

### Prediction Response
```json
{
  "prediction": "No",
  "probability": 0.3245,
  "confidence": "High",
  "risk_factors": [
    "Month-to-month contract (highest churn risk)",
    "Short tenure (12 months < 24)"
  ]
}
```

### Model Info Response
```json
{
  "model_name": "RandomForest",
  "model_version": "1.0",
  "training_date": "2024-01-28",
  "total_samples_trained": 7043,
  "metrics": {
    "accuracy": 0.8011,
    "precision": 0.6456,
    "recall": 0.5231,
    "f1_score": 0.5759,
    "auc_score": 0.8432
  }
}
```

### Feature Importance Response
```json
{
  "features": [
    {
      "feature": "tenure",
      "importance": 0.2847
    },
    {
      "feature": "Contract",
      "importance": 0.2134
    },
    {
      "feature": "MonthlyCharges",
      "importance": 0.1856
    }
  ]
}
```

## Technology Stack Overview

### Frontend
```
┌─────────────────────────────────┐
│      React 19 / Next.js 16      │
├─────────────────────────────────┤
│  • TypeScript 5 (Type safety)   │
│  • Tailwind CSS 4 (Styling)     │
│  • Shadcn/ui (Components)       │
│  • Recharts (Charts)            │
│  • React Hooks (State)          │
│  • Async/await (API calls)      │
└─────────────────────────────────┘
```

### Backend
```
┌──────────────────────────────────┐
│     FastAPI (Async Python)       │
├──────────────────────────────────┤
│  • Pydantic (Validation)         │
│  • scikit-learn (ML model)       │
│  • pandas (Data processing)      │
│  • numpy (Numerical computing)   │
│  • joblib (Serialization)        │
│  • CORS middleware (Cross-origin)│
│  • loguru (Logging)              │
└──────────────────────────────────┘
```

## Deployment Architecture (Production)

```
Internet
   │
   ▼
┌─────────────────────┐
│   Load Balancer     │     (Optional)
│  (AWS ELB/ALB)      │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ Web   │ │ Web   │
│ Front-│ │Front- │  (Vercel, Netlify, AWS S3+CloudFront)
│ end   │ │ end   │
└───────┘ └───────┘
    │         │
    └────┬────┘
         │
    ┌────────────────────────┐
    │  API Gateway           │
    │  (AWS API Gateway)     │
    └───┬────────────────────┘
        │
    ┌───┴─────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌────────────┐              ┌────────────────┐
│ FastAPI    │◄─────────────►│  Database      │
│ Backend #1 │              │  (Optional)    │
└────────────┘              └────────────────┘
    │
    ├─ Models cache
    ├─ Preprocessor cache
    └─ Feature importance cache
```

---

## Summary

The system is composed of:
1. **Frontend**: Modern React/Next.js application in browser
2. **API Layer**: TypeScript service for backend communication
3. **Backend**: FastAPI serving ML predictions
4. **ML Pipeline**: Trained model with preprocessing

All components communicate via REST API with JSON payloads, providing a complete customer churn prediction system.
