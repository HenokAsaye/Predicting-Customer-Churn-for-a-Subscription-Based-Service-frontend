# 📊 Customer Churn Prediction for Subscription-Based Service

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning solution for predicting customer churn in subscription-based services. This project implements a complete ML pipeline from data preprocessing to model deployment with an interactive web interface.

![Customer Churn Prediction](https://img.icons8.com/color/96/000000/customer-insight.png)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)

## 🎯 Overview

Customer churn prediction is crucial for subscription-based businesses to identify at-risk customers and implement proactive retention strategies. This project provides:

- **Machine Learning Pipeline**: Complete data preprocessing, feature engineering, and model training
- **Multiple Models**: Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- **REST API**: FastAPI backend for serving predictions
- **Web Interface**: Interactive Streamlit dashboard for real-time predictions
- **Docker Support**: Containerized deployment for production environments

## ✨ Features

### 🔬 Data Science
- Comprehensive exploratory data analysis (EDA)
- Automated data cleaning and preprocessing
- Feature engineering with domain knowledge
- Class imbalance handling with SMOTE
- Cross-validation and hyperparameter tuning

### 🤖 Machine Learning
- Multiple model implementations (Logistic Regression, Random Forest, XGBoost)
- Model comparison and selection
- Feature importance analysis
- Model evaluation with multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

### 🌐 Backend API
- RESTful API with FastAPI
- Single and batch prediction endpoints
- Model information and feature importance endpoints
- Automatic API documentation (Swagger/OpenAPI)
- Health check and model reload endpoints

### 🎨 Frontend
- Interactive Streamlit web application
- Real-time churn predictions
- Risk factor identification
- Visualization of model performance
- Feature importance charts

### 🚀 Deployment
- Docker and Docker Compose support
- Heroku deployment configuration
- Streamlit Cloud ready
- CI/CD pipeline ready

## 📁 Project Structure

```
Predicting-Customer-Churn-for-a-Subscription-Based-Service/
│
├── 📂 api/                          # FastAPI backend
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   └── schemas.py                   # Pydantic schemas
│
├── 📂 app/                          # Streamlit frontend
│   ├── __init__.py
│   └── streamlit_app.py             # Streamlit application
│
├── 📂 data/                         # Data directory
│   ├── raw/                         # Raw data files
│   └── processed/                   # Processed data files
│
├── 📂 models/                       # Trained models
│   ├── best_model.joblib            # Best performing model
│   ├── preprocessor.joblib          # Data preprocessor
│   ├── model_metrics.json           # Model performance metrics
│   └── feature_importance.json      # Feature importance scores
│
├── 📂 notebooks/                    # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
│
├── 📂 reports/                      # Generated reports
│   └── figures/                     # Visualization outputs
│
├── 📂 src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessing.py             # Data preprocessing
│   ├── model_training.py            # Model training pipeline
│   └── evaluation.py                # Model evaluation
│
├── 📂 tests/                        # Unit tests
│
├── 📂 .streamlit/                   # Streamlit configuration
│   └── config.toml
│
├── config.py                        # Project configuration
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
├── Procfile                         # Heroku configuration
├── runtime.txt                      # Python runtime version
├── .gitignore                       # Git ignore file
└── README.md                        # Project documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Predicting-Customer-Churn-for-a-Subscription-Based-Service.git
cd Predicting-Customer-Churn-for-a-Subscription-Based-Service
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it in `data/raw/telco_customer_churn.csv`.

*Note: If no data file is found, the system will generate a sample dataset for demonstration.*

## 🚀 Quick Start

### 1. Train the Model

```bash
python src/model_training.py
```

This will:
- Load and preprocess the data
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Select the best performing model
- Save the model and preprocessor to the `models/` directory

### 2. Start the FastAPI Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Start the Streamlit Frontend

In a new terminal:

```bash
streamlit run app/streamlit_app.py
```

The web interface will be available at `http://localhost:8501`

## 📖 Usage

### Using the Web Interface

1. Open `http://localhost:8501` in your browser
2. Navigate to "Predict Churn" page
3. Enter customer information in the form
4. Click "Predict Churn" to get the prediction
5. View the prediction result, probability, and risk factors

### Using the API

#### Single Prediction

```python
import requests

customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20
}

response = requests.post(
    "http://localhost:8000/predict",
    json=customer_data
)
print(response.json())
```

#### Batch Prediction

```python
import requests

batch_data = {
    "customers": [customer_data1, customer_data2, ...]
}

response = requests.post(
    "http://localhost:8000/predict/batch",
    json=batch_data
)
print(response.json())
```

## 📚 API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch prediction |
| `/model/info` | GET | Model information |
| `/model/features` | GET | Feature importance |
| `/model/reload` | GET | Reload model from disk |

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.65 | 0.54 | 0.59 | 0.84 |
| Random Forest | 0.79 | 0.63 | 0.48 | 0.54 | 0.82 |
| **XGBoost** | **0.81** | **0.67** | **0.52** | **0.59** | **0.85** |

*Note: Actual performance may vary based on the dataset used.*

### Key Insights from EDA

1. **Contract Type**: Month-to-month contracts have ~43% churn rate vs ~3% for two-year contracts
2. **Tenure**: New customers (< 12 months) have higher churn risk
3. **Payment Method**: Electronic check users have the highest churn rate
4. **Internet Service**: Fiber optic customers churn more (possibly due to higher costs)
5. **Tech Support**: Customers without tech support are more likely to churn

## 🚢 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the services:
# - API: http://localhost:8000
# - Streamlit: http://localhost:8501
```

### Heroku Deployment

```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Deploy
git push heroku main

# Open the app
heroku open
```

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Select `app/streamlit_app.py` as the main file
5. Deploy!

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Team

This project was developed as part of a Machine Learning Lab course project.

| Role | Responsibilities |
|------|-----------------|
| Data Engineer | Data preprocessing, feature engineering |
| ML Engineer | Model training, evaluation, optimization |
| Backend Developer | FastAPI development, API design |
| Frontend Developer | Streamlit interface, visualizations |
| DevOps | Docker, deployment, CI/CD |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- [FastAPI](https://fastapi.tiangolo.com/) for the awesome web framework
- [Streamlit](https://streamlit.io/) for the beautiful frontend framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools

---

<p align="center">
  Made with ❤️ for Machine Learning Education
</p>
