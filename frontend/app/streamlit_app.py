"""
Streamlit Frontend for Customer Churn Prediction.
Interactive web application for predicting customer churn.
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import sys

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# API URL
API_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-yes {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 5px;
    }
    .prediction-no {
        background-color: #e8f5e9;
        border-left: 5px solid #2ecc71;
        padding: 15px;
        border-radius: 5px;
    }
    .risk-factor {
        background-color: #fff3e0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(customer_data: dict):
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_feature_importance():
    """Get feature importance from API."""
    try:
        response = requests.get(f"{API_URL}/model/features", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def create_gauge_chart(probability: float):
    """Create a gauge chart for churn probability."""
    color = "#e74c3c" if probability >= 0.5 else "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(features: list):
    """Create feature importance bar chart."""
    if not features:
        return None
    
    df = pd.DataFrame(features).head(15)
    
    fig = px.bar(
        df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 Feature Importance',
        color='importance',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_layout(
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">📊 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🎯 Predict Churn", "📈 Model Performance", "ℹ️ About"]
        )
        
        st.markdown("---")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info("Please start the API server:\n```\nuvicorn api.main:app --reload\n```")
    
    # Main content based on page selection
    if page == "🎯 Predict Churn":
        predict_page()
    elif page == "📈 Model Performance":
        performance_page()
    else:
        about_page()


def predict_page():
    """Customer churn prediction page."""
    st.markdown('<p class="sub-header">Enter Customer Information</p>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("📞 Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )
        
        with col3:
            st.subheader("🔧 Additional Services")
            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )
        
        st.markdown("---")
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("📋 Contract & Billing")
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )
        
        with col5:
            st.subheader("💰 Account Information")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=200.0, 
                value=70.0,
                step=0.01
            )
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=tenure * monthly_charges,
                step=0.01
            )
        
        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    if submitted:
        # Prepare data
        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        with st.spinner("Making prediction..."):
            result = get_prediction(customer_data)
        
        if result:
            st.markdown("---")
            st.markdown('<p class="sub-header">📊 Prediction Result</p>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Prediction result
                if result["prediction"] == "Yes":
                    st.markdown(
                        f"""
                        <div class="prediction-yes">
                            <h2>⚠️ High Churn Risk</h2>
                            <p>This customer is <strong>likely to churn</strong>.</p>
                            <p>Probability: <strong>{result['probability']*100:.1f}%</strong></p>
                            <p>Confidence: <strong>{result['confidence']}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-no">
                            <h2>✅ Low Churn Risk</h2>
                            <p>This customer is <strong>likely to stay</strong>.</p>
                            <p>Probability: <strong>{result['probability']*100:.1f}%</strong></p>
                            <p>Confidence: <strong>{result['confidence']}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Risk factors
                if result.get("risk_factors"):
                    st.markdown("### 🚨 Risk Factors Identified")
                    for factor in result["risk_factors"]:
                        st.markdown(f"""
                        <div class="risk-factor">
                            ⚡ {factor}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                # Gauge chart
                fig = create_gauge_chart(result["probability"])
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if result["prediction"] == "Yes":
                st.info("""
                **To reduce churn risk, consider:**
                - Offering a contract upgrade with incentives
                - Providing personalized support or discounts
                - Adding value-added services like Tech Support or Online Security
                - Contacting the customer proactively to address concerns
                """)
            else:
                st.success("""
                **To maintain customer loyalty:**
                - Continue providing excellent service
                - Consider loyalty rewards or referral programs
                - Regularly check in with satisfaction surveys
                - Offer early renewal incentives
                """)


def performance_page():
    """Model performance page."""
    st.markdown('<p class="sub-header">📈 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    # Get model info
    model_info = get_model_info()
    feature_importance = get_feature_importance()
    
    if model_info:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", model_info.get("model_name", "N/A"))
        with col2:
            st.metric("Accuracy", f"{model_info.get('accuracy', 0)*100:.1f}%")
        with col3:
            st.metric("ROC-AUC", f"{model_info.get('roc_auc', 0)*100:.1f}%")
        with col4:
            st.metric("Features", len(model_info.get("features", [])))
        
        st.markdown("---")
    else:
        st.warning("Could not retrieve model information. Please ensure the API is running.")
    
    # Feature importance
    st.markdown('<p class="sub-header">🔍 Feature Importance</p>', unsafe_allow_html=True)
    
    if feature_importance:
        fig = create_feature_importance_chart(feature_importance)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature table
        with st.expander("View All Features"):
            df = pd.DataFrame(feature_importance)
            df['importance'] = df['importance'].apply(lambda x: f"{x:.6f}")
            st.dataframe(df, use_container_width=True)
    else:
        st.info("Feature importance data not available. Please train the model first.")
    
    # Model features list
    if model_info and model_info.get("features"):
        st.markdown("---")
        st.markdown('<p class="sub-header">📋 Model Features</p>', unsafe_allow_html=True)
        
        features = model_info["features"]
        cols = st.columns(4)
        for i, feature in enumerate(features):
            cols[i % 4].markdown(f"- {feature}")


def about_page():
    """About page."""
    st.markdown('<p class="sub-header">ℹ️ About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Customer Churn Prediction System
    
    This application predicts whether a customer is likely to churn (leave) a subscription-based service 
    using machine learning.
    
    ### 🎯 Purpose
    - Identify customers at risk of churning
    - Enable proactive retention strategies
    - Reduce customer acquisition costs
    
    ### 🔧 Technology Stack
    - **Frontend**: Streamlit
    - **Backend**: FastAPI
    - **ML Models**: Logistic Regression, Random Forest, XGBoost
    - **Preprocessing**: scikit-learn, pandas
    
    ### 📊 Dataset
    Based on the Telco Customer Churn dataset, featuring:
    - Customer demographics
    - Service subscriptions
    - Account information
    - Contract details
    
    ### 👥 Team
    This project was developed as part of a Machine Learning Lab course project.
    
    ### 📚 How to Use
    1. Navigate to the **Predict Churn** page
    2. Enter customer information
    3. Click **Predict Churn** to get the prediction
    4. View risk factors and recommendations
    
    ### 🚀 Getting Started
    
    1. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
    2. **Train the model**:
    ```bash
    python src/model_training.py
    ```
    
    3. **Start the API**:
    ```bash
    uvicorn api.main:app --reload
    ```
    
    4. **Start the Streamlit app**:
    ```bash
    streamlit run app/streamlit_app.py
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📧 Contact
    For questions or feedback, please open an issue on GitHub.
    
    ---
    *Built with ❤️ using Python, Streamlit, and FastAPI*
    """)


if __name__ == "__main__":
    main()
