import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_PATH = Path(__file__).resolve().parents[1]
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

API_BASE_URL = "http://localhost:8000"

def inject_custom_assets():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
            .stMetric { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
            .status-online { color: #00ffaa; font-weight: bold; }
            .status-offline { color: #ff4b4b; font-weight: bold; }
            .card-container { padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; }
            .churn-alert { background-color: rgba(255, 75, 75, 0.1); border-left: 5px solid #ff4b4b; }
            .retention-success { background-color: rgba(0, 255, 170, 0.1); border-left: 5px solid #00ffaa; }
        </style>
    """, unsafe_allow_html=True)

class ChurnClient:
    @staticmethod
    def ping():
        try:
            return requests.get(f"{API_BASE_URL}/health", timeout=2).status_code == 200
        except:
            return False

    @staticmethod
    def fetch_data(endpoint):
        try:
            response = requests.get(f"{API_BASE_URL}/{endpoint}", timeout=5)
            return response.json() if response.status_code == 200 else None
        except:
            return None

    @staticmethod
    def post_inference(payload):
        return requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)

def render_sidebar():
    with st.sidebar:
        st.title("RetainIQ")
        st.caption("Advanced Churn Intelligence")
        
        is_alive = ChurnClient.ping()
        status_text = "SYSTEM ONLINE" if is_alive else "SYSTEM OFFLINE"
        status_class = "status-online" if is_alive else "status-offline"
        st.markdown(f"Status: <span class='{status_class}'>{status_text}</span>", unsafe_allow_html=True)
        
        st.divider()
        nav = st.selectbox("Intelligence Hub", ["Inference Engine", "Analytics Vault", "System Documentation"])
        
        if not is_alive:
            st.warning("Gateway unreachable. Ensure the FastAPI backend is listening on port 8000.")
        
        return nav

def render_inference_engine():
    st.header("🎯 Customer Inference Engine")
    
    with st.expander("Customer Profile Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.toggle("Senior Citizen")
            partner = st.selectbox("Partnered", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with c2:
            contract = st.select_slider("Contract Term", options=["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Network Provider", ["Fiber optic", "DSL", "No"])
            streaming_tv = st.checkbox("Streaming TV")
            streaming_mov = st.checkbox("Streaming Movies")
        with c3:
            tenure = st.number_input("Tenure (Months)", 0, 72, 12)
            monthly = st.slider("Monthly Subscription ($)", 18.0, 120.0, 65.0)
            paperless = st.toggle("Paperless Billing", True)

    if st.button("Execute Risk Analysis", use_container_width=True, type="primary"):
        payload = {
            "gender": gender, "SeniorCitizen": int(senior), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": internet, "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "Yes" if streaming_tv else "No",
            "StreamingMovies": "Yes" if streaming_mov else "No",
            "Contract": contract, "PaperlessBilling": "Yes" if paperless else "No",
            "PaymentMethod": "Electronic check", "MonthlyCharges": monthly,
            "TotalCharges": monthly * tenure
        }
        
        with st.spinner("Analyzing behavioral patterns..."):
            response = ChurnClient.post_inference(payload)
            
            if response.status_code == 200:
                data = response.json()
                prob = data["probability"]
                is_churn = data["prediction"] == "Yes"
                
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        title = {'text': "Attrition Risk %"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ff4b4b" if is_churn else "#00ffaa"},
                            'steps': [{'range': [0, 50], 'color': "gray"}, {'range': [50, 100], 'color': "darkgray"}]
                        }
                    ))
                    gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(gauge, use_container_width=True)

                with res_col2:
                    style = "churn-alert" if is_churn else "retention-success"
                    msg = "HIGH RISK" if is_churn else "LOW RISK"
                    st.markdown(f"""<div class='card-container {style}'><h3>{msg} Detected</h3>
                                <p>Probability of churn: {prob:.2%}</p>
                                <p>Confidence Score: {data.get('confidence', 'High')}</p></div>""", unsafe_allow_html=True)
                    
                    if is_churn:
                        st.error("Action Required: Targeted retention offer recommended.")
                    else:
                        st.success("Customer stable. Focus on upsell opportunities.")
            else:
                st.error("Analysis Failed. Check payload schema.")

def render_analytics_vault():
    st.header("📈 Model Diagnostics")
    info = ChurnClient.fetch_data("model/info")
    features = ChurnClient.fetch_data("model/features")
    
    if info:
        m1, m2, m3 = st.columns(3)
        m1.metric("Algorithm", info.get("model_name"))
        m2.metric("Accuracy Score", f"{info.get('accuracy', 0):.2%}")
        m3.metric("AUC Metric", f"{info.get('roc_auc', 0):.3f}")

    if features:
        df = pd.DataFrame(features).sort_values("importance", ascending=True).tail(10)
        fig = px.bar(df, x="importance", y="feature", orientation='h', 
                     title="Key Drivers of Attrition", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="RetainIQ | Churn AI", layout="wide")
    inject_custom_assets()
    
    view = render_sidebar()
    
    if view == "Inference Engine":
        render_inference_engine()
    elif view == "Analytics Vault":
        render_analytics_vault()
    else:
        st.write("RetainIQ leverages Deep Neural Networks and XGBoost ensembles to predict customer turnover.")

if __name__ == "__main__":
    main()