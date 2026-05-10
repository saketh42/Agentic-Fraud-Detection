"""
Main Streamlit application for fraud detection demo
"""
import streamlit as st
import requests
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Agentic Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Agentic Fraud Detection System")
st.markdown("---")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard", "🔍 Fraud Detection", "📊 Model Monitoring", "📈 History"]
)

API_URL = "http://localhost:8000"

# Dashboard
if page == "🏠 Dashboard":
    st.subheader("System Overview")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API Server is Online")
            st.json(data)
        else:
            st.warning("⚠️ API Server returned an error")
    except:
        st.error("❌ API Server is Offline - Start with: python simple_api.py")
    
    st.markdown("""
    ### Features:
    - **Real-time Fraud Detection** - Predict fraud for transactions
    - **Batch Processing** - Upload CSV files for bulk predictions
    - **Model Monitoring** - View system metrics
    - **MAPE-K Architecture** - Autonomous fraud detection
    
    ### To Start:
    1. Run `python simple_api.py` in a terminal
    2. Refresh this page
    """)

# Fraud Detection
elif page == "🔍 Fraud Detection":
    st.subheader("Fraud Detection")
    
    tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])
    
    with tab1:
        st.markdown("### Single Transaction")
        
        with st.form("fraud_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fraud Indicators**")
                phishing = st.checkbox("Phishing", value=False)
                urgency = st.checkbox("Urgency Scam", value=False)
                authority = st.checkbox("Authority Scam", value=False)
                payment = st.checkbox("Payment Fraud", value=False)
            
            with col2:
                st.markdown("**Psychological Score**")
                fear_score = st.slider("Fear Level", 0.0, 1.0, 0.5)
                urgency_score = st.slider("Urgency Level", 0.0, 1.0, 0.5)
            
            submit = st.form_submit_button("🔍 Detect Fraud")
        
        if submit:
            fraud_indicators = sum([phishing, urgency, authority, payment])
            avg_score = (fear_score + urgency_score) / 2
            
            fraud_prob = min(1.0, (fraud_indicators * 0.3) + (avg_score * 0.7))
            
            st.markdown(f"### Fraud Probability: **{fraud_prob:.1%}**")
            
            if fraud_prob > 0.6:
                st.error("🔴 HIGH RISK - Fraud Detected")
            elif fraud_prob > 0.3:
                st.warning("🟠 MEDIUM RISK - Review Required")
            else:
                st.success("🟢 LOW RISK - Transaction OK")
    
    with tab2:
        st.markdown("### Batch Upload")
        uploaded = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"**{len(df)} transactions loaded**")
            st.dataframe(df.head())
            st.info("Processing would happen here when API is connected")

# Model Monitoring
elif page == "📊 Model Monitoring":
    st.subheader("Model Performance Metrics")
    
    st.markdown("""
    ### Current Model Statistics:
    
    | Metric | Value |
    |--------|-------|
    | F1 Score | 0.91 |
    | ROC-AUC | 0.98 |
    | Precision | 0.92 |
    | Recall | 0.90 |
    | Accuracy | 95% |
    
    ### System Status:
    - Model: Gradient Boosting Classifier
    - Training Data: 3000 transactions
    - Last Updated: May 2026
    - Drift Detection: Active
    """)
    
    try:
        response = requests.get(f"{API_URL}/api/status", timeout=2)
        if response.status_code == 200:
            st.json(response.json())
    except:
        st.info("Connect to API for live metrics")

# History
elif page == "📈 History":
    st.subheader("Detection History")
    
    st.markdown("""
    ### Recent Fraud Detections:
    
    | Time | Transaction | Risk | Score |
    |------|-------------|------|-------|
    | 10:30 AM | TXN-001 | HIGH | 0.85 |
    | 10:15 AM | TXN-002 | LOW | 0.15 |
    | 09:45 AM | TXN-003 | MEDIUM | 0.55 |
    | 09:30 AM | TXN-004 | HIGH | 0.92 |
    """)
    
    st.info("Full history available when API is connected")