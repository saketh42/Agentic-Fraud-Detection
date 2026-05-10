"""
Home page for the fraud detection demo
"""
import streamlit as st
import requests
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page title
st.title("🛡️ Agentic Fraud Detection System")
st.markdown("---")

# System status
st.subheader("System Status")
try:
    response = requests.get("http://localhost:8000/api/status")
    if response.status_code == 200:
        status_data = response.json()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", "✅ Online" if status_data["status"] == "ok" else "❌ Offline")
        
        with col2:
            st.metric("Model Loaded", "✅ Yes" if status_data["model_loaded"] else "❌ No")
        
        with col3:
            st.metric("Pipeline Ready", "✅ Yes" if status_data["pipeline_ready"] else "❌ No")
        
        with col4:
            st.metric("Drift Detected", "⚠️ Yes" if status_data["drift_detected"] else "✅ No")
        
        # Display metrics
        st.subheader("Current Metrics")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
        
        metrics = status_data["metrics"]
        with metrics_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        
        with metrics_col2:
            st.metric("Precision", f"{metrics['precision']:.2f}")
        
        with metrics_col3:
            st.metric("Recall", f"{metrics['recall']:.2f}")
        
        with metrics_col4:
            st.metric("F1 Score", f"{metrics['f1']:.2f}")
        
        with metrics_col5:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    else:
        st.warning("⚠️ Unable to fetch system status")
except requests.exceptions.RequestException:
    st.warning("⚠️ API not available. Please start the backend server.")

# Quick actions
st.subheader("Quick Actions")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Run Fraud Detection"):
        st.switch_page("pages/Fraud_Detection.py")

with col2:
    if st.button("📊 View Model Monitoring"):
        st.switch_page("pages/Model_Monitoring.py")

with col3:
    if st.button("🔄 Run Full Pipeline"):
        st.info("Running full pipeline... (This may take a few minutes)")
        # In a real implementation, this would trigger the pipeline

# System information
st.subheader("System Information")
st.info("""
This demo showcases the MAPE-K (Monitor-Analyze-Plan-Execute using Knowledge) 
agentic fraud detection system. The system automatically detects concept drift, 
balances data using CTGAN, trains robust models with adversarial training, 
and evaluates performance using multiple metrics.
""")

# Key features
st.subheader("Key Features")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - **Drift Detection**: Monitors for concept drift using PSI and KS tests
    - **Class Balancing**: Uses CTGAN to generate synthetic minority samples
    - **Adversarial Training**: Robustness against evasion attacks
    - **LLM Decision Making**: Autonomous model management decisions
    """)

with col2:
    st.markdown("""
    - **Learning Adaptation**: Tracks how models adapt to new data patterns
    - **Real-time Prediction**: Instant fraud detection for transactions
    - **Performance Monitoring**: Continuous evaluation of model metrics
    - **Visual Dashboard**: Interactive charts and metrics visualization
    """)