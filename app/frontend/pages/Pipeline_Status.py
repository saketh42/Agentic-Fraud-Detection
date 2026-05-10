"""
Pipeline status page for the fraud detection demo
"""
import streamlit as st
import requests
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("🔄 Pipeline Status")
st.markdown("---")

# Fetch system status
try:
    response = requests.get("http://localhost:8000/api/status")
    if response.status_code == 200:
        status_data = response.json()
        
        st.subheader("System Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("System Status", "✅ Online" if status_data["status"] == "ok" else "❌ Offline")
        
        with col2:
            st.metric("Model Loaded", "✅ Yes" if status_data["model_loaded"] else "❌ No")
        
        with col3:
            st.metric("Pipeline Ready", "✅ Yes" if status_data["pipeline_ready"] else "❌ No")
        
        with col4:
            st.metric("Drift Detected", "⚠️ Yes" if status_data["drift_detected"] else "✅ No")
        
    else:
        st.error("Failed to fetch system status")
except requests.exceptions.RequestException:
    st.warning("API not available. Please start the backend server.")

# MAPE-K Pipeline Visualization
st.subheader("MAPE-K Pipeline Visualization")

# Create a visual representation of the MAPE-K loop
st.markdown("""
<div style="text-align: center;">
    <h3>MAPE-K Loop Architecture</h3>
    <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 20px; width: 80%; background-color: #f9f9f9;">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div style="text-align: center; padding: 10px; border: 1px solid #2196F3; border-radius: 5px; background-color: #e3f2fd;">
                    <h4>Monitor</h4>
                    <p>Drift Agent</p>
                </div>
                <div>→</div>
                <div style="text-align: center; padding: 10px; border: 1px solid #FF9800; border-radius: 5px; background-color: #fff3e0;">
                    <h4>Analyze</h4>
                    <p>Evaluation Agent</p>
                </div>
                <div>→</div>
                <div style="text-align: center; padding: 10px; border: 1px solid #9C27B0; border-radius: 5px; background-color: #f3e5f5;">
                    <h4>Plan</h4>
                    <p>Training Agent</p>
                </div>
                <div>→</div>
                <div style="text-align: center; padding: 10px; border: 1px solid #F44336; border-radius: 5px; background-color: #ffebee;">
                    <h4>Execute</h4>
                    <p>Balance Agent</p>
                </div>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <div style="border: 1px solid #607D8B; border-radius: 5px; padding: 10px; background-color: #eceff1;">
                    <h4>Knowledge Base</h4>
                    <p>Stores history, drift records, metrics</p>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Pipeline controls
st.subheader("Pipeline Controls")

# Run pipeline button
if st.button("🚀 Run Full Pipeline"):
    try:
        response = requests.get("http://localhost:8000/api/pipeline/run")
        if response.status_code == 200:
            st.success("Pipeline started successfully!")
            st.info("The full MAPE-K pipeline is now running. This may take several minutes.")
        else:
            st.error(f"Failed to start pipeline: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API: {str(e)}")

# Pipeline configuration
st.subheader("Pipeline Configuration")
with st.expander("Advanced Configuration"):
    st.write("Configure pipeline parameters:")
    
    # Drift detection thresholds
    st.markdown("#### Drift Detection Thresholds")
    psi_threshold = st.slider("PSI Threshold", 0.0, 1.0, 0.20, 0.01)
    ks_threshold = st.slider("KS Threshold", 0.0, 0.5, 0.05, 0.01)
    
    # Training parameters
    st.markdown("#### Training Parameters")
    model_type = st.selectbox("Model Type", ["gradient_boosting", "random_forest"])
    adversarial_training = st.checkbox("Adversarial Training", value=True)
    fgsm_epsilon = st.slider("FGSM Epsilon", 0.0, 0.5, 0.05, 0.01)
    
    # Evaluation thresholds
    st.markdown("#### Evaluation Thresholds")
    min_f1 = st.slider("Minimum F1 Score", 0.0, 1.0, 0.70, 0.01)
    min_roc_auc = st.slider("Minimum ROC-AUC", 0.0, 1.0, 0.75, 0.01)
    min_robustness = st.slider("Minimum Robustness", 0.0, 1.0, 0.60, 0.01)

st.info("The MAPE-K (Monitor-Analyze-Plan-Execute using Knowledge) pipeline automatically adapts to concept drift "
        "and maintains model performance over time. The visualization above shows how the different agents "
        "work together in a closed-loop system to detect fraud and maintain model accuracy.")