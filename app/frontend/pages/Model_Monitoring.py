"""
Model monitoring page for the fraud detection demo
"""
import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("📊 Model Monitoring")
st.markdown("---")

# Fetch current metrics from the API
try:
    st.subheader("Current Model Metrics")
    
    response = requests.get("http://localhost:8000/api/metrics")
    if response.status_code == 200:
        metrics_data = response.json()
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics_data['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics_data['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics_data['recall']:.3f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics_data['f1']:.3f}")
        
        with col5:
            st.metric("ROC-AUC", f"{metrics_data['roc_auc']:.3f}")
        
        # Display metrics table
        st.subheader("Detailed Metrics")
        metrics_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Value": [
                metrics_data['accuracy'],
                metrics_data['precision'],
                metrics_data['recall'],
                metrics_data['f1'],
                metrics_data['roc_auc']
            ]
        })
        st.table(metrics_table)
        
        # Display robustness metrics
        st.subheader("Robustness Analysis")
        st.write(f"Model is considered robust: **{metrics_data.get('is_robust', 'Unknown')}**")
        
        # Display robustness curve
        if 'robustness_curve' in metrics_data:
            st.write("Robustness curve (F1 score vs. adversarial attack strength):")
            robustness_df = pd.DataFrame(metrics_data['robustness_curve'])
            st.line_chart(robustness_df.set_index('epsilon'))
            
            # Additional robustness metrics
            st.write(f"Clean F1 Score: {metrics_data.get('clean_f1', 0):.3f}")
            st.write(f"Worst F1 Score: {metrics_data.get('worst_f1', 0):.3f}")
            st.write(f"Average F1 Score: {metrics_data.get('avg_f1', 0):.3f}")
            st.write(f"F1 Drop: {metrics_data.get('f1_drop', 0):.3f}")
        
    else:
        st.error("Failed to fetch metrics")
except requests.exceptions.RequestException as e:
    st.error(f"Failed to connect to API: {str(e)}")
except Exception as e:
    st.error(f"Error processing metrics: {str(e)}")

# Confusion matrix
st.subheader("Confusion Matrix")
try:
    response = requests.get("http://localhost:8000/api/metrics")
    if response.status_code == 200:
        metrics_data = response.json()
        if all(key in metrics_data for key in ['true_negatives', 'false_positives', 'false_negatives', 'true_positives']):
            tn = metrics_data['true_negatives']
            fp = metrics_data['false_positives']
            fn = metrics_data['false_negatives']
            tp = metrics_data['true_positives']
            
            # Create confusion matrix data
            confusion_data = pd.DataFrame({
                'Predicted Negative': [tn, fn],
                'Predicted Positive': [fp, tp]
            }, index=['Actual Negative', 'Actual Positive'])
            
            st.table(confusion_data)
            
            # Calculate additional metrics
            st.write("Additional Metrics:")
            st.write(f"False Positive Rate: {metrics_data.get('fpr', 0):.3f}")
    else:
        st.warning("Confusion matrix data not available")
except:
    st.warning("Unable to fetch confusion matrix data")

st.info("This page displays the current performance metrics of the fraud detection model. "
        "The metrics are updated after each training cycle and provide insights into the model's accuracy, "
        "precision, recall, and robustness against adversarial attacks.")