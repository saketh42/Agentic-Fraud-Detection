"""
History page for the fraud detection demo
"""
import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("📈 History")
st.markdown("---")

# Sample history data (in a real implementation, this would come from the API)
sample_history = [
    {
        "run_id": "20260508_013608",
        "timestamp": "2026-05-08 01:36:08",
        "status": "Success",
        "f1_score": 1.000,
        "roc_auc": 1.000,
        "drift_detected": False,
        "iterations": 2
    },
    {
        "run_id": "20260507_152245",
        "timestamp": "2026-05-07 15:22:45",
        "status": "Success",
        "f1_score": 0.985,
        "roc_auc": 0.992,
        "drift_detected": True,
        "iterations": 3
    },
    {
        "run_id": "20260506_091532",
        "timestamp": "2026-05-06 09:15:32",
        "status": "Success",
        "f1_score": 0.978,
        "roc_auc": 0.987,
        "drift_detected": False,
        "iterations": 1
    }
]

# Display history table
st.subheader("Pipeline Run History")
history_df = pd.DataFrame(sample_history)

# Style the dataframe
def style_status(status):
    if status == "Success":
        return "✅ Success"
    else:
        return "❌ Failed"

# Apply styling
history_df["status"] = history_df["status"].apply(style_status)

st.dataframe(history_df, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Runs", len(sample_history))

with col2:
    successful_runs = len([run for run in sample_history if run["status"] == "✅ Success"])
    st.metric("Successful Runs", successful_runs)

with col3:
    drift_runs = len([run for run in sample_history if run["drift_detected"]])
    st.metric("Drift Detected Runs", drift_runs)

with col4:
    avg_f1 = sum([run["f1_score"] for run in sample_history]) / len(sample_history)
    st.metric("Average F1 Score", f"{avg_f1:.3f}")

# Performance trends
st.subheader("Performance Trends")
st.line_chart(history_df.set_index("timestamp")[["f1_score", "roc_auc"]])

# Drift detection history
st.subheader("Drift Detection History")
drift_history = []
for run in sample_history:
    drift_history.append({
        "timestamp": run["timestamp"],
        "drift_detected": run["drift_detected"]
    })

drift_df = pd.DataFrame(drift_history)
st.bar_chart(drift_df.set_index("timestamp")["drift_detected"].astype(int))

# Run details
st.subheader("Run Details")
selected_run = st.selectbox("Select a run to view details", 
                           [run["run_id"] for run in sample_history])

if selected_run:
    run_details = next((run for run in sample_history if run["run_id"] == selected_run), None)
    if run_details:
        st.write(f"**Run ID:** {run_details['run_id']}")
        st.write(f"**Timestamp:** {run_details['timestamp']}")
        st.write(f"**Status:** {run_details['status']}")
        st.write(f"**F1 Score:** {run_details['f1_score']:.3f}")
        st.write(f"**ROC-AUC:** {run_details['roc_auc']:.3f}")
        st.write(f"**Drift Detected:** {'Yes' if run_details['drift_detected'] else 'No'}")
        st.write(f"**Iterations:** {run_details['iterations']}")

st.info("This page shows the history of pipeline runs, including performance metrics and drift detection results. "
        "Use this information to track model performance over time and identify trends in fraud patterns.")