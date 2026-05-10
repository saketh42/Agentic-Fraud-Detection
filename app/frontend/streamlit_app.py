"""
Main Streamlit application for fraud detection demo
"""
import streamlit as st
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config
st.set_page_config(
    page_title="Agentic Fraud Detection Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .stAlert {
        background-color: #f0f8ff;
        border: 1px solid #4b9cd3;
        border-radius: 5px;
        padding: 10px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .fraud-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .fraud-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .fraud-low {
        color: #00cc00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🛡️ Fraud Detection Demo")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🔍 Fraud Detection",
        "📊 Model Monitoring",
        "🔄 Pipeline Status",
        "📈 History"
    ]
)

# API client
API_BASE_URL = "http://localhost:8000"

# Page routing
if page == "🏠 Dashboard":
    import pages.Home
elif page == "🔍 Fraud Detection":
    import pages.Fraud_Detection
elif page == "📊 Model Monitoring":
    import pages.Model_Monitoring
elif page == "🔄 Pipeline Status":
    import pages.Pipeline_Status
elif page == "📈 History":
    import pages.History