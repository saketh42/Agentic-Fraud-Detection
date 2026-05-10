"""
Fraud detection page for the demo
"""
import streamlit as st
import requests
import pandas as pd
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.title("🔍 Fraud Detection")
st.markdown("---")

# Tabs for single and batch prediction
tab1, tab2 = st.tabs(["Single Transaction", "Batch Processing"])

# Single transaction prediction
with tab1:
    st.subheader("Single Transaction Fraud Detection")
    
    # Create input form for transaction features
    with st.form("fraud_detection_form"):
        st.markdown("### Transaction Features")
        
        # Fraud labels section
        st.markdown("#### Fraud Labels")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            transaction_upi_fraud = st.checkbox("UPI Fraud", value=False)
            transaction_card_fraud = st.checkbox("Card Fraud", value=False)
        
        with col2:
            transaction_bank_transfer = st.checkbox("Bank Transfer Fraud", value=False)
            commerce_nondelivery = st.checkbox("Commerce Non-delivery", value=False)
        
        with col3:
            commerce_fake_seller = st.checkbox("Fake Seller", value=False)
            credential_phishing = st.checkbox("Credential Phishing", value=False)
        
        with col4:
            social_authority_scam = st.checkbox("Authority Scam", value=False)
            social_urgency_scam = st.checkbox("Urgency Scam", value=False)
        
        with col5:
            meta_victim_story = st.checkbox("Victim Story", value=False)
            meta_fraud_question = st.checkbox("Fraud Question", value=False)
        
        # Key features section
        st.markdown("#### Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            payment_method = st.selectbox("Payment Method", 
                                        ["unknown", "upi", "card", "bank", "crypto", "cash"])
            fraud_channel = st.selectbox("Fraud Channel", 
                                        ["unknown", "email", "website", "phone", "social_media", "direct_message"])
            victim_action = st.selectbox("Victim Action", 
                                       ["unknown", "sent_money", "shared_credentials", "clicked_link", "installed_app"])
        
        with col2:
            request_type = st.selectbox("Request Type", 
                                      ["unknown", "payment", "verification", "commerce", "phishing"])
            impersonated_entity = st.text_input("Impersonated Entity", "unknown")
            amount_mentioned = st.text_input("Amount Mentioned", "unknown")
        
        with col3:
            currency = st.selectbox("Currency", 
                                  ["unknown", "INR", "USD", "EUR", "GBP"])
            urgency_level = st.select_slider("Urgency Level", 
                                             options=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], 
                                             value="0.6")
        
        # Psychological tactics section
        st.markdown("#### Psychological Tactics Scores (0-1)")
        col1, col2 = st.columns(2)
        
        with col1:
            urgency = st.slider("Urgency", 0.0, 1.0, 0.5)
            fear = st.slider("Fear", 0.0, 1.0, 0.5)
        
        with col2:
            authority = st.slider("Authority", 0.0, 1.0, 0.5)
            reward = st.slider("Reward", 0.0, 1.0, 0.5)
        
        # Amount features
        st.markdown("#### Amount Features")
        col1, col2 = st.columns(2)
        
        with col1:
            amount_normalized = st.number_input("Normalized Amount", value=0.0, min_value=0.0)
        
        with col2:
            has_amount = st.checkbox("Has Amount", value=False)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Fraud")
        
        if submitted:
            transaction_data = {
                "transaction_upi_fraud": 1 if transaction_upi_fraud else 0,
                "transaction_card_fraud": 1 if transaction_card_fraud else 0,
                "transaction_bank_transfer": 1 if transaction_bank_transfer else 0,
                "commerce_nondelivery": 1 if commerce_nondelivery else 0,
                "commerce_fake_seller": 1 if commerce_fake_seller else 0,
                "credential_phishing": 1 if credential_phishing else 0,
                "social_authority_scam": 1 if social_authority_scam else 0,
                "social_urgency_scam": 1 if social_urgency_scam else 0,
                "meta_victim_story": 1 if meta_victim_story else 0,
                "meta_fraud_question": 1 if meta_fraud_question else 0,
                "payment_method": payment_method,
                "fraud_channel": fraud_channel,
                "victim_action": victim_action,
                "request_type": request_type,
                "impersonated_entity": impersonated_entity,
                "amount_mentioned": amount_mentioned,
                "currency": currency,
                "urgency_level": urgency_level,
                "urgency": urgency,
                "fear": fear,
                "authority": authority,
                "reward": reward,
                "amount_normalized": amount_normalized,
                "has_amount": 1 if has_amount else 0
            }
            
            try:
                response = requests.post(
                    "http://localhost:8000/api/predict/single",
                    json=transaction_data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Prediction Complete!")
                    st.markdown(f"### Fraud Probability: **{result['fraud_probability']:.2%}**")
                    st.markdown(f"### Risk Level: **{result['risk_level']}**")
                    st.markdown(f"### Model Confidence: **{result['model_confidence']:.2%}**")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

# Batch processing
with tab2:
    st.subheader("Batch Fraud Detection")
    st.info("Upload a CSV file with transaction data for batch processing")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded file contains **{len(df)}** transactions")
        st.dataframe(df.head(10))
        
        if st.button("🚀 Process Batch"):
            st.info("Processing... (Connect API for full functionality)")