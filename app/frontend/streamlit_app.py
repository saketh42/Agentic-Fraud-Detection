"""
MAPE-K Agentic Fraud Detection - Demo
Final Year Project 2026
"""
import streamlit as st
import requests

st.set_page_config(page_title="MAPE-K Fraud Detection", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
* { font-family: 'Segoe UI', sans-serif; }
.stApp { background-color: #0a0a0f; color: #ffffff; }
[data-testid="stSidebar"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 3rem; padding-left: 3rem; padding-right: 3rem; max-width: 800px !important; }
textarea { background-color: #1f2937 !important; color: #fff !important; border: 1px solid #374151 !important; font-size: 16px !important; min-height: 150px !important; border-radius: 12px !important; }
textarea::placeholder { color: #6b7280 !important; }
.stButton > button { background-color: #6366f1; color: white; border: none; border-radius: 12px; padding: 1rem 3rem; font-size: 1.2rem; font-weight: 600; width: 100%; }
.stButton > button:hover { background-color: #4f46e5; }
.fraud-result { background: linear-gradient(135deg, #dc2626, #7f1d1d); padding: 3rem; border-radius: 20px; text-align: center; color: white; margin: 2rem 0; }
.safe-result { background: linear-gradient(135deg, #059669, #064e3b); padding: 3rem; border-radius: 20px; text-align: center; color: white; margin: 2rem 0; }
.result-big { font-size: 3rem; font-weight: 700; margin: 0; color: white; }
.result-sub { font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
.score-bar { background: #374151; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
.score-label { color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.25rem; }
.score-value { color: #f3f4f6; font-size: 1.2rem; font-weight: 600; }
.stProgress > div > div { background-color: #6366f1; }
#MainMenu, footer { visibility: hidden; }
.stTabs { display: none; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except:
        return False

def text_to_features(text):
    """Convert raw text into model features - same schema as dataset"""
    text_lower = text.lower()
    
    urgency_kw = ['urgent', 'immediately', 'act now', 'limited time', 'expires', 'hurry', 'asap', 'right away', 'final notice', 'deadline', 'today only', 'quick', 'emergency', 'only few']
    fear_kw = ['suspended', 'blocked', 'arrested', 'legal action', 'jail', 'police', 'court', 'lawsuit', 'fined', 'penalty', 'terminate', 'cancel', 'verify', 'unauthorized', 'danger', 'warning']
    authority_kw = ['irs', 'fedex', 'amazon', 'paypal', 'google', 'microsoft', 'apple', 'bank', 'wells fargo', 'social security', 'fbi', 'government', 'official', 'ceo', 'security team', 'account team']
    reward_kw = ['congratulations', 'winner', 'won', 'claim', 'prize', 'lottery', 'gift card', 'free', 'reward', 'bonus', 'cash prize', 'exclusive', 'get paid', 'make money']
    money_kw = ['bitcoin', 'wire transfer', 'western union', 'gift card', 'moneygram', 'cashapp', 'venmo', 'zelle', 'crypto', 'money order', 'payment', 'send money', 'paypal']
    
    uc = sum(1 for w in urgency_kw if w in text_lower)
    fc = sum(1 for w in fear_kw if w in text_lower)
    ac = sum(1 for w in authority_kw if w in text_lower)
    rc = sum(1 for w in reward_kw if w in text_lower)
    mc = sum(1 for w in money_kw if w in text_lower)
    
    has_link = 1 if bool(requests.utils.urlparse(text_lower).scheme or 'http' in text_lower or 'www.' in text_lower or '.com' in text_lower or '.net' in text_lower or '.org' in text_lower or '://' in text_lower) else 0
    has_phone = 1 if bool(__import__('re').search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text)) else 0
    has_amount = 1 if bool(__import__('re').search(r'\$\d+|\d+\s?(dollars|rupees|euros|pounds)', text_lower)) else 0
    
    urgency = min(1.0, (uc + fc) / 5)
    fear = min(1.0, fc / 3)
    authority = min(1.0, ac / 2)
    reward = min(1.0, rc / 2)
    
    return {
        'urgency': urgency,
        'fear': fear,
        'authority': authority,
        'reward': reward,
        'urgency_level': urgency,
        'amount_normalized': 100 if has_amount else 0,
        'has_amount': has_amount,
        'transaction_upi_fraud': 1 if (uc >= 2 and fc >= 1) or 'upi' in text_lower else 0,
        'transaction_card_fraud': 1 if mc >= 2 or 'card' in text_lower else 0,
        'transaction_bank_transfer': 1 if (mc >= 1 and ac >= 1) or 'bank' in text_lower else 0,
        'commerce_nondelivery': 1 if ('free' in text_lower and 'shipping' in text_lower) or ('order' in text_lower and 'never arrived' in text_lower) else 0,
        'commerce_fake_seller': 1 if (rc >= 1 and uc >= 1) or ('fake' in text_lower and 'seller' in text_lower) else 0,
        'credential_phishing': 1 if has_link and ('verify' in text_lower or 'account' in text_lower or 'password' in text_lower or 'login' in text_lower or 'link' in text_lower) else 0,
        'social_authority_scam': 1 if (ac >= 1 and uc >= 1) or ('irs' in text_lower or 'fbi' in text_lower or 'police' in text_lower) else 0,
        'social_urgency_scam': 1 if (uc >= 2) or ('limited time' in text_lower and ('offer' in text_lower or 'deal' in text_lower)) else 0,
        'meta_victim_story': 1 if ('help' in text_lower and 'please' in text_lower) or ('urgent' in text_lower and 'money' in text_lower) or ('stranded' in text_lower and 'need' in text_lower) else 0,
        'meta_fraud_question': 1 if ('confirm' in text_lower and 'account' in text_lower) or ('verify' in text_lower and 'identity' in text_lower) else 0,
    }

def predict(data):
    try:
        r = requests.post(f"{API_URL}/api/predict/single", json=data, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def rule_based_verdict(features):
    """Rule-based detection based on dataset patterns (non-fraud = urgency_level 0, fraud_type none, no fraud labels)"""
    urgency = features.get('urgency', 0)
    fear = features.get('fear', 0)
    authority = features.get('authority', 0)
    reward = features.get('reward', 0)
    urgency_level = features.get('urgency_level', 0)
    
    fraud_labels = sum([
        features.get('transaction_upi_fraud', 0),
        features.get('transaction_card_fraud', 0),
        features.get('transaction_bank_transfer', 0),
        features.get('commerce_nondelivery', 0),
        features.get('commerce_fake_seller', 0),
        features.get('credential_phishing', 0),
        features.get('social_authority_scam', 0),
        features.get('social_urgency_scam', 0),
        features.get('meta_victim_story', 0),
        features.get('meta_fraud_question', 0)
    ])
    
    total_score = urgency + fear + authority + reward + urgency_level * 2 + fraud_labels * 3
    
    if total_score >= 3:
        return 'FRAUD', total_score
    elif total_score >= 1.5:
        return 'SUSPICIOUS', total_score
    else:
        return 'NOT FRAUD', total_score

st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>🛡️ MAPE-K Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 1.1rem;'>Enter any message or text. The MAPE-K system analyzes and detects fraud.</p>", unsafe_allow_html=True)

if not check_api():
    st.error("🔴 API not running. Run: `python simple_api.py` first", icon="⚠️")
    st.info("In Terminal 1: `cd /mnt/c/Users/Priti/Desktop/Agentic-Fraud-Detection && python simple_api.py`", icon="💡")
    st.stop()

st.markdown("---")
text_input = st.text_area(
    "📩 Paste message, email, or text here:",
    placeholder="Example: URGENT! Your bank account has been suspended. Click here immediately to verify: http://bank-verify.com",
    height=160,
    label_visibility="collapsed"
)

analyze = st.button("🔍 Analyze with MAPE-K", type="primary")

if analyze and text_input.strip():
    features = text_to_features(text_input)
    result = predict(features)
    
    if result:
        st.markdown("---")
        
        model_prob = result.get('fraud_probability', 0)
        model_is_fraud = result.get('is_fraud', False)
        rule_verdict, rule_score = rule_based_verdict(features)
        
        if rule_verdict == 'FRAUD':
            final = 'FRAUD'
            prob = max(model_prob, 0.85)
        elif rule_verdict == 'SUSPICIOUS':
            final = 'FRAUD' if model_prob > 0.5 else 'NOT FRAUD'
            prob = model_prob
        else:
            final = 'NOT FRAUD'
            prob = min(model_prob, 0.15)
        
        if final == 'FRAUD':
            st.markdown(f"""
            <div class="fraud-result">
                <p class="result-big">🚨 FRAUD</p>
                <p class="result-sub">MAPE-K detected fraud patterns in this message</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-result">
                <p class="result-big">✅ NOT FRAUD</p>
                <p class="result-sub">MAPE-K classified this as a legitimate message</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="score-bar">
            <div class="score-label">Fraud Probability</div>
            <div class="score-value">{prob:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Pattern:** `{result.get('pattern_name', 'N/A')}`")
            st.markdown(f"**Risk Level:** `{result.get('risk_level', 'N/A')}`")
        with col2:
            st.markdown(f"**Urgency:** `{features.get('urgency', 0):.2f}`")
            st.markdown(f"**Fear:** `{features.get('fear', 0):.2f}`")
            st.markdown(f"**Authority:** `{features.get('authority', 0):.2f}`")
            st.markdown(f"**Reward:** `{features.get('reward', 0):.2f}`")
        
        with st.expander("📋 Full Model Response"):
            st.json(result)
    
    elif analyze and not text_input.strip():
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #4b5563; font-size: 0.8rem;'>MAPE-K: Monitor → Analyze → Plan → Execute → Knowledge | Final Year Project 2026</p>", unsafe_allow_html=True)