"""
MAPE-K Agentic Fraud Detection - Demo
Final Year Project 2026
"""
import streamlit as st
import requests
import re

st.set_page_config(page_title="MAPE-K Fraud Detection", page_icon=":shield:", layout="centered")

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
    text_lower = text.lower()

    urgency_kw = ['urgent', 'immediately', 'act now', 'limited time', 'expires', 'hurry', 'asap', 'right away', 'final notice', 'deadline', 'today only', 'quick', 'emergency', 'only few', 'warning']
    fear_kw = ['suspended', 'blocked', 'arrested', 'legal action', 'jail', 'fined', 'penalty', 'terminate', 'cancel', 'unauthorized', 'danger', 'deactivated', 'closed', 'restrict', 'locked', 'compromised']
    authority_kw = ['police', 'irs', 'fedex', 'amazon', 'paypal', 'google', 'microsoft', 'apple', 'wells fargo', 'social security', 'fbi', 'government', 'official', 'ceo', 'security team', 'account team', 'bank', 'sbi', 'hdfc', 'icici', 'axis', 'court', 'judge', 'authority', 'officer', 'ministry', 'income tax', 'enforcement', 'regulatory', 'legal notice', 'summons']
    reward_kw = ['congratulations', 'winner', 'won', 'claim', 'prize', 'lottery', 'gift card', 'free', 'reward', 'bonus', 'cash prize', 'exclusive', 'get paid', 'make money']
    money_kw = ['bitcoin', 'wire transfer', 'western union', 'gift card', 'moneygram', 'cashapp', 'venmo', 'zelle', 'crypto', 'money order', 'send money', 'paypal', 'pay', 'send', 'transfer', 'deposit', 'wire']

    uc = sum(1 for w in urgency_kw if w in text_lower)
    fc = sum(1 for w in fear_kw if w in text_lower)
    ac = sum(1 for w in authority_kw if w in text_lower)
    rc = sum(1 for w in reward_kw if w in text_lower)
    mc = sum(1 for w in money_kw if w in text_lower)

    has_link = 1 if ('http' in text_lower or 'www.' in text_lower or '.com' in text_lower or '://' in text_lower) else 0
    has_phone = 1 if re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text) else 0
    has_amount = 1 if re.search(r'(\$|rs\.?|inr|usd|eur|gbp)\s*\d+|\d+\s?(dollars|rupees|euros|pounds|rs)', text_lower) else 0
    amount_value = 0.0
    amount_match = re.search(r'(\d[\d,]*)\s*(dollars|rupees|euros|pounds|rs|\$|inr)?', text_lower)
    if amount_match:
        amount_value = float(amount_match.group(1).replace(',', ''))

    has_account_threat = ('account' in text_lower and any(w in text_lower for w in ['blocked', 'suspended', 'closed', 'locked', 'deactivated', 'terminate', 'cancel']))
    has_or_else = ' or ' in text_lower and any(w in text_lower for w in ['blocked', 'suspended', 'lose', 'locked', 'deleted', 'closed', 'jail', 'arrested', 'court'])

    urgency = min(1.0, (uc + fc) / 5)
    fear = min(1.0, fc / 3 + (0.3 if has_account_threat else 0))
    authority = min(1.0, ac / 2 + (0.3 if has_account_threat else 0))
    reward = min(1.0, rc / 2)

    amount = amount_value if amount_value > 0 else (100.0 if has_amount else 0.0)

    impersonated = 'unknown'
    if any(w in text_lower for w in ['bank', 'sbi', 'hdfc', 'icici', 'axis', 'paypal', 'irs', 'google', 'amazon', 'microsoft']):
        impersonated = 'bank' if any(w in text_lower for w in ['bank', 'sbi', 'hdfc', 'icici', 'axis', 'account']) else 'service_provider'

    victim_action = 'transfer_money' if has_amount else ('click_link' if has_link else ('respond' if has_account_threat else 'none'))

    channel = 'email' if has_link else ('phone' if has_phone else 'message')
    request_type = 'make_payment' if amount_value > 0 else ('verify_account' if ('verify' in text_lower or 'confirm' in text_lower) else 'threat_demand')

    credential_phishing = 1 if (has_link and ('verify' in text_lower or 'account' in text_lower or 'password' in text_lower or 'login' in text_lower or 'link' in text_lower)) else (
        1 if (has_account_threat and ('verify' in text_lower or 'password' in text_lower or 'login' in text_lower)) else 0
    )
    bank_transfer = 1 if (mc >= 1 and (ac >= 1 or fc >= 1)) or any(w in text_lower for w in ['bank', 'sbi', 'hdfc', 'icici', 'axis', 'transfer']) or (
        has_amount and (has_account_threat or ac >= 1 or uc >= 1)
    ) else 0
    upi_fraud = 1 if ('upi' in text_lower) or (has_amount and 'pay' in text_lower and uc >= 1) else 0
    card_fraud = 1 if mc >= 2 or 'card' in text_lower or ('pay' in text_lower and amount_value > 0) else 0

    return {
        'transaction_id': f"TXN-{abs(hash(text)) % 100000:05d}",
        'urgency': round(urgency, 4),
        'fear': round(fear, 4),
        'authority': round(authority, 4),
        'reward': round(reward, 4),
        'urgency_level': 'high' if urgency > 0.4 else 'medium' if urgency > 0.2 else 'low',
        'amount_mentioned_value': amount,
        'transaction_upi_fraud': upi_fraud,
        'transaction_card_fraud': card_fraud,
        'transaction_bank_transfer': bank_transfer,
        'commerce_nondelivery': 1 if ('free' in text_lower and 'shipping' in text_lower) or ('order' in text_lower and 'never arrived' in text_lower) else 0,
        'commerce_fake_seller': 1 if (rc >= 1 and uc >= 1) or ('fake' in text_lower and 'seller' in text_lower) or (rc >= 1 and has_amount) or ('free' in text_lower and 'pay' in text_lower) or ('win' in text_lower and 'fee' in text_lower) else 0,
        'credential_phishing': credential_phishing,
        'social_authority_scam': 1 if (ac >= 1 and (uc >= 1 or fc >= 1)) or has_or_else or ('authority' in text_lower or 'official' in text_lower) or (
            has_account_threat
        ) else 0,
        'social_urgency_scam': 1 if (uc >= 2) or ('limited time' in text_lower and ('offer' in text_lower or 'deal' in text_lower)) or (
            has_account_threat and 'immediately' in text_lower
        ) else 0,
        'meta_victim_story': 1 if ('help' in text_lower and 'please' in text_lower) or ('urgent' in text_lower and 'money' in text_lower) or ('stranded' in text_lower and 'need' in text_lower) else 0,
        'meta_fraud_question': 1 if ('confirm' in text_lower and 'account' in text_lower) or ('verify' in text_lower and 'identity' in text_lower) else 0,
        'impersonated_entity': impersonated,
        'victim_action': victim_action,
        'payment_method': 'unknown',
        'fraud_channel': channel,
        'request_type': request_type,
        'amount_mentioned': 'yes' if amount > 0 else 'no',
        'currency': 'USD',
    }

def analyze_transaction(data):
    try:
        r = requests.post(f"{API_URL}/transaction/process", json=data, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None

st.markdown("<h1 style='text-align: center; font-size: 2.5rem;'>MAPE-K Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 1.1rem;'>Enter any message or text. The MAPE-K system analyzes and detects fraud.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    api_ok = check_api()
    st.markdown(f"{'API: Connected' if api_ok else 'API: Disconnected'}")
with col2:
    st.markdown(f"[Swagger Docs]({API_URL}/docs)")

if not api_ok:
    st.error("API not running. Run: `python run.py` first", icon=":warning:")
    st.stop()

st.markdown("---")
text_input = st.text_area(
    "Paste message, email, or text here:",
    placeholder="Example: URGENT! Your bank account has been suspended. Click here immediately to verify: http://bank-verify.com",
    height=160,
    label_visibility="collapsed"
)

analyze = st.button("Analyze with MAPE-K", type="primary")

if analyze and text_input.strip():
    features = text_to_features(text_input)
    result = analyze_transaction(features)

    if result and "error" not in result:
        st.markdown("---")

        prediction = result.get('prediction', {})
        pattern = result.get('pattern_learning', {})
        reasoning = result.get('reasoning', {})
        plan = result.get('plan', {})
        adversarial = result.get('adversarial_simulation', {})

        fraud_score = prediction.get('fraud_score', 0)
        risk_level = prediction.get('risk_level', 'LOW')
        pattern_name = pattern.get('detected_pattern', 'UNKNOWN')
        pattern_conf = pattern.get('pattern_confidence', 0)
        adv_risk = adversarial.get('adversarial_risk', 'LOW')
        actions = plan.get('actions', [])

        is_fraud = fraud_score > 0.5

        if is_fraud:
            st.markdown(f"""
            <div class="fraud-result">
                <p class="result-big">FRAUD</p>
                <p class="result-sub">MAPE-K detected fraud patterns in this message</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-result">
                <p class="result-big">NOT FRAUD</p>
                <p class="result-sub">MAPE-K classified this as a legitimate message</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="score-bar">
            <div class="score-label">Fraud Score</div>
            <div class="score-value">{fraud_score:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(fraud_score)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Pattern:** `{pattern_name}` ({pattern_conf:.0%})")
            st.markdown(f"**Risk Level:** `{risk_level}`")
            st.markdown(f"**Adversarial Risk:** `{adv_risk}`")
        with c2:
            st.markdown(f"**Urgency:** `{features.get('urgency', 0):.2f}`")
            st.markdown(f"**Fear:** `{features.get('fear', 0):.2f}`")
            st.markdown(f"**Authority:** `{features.get('authority', 0):.2f}`")
            st.markdown(f"**Reward:** `{features.get('reward', 0):.2f}`")

        st.markdown(f"**Actions Taken:** `{', '.join(actions)}`")

        with st.expander("Full MAPE-K Response"):
            st.json(result)
    else:
        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

elif analyze and not text_input.strip():
    st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #4b5563; font-size: 0.8rem;'>MAPE-K: Monitor  Analyze  Plan  Execute  Knowledge | Final Year Project 2026</p>", unsafe_allow_html=True)
