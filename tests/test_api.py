import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from fastapi.testclient import TestClient

# Check if TestClient works, skip API tests if not
try:
    from api.server import app
    client = TestClient(app)
    TEST_API = True
except Exception:
    TEST_API = False

pytestmark = pytest.mark.skipif(not TEST_API, reason="TestClient unavailable")


SAMPLE_PAYLOAD = {
    "transaction_id": "TXN-API-TEST-001",
    "transaction_upi_fraud": 0,
    "transaction_card_fraud": 0,
    "transaction_bank_transfer": 0,
    "commerce_nondelivery": 0,
    "commerce_fake_seller": 0,
    "credential_phishing": 1,
    "social_authority_scam": 1,
    "social_urgency_scam": 0,
    "meta_victim_story": 0,
    "meta_fraud_question": 0,
    "payment_method": "credit_card",
    "fraud_channel": "email",
    "victim_action": "click_link",
    "request_type": "verify_account",
    "impersonated_entity": "bank",
    "amount_mentioned": "yes",
    "currency": "USD",
    "urgency_level": "high",
    "amount_mentioned_value": 5000.0,
    "urgency": 0.8,
    "fear": 0.6,
    "authority": 0.7,
    "reward": 0.0,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_process_transaction():
    resp = client.post("/transaction/process", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "transaction_id" in data
    assert "feature_profile" in data
    assert "prediction" in data
    assert "pattern_learning" in data
    assert "adversarial_simulation" in data
    assert "reasoning" in data
    assert "plan" in data
    assert "execution" in data


def test_process_invalid():
    resp = client.post("/transaction/process", json={})
    assert resp.status_code == 422


def test_feedback():
    resp = client.post("/feedback", json={
        "transaction_id": "****test",
        "is_correct": True,
        "feedback_type": "correction"
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_get_patterns():
    resp = client.get("/patterns")
    assert resp.status_code == 200
    assert "patterns" in resp.json()


def test_get_transaction_not_found():
    resp = client.get("/transaction/NONEXISTENT")
    assert resp.status_code == 404


def test_analytics():
    resp = client.get("/analytics")
    assert resp.status_code == 200
