import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
from pipeline import MAPEKPipeline


SAMPLE_TXN = {
    "transaction_id": "TXN-001-test",
    "is_fraud": None,
    "fraud_type": None,
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


def test_full_pipeline():
    pipeline = MAPEKPipeline(db_path=":memory:", llm_mock_mode=True)
    result = pipeline.process_transaction(SAMPLE_TXN)

    assert "error" not in result
    assert result["transaction_id"] == "****test"
    assert "feature_profile" in result
    assert "prediction" in result
    assert "pattern_learning" in result
    assert "adversarial_simulation" in result
    assert "reasoning" in result
    assert "plan" in result
    assert "execution" in result
    assert len(result["execution"]) > 0
    assert result["memory_update"] == "SUCCESS"

    for action_result in result["execution"]:
        assert "action" in action_result
        assert "status" in action_result


def test_pipeline_low_risk():
    clean_txn = SAMPLE_TXN.copy()
    clean_txn["transaction_id"] = "TXN-002-clean"
    for key in ['credential_phishing', 'social_authority_scam',
                 'transaction_upi_fraud', 'transaction_card_fraud',
                 'transaction_bank_transfer', 'commerce_nondelivery',
                 'commerce_fake_seller', 'meta_victim_story', 'meta_fraud_question']:
        clean_txn[key] = 0
    clean_txn["urgency"] = 0.1
    clean_txn["fear"] = 0.0
    clean_txn["authority"] = 0.0
    clean_txn["reward"] = 0.0

    pipeline = MAPEKPipeline(db_path=":memory:", llm_mock_mode=True)
    result = pipeline.process_transaction(clean_txn)

    assert "error" not in result
    assert "ALLOW_TRANSACTION" in result["plan"]["actions"]


def test_pipeline_high_risk():
    high_txn = SAMPLE_TXN.copy()
    high_txn["transaction_id"] = "TXN-003-high"
    high_txn["credential_phishing"] = 1
    high_txn["social_authority_scam"] = 1
    high_txn["social_urgency_scam"] = 1
    high_txn["transaction_bank_transfer"] = 1
    high_txn["meta_victim_story"] = 1

    pipeline = MAPEKPipeline(db_path=":memory:", llm_mock_mode=True)
    result = pipeline.process_transaction(high_txn)

    assert "error" not in result
    assert "BLOCK_TRANSACTION" in result["plan"]["actions"] or \
           "TRIGGER_MFA" in result["plan"]["actions"]


if __name__ == "__main__":
    test_full_pipeline()
    test_pipeline_low_risk()
    test_pipeline_high_risk()
    print("All pipeline tests passed!")
