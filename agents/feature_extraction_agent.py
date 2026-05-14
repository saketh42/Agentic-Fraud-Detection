"""
Feature Extraction Agent - Analyze phase of MAPE-K
Parses transaction data to extract fraud labels, key features, and psychological tactics.
"""
from .base import BaseAgent, AgentResult


FRAUD_LABELS = [
    'transaction_upi_fraud', 'transaction_card_fraud', 'transaction_bank_transfer',
    'commerce_nondelivery', 'commerce_fake_seller', 'credential_phishing',
    'social_authority_scam', 'social_urgency_scam', 'meta_victim_story', 'meta_fraud_question'
]

KEY_FEATURES = [
    'payment_method', 'fraud_channel', 'victim_action', 'request_type',
    'impersonated_entity', 'amount_mentioned', 'currency', 'urgency_level'
]

PSYCHOLOGICAL_TACTICS = ['urgency', 'fear', 'authority', 'reward']


class FeatureExtractionAgent(BaseAgent):
    def __init__(self):
        super().__init__("FeatureExtractionAgent")

    def run(self, state: dict) -> AgentResult:
        self.log("Extracting fraud features...")

        txn = state.get('monitored_transaction', state.get('transaction', {}))
        if not txn:
            return AgentResult(success=False, message="No transaction data")

        active_labels = [l for l in FRAUD_LABELS if int(txn.get(l, 0)) == 1]
        active_tactics = [t for t in PSYCHOLOGICAL_TACTICS if float(txn.get(t, 0)) > 0.3]
        active_features = {k: txn.get(k, "unknown") for k in KEY_FEATURES if k in txn}

        label_count = len(active_labels)
        tactic_count = len(active_tactics)

        if label_count >= 3:
            semantic_profile = "high_indicator_multi_fraud"
        elif label_count >= 1 and tactic_count >= 2:
            semantic_profile = "combined_psychological_pressure"
        elif label_count >= 1:
            semantic_profile = "single_indicator_fraud"
        else:
            semantic_profile = "low_indicator_transaction"

        extraction = {
            'active_fraud_labels': active_labels,
            'active_key_features': active_features,
            'active_psychological_tactics': active_tactics,
            'label_count': label_count,
            'tactic_count': tactic_count,
            'semantic_profile': semantic_profile,
            'tactic_values': {t: float(txn.get(t, 0)) for t in PSYCHOLOGICAL_TACTICS}
        }

        self.log(f"Extracted {label_count} labels, {tactic_count} tactics — {semantic_profile}")
        return AgentResult(
            success=True,
            data=extraction,
            message="Feature extraction complete",
            metrics={'label_count': label_count, 'tactic_count': tactic_count, 'profile': semantic_profile}
        )
