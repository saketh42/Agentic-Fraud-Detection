"""
Monitor Agent - Monitor phase of MAPE-K
Validates incoming transactions and masks sensitive IDs.
"""
from .base import BaseAgent, AgentResult
from datetime import datetime


REQUIRED_FRAUD_LABELS = [
    'transaction_upi_fraud', 'transaction_card_fraud', 'transaction_bank_transfer',
    'commerce_nondelivery', 'commerce_fake_seller', 'credential_phishing',
    'social_authority_scam', 'social_urgency_scam', 'meta_victim_story', 'meta_fraud_question'
]

REQUIRED_KEY_FEATURES = [
    'payment_method', 'fraud_channel', 'victim_action', 'request_type',
    'impersonated_entity', 'amount_mentioned', 'currency', 'urgency_level'
]

REQUIRED_TACTICS = ['urgency', 'fear', 'authority', 'reward']


class MonitorAgent(BaseAgent):
    def __init__(self):
        super().__init__("MonitorAgent")

    def run(self, state: dict) -> AgentResult:
        self.log("Monitoring incoming transaction...")

        transaction = state.get('transaction', {})
        if not transaction:
            return AgentResult(success=False, message="No transaction data provided")

        errors = self._validate_schema(transaction)
        if errors:
            self.log(f"Validation failed: {errors}")
            return AgentResult(success=False, message=f"Validation errors: {errors}")

        masked_id = self._mask_id(transaction.get('transaction_id', 'unknown'))
        timestamp = datetime.now().isoformat()

        cleaned = {
            'transaction_id': transaction.get('transaction_id'),
            'masked_transaction_id': masked_id,
            'timestamp': timestamp,
            'is_fraud': transaction.get('is_fraud'),
            'fraud_type': transaction.get('fraud_type'),
        }

        for label in REQUIRED_FRAUD_LABELS:
            cleaned[label] = int(transaction.get(label, 0))
        for tactic in REQUIRED_TACTICS:
            cleaned[tactic] = float(transaction.get(tactic, 0.0))
        for feature in REQUIRED_KEY_FEATURES:
            cleaned[feature] = transaction.get(feature, "unknown")
        cleaned['amount_mentioned_value'] = float(transaction.get('amount_mentioned_value', 0))

        monitor_data = {
            'monitored_transaction': cleaned,
            'masked_transaction_id': masked_id,
            'processing_timestamp': timestamp,
            'validation_passed': True
        }

        self.log(f"Transaction {masked_id} validated successfully")
        return AgentResult(
            success=True,
            data=monitor_data,
            message=f"Transaction {masked_id} validated",
            metrics={'validation_passed': True}
        )

    def _validate_schema(self, txn: dict) -> list:
        errors = []
        if 'transaction_id' not in txn or not txn.get('transaction_id'):
            errors.append("Missing transaction_id")
        for label in REQUIRED_FRAUD_LABELS:
            txn.setdefault(label, 0)
        for tactic in REQUIRED_TACTICS:
            txn.setdefault(tactic, 0.0)
        for feature in REQUIRED_KEY_FEATURES:
            txn.setdefault(feature, "unknown")
        txn.setdefault('amount_mentioned_value', 0.0)
        return errors

    def _mask_id(self, txn_id: str) -> str:
        if len(txn_id) > 4:
            return "****" + txn_id[-4:]
        return "****" + txn_id
