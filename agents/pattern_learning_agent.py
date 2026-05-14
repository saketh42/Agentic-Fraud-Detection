"""
Pattern Learning Agent - Analyze phase of MAPE-K
Maps signals to known fraud patterns using spec rules.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


PATTERN_RULES = {
    'HYBRID_SOCIAL_ENGINEERING': {
        'signals': ['credential_phishing', 'social_authority_scam', 'social_urgency_scam', 'urgency'],
        'type': 'HYBRID',
        'min_match': 2
    },
    'FAKE_SELLER_SCAM': {
        'signals': ['commerce_fake_seller', 'reward'],
        'type': 'COMMERCE_SCAM',
        'min_match': 2
    },
    'AUTHORITY_SCAM': {
        'signals': ['social_authority_scam', 'authority', 'fear'],
        'type': 'SOCIAL_ENGINEERING',
        'min_match': 2
    },
    'BANK_TRANSFER_FRAUD': {
        'signals': ['transaction_bank_transfer', 'urgency'],
        'type': 'TRANSACTION_FRAUD',
        'min_match': 2
    },
    'PHISHING': {
        'signals': ['credential_phishing', 'urgency'],
        'type': 'PHISHING',
        'min_match': 1
    },
    'UPI_FRAUD': {
        'signals': ['transaction_upi_fraud'],
        'type': 'TRANSACTION_FRAUD',
        'min_match': 1
    },
    'CARD_FRAUD': {
        'signals': ['transaction_card_fraud'],
        'type': 'TRANSACTION_FRAUD',
        'min_match': 1
    },
    'COMMERCE_SCAM': {
        'signals': ['commerce_nondelivery', 'commerce_fake_seller'],
        'type': 'COMMERCE_SCAM',
        'min_match': 1
    },
}


class PatternLearningAgent(BaseAgent):
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("PatternLearningAgent")
        self.knowledge = KnowledgeStore(db_path)

    def run(self, state: dict) -> AgentResult:
        self.log("Learning fraud patterns...")

        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])

        if not active_labels and not active_tactics:
            return AgentResult(success=False, message="No signals to match")

        all_signals = active_labels + active_tactics
        matches = self._match_patterns(all_signals)

        if matches:
            best = matches[0]
            pattern_name = best['name']
            pattern_type = best['type']
            confidence = best['confidence']
            pattern_evidence = [m['name'] for m in matches[:3]]
        else:
            pattern_name = 'UNKNOWN_EMERGING_PATTERN'
            pattern_type = 'EMERGING'
            confidence = 0.3
            pattern_evidence = []

        is_hybrid = len(active_labels) >= 2 and len(active_tactics) >= 1
        if is_hybrid and pattern_name != 'UNKNOWN_EMERGING_PATTERN':
            pattern_name = 'HYBRID_SOCIAL_ENGINEERING'
            pattern_type = 'HYBRID'
            confidence = max(confidence, 0.8)

        historical_freq = self.knowledge.get_pattern_frequency(pattern_name)
        is_emerging = historical_freq < 3
        success_rate = self.knowledge.get_pattern_success_rate(pattern_name) if historical_freq > 0 else 0.5

        self.knowledge.store_pattern(pattern_name, pattern_type, is_emerging)

        result = {
            'detected_pattern': pattern_name,
            'pattern_type': pattern_type,
            'pattern_confidence': round(confidence, 4),
            'matched_historical_cases': historical_freq,
            'is_emerging_pattern': is_emerging,
            'pattern_summary': f"Detected {pattern_name} ({pattern_type}) with {confidence:.0%} confidence",
            'pattern_evidence': pattern_evidence,
            'success_rate': success_rate,
            'signals': all_signals
        }

        # Update state for adversarial simulations
        result['active_labels'] = active_labels
        result['active_tactics'] = active_tactics

        self.log(f"Pattern: {pattern_name} (confidence: {confidence:.2f})")
        return AgentResult(
            success=True, data=result,
            message="Pattern learning complete",
            metrics={'pattern': pattern_name, 'confidence': confidence, 'emerging': is_emerging}
        )

    def _match_patterns(self, signals: list) -> list:
        matches = []
        for name, rule in PATTERN_RULES.items():
            matched = sum(1 for s in rule['signals'] if s in signals)
            if matched >= rule['min_match']:
                confidence = matched / len(rule['signals'])
                matches.append({'name': name, 'type': rule['type'], 'confidence': confidence})
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
