"""
Pattern Learning Agent - Analyze phase of MAPE-K
Maps extracted signals to known or emerging fraud patterns.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


PATTERN_RULES = {
    'PHISHING': ['credential_phishing'],
    'SOCIAL_AUTHORITY_SCAM': ['social_authority_scam', 'authority', 'fear'],
    'SOCIAL_URGENCY_SCAM': ['social_urgency_scam', 'urgency'],
    'FAKE_SELLER_SCAM': ['commerce_fake_seller', 'reward'],
    'BANK_TRANSFER_FRAUD': ['transaction_bank_transfer', 'urgency'],
    'CARD_FRAUD': ['transaction_card_fraud'],
    'UPI_FRAUD': ['transaction_upi_fraud'],
    'HYBRID_SOCIAL_ENGINEERING': ['credential_phishing', 'social_authority_scam', 'urgency'],
    'COMMERCE_SCAM': ['commerce_nondelivery', 'commerce_fake_seller'],
    'UNKNOWN_EMERGING_PATTERN': []
}


class PatternLearningAgent(BaseAgent):
    """
    Maps fraud indicators to known patterns or identifies emerging ones.
    
    Outputs:
    - pattern_name: Mapped pattern
    - pattern_type: Category
    - is_emerging: New pattern flag
    - confidence: Pattern match confidence
    - evidence: Why this pattern was selected
    """
    
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("PatternLearningAgent")
        self.knowledge = KnowledgeStore(db_path)
        self.pattern_rules = PATTERN_RULES
    
    def run(self, state: dict) -> AgentResult:
        self.log("Learning fraud patterns...")
        
        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        semantic_profile = extraction.get('semantic_profile', '')
        
        if not active_labels and not active_tactics:
            return AgentResult(
                success=False,
                message="No features to pattern match"
            )
        
        # Combine all signals
        all_signals = active_labels + active_tactics
        
        # Try to match patterns
        matched_patterns = self._match_patterns(all_signals)
        
        if matched_patterns:
            # Use best match
            best_pattern = matched_patterns[0]
            pattern_name = best_pattern['name']
            pattern_type = best_pattern['type']
            confidence = best_pattern['confidence']
        else:
            # Mark as emerging
            pattern_name = 'UNKNOWN_EMERGING_PATTERN'
            pattern_type = 'EMERGING'
            confidence = 0.3
        
        # Check if this pattern has appeared before
        historical_frequency = self.knowledge.get_pattern_frequency(pattern_name)
        is_emerging = historical_frequency < 3
        
        # Get success rate
        if historical_frequency > 0:
            success_rate = self.knowledge.get_pattern_success_rate(pattern_name)
        else:
            success_rate = 0.5  # Unknown, assume 50%
        
        # Store/update pattern in knowledge
        self.knowledge.store_pattern(
            pattern_name=pattern_name,
            pattern_type=pattern_type,
            is_emerging=is_emerging
        )
        
        pattern_learning = {
            'pattern_name': pattern_name,
            'pattern_type': pattern_type,
            'is_emerging': is_emerging,
            'confidence': confidence,
            'historical_frequency': historical_frequency,
            'success_rate': success_rate,
            'matched_patterns': matched_patterns[:3],
            'signals': all_signals
        }
        
        self.log("Pattern: " + pattern_name + " (confidence: " + str(round(confidence, 2)) + ")")
        
        return AgentResult(
            success=True,
            data=pattern_learning,
            message="Pattern learning complete",
            metrics={
                'pattern': pattern_name,
                'is_emerging': is_emerging,
                'confidence': confidence,
                'historical_frequency': historical_frequency
            }
        )
    
    def _match_patterns(self, signals: list) -> list:
        """Match signals to known patterns"""
        matches = []
        
        for pattern_name, required_signals in self.pattern_rules.items():
            if not required_signals:
                continue
            
            # Count matches
            matched = sum(1 for sig in required_signals if sig in signals)
            if matched > 0:
                confidence = matched / len(required_signals)
                matches.append({
                    'name': pattern_name,
                    'type': self._get_pattern_type(pattern_name),
                    'confidence': confidence,
                    'matched_signals': matched,
                    'total_required': len(required_signals)
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        return matches
    
    def _get_pattern_type(self, pattern_name: str) -> str:
        """Get pattern category"""
        if 'PHISHING' in pattern_name:
            return 'PHISHING'
        elif 'SOCIAL' in pattern_name:
            return 'SOCIAL_ENGINEERING'
        elif 'FAKE' in pattern_name or 'COMMERCE' in pattern_name:
            return 'COMMERCE_SCAM'
        elif 'BANK' in pattern_name or 'CARD' in pattern_name or 'UPI' in pattern_name:
            return 'TRANSACTION_FRAUD'
        elif 'HYBRID' in pattern_name:
            return 'HYBRID'
        else:
            return 'UNKNOWN'