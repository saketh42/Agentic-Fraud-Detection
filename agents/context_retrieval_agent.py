"""
Context Retrieval Agent - Analyze phase of MAPE-K
Queries historical knowledge for contextual decision-making.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


class ContextRetrievalAgent(BaseAgent):
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("ContextRetrievalAgent")
        self.knowledge = KnowledgeStore(db_path)

    def run(self, state: dict) -> AgentResult:
        self.log("Retrieving contextual knowledge...")

        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        semantic_profile = extraction.get('semantic_profile', '')

        historical_freq = {}
        for label in active_labels:
            historical_freq[label] = self.knowledge.get_pattern_frequency(label)

        tactic_rates = {}
        for tactic in active_tactics:
            tactic_rates[tactic] = self.knowledge.get_tactic_success_rate(tactic)

        similar_cases = []
        if semantic_profile:
            similar_cases = self.knowledge.get_similar_transactions(semantic_profile, limit=5)

        impersonation_patterns = self.knowledge.get_impersonation_patterns()
        all_patterns = self.knowledge.get_all_patterns()
        recent = self.knowledge.get_recent_transactions(limit=10)

        evidence = []
        if historical_freq:
            evidence.append({
                'type': 'label_frequency',
                'value': historical_freq,
                'description': f"Historical frequency of detected labels"
            })
        if tactic_rates:
            evidence.append({
                'type': 'tactic_success_rates',
                'value': tactic_rates,
                'description': f"Success rates of detected tactics"
            })
        if similar_cases:
            fraud_count = sum(1 for c in similar_cases if c.get('is_fraud') == 1)
            evidence.append({
                'type': 'similar_cases',
                'value': len(similar_cases),
                'description': f"{len(similar_cases)} similar cases, {fraud_count} were fraud"
            })
        if impersonation_patterns:
            evidence.append({
                'type': 'impersonation_patterns',
                'value': impersonation_patterns,
                'description': f"{len(impersonation_patterns)} known impersonation patterns"
            })
        if all_patterns:
            evidence.append({
                'type': 'known_patterns',
                'value': len(all_patterns),
                'description': f"{len(all_patterns)} known patterns in knowledge store"
            })

        history_count = len(recent)
        context_confidence = min(1.0, history_count / 10.0)

        context = {
            'historical_label_frequencies': historical_freq,
            'tactic_success_rates': tactic_rates,
            'similar_cases': similar_cases,
            'impersonation_patterns': impersonation_patterns,
            'known_patterns': all_patterns,
            'context_confidence': context_confidence,
            'evidence': evidence,
            'history_available': history_count > 0
        }

        self.log(f"Retrieved context — {history_count} past transactions, confidence={context_confidence}")
        return AgentResult(
            success=True,
            data=context,
            message="Context retrieval complete",
            metrics={'history_count': history_count, 'confidence': context_confidence}
        )
