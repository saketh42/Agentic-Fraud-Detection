"""
Context Retrieval Agent - Analyze phase of MAPE-K
Queries historical knowledge for contextual decision-making.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


class ContextRetrievalAgent(BaseAgent):
    """
    Retrieves historical context for current transaction.
    
    Queries:
    - Similar semantic profiles
    - Pattern frequencies
    - Past similar transactions
    - Tactic success rates
    - Impersonation patterns
    """
    
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("ContextRetrievalAgent")
        self.knowledge = KnowledgeStore(db_path)
    
    def run(self, state: dict) -> AgentResult:
        self.log("Retrieving contextual knowledge...")
        
        semantic_profile = state.get('semantic_profile')
        active_labels = state.get('active_fraud_labels', [])
        active_tactics = state.get('active_psychological_tactics', [])
        
        if not semantic_profile and not active_labels:
            return AgentResult(
                success=False,
                message="No profile or labels provided"
            )
        
        # Get recent history
        recent_transactions = self.knowledge.get_recent_transactions(limit=10)
        
        # Get similar cases
        if semantic_profile:
            similar_cases = self.knowledge.get_similar_transactions(semantic_profile, limit=5)
        else:
            similar_cases = []
        
        # Get pattern frequencies
        label_frequencies = self.knowledge.get_label_frequencies()
        
        # Get all patterns
        patterns = self.knowledge.get_all_patterns()
        
        # Calculate context confidence based on history
        history_count = len(recent_transactions)
        context_confidence = min(1.0, history_count / 10.0)
        
        # Build evidence from history
        evidence = []
        if recent_transactions:
            fraud_count = sum(1 for t in recent_transactions if t.get('is_fraud') == 1)
            fraud_rate = fraud_count / len(recent_transactions) if recent_transactions else 0.0
            evidence.append({
                'type': 'historical_prevalence',
                'value': fraud_rate,
                'description': f'Historical fraud rate: {fraud_rate:.2%}'
            })
        
        if patterns:
            evidence.append({
                'type': 'pattern_match',
                'value': len(patterns),
                'description': f'{len(patterns)} known patterns available'
            })
        
        if similar_cases:
            recent_fraud = sum(1 for c in similar_cases if c.get('is_fraud') == 1)
            evidence.append({
                'type': 'similar_cases',
                'value': recent_fraud / len(similar_cases) if similar_cases else 0.0,
                'description': f'{len(similar_cases)} similar cases, {recent_fraud} were fraud'
            })
        
        context = {
            'recent_transactions': recent_transactions,
            'similar_cases': similar_cases,
            'label_frequencies': label_frequencies,
            'patterns': patterns,
            'context_confidence': context_confidence,
            'evidence': evidence,
            'history_available': history_count > 0
        }
        
        self.log("Retrieved context: " + str(history_count) + " transactions, confidence: " + str(context_confidence))
        
        return AgentResult(
            success=True,
            data=context,
            message="Context retrieval complete",
            metrics={
                'history_count': history_count,
                'context_confidence': context_confidence,
                'similar_cases': len(similar_cases),
                'known_patterns': len(patterns)
            }
        )