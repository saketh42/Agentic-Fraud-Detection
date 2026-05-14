"""
Feedback Agent - Knowledge phase of MAPE-K
Processes human feedback for continuous learning.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore


class FeedbackAgent(BaseAgent):
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("FeedbackAgent")
        self.knowledge = KnowledgeStore(db_path)

    def run(self, state: dict) -> AgentResult:
        action = state.get('feedback_action', 'store')

        if action == 'store':
            return self._store_feedback(state)
        elif action == 'review':
            return self._review_feedback(state)
        elif action == 'improvement':
            return self._calculate_improvement(state)
        else:
            return AgentResult(success=False, message=f"Unknown feedback action: {action}")

    def _store_feedback(self, state: dict) -> AgentResult:
        txn_id = state.get('transaction_id')
        feedback_type = state.get('feedback_type', 'correction')
        is_correct = state.get('is_correct', True)

        if not txn_id:
            return AgentResult(success=False, message="No transaction_id")

        ok = self.knowledge.store_feedback(txn_id, feedback_type, is_correct)
        if not ok:
            return AgentResult(success=False, message="Failed to store feedback")

        self.log(f"Feedback stored for {txn_id} — correct={is_correct}")
        result = {'feedback_stored': True}
        if not is_correct:
            result['review_triggered'] = True
            result['action'] = 'TRIGGER_RETRAINING_REVIEW'

        return AgentResult(
            success=True, data=result,
            message="Feedback stored" + (" — retraining review triggered" if not is_correct else "")
        )

    def _review_feedback(self, state: dict) -> AgentResult:
        analytics = self.knowledge.get_analytics()
        accuracy = analytics.get('feedback_accuracy', 0.0)
        total = analytics.get('total_feedback', 0)

        if total == 0:
            return AgentResult(success=True, data={'review_complete': False, 'reason': 'No feedback'})

        needs_retraining = accuracy < 0.70
        review = {
            'review_complete': True,
            'feedback_accuracy': accuracy,
            'total_feedback': total,
            'needs_retraining': needs_retraining,
            'recommended_action': 'TRIGGER_RETRAINING_REVIEW' if needs_retraining else 'CONTINUE'
        }

        self.log(f"Feedback review: accuracy={accuracy:.2f}, needs_retraining={needs_retraining}")
        return AgentResult(success=True, data=review, message="Review complete")

    def _calculate_improvement(self, state: dict) -> AgentResult:
        rate = self.knowledge.calculate_learning_improvement()

        if rate > 0.7:
            assessment = 'IMPROVING'
        elif rate > 0.5:
            assessment = 'STABLE'
        else:
            assessment = 'DEGRADING'

        result = {'improvement_rate': rate, 'assessment': assessment}
        self.log(f"Learning improvement: {assessment} ({rate:.2f})")
        return AgentResult(success=True, data=result, message="Improvement calculated")
