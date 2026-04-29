"""
Feedback Agent - Knowledge phase of MAPE-K
Processes human feedback for continuous learning.
"""
from .base import BaseAgent, AgentResult
from .knowledge_store import KnowledgeStore
from datetime import datetime


class FeedbackAgent(BaseAgent):
    """
    Processes human feedback to improve future decisions.
    
    Actions:
    - Store feedback in memory
    - Update pattern success rates
    - Trigger retraining review if needed
    - Calculate learning improvement
    """
    
    def __init__(self, db_path: str = "knowledge_store.db"):
        super().__init__("FeedbackAgent")
        self.knowledge = KnowledgeStore(db_path)
    
    def run(self, state: dict) -> AgentResult:
        """Process feedback"""
        action = state.get('feedback_action', 'store')
        
        if action == 'store':
            return self._store_feedback(state)
        elif action == 'review':
            return self._review_feedback(state)
        elif action == 'improvement':
            return self._calculate_improvement(state)
        else:
            return AgentResult(success=False, message="Unknown action")
    
    def _store_feedback(self, state: dict) -> AgentResult:
        """Store human feedback in memory"""
        transaction_id = state.get('transaction_id')
        feedback_type = state.get('feedback_type', 'correction')
        feedback_text = state.get('feedback_text', '')
        is_correct = state.get('is_correct', True)
        
        if not transaction_id:
            return AgentResult(success=False, message="No transaction_id")
        
        # Store in knowledge
        success = self.knowledge.store_feedback(
            transaction_id=transaction_id,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            is_correct=is_correct
        )
        
        if success:
            self.log("Feedback stored for transaction: " + transaction_id)
            
            # If incorrect, trigger review
            if not is_correct:
                self.log("Incorrect prediction - marking for review")
                return AgentResult(
                    success=True,
                    data={
                        'feedback_stored': True,
                        'review_triggered': True,
                        'action': 'TRIGGER_RETRAINING_REVIEW'
                    },
                    message="Feedback stored, retraining review triggered"
                )
            
            return AgentResult(
                success=True,
                data={'feedback_stored': True},
                message="Feedback stored successfully"
            )
        
        return AgentResult(success=False, message="Failed to store feedback")
    
    def _review_feedback(self, state: dict) -> AgentResult:
        """Review recent feedback for patterns"""
        analytics = self.knowledge.get_analytics()
        
        feedback_accuracy = analytics.get('feedback_accuracy', 0.0)
        total_feedback = analytics.get('total_feedback', 0)
        
        if total_feedback == 0:
            return AgentResult(
                success=True,
                data={'review_complete': False, 'reason': 'No feedback yet'},
                message="No feedback to review"
            )
        
        # Check if we need retraining
        needs_retraining = feedback_accuracy < 0.70
        
        review = {
            'review_complete': True,
            'feedback_accuracy': feedback_accuracy,
            'total_feedback': total_feedback,
            'needs_retraining': needs_retraining,
            'recommended_action': 'TRIGGER_RETRAINING_REVIEW' if needs_retraining else 'CONTINUE'
        }
        
        self.log("Feedback review: accuracy=" + str(round(feedback_accuracy, 2)) + 
              ", needs_retraining=" + str(needs_retraining))
        
        return AgentResult(
            success=True,
            data=review,
            message="Review complete",
            metrics={'accuracy': feedback_accuracy, 'needs_retraining': needs_retraining}
        )
    
    def _calculate_improvement(self, state: dict) -> AgentResult:
        """Calculate if feedback improves decisions over time"""
        improvement_rate = self.knowledge.calculate_learning_improvement()
        
        if improvement_rate > 0.7:
            assessment = 'IMPROVING'
        elif improvement_rate > 0.5:
            assessment = 'STABLE'
        else:
            assessment = 'DEGRADING'
        
        result = {
            'improvement_rate': improvement_rate,
            'assessment': assessment,
            'interpretation': self._interpret_improvement(improvement_rate)
        }
        
        self.log("Learning improvement: " + assessment + " (" + str(round(improvement_rate, 2)) + ")")
        
        return AgentResult(
            success=True,
            data=result,
            message="Improvement calculation complete"
        )
    
    def _interpret_improvement(self, rate: float) -> str:
        """Interpret improvement rate"""
        if rate >= 0.80:
            return "System is learning effectively from feedback"
        elif rate >= 0.60:
            return "System is maintaining performance with feedback"
        else:
            return "Consider reviewing decision logic"


def submit_feedback(transaction_id: str, is_correct: bool, 
               feedback_text: str = '', db_path: str = "knowledge_store.db") -> bool:
    """Convenience function to submit feedback"""
    agent = FeedbackAgent(db_path)
    state = {
        'feedback_action': 'store',
        'transaction_id': transaction_id,
        'feedback_type': 'correction',
        'feedback_text': feedback_text,
        'is_correct': is_correct
    }
    result = agent.run(state)
    return result.success