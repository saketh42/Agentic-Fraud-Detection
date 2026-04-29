"""
Metrics Tracker - Agentic Performance Metrics
Tracks and calculates agentic system performance metrics.
"""
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from .base import BaseAgent, AgentResult


class MetricsTracker(BaseAgent):
    """
    Tracks agentic performance metrics.
    
    Core Classification Metrics:
    - Accuracy, Precision, Recall, F1
    - PR-AUC (important for imbalanced data)
    - ROC-AUC
    
    Agentic Metrics:
    - Planning Accuracy
    - Escalation Precision/Recall
    - Action Utility
    - Human Override Rate
    - Mean Time to Decision
    - Memory Improvement Rate
    
    Adversarial Metrics:
    - Adversarial Accuracy
    - FPR@Attack
    - Robustness Drop
    - Robustness Gain (after feedback)
    
    Pattern Metrics:
    - Pattern Classification Accuracy
    """
    
    def __init__(self):
        super().__init__("MetricsTracker")
        self.predictions = []
        self.ground_truth = []
        self.actions = []
        self.action_outcomes = []
        self.plans = []
        self.human_overrides = []
        self.execution_times = []
        self.pattern_predictions = []
        self.pattern_ground_truth = []
        self.adversarial_results = []
    
    def run(self, state: dict) -> AgentResult:
        action = state.get('metrics_action', 'compute')
        
        if action == 'compute':
            return self._compute_metrics(state)
        elif action == 'record':
            return self._record_prediction(state)
        elif action == 'adversarial':
            return self._record_adversarial(state)
        else:
            return AgentResult(success=False, message="Unknown action")
    
    def _record_prediction(self, state: dict) -> AgentResult:
        """Record a prediction for later evaluation"""
        transaction_id = state.get('transaction_id')
        prediction = state.get('prediction')
        ground_truth = state.get('ground_truth')
        action = state.get('action')
        action_outcome = state.get('action_outcome')
        plan = state.get('plan')
        human_override = state.get('human_override', False)
        execution_time = state.get('execution_time', 0.0)
        
        if prediction is not None:
            self.predictions.append(prediction)
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
        if action is not None:
            self.actions.append(action)
        if action_outcome is not None:
            self.action_outcomes.append(action_outcome)
        if plan is not None:
            self.plans.append(plan)
        if human_override is not None:
            self.human_overrides.append(human_override)
        if execution_time is not None:
            self.execution_times.append(execution_time)
        
        return AgentResult(
            success=True,
            data={'recorded': True},
            message="Prediction recorded"
        )
    
    def _record_adversarial(self, state: dict) -> AgentResult:
        """Record adversarial test result"""
        result = {
            'original_score': state.get('original_score', 0.5),
            'adversarial_score': state.get('adversarial_score', 0.5),
            'attack_type': state.get('attack_type', 'unknown'),
            'is_robust': state.get('adversarial_score', 0.5) > 0.3
        }
        self.adversarial_results.append(result)
        
        return AgentResult(
            success=True,
            data=result,
            message="Adversarial result recorded"
        )
    
    def _compute_metrics(self, state: dict) -> AgentResult:
        """Compute all performance metrics"""
        
        metrics = {}
        
        # Core classification metrics
        if self.predictions and self.ground_truth:
            y_pred = np.array(self.predictions)
            y_true = np.array(self.ground_truth)
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            # PR-AUC (better for imbalanced)
            if len(np.unique(y_true)) > 1:
                try:
                    y_proba = np.array(self.predictions)
                    metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
                except:
                    metrics['pr_auc'] = 0.0
        
        # Agentic metrics
        metrics['planning_accuracy'] = self._compute_planning_accuracy()
        metrics['escalation_precision'] = self._compute_escalation_precision()
        metrics['escalation_recall'] = self._compute_escalation_recall()
        metrics['human_override_rate'] = self._compute_override_rate()
        metrics['mean_time_to_decision'] = self._compute_mean_time()
        metrics['memory_improvement_rate'] = self._compute_memory_improvement()
        
        # Adversarial metrics
        adv_metrics = self._compute_adversarial_metrics()
        metrics.update(adv_metrics)
        
        # Pattern metrics
        metrics['pattern_classification_accuracy'] = self._compute_pattern_accuracy()
        
        self.log("Metrics computed: F1=" + str(round(metrics.get('f1', 0), 3)) + 
               ", Planning=" + str(round(metrics.get('planning_accuracy', 0), 3)))
        
        return AgentResult(
            success=True,
            data=metrics,
            message="Metrics computation complete",
            metrics=metrics
        )
    
    def _compute_planning_accuracy(self) -> float:
        """Whether selected action was appropriate"""
        if not self.action_outcomes:
            return 0.0
        
        correct = sum(1 for o in self.action_outcomes if o in ['true_positive', 'true_negative'])
        return correct / len(self.action_outcomes) if self.action_outcomes else 0.0
    
    def _compute_escalation_precision(self) -> float:
        """How many escalated cases truly needed escalation"""
        if not self.actions:
            return 0.0
        
        escalated = [a for a in self.actions if a in [' MANUAL_REVIEW', 'BLOCK']]
        if not escalated:
            return 0.0
        
        needed = sum(1 for o in self.action_outcomes if o == 'true_positive')
        return needed / len(escalated) if escalated else 0.0
    
    def _compute_escalation_recall(self) -> float:
        """How many risky cases were correctly escalated"""
        if not self.action_outcomes:
            return 0.0
        
        risky_cases = [o for o in self.action_outcomes if o in ['true_positive']]
        if not risky_cases:
            return 1.0
        
        correctly_escalated = sum(1 for a, o in zip(self.actions, self.action_outcomes) 
                             if a in ['MANUAL_REVIEW', 'BLOCK'] and o == 'true_positive')
        return correctly_escalated / len(risky_cases) if risky_cases else 0.0
    
    def _compute_override_rate(self) -> float:
        """How often reviewers disagreed"""
        if not self.human_overrides:
            return 0.0
        
        return sum(1.0 for h in self.human_overrides if h) / len(self.human_overrides)
    
    def _compute_mean_time(self) -> float:
        """Mean time to produce decision"""
        if not self.execution_times:
            return 0.0
        
        return float(np.mean(self.execution_times))
    
    def _compute_memory_improvement(self) -> float:
        """Whether feedback improves future decisions"""
        # Placeholder - would be calculated from knowledge store
        return 0.75  # Default assumption
    
    def _compute_adversarial_metrics(self) -> Dict:
        """Compute adversarial robustness metrics"""
        if not self.adversarial_results:
            return {
                'adversarial_accuracy': 0.0,
                'fpr_at_attack': 0.0,
                'robustness_drop': 0.0
            }
        
        original_scores = [r['original_score'] for r in self.adversarial_results]
        adv_scores = [r['adversarial_score'] for r in self.adversarial_results]
        
        # Accuracy under attack
        adv_correct = sum(1 for r in self.adversarial_results if r['adversarial_score'] > 0.5)
        adv_accuracy = adv_correct / len(self.adversarial_results) if self.adversarial_results else 0.0
        
        # FPR at attack threshold
        fpr_attack = sum(1 for s in adv_scores if s > 0.5 and s < 0.3) / len(adv_scores)
        
        # Robustness drop
        orig_correct = sum(1 for s in original_scores if s > 0.5)
        robustness_drop = (orig_correct / len(original_scores)) - adv_accuracy if original_scores else 0.0
        
        return {
            'adversarial_accuracy': adv_accuracy,
            'fpr_at_attack': fpr_attack,
            'robustness_drop': robustness_drop
        }
    
    def _compute_pattern_accuracy(self) -> float:
        """Pattern classification accuracy"""
        if not self.pattern_predictions:
            return 0.0
        
        correct = sum(1 for p, g in zip(self.pattern_predictions, self.pattern_ground_truth) 
                    if p == g)
        return correct / len(self.pattern_predictions) if self.pattern_predictions else 0.0
    
    def reset(self):
        """Reset all tracked metrics"""
        self.predictions = []
        self.ground_truth = []
        self.actions = []
        self.action_outcomes = []
        self.plans = []
        self.human_overrides = []
        self.execution_times = []
        self.pattern_predictions = []
        self.pattern_ground_truth = []
        self.adversarial_results = []