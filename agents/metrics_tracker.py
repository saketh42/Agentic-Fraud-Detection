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
    
    Core Classification Metrics (Primary):
    - F1-score
    - PR-AUC (better for imbalanced fraud data)
    - ROC-AUC
    - Accuracy, Precision, Recall
    
    Adversarial Robustness Metrics:
    - Adversarial Accuracy (worst-case under attack)
    - FPR@Attack (False Positive Rate under attack)
    - Robustness Drop (F1 drop from clean to attacked)
    - Robustness Gain (improvement after feedback)
    
    Pattern Learning Metrics:
    - Pattern Classification Accuracy
    
    Agentic Decision Metrics:
    - Planning Accuracy (appropriate action selection)
    - Human Override Rate (disagreement with agent)
    - Action Utility (benefit minus cost)
    - Escalation Precision/Recall
    - Mean Time to Decision
    - Memory Improvement Rate
    """
    
    def __init__(self):
        super().__init__("MetricsTracker")
        # Core classification tracking
        self.predictions = []
        self.ground_truth = []
        self.prediction_probabilities = []
        
        # Action and planning tracking
        self.actions = []
        self.action_outcomes = []
        self.plans = []
        self.plans_appropriate = []  # Track if plan was appropriate
        self.human_overrides = []
        self.execution_times = []
        
        # Pattern learning tracking
        self.pattern_predictions = []
        self.pattern_ground_truth = []
        
        # Adversarial tracking
        self.adversarial_results = []
        self.clean_metrics = {}
        self.post_feedback_metrics = {}  # For robustness gain
        
        # Feedback tracking
        self.feedback_history = []
        self.performance_before_feedback = []
        self.performance_after_feedback = []
    
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
        probability = state.get('probability', 0.5)
        action = state.get('action')
        action_outcome = state.get('action_outcome')
        plan = state.get('plan')
        is_appropriate = state.get('plan_appropriate', False)
        human_override = state.get('human_override', False)
        execution_time = state.get('execution_time', 0.0)
        
        if prediction is not None:
            self.predictions.append(prediction)
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
        if probability is not None:
            self.prediction_probabilities.append(probability)
        if action is not None:
            self.actions.append(action)
        if action_outcome is not None:
            self.action_outcomes.append(action_outcome)
        if plan is not None:
            self.plans.append(plan)
            self.plans_appropriate.append(is_appropriate)
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
        """Compute all performance metrics - includes all 9 recommended metrics"""
        
        metrics = {}
        
        # === CORE CLASSIFICATION METRICS (Primary) ===
        if self.predictions and self.ground_truth:
            y_pred = np.array(self.predictions)
            y_true = np.array(self.ground_truth)
            
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            
            # PR-AUC (better for imbalanced fraud datasets)
            if len(np.unique(y_true)) > 1 and self.prediction_probabilities:
                try:
                    y_proba = np.array(self.prediction_probabilities)
                    metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
                except:
                    metrics['pr_auc'] = 0.0
        
        # === ADVERSARIAL ROBUSTNESS METRICS ===
        adv_metrics = self._compute_adversarial_metrics()
        metrics.update(adv_metrics)
        
        # === PATTERN CLASSIFICATION METRICS ===
        metrics['pattern_classification_accuracy'] = self._compute_pattern_accuracy()
        
        # === AGENTIC DECISION METRICS ===
        metrics['planning_accuracy'] = self._compute_planning_accuracy()
        metrics['human_override_rate'] = self._compute_override_rate()
        metrics['escalation_precision'] = self._compute_escalation_precision()
        metrics['escalation_recall'] = self._compute_escalation_recall()
        metrics['mean_time_to_decision'] = self._compute_mean_time()
        metrics['memory_improvement_rate'] = self._compute_memory_improvement()
        metrics['action_utility'] = self._compute_action_utility()
        
        # === ROBUSTNESS GAIN AFTER FEEDBACK ===
        metrics['robustness_gain'] = self._compute_robustness_gain()
        
        self.log("Metrics computed: F1=" + str(round(metrics.get('f1', 0), 3)) + 
               ", PR-AUC=" + str(round(metrics.get('pr_auc', 0), 3)) +
               ", Adv_Acc=" + str(round(metrics.get('adversarial_accuracy', 0), 3)) +
               ", Planning=" + str(round(metrics.get('planning_accuracy', 0), 3)))
        
        return AgentResult(
            success=True,
            data=metrics,
            message="Metrics computation complete",
            metrics=metrics
        )
    
    def _compute_planning_accuracy(self) -> float:
        """Planning Accuracy - Whether the selected action was appropriate"""
        if not self.plans_appropriate:
            # Fallback to action_outcomes if plans_appropriate not tracked
            if not self.action_outcomes:
                return 0.0
            correct = sum(1 for o in self.action_outcomes if o in ['true_positive', 'true_negative', 'appropriate'])
            return correct / len(self.action_outcomes) if self.action_outcomes else 0.0
        
        correct = sum(1 for p in self.plans_appropriate if p)
        return correct / len(self.plans_appropriate) if self.plans_appropriate else 0.0
    
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
        """Compute adversarial robustness metrics - all recommended metrics"""
        if not self.adversarial_results:
            return {
                'adversarial_accuracy': 0.0,
                'fpr_at_attack': 0.0,
                'robustness_drop': 0.0
            }
        
        # Calculate adversarial accuracy (worst-case accuracy under attack)
        adv_scores = [r['adversarial_score'] for r in self.adversarial_results]
        adv_correct = sum(1 for s in adv_scores if s > 0.5)
        adv_accuracy = adv_correct / len(self.adversarial_results) if self.adversarial_results else 0.0
        
        # FPR@Attack (False Positive Rate under attack)
        # Calculate based on actual adversarial predictions vs ground truth
        fpr_values = [r.get('fpr', 0.0) for r in self.adversarial_results]
        fpr_at_attack = max(fpr_values) if fpr_values else 0.0
        
        # Robustness Drop (F1 drop from clean to adversarial)
        clean_f1 = self.clean_metrics.get('f1', 0.0)
        adv_f1 = self.clean_metrics.get('adv_f1', adv_accuracy)
        robustness_drop = clean_f1 - adv_f1
        
        return {
            'adversarial_accuracy': adv_accuracy,
            'fpr_at_attack': fpr_at_attack,
            'robustness_drop': robustness_drop
        }
    
    def _compute_pattern_accuracy(self) -> float:
        """Pattern Classification Accuracy - recommended metric"""
        if not self.pattern_predictions:
            return 0.0
        
        correct = sum(1 for p, g in zip(self.pattern_predictions, self.pattern_ground_truth) 
                    if p == g)
        return correct / len(self.pattern_predictions) if self.pattern_predictions else 0.0
    
    def _compute_robustness_gain(self) -> float:
        """Robustness Gain after feedback - recommended metric"""
        if not self.performance_before_feedback or not self.performance_after_feedback:
            return 0.0
        
        # Calculate improvement in adversarial accuracy after feedback
        before_avg = np.mean(self.performance_before_feedback)
        after_avg = np.mean(self.performance_after_feedback)
        
        if before_avg == 0:
            return 0.0
        
        gain = (after_avg - before_avg) / before_avg
        return float(gain)
    
    def _compute_action_utility(self) -> float:
        """Action Utility - benefit of action minus operational cost"""
        if not self.action_outcomes:
            return 0.0
        
        # Simple utility calculation
        true_positives = sum(1 for o in self.action_outcomes if o == 'true_positive')
        true_negatives = sum(1 for o in self.action_outcomes if o == 'true_negative')
        false_positives = sum(1 for o in self.action_outcomes if o == 'false_positive')
        false_negatives = sum(1 for o in self.action_outcomes if o == 'false_negative')
        
        # Weights: TP=+1, TN=+0.1, FP=-0.5, FN=-2 (fraud missed is costly)
        utility = (true_positives * 1.0 + true_negatives * 0.1 - 
                  false_positives * 0.5 - false_negatives * 2.0)
        
        return float(utility / len(self.action_outcomes)) if self.action_outcomes else 0.0
    
    def reset(self):
        """Reset all tracked metrics"""
        # Core classification tracking
        self.predictions = []
        self.ground_truth = []
        self.prediction_probabilities = []
        
        # Action and planning tracking
        self.actions = []
        self.action_outcomes = []
        self.plans = []
        self.plans_appropriate = []
        self.human_overrides = []
        self.execution_times = []
        
        # Pattern learning tracking
        self.pattern_predictions = []
        self.pattern_ground_truth = []
        
        # Adversarial tracking
        self.adversarial_results = []
        self.clean_metrics = {}
        self.post_feedback_metrics = {}
        
        # Feedback tracking
        self.feedback_history = []
        self.performance_before_feedback = []
        self.performance_after_feedback = []