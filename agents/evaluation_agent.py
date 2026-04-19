"""
Evaluation Agent - Analyze phase of MAPE-K
Evaluates model performance and robustness.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report
)
from .base import BaseAgent, AgentResult

class EvaluationAgent(BaseAgent):
    """
    Evaluates trained model on test data.
    
    Computes standard metrics (accuracy, F1, ROC-AUC) and
    robustness metrics under adversarial attacks.
    """
    
    def __init__(self,
                 robustness_thresholds: dict = None):
        super().__init__("EvaluationAgent")
        self.robustness_thresholds = robustness_thresholds or {
            "min_f1": 0.70,
            "min_roc_auc": 0.75,
            "min_robustness": 0.60
        }
    
    def run(self, state: dict) -> AgentResult:
        self.log("Evaluating model performance...")
        
        model = state.get("model")
        X_test = state.get("test_features")
        y_test = state.get("test_labels")
        
        if model is None or X_test is None or y_test is None:
            return AgentResult(
                success=False,
                message="Model or test data not provided"
            )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Standard metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        })
        
        self.log(f"Clean F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Adversarial robustness test
        robustness = self._test_robustness(model, X_test, y_test)
        metrics.update(robustness)
        
        self.log(f"Robustness: F1_drop={robustness.get('f1_drop', 0):.4f}, "
                 f"robust={robustness.get('is_robust', False)}")
        
        # Pass/fail based on thresholds
        passed = (
            metrics["f1"] >= self.robustness_thresholds["min_f1"] and
            metrics["roc_auc"] >= self.robustness_thresholds["min_roc_auc"] and
            robustness.get("is_robust", False)
        )
        
        return AgentResult(
            success=True,
            data={
                "evaluation_metrics": metrics,
                "passed": passed
            },
            message=f"Evaluation complete - {'PASSED' if passed else 'FAILED'}",
            metrics=metrics
        )
    
    def _test_robustness(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Test model robustness under FGSM attack"""
        from .training_agent import fgsm_attack
        
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
        robustness_scores = []
        
        for eps in epsilons:
            if eps == 0.0:
                X_adv = X_test
            else:
                X_adv = fgsm_attack(model, X_test, epsilon=eps)
            
            y_pred = model.predict(X_adv)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            robustness_scores.append({"epsilon": eps, "f1": f1})
        
        clean_f1 = robustness_scores[0]["f1"]
        worst_f1 = min(s["f1"] for s in robustness_scores)
        avg_f1 = np.mean([s["f1"] for s in robustness_scores])
        
        f1_drop = clean_f1 - worst_f1
        is_robust = (avg_f1 / clean_f1) >= self.robustness_thresholds["min_robustness"]
        
        return {
            "robustness_curve": robustness_scores,
            "clean_f1": clean_f1,
            "worst_f1": worst_f1,
            "avg_f1": avg_f1,
            "f1_drop": f1_drop,
            "is_robust": is_robust
        }