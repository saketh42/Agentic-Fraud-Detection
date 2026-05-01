"""
Training Agent - Plan phase of MAPE-K
Trains classifier with fixed learning rate, but tracks learning adaptation.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score
)
from .base import BaseAgent, AgentResult


class TrainingAgent(BaseAgent):
    """
    Trains a classifier on balanced data.
    
    Tracks learning adaptation across runs to demonstrate:
    - How quickly model adapts to new data patterns
    - How performance changes under drift conditions
    - The learning rate of the system itself (not model LR)
    """
    
    def __init__(self,
                 model_type: str = "gradient_boosting",
                 adversarial_training: bool = True,
                 fgsm_epsilon: float = 0.05,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100):
        super().__init__("TrainingAgent")
        self.model_type = model_type
        self.adversarial_training = adversarial_training
        self.fgsm_epsilon = fgsm_epsilon
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        
        # Fixed learning rate (not adaptive)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        # Track learning adaptation
        self.performance_history = []
        self.adaptation_rate = None
    
    def run(self, state: dict) -> AgentResult:
        self.log("Training classifier...")
        
        data = state.get("balanced_data")
        target_col = state.get("target_col", "is_fraud")
        feature_cols = state.get("feature_cols", [])
        
        if data is None or len(data) == 0:
            return AgentResult(
                success=False,
                message="No balanced data provided to TrainingAgent"
            )
        
        if len(feature_cols) == 0:
            feature_cols = [c for c in data.columns if c != target_col]
        
        # Prepare features and target
        # Convert to DataFrame for easier preprocessing
        feature_df = data[feature_cols].copy()
        
        # Encode string columns to numeric
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                # Label encode string columns
                codes = pd.Categorical(feature_df[col]).codes
                feature_df[col] = codes
        
        # Convert entire DataFrame to numeric
        X = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        y = data[target_col].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.log(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Generate adversarial samples if enabled
        if self.adversarial_training:
            self.log(f"Generating FGSM adversarial samples (ε={self.fgsm_epsilon})...")
            X_train_adv, y_train_adv = self._fgsm_augment(X_train, y_train)
            X_train = np.vstack([X_train, X_train_adv])
            y_train = np.concatenate([y_train, y_train_adv])
            self.log(f"Augmented training set: {len(X_train)} samples")
        
        # Train model with fixed learning rate
        self.model = self._create_model()
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        
        # === TRACK LEARNING ADAPTATION ===
        # Get prior performance from state (if available)
        prior_f1 = state.get("prior_f1", None)
        drift_detected = state.get("drift_detected", False)
        
        # Calculate adaptation metrics
        if prior_f1 is not None:
            # Performance change
            f1_change = f1 - prior_f1
            
            # Adaptation rate: how fast model improved from prior
            # If drift detected and F1 improved = fast adaptation
            if drift_detected:
                if f1 >= prior_f1:
                    self.adaptation_rate = "fast_adaptation"
                    adaptation_msg = "Model adapted quickly to new data pattern"
                else:
                    self.adaptation_rate = "slow_adaptation"
                    adaptation_msg = "Model struggling with new pattern"
            else:
                self.adaptation_rate = "stable_learning"
                adaptation_msg = "Model in stable learning phase"
            
            self.log(f"Learning adaptation: {adaptation_msg} (F1: {prior_f1:.4f} -> {f1:.4f})")
        else:
            f1_change = 0.0
            self.adaptation_rate = "initial_training"
            adaptation_msg = "First training run"
        
        # Track performance history
        self.performance_history.append({
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
            "f1_change": f1_change,
            "drift_detected": drift_detected,
            "adaptation_rate": self.adaptation_rate
        })
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "adversarial_training": self.adversarial_training,
            "adversarial_samples": len(X_train) - len(X_test) if self.adversarial_training else 0,
            # Learning adaptation tracking
            "prior_f1": prior_f1,
            "f1_change": float(f1_change),
            "adaptation_rate": self.adaptation_rate,
            "performance_history": self.performance_history,
            "learning_rate": self.learning_rate,  # Fixed LR
            "n_estimators": self.n_estimators
        }
        
        self.log(f"Test F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return AgentResult(
            success=True,
            data={
                "model": self.model,
                "feature_cols": feature_cols,
                "test_features": X_test,
                "test_labels": y_test,
                "training_report": metrics,
                "current_f1": f1  # Store for next run
            },
            message=f"Model trained with F1={f1:.4f}, adaptation={self.adaptation_rate}",
            metrics=metrics
        )
    
    def _create_model(self):
        """Create the classifier"""
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=5,
                random_state=self.random_state
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                random_state=self.random_state
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
    
    def _fgsm_augment(self, X: np.ndarray, y: np.ndarray, feature_cols: list = None) -> tuple:
        """
        Generate FGSM adversarial samples.
        Creates perturbed versions of samples with label flipping.
        Only perturbs numeric features, leaves categorical features unchanged.
        """
        X_adv = []
        y_adv = []
        
        # Determine which features are numeric (assuming feature_cols is passed)
        # For now, try to detect numeric features by checking if the column contains strings
        for i in range(len(X)):
            features = X[i].copy()
            perturbation = np.random.uniform(-self.fgsm_epsilon, self.fgsm_epsilon, X[i].shape)
            
            # Only apply perturbation to numeric features (try to detect)
            # If feature is string-like, don't perturb
            for j in range(len(features)):
                if isinstance(features[j], (np.str_, str)):
                    perturbation[j] = 0  # Don't perturb string features
            
            adv_sample = features + perturbation
            X_adv.append(adv_sample)
            y_adv.append(1 - y[i])
        
        return np.array(X_adv), np.array(y_adv)


def fgsm_attack(model, X: np.ndarray, epsilon: float = 0.05) -> np.ndarray:
    """Generate FGSM adversarial examples for evaluation"""
    X_adv = X.copy()
    for i in range(len(X)):
        noise = np.random.uniform(-epsilon, epsilon, X[i].shape)
        X_adv[i] = X[i] + noise
    return X_adv