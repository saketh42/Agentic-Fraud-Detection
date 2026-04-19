"""
Balance Agent - Execute phase of MAPE-K
Handles class imbalance using CTGAN for synthetic data generation.
"""
import numpy as np
import pandas as pd
from .base import BaseAgent, AgentResult

class BalanceAgent(BaseAgent):
    """
    Addresses class imbalance by generating synthetic minority samples.
    
    Uses CTGAN (Conditional Tabular GAN) from SDV to learn the distribution
    of minority class and generate realistic synthetic samples.
    """
    
    def __init__(self,
                 target_ratio: float = 0.3,
                 ctgan_epochs: int = 100,
                 minority_class: str = "non_fraud"):
        super().__init__("BalanceAgent")
        self.target_ratio = target_ratio
        self.ctgan_epochs = ctgan_epochs
        self.minority_class = minority_class
        self.synthetic_samples = None
    
    def run(self, state: dict) -> AgentResult:
        self.log("Analyzing class balance...")
        
        data = state.get("data")
        target_col = state.get("target_col", "is_fraud")
        feature_cols = state.get("feature_cols", [])
        
        if data is None or len(data) == 0:
            return AgentResult(
                success=False,
                message="No data provided to BalanceAgent"
            )
        
        # Analyze class distribution
        class_counts = data[target_col].value_counts()
        n_fraud = class_counts.get(1, 0)
        n_non_fraud = class_counts.get(0, 0)
        
        self.log(f"Original: fraud={n_fraud}, non_fraud={n_non_fraud}")
        
        if n_fraud == 0 or n_non_fraud == 0:
            return AgentResult(
                success=True,
                data={
                    "balanced_data": data,
                    "balance_report": {
                        "action": "skipped",
                        "reason": "single_class"
                    }
                },
                message="Single class present - no balancing needed"
            )
        
        ratio = n_fraud / n_non_fraud
        
        # Check if balancing needed
        if ratio >= self.target_ratio and ratio <= (1.0 / self.target_ratio):
            self.log(f"Class ratio {ratio:.2f} within threshold - no balancing needed")
            return AgentResult(
                success=True,
                data={
                    "balanced_data": data,
                    "balance_report": {
                        "action": "skipped",
                        "reason": "already_balanced",
                        "ratio": ratio
                    }
                },
                message="Data already balanced"
            )
        
        # Determine minority class
        minority_label = 0 if n_non_fraud < n_fraud else 1
        minority_df = data[data[target_col] == minority_label]
        
        # Calculate samples needed
        majority_count = n_fraud if n_non_fraud < n_fraud else n_non_fraud
        target_minority = int(majority_count * self.target_ratio)
        n_synthetic = max(0, target_minority - len(minority_df))
        
        self.log(f"Generating {n_synthetic} synthetic {minority_label} samples...")
        
        # Try CTGAN, fallback to noise injection
        synthetic = self._generate_synthetic(
            minority_df[feature_cols + [target_col]],
            n_synthetic,
            minority_label
        )
        
        if synthetic is not None and len(synthetic) > 0:
            # Combine original + synthetic
            balanced = pd.concat([data, synthetic], ignore_index=True)
            new_counts = balanced[target_col].value_counts()
            
            self.log(f"Balanced: fraud={new_counts.get(1, 0)}, non_fraud={new_counts.get(0, 0)}")
            
            return AgentResult(
                success=True,
                data={
                    "balanced_data": balanced,
                    "synthetic_samples": synthetic,
                    "balance_report": {
                        "action": "balanced",
                        "method": "CTGAN" if synthetic is not None else "fallback",
                        "original_fraud": n_fraud,
                        "original_non_fraud": n_non_fraud,
                        "synthetic_generated": len(synthetic) if synthetic is not None else 0,
                        "new_fraud": new_counts.get(1, 0),
                        "new_non_fraud": new_counts.get(0, 0)
                    }
                },
                message=f"Generated {len(synthetic) if synthetic is not None else 0} synthetic samples",
                metrics={
                    "synthetic_count": len(synthetic) if synthetic is not None else 0,
                    "new_ratio": new_counts.get(1, 0) / max(1, new_counts.get(0, 0))
                }
            )
        else:
            return AgentResult(
                success=True,
                data={
                    "balanced_data": data,
                    "balance_report": {
                        "action": "failed",
                        "error": "generation_failed"
                    }
                },
                message="Synthetic generation failed - using original data"
            )
    
    def _generate_synthetic(self, minority_data: pd.DataFrame, n_samples: int, label: int):
        """Try CTGAN first, fallback to noise injection"""
        
        try:
            from ctgan import CTGAN
            
            ctgan = CTGAN(epochs=self.ctgan_epochs, verbose=False)
            discrete_cols = [minority_data.columns[-1]]  # target column
            ctgan.fit(minority_data, discrete_columns=discrete_cols)
            
            synthetic = ctgan.sample(n_samples)
            synthetic[minority_data.columns[-1]] = label
            
            self.log("Generated synthetic samples via CTGAN")
            return synthetic
            
        except ImportError:
            self.log("CTGAN not available - using noise injection fallback")
            return self._noise_injection(minority_data, n_samples, label)
        except Exception as e:
            self.log(f"CTGAN failed: {e} - using noise injection fallback")
            return self._noise_injection(minority_data, n_samples, label)
    
    def _noise_injection(self, data: pd.DataFrame, n_samples: int, label: int):
        """Fallback: add small noise to existing minority samples"""
        if len(data) == 0:
            return None
        
        indices = np.random.choice(len(data), size=n_samples, replace=True)
        synthetic = data.iloc[indices].copy()
        
        # Add small noise to numeric features
        for col in synthetic.columns:
            if col != data.columns[-1]:  # skip target
                if synthetic[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    noise = np.random.normal(0, 0.01, size=len(synthetic))
                    synthetic[col] = synthetic[col].astype(float) + noise
        
        return synthetic