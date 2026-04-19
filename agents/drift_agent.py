"""
Drift Agent - Monitor phase of MAPE-K
Detects concept drift using Population Stability Index (PSI) and Kolmogorov-Smirnov test.
"""
import numpy as np
import pandas as pd
from scipy import stats
from .base import BaseAgent, AgentResult

class DriftAgent(BaseAgent):
    """
    Detects data drift between reference and current distributions.
    
    Uses PSI (Population Stability Index) for categorical/summary drift
    and KS test for continuous feature drift.
    """
    
    def __init__(self, 
                 psi_threshold: float = 0.20,
                 ks_threshold: float = 0.05,
                 reference_data: pd.DataFrame = None):
        super().__init__("DriftAgent")
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.reference_data = reference_data
        self.drift_history = []
    
    def _compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index"""
        eps = 1e-4
        ref_min, ref_max = reference.min(), reference.max()
        
        if ref_min == ref_max:
            return 0.0
        
        breakpoints = np.linspace(ref_min, ref_max, bins + 1)
        ref_counts = np.histogram(reference, bins=breakpoints)[0] + eps
        cur_counts = np.histogram(current, bins=breakpoints)[0] + eps
        
        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()
        
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    
    def _compute_ks(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic"""
        ks_stat, _ = stats.ks_2samp(reference, current)
        return ks_stat
    
    def run(self, state: dict) -> AgentResult:
        self.log("Checking for concept drift...")
        
        current_data = state.get("data")
        
        if current_data is None or len(current_data) == 0:
            return AgentResult(
                success=False,
                message="No data provided to DriftAgent"
            )
        
        # Use provided reference or first data as reference
        if self.reference_data is None and state.get("reference_data") is not None:
            self.reference_data = state["reference_data"]
        
        if self.reference_data is None:
            # First run - set reference and report no drift
            self.reference_data = current_data.copy()
            self.log("First run - setting reference data")
            return AgentResult(
                success=True,
                data={
                    "drift_detected": False,
                    "reference_data": self.reference_data,
                    "psi_score": 0.0,
                    "ks_score": 0.0
                },
                message="Reference established",
                metrics={"drift": 0.0}
            )
        
        # Compute drift metrics
        feature_cols = [c for c in current_data.columns if c != state.get("target_col", "is_fraud")]
        
        psi_scores = []
        ks_scores = []
        
        for col in feature_cols:
            if current_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                ref_vals = self.reference_data[col].dropna().values
                cur_vals = current_data[col].dropna().values
                
                if len(ref_vals) > 10 and len(cur_vals) > 10:
                    psi = self._compute_psi(ref_vals, cur_vals)
                    ks = self._compute_ks(ref_vals, cur_vals)
                    psi_scores.append(psi)
                    ks_scores.append(ks)
        
        avg_psi = np.mean(psi_scores) if psi_scores else 0.0
        avg_ks = np.mean(ks_scores) if ks_scores else 0.0
        
        drift_detected = (avg_psi > self.psi_threshold) or (avg_ks > self.ks_threshold)
        
        # Store in history
        self.drift_history.append({
            "psi": avg_psi,
            "ks": avg_ks,
            "drift_detected": drift_detected
        })
        
        self.log(f"Drift check: PSI={avg_psi:.4f}, KS={avg_ks:.4f}, Detected={drift_detected}")
        
        return AgentResult(
            success=True,
            data={
                "drift_detected": drift_detected,
                "psi_score": avg_psi,
                "ks_score": avg_ks,
                "drift_history": self.drift_history
            },
            message=f"Drift {'detected' if drift_detected else 'not detected'}",
            metrics={
                "psi": avg_psi,
                "ks": avg_ks,
                "threshold_psi": self.psi_threshold,
                "threshold_ks": self.ks_threshold
            }
        )


def run_drift_detection(data: pd.DataFrame, reference: pd.DataFrame = None) -> dict:
    """Standalone function for drift detection"""
    agent = DriftAgent(reference_data=reference)
    result = agent.run({"data": data})
    return result.metrics if result.success else {"error": result.message}