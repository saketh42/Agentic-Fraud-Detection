"""
Decision Agent - LLM-based Decision Node for MAPE-K Pipeline
Uses Ollama Llama model to reason over system state and KB history.
"""
import json
import requests
from .base import BaseAgent, AgentResult


class DecisionAgent(BaseAgent):
    """
    Decision agent that uses LLM to make contextual decisions.
    
    Replaces hardcoded threshold-based rules with LLM reasoning
    over system state and knowledge base history.
    """
    
    def __init__(self,
                 model: str = "llama3",
                 mock_mode: bool = True,
                 ollama_url: str = "http://localhost:11434",
                 max_tokens: int = 512):
        super().__init__("DecisionAgent")
        self.model = model
        self.mock_mode = mock_mode
        self.ollama_url = ollama_url
        self.max_tokens = max_tokens
    
    def run(self, state: dict) -> AgentResult:
        self.log("Making decision via LLM...")
        
        drift_detected = state.get('drift_detected', False)
        metrics = state.get('evaluation_metrics', {})
        prior_f1 = state.get('prior_f1', None)
        iteration = state.get('supervisor_iterations', 1)
        max_iterations = state.get('supervisor_max', 3)
        kb_recent = state.get('kb_recent', [])
        
        if self.mock_mode:
            return self._mock_decision(drift_detected, metrics, prior_f1, iteration, max_iterations)
        
        prompt = self._build_prompt(
            drift_detected=drift_detected,
            metrics=metrics,
            prior_f1=prior_f1,
            iteration=iteration,
            max_iterations=max_iterations,
            kb_recent=kb_recent
        )
        
        try:
            decision = self._call_ollama(prompt)
            self.log(f"LLM Decision: {decision.get('action')} - {decision.get('reason')}")
            
            return AgentResult(
                success=True,
                data={
                    "decision": decision
                },
                message=f"Decision: {decision.get('action')}",
                metrics=decision
            )
        except Exception as e:
            self.log(f"Ollama error: {e}, falling back to mock decision")
            return self._mock_decision(drift_detected, metrics, prior_f1, iteration, max_iterations)
    
    def _build_prompt(self, drift_detected: bool, metrics: dict, prior_f1: float,
                    iteration: int, max_iterations: int, kb_recent: list) -> str:
        """Build prompt for LLM with system context."""
        
        f1 = metrics.get('f1', 0.0)
        roc_auc = metrics.get('roc_auc', 0.0)
        is_robust = metrics.get('is_robust', False)
        
        kb_summary = ""
        if kb_recent:
            runs = [f"Run {i+1}: F1={m.get('f1', 0):.3f}" for i, m in enumerate(kb_recent[-3:])]
            kb_summary = ", ".join(runs)
        
        prompt = f"""You are a MAPE-K supervisor for a fraud detection system.
Given the following system state, make a decision:

- Drift Detected: {drift_detected}
- Current Metrics: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Robust={is_robust}
- Prior F1: {prior_f1}
- Iteration: {iteration}/{max_iterations}
- KB History: {kb_summary}

Choose ONE action:
1. "deploy" - Model is acceptable, deploy it
2. "retrain_L2" - Retrain model with same strategy
3. "retrain_new_strategy" - Retrain with different hyperparameters
4. "stop" - Unstable, stop iterations

Respond ONLY in this exact JSON format:
{{"action": "deploy|retrain_L2|retrain_new_strategy|stop", "reason": "brief explanation", "params": {{}}}}"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> dict:
        """Call Ollama API and parse response."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        result = response.json()
        text = result.get('response', '').strip()
        
        try:
            decision = json.loads(text)
            if 'action' not in decision:
                decision = self._parse_fallback(text)
            return decision
        except json.JSONDecodeError:
            return self._parse_fallback(text)
    
    def _parse_fallback(self, text: str) -> dict:
        """Parse LLM response if JSON parsing fails."""
        text = text.lower()
        
        if 'deploy' in text:
            return {"action": "deploy", "reason": "Model passes quality thresholds"}
        elif 'stop' in text:
            return {"action": "stop", "reason": "System unstable"}
        elif 'new_strategy' in text:
            return {"action": "retrain_new_strategy", "reason": "Need different approach"}
        else:
            return {"action": "retrain_L2", "reason": "Model needs retraining"}
    
    def _mock_decision(self, drift_detected: bool, metrics: dict, prior_f1: float,
                      iteration: int, max_iterations: int) -> AgentResult:
        """Mock decision - replicates original hardcoded logic for reproducibility."""
        
        f1 = metrics.get('f1', 0.0)
        roc_auc = metrics.get('roc_auc', 0.0)
        robustness = metrics.get('is_robust', False)
        
        passed = (f1 >= 0.70 and roc_auc >= 0.75 and robustness)
        
        if passed and not drift_detected:
            action = "deploy"
            reason = "Model meets all thresholds and no drift detected"
        elif iteration >= max_iterations:
            action = "stop"
            reason = "Max iterations reached"
        elif prior_f1 and f1 > 0 and (f1 - prior_f1) < 0.01:
            action = "stop"
            reason = "Converged - minimal improvement"
        elif f1 < 0.30 and iteration >= 2:
            action = "stop"
            reason = "Unstable - very poor F1"
        elif drift_detected:
            action = "retrain_new_strategy"
            reason = "Drift detected - need new strategy"
        else:
            action = "retrain_L2"
            reason = "Metrics below thresholds - retrain"
        
        self.log("Mock Decision: " + action + " - " + reason)
        
        return AgentResult(
            success=True,
            data={
                "decision": {
                    "action": action,
                    "reason": reason,
                    "params": {}
                }
            },
            message=f"Decision: {action}",
            metrics={"action": action, "reason": reason}
        )