"""
Critic Agent - Validates and Refines Decision Agent Proposals
Second LLM agent in the dual-agent architecture.
Creates interaction: Decision Agent proposes → Critic Agent evaluates → Final Action
"""
import json
import requests
from .base import BaseAgent, AgentResult


class CriticAgent(BaseAgent):
    """
    Critic Agent that evaluates and validates decisions from the Decision Agent.
    
    Role:
    - Receives: Decision Agent's proposal + original system state
    - Evaluates: appropriateness, risks, confidence
    - Outputs: approved/modified/rejected with reasoning
    
    This creates the key agentic interaction - two LLMs collaborating,
    not just sequential execution.
    """
    
    def __init__(self,
                 model: str = "llama3",
                 mock_evaluation: bool = False,
                 ollama_url: str = "http://localhost:11434",
                 max_tokens: int = 512):
        super().__init__("CriticAgent")
        self.model = model
        self.mock_evaluation = mock_evaluation
        self.ollama_url = ollama_url
        self.max_tokens = max_tokens

    def run(self, state: dict) -> AgentResult:
        self.log("Critic Agent evaluating Decision Agent proposal...")
        
        llm_decision = state.get('llm_decision', {})
        evaluation_metrics = state.get('evaluation_metrics', {})
        drift_detected = state.get('drift_detected', False)
        prior_f1 = state.get('prior_f1', 0.0)
        iteration = state.get('supervisor_iterations', 1)
        max_iterations = state.get('supervisor_max', 3)
        kb_recent = state.get('kb_recent', [])
        
        if self.mock_evaluation:
            return self._mock_critique(llm_decision, evaluation_metrics, drift_detected)
        
        prompt = self._build_critic_prompt(
            decision_proposal=llm_decision,
            metrics=evaluation_metrics,
            drift_detected=drift_detected,
            prior_f1=prior_f1,
            iteration=iteration,
            max_iterations=max_iterations,
            kb_history=kb_recent
        )
        
        try:
            critique = self._call_ollama(prompt)
            self.log(f"Critic Evaluation: {critique.get('approved')} - {critique.get('critic_reason')}")
            
            final_action = critique.get('modified_action') or llm_decision.get('action', 'deploy')
            
            return AgentResult(
                success=True,
                data={
                    "critic_evaluation": critique,
                    "decision_proposal": llm_decision,
                    "final_action": final_action,
                    "was_modified": critique.get('approved', True) is False
                },
                message=f"Critic: {critique.get('approved')}, Final: {final_action}",
                metrics={
                    "approved": critique.get('approved'),
                    "modified_action": critique.get('modified_action'),
                    "confidence": critique.get('confidence', 0.0),
                    "final_action": final_action
                }
            )
        except Exception as e:
            self.log(f"Critic LLM error: {e}, using decision proposal")
            return AgentResult(
                success=True,
                data={
                    "critic_evaluation": {"approved": True, "critic_reason": "LLM error, accepting proposal"},
                    "decision_proposal": llm_decision,
                    "final_action": llm_decision.get('action', 'deploy'),
                    "was_modified": False
                },
                message=f"Critic: fallback to {llm_decision.get('action')}",
                metrics={"approved": True, "final_action": llm_decision.get('action')}
            )

    def _build_critic_prompt(self, decision_proposal: dict, metrics: dict,
                        drift_detected: bool, prior_f1: float,
                        iteration: int, max_iterations: int,
                        kb_history: list) -> str:
        """Build prompt for Critic LLM evaluation."""
        
        f1 = metrics.get('f1', 0.0)
        roc_auc = metrics.get('roc_auc', 0.0)
        is_robust = metrics.get('is_robust', False)
        
        action = decision_proposal.get('action', 'unknown')
        reason = decision_proposal.get('reason', 'no reason provided')
        
        kb_summary = "No history"
        if kb_history:
            runs = [f"Run {i+1}: F1={m.get('f1', 0):.3f}" for i, m in enumerate(kb_history[-3:])]
            kb_summary = ", ".join(runs)
        
        prompt = f"""You are a Critic Agent for a MAPE-K fraud detection system.
Your role is to evaluate the Decision Agent's proposal and either approve, modify, or reject it.

ORIGINAL SYSTEM STATE:
- F1 Score: {f1:.4f}
- ROC-AUC: {roc_auc:.4f}
- Robustness: {is_robust}
- Drift Detected: {drift_detected}
- Prior F1: {prior_f1}
- Iteration: {iteration}/{max_iterations}
- KB History: {kb_summary}

DECISION AGENT PROPOSAL:
- Suggested Action: "{action}"
- Reasoning: "{reason}"

CRITIQUE CRITERIA:
1. Is the proposed action appropriate given the metrics?
2. Are there potential risks or issues?
3. Should we approve, modify, or reject?

Consider:
- If F1 > 0.90 and no drift, deploy is appropriate
- If drift detected and low F1, retrain_new_strategy may be better
- If marginal metrics, consider retrain_L2
- If iterations maxed out, stop or deploy current model

Respond ONLY with valid JSON (no markdown):
{{"approved": true or false, "modified_action": "deploy|retrain_L2|retrain_new_strategy|stop|null", "critic_reason": "brief 1-2 sentence explanation", "confidence": 0.0 to 1.0}}"""
        
        return prompt

    def _call_ollama(self, prompt: str) -> dict:
        """Call Ollama API and parse response."""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": self.max_tokens}
        }
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json=payload,
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Critic Ollama API error: {response.status_code}")
        
        result = response.json()
        text = result.get('response', '').strip()
        
        try:
            critique = json.loads(text)
            if 'approved' not in critique:
                critique = self._parse_fallback(text, critique)
            return critique
        except json.JSONDecodeError:
            self.log(f"JSON parse failed, using fallback")
            return self._parse_fallback(text, {})

    def _parse_fallback(self, text: str, partial: dict) -> dict:
        """Parse LLM response if JSON parsing fails."""
        text_lower = text.lower()
        
        approved = 'reject' not in text_lower and 'modify' not in text_lower
        modified_action = None
        
        if 'retrain' in text_lower and 'new' in text_lower:
            modified_action = 'retrain_new_strategy'
        elif 'retrain_l2' in text_lower:
            modified_action = 'retrain_l2'
        elif 'stop' in text_lower:
            modified_action = 'stop'
            approved = False
        elif 'deploy' in text_lower:
            modified_action = 'deploy'
        
        return {
            "approved": approved,
            "modified_action": modified_action,
            "critic_reason": f"Parsed from: {text[:100]}",
            "confidence": 0.7
        }

    def _mock_critique(self, decision_proposal: dict, metrics: dict, drift_detected: bool) -> AgentResult:
        """Mock critique - simple threshold-based validation."""
        
        f1 = metrics.get('f1', 0.0)
        roc_auc = metrics.get('roc_auc', 0.0)
        action = decision_proposal.get('action', 'deploy')
        
        passed = f1 >= 0.70 and roc_auc >= 0.75
        
        if passed:
            critique = {
                "approved": True,
                "modified_action": None,
                "critic_reason": "Metrics meet thresholds, approving Decision Agent proposal",
                "confidence": 0.85
            }
        else:
            critique = {
                "approved": False,
                "modified_action": "retrain_l2",
                "critic_reason": f"Low metrics (F1={f1:.2f}, ROC={roc_auc:.2f}), overriding to retrain",
                "confidence": 0.90
            }
        
        self.log(f"Mock Critic: {critique['approved']} - {critique['critic_reason']}")
        
        return AgentResult(
            success=True,
            data={
                "critic_evaluation": critique,
                "decision_proposal": decision_proposal,
                "final_action": critique.get('modified_action') or action,
                "was_modified": not critique['approved']
            },
            message=f"Critic: {critique['approved']}",
            metrics=critique
        )