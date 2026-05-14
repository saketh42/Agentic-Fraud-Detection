"""
LLM Reasoning Agent - Analyze phase of MAPE-K
Uses Ollama (or mock mode) for contextual reasoning over fraud signals.
Returns JSON only — no tool execution, no chain-of-thought.
"""
import json
import requests
from .base import BaseAgent, AgentResult


class LLMReasoningAgent(BaseAgent):
    def __init__(self, model: str = "llama3", mock_mode: bool = False,
                 ollama_url: str = "http://localhost:11434"):
        super().__init__("LLMReasoningAgent")
        self.model = model
        self.mock_mode = mock_mode
        self.ollama_url = ollama_url

    def run(self, state: dict) -> AgentResult:
        extraction = state.get('extraction', {})
        scoring = state.get('scoring', {})
        context = state.get('context', {})
        pattern = state.get('pattern_learning', {})

        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        fraud_score = scoring.get('fraud_score', 0.0)
        risk_level = scoring.get('risk_level', 'LOW')
        detected_pattern = pattern.get('pattern_name', 'UNKNOWN')
        is_emerging = pattern.get('is_emerging', False)

        if self.mock_mode:
            return self._mock_reasoning(active_labels, active_tactics, fraud_score,
                                       risk_level, detected_pattern, is_emerging, context)

        prompt = self._build_prompt(active_labels, active_tactics, fraud_score,
                                   risk_level, detected_pattern, is_emerging, context)
        try:
            response = self._call_ollama(prompt)
            data = json.loads(response) if isinstance(response, str) else response
            validated = self._validate_output(data)
            return AgentResult(
                success=True, data=validated,
                message=f"LLM reasoning: {validated.get('risk_level')}",
                metrics={'risk_level': validated.get('risk_level'), 'pattern': validated.get('fraud_pattern')}
            )
        except Exception as e:
            self.log(f"Ollama error: {e}, falling back to mock")
            return self._mock_reasoning(active_labels, active_tactics, fraud_score,
                                       risk_level, detected_pattern, is_emerging, context)

    def _build_prompt(self, labels: list, tactics: list, score: float,
                     risk: str, pattern: str, emerging: bool, context: dict) -> str:
        evidence_list = context.get('evidence', [])
        evidence_str = "; ".join(
            e.get('description', str(e)) if isinstance(e, dict) else str(e)
            for e in evidence_list[:3]
        )

        prompt = f"""Assess this fraud case. Return ONLY valid JSON with these exact keys:
- risk_level: string (LOW/MEDIUM/HIGH/CRITICAL)
- final_risk_score: float (0.0 to 1.0)
- reasoning_summary: string (brief analysis)
- fraud_pattern: string (pattern name)
- adversarial_risk: string (LOW/MEDIUM/HIGH)
- recommended_next_step: string

Input signals:
- Labels: {labels}
- Tactics: {tactics}
- Score: {score}
- Rule risk: {risk}
- Pattern: {pattern}
- Emerging: {emerging}
- Evidence: {evidence_str}

JSON:"""
        return prompt

    def _call_ollama(self, prompt: str) -> dict:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "format": "json"}
        resp = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
        if resp.status_code != 200:
            raise Exception(f"Ollama error: {resp.status_code}")
        text = resp.json().get('response', '').strip()
        return json.loads(text) if text else {}

    def _validate_output(self, data: dict) -> dict:
        required = ['risk_level', 'final_risk_score', 'reasoning_summary',
                    'evidence', 'fraud_pattern', 'adversarial_risk', 'recommended_next_step']
        for key in required:
            if key not in data:
                data[key] = self._default_value(key)
        return data

    def _default_value(self, key: str):
        defaults = {
            'risk_level': 'MEDIUM',
            'final_risk_score': 0.5,
            'reasoning_summary': 'LLM output incomplete, using defaults',
            'evidence': [],
            'fraud_pattern': 'UNKNOWN',
            'adversarial_risk': 'MEDIUM',
            'recommended_next_step': 'TRIGGER_MFA'
        }
        return defaults.get(key, '')

    def _mock_reasoning(self, labels: list, tactics: list, score: float,
                       risk: str, pattern: str, emerging: bool, context: dict) -> AgentResult:
        has_phishing = 'credential_phishing' in labels
        has_authority = 'social_authority_scam' in labels
        has_urgency = 'urgency' in tactics
        has_fear = 'fear' in tactics
        has_reward = 'reward' in tactics

        fraud_pattern = 'hybrid'
        if has_phishing and has_urgency:
            fraud_pattern = 'phishing'
        elif has_authority and has_fear:
            fraud_pattern = 'social_engineering'
        elif has_reward:
            fraud_pattern = 'commerce_scam'

        contexts = context.get('evidence', [])
        evidence = []
        for c in contexts:
            if isinstance(c, dict):
                evidence.append(c.get('description', str(c)))
            else:
                evidence.append(str(c))

        label_descriptions = {
            'credential_phishing': 'Credential harvesting attempt',
            'social_authority_scam': 'Authority figure impersonation',
            'social_urgency_scam': 'Urgency-based social engineering',
            'transaction_bank_transfer': 'Unauthorized bank transfer request',
            'transaction_upi_fraud': 'UPI payment fraud',
            'transaction_card_fraud': 'Card payment fraud',
            'commerce_fake_seller': 'Fake seller / marketplace scam',
            'commerce_nondelivery': 'Non-delivery of goods',
            'meta_victim_story': 'Victim story emotional manipulation',
            'meta_fraud_question': 'Fraudulent verification question'
        }
        reasoning_parts = [label_descriptions.get(l, l.replace('_', ' ')) for l in labels]
        reasoning_parts.extend([f"{t} tactic detected" for t in tactics])
        reasoning_summary = "; ".join(reasoning_parts) if reasoning_parts else "Minimal fraud indicators"

        if score >= 0.75:
            adv_risk = 'HIGH'
        elif score >= 0.50:
            adv_risk = 'MEDIUM'
        else:
            adv_risk = 'LOW'

        if risk == 'HIGH' and (has_authority or has_urgency):
            next_step = 'CREATE_MANUAL_REVIEW_CASE'
        elif score >= 0.92:
            next_step = 'BLOCK_TRANSACTION'
        elif risk == 'MEDIUM':
            next_step = 'TRIGGER_MFA'
        else:
            next_step = 'ALLOW_TRANSACTION'

        reasoning = {
            'risk_level': risk,
            'final_risk_score': round(score, 4),
            'reasoning_summary': reasoning_summary,
            'evidence': evidence,
            'fraud_pattern': fraud_pattern,
            'adversarial_risk': adv_risk,
            'recommended_next_step': next_step
        }

        self.log(f"Mock reasoning — risk: {risk}, pattern: {fraud_pattern}, adv_risk: {adv_risk}")
        return AgentResult(
            success=True, data=reasoning,
            message=f"LLM reasoning: {risk}",
            metrics={'risk_level': risk, 'pattern': fraud_pattern}
        )
