"""
Planning Agent - Plan phase of MAPE-K
Selects actions based on fraud score, risk level, pattern, and adversarial risk.
"""
from .base import BaseAgent, AgentResult


PLANNING_RULES = [
    {'condition': 'score >= 0.92', 'actions': ['BLOCK_TRANSACTION']},
    {'condition': 'risk == HIGH', 'actions': ['TRIGGER_MFA', 'CREATE_MANUAL_REVIEW_CASE']},
    {'condition': 'risk == MEDIUM', 'actions': ['TRIGGER_MFA']},
    {'condition': 'risk == LOW', 'actions': ['ALLOW_TRANSACTION']},
    {'condition': 'emerging_pattern', 'actions': ['RUN_DRIFT_CHECK']},
    {'condition': 'adversarial_risk == HIGH', 'actions': ['TRIGGER_RETRAINING_REVIEW']},
    {'condition': 'hybrid_and_authority_and_urgency', 'actions': ['TRIGGER_MFA', 'CREATE_MANUAL_REVIEW_CASE']},
]


class PlanningAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlanningAgent")

    def run(self, state: dict) -> AgentResult:
        self.log("Planning actions...")

        scoring = state.get('scoring', {})
        reasoning = state.get('llm_reasoning', {})
        pattern = state.get('pattern_learning', {})
        adversarial = state.get('adversarial_simulation', {})

        fraud_score = scoring.get('fraud_score', 0.0)
        risk_level = reasoning.get('risk_level', scoring.get('risk_level', 'LOW'))
        is_emerging = pattern.get('is_emerging_pattern', pattern.get('is_emerging', False))
        adversarial_risk = adversarial.get('adversarial_risk', reasoning.get('adversarial_risk', 'LOW'))

        active_labels = pattern.get('active_labels', state.get('extraction', {}).get('active_fraud_labels', []))
        active_tactics = pattern.get('active_tactics', state.get('extraction', {}).get('active_psychological_tactics', []))

        is_hybrid = pattern.get('detected_pattern', '') == 'HYBRID_SOCIAL_ENGINEERING'
        has_authority = 'social_authority_scam' in active_labels or 'authority' in active_tactics
        has_urgency = 'social_urgency_scam' in active_labels or 'urgency' in active_tactics

        actions = set()

        if fraud_score >= 0.92:
            actions.add('BLOCK_TRANSACTION')

        if risk_level == 'HIGH':
            actions.add('TRIGGER_MFA')
            actions.add('CREATE_MANUAL_REVIEW_CASE')
        elif risk_level == 'MEDIUM':
            actions.add('TRIGGER_MFA')
        elif risk_level == 'LOW':
            actions.add('ALLOW_TRANSACTION')

        if is_emerging:
            actions.add('RUN_DRIFT_CHECK')

        if adversarial_risk == 'HIGH':
            actions.add('TRIGGER_RETRAINING_REVIEW')

        if is_hybrid and has_authority and has_urgency:
            actions.add('TRIGGER_MFA')
            actions.add('CREATE_MANUAL_REVIEW_CASE')

        if not actions:
            actions.add('ALLOW_TRANSACTION')

        justification_parts = []
        if 'BLOCK_TRANSACTION' in actions:
            justification_parts.append(f"Score {fraud_score:.3f} >= 0.92 threshold")
        if 'TRIGGER_MFA' in actions:
            justification_parts.append(f"Risk level is {risk_level}")
        if 'CREATE_MANUAL_REVIEW_CASE' in actions:
            justification_parts.append(f"Additional verification needed for risk level {risk_level}")
        if 'RUN_DRIFT_CHECK' in actions:
            justification_parts.append("Emerging pattern detected — checking for drift")
        if 'TRIGGER_RETRAINING_REVIEW' in actions:
            justification_parts.append(f"Adversarial risk is {adversarial_risk}")
        if 'ALLOW_TRANSACTION' in actions:
            justification_parts.append("Risk level is LOW — allowing transaction")

        plan = {
            'actions': sorted(list(actions)),
            'justification': '; '.join(justification_parts),
            'fraud_score': fraud_score,
            'risk_level': risk_level,
            'is_emerging_pattern': is_emerging,
            'adversarial_risk': adversarial_risk,
            'planning_rules_applied': [
                'score >= 0.92 -> BLOCK' if fraud_score >= 0.92 else None,
                f'risk == {risk_level} -> {risk_level} actions',
                'emerging -> DRIFT_CHECK' if is_emerging else None,
                f'adversarial risk == {adversarial_risk} -> RETRAIN' if adversarial_risk == 'HIGH' else None,
                'hybrid + authority + urgency -> MFA + REVIEW' if is_hybrid and has_authority and has_urgency else None,
            ]
        }

        self.log(f"Plan: {plan['actions']}")
        return AgentResult(
            success=True, data=plan,
            message=f"Planned actions: {plan['actions']}",
            metrics={'action_count': len(actions), 'primary_action': plan['actions'][0] if plan['actions'] else 'NONE'}
        )
