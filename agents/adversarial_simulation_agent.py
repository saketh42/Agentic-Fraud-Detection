"""
Adversarial Simulation Agent - Knowledge phase of MAPE-K
Generates SAFE boolean toggles of features for robustness testing.
STRICT: No fraud instructions, no scam text generation.
"""
from .base import BaseAgent, AgentResult


class AdversarialSimulationAgent(BaseAgent):
    def __init__(self):
        super().__init__("AdversarialSimulationAgent")

    def run(self, state: dict) -> AgentResult:
        self.log("Generating adversarial variants...")

        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        scoring = state.get('scoring', {})
        fraud_score = scoring.get('fraud_score', 0.0)

        variants = []
        variant_scores = []

        for label in active_labels[:3]:
            variant = {'type': 'toggle_label', 'original': label, 'new_value': 0}
            var_score = max(0, fraud_score - 0.15)
            variants.append(variant)
            variant_scores.append(var_score)

        if 'urgency' in active_tactics:
            variant = {
                'type': 'swap_tactic',
                'original': 'urgency',
                'new_tactic': 'reward',
                'description': 'Swapped urgency for reward tactic'
            }
            variants.append(variant)
            variant_scores.append(max(0, fraud_score - 0.05))
        if 'reward' in active_tactics:
            variant = {
                'type': 'swap_tactic',
                'original': 'reward',
                'new_tactic': 'urgency',
                'description': 'Swapped reward for urgency tactic'
            }
            variants.append(variant)
            variant_scores.append(min(1.0, fraud_score + 0.05))

        if active_tactics:
            variant = {
                'type': 'remove_tactic',
                'original': active_tactics[0],
                'new_value': 0,
                'description': f"Removed {active_tactics[0]} tactic"
            }
            variants.append(variant)
            variant_scores.append(max(0, fraud_score - 0.10))

        if active_labels:
            variant = {
                'type': 'flip_labels',
                'original': list(active_labels),
                'new_value': [],
                'description': 'All fraud labels set to 0'
            }
            variants.append(variant)
            variant_scores.append(max(0, fraud_score - 0.25))

        score_drop = fraud_score - (sum(variant_scores) / len(variant_scores)) if variant_scores else 0
        is_robust = score_drop < 0.15

        if is_robust:
            adversarial_risk = 'LOW'
            robustness_summary = f"Score drops by {score_drop:.3f} — model is robust to feature toggles"
        elif score_drop < 0.25:
            adversarial_risk = 'MEDIUM'
            robustness_summary = f"Score drops by {score_drop:.3f} — moderate sensitivity to feature changes"
        else:
            adversarial_risk = 'HIGH'
            robustness_summary = f"Score drops by {score_drop:.3f} — high sensitivity, review decision logic"

        hardening = []
        if not is_robust:
            hardening.append("Add additional verification for single-indicator fraud")
            hardening.append("Implement ensemble scoring with multiple signal sources")
            if score_drop > 0.2:
                hardening.append("Consider retraining with adversarially augmented data")

        result = {
            'generated_variants': variants,
            'adversarial_risk': adversarial_risk,
            'robustness_summary': robustness_summary,
            'hardening_recommendations': hardening,
            'original_score': round(fraud_score, 4),
            'variant_scores': [round(s, 4) for s in variant_scores],
            'score_drop': round(score_drop, 4),
            'is_robust': is_robust
        }

        self.log(f"Adversarial risk: {adversarial_risk}, robust: {is_robust}, drop: {score_drop:.3f}")
        return AgentResult(
            success=True, data=result,
            message=f"Adversarial simulation: {adversarial_risk} risk",
            metrics={'risk': adversarial_risk, 'robust': is_robust, 'drop': score_drop}
        )
