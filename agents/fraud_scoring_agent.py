"""
Fraud Scoring Agent - Analyze phase of MAPE-K
Rule-based scoring with exact weights from specification.
"""
from .base import BaseAgent, AgentResult


FRAUD_LABEL_WEIGHTS = {
    'credential_phishing': 0.25,
    'social_authority_scam': 0.25,
    'social_urgency_scam': 0.20,
    'commerce_fake_seller': 0.20,
    'commerce_nondelivery': 0.15,
    'transaction_bank_transfer': 0.15,
    'transaction_upi_fraud': 0.15,
    'transaction_card_fraud': 0.15,
}

TACTIC_WEIGHTS = {
    'urgency': 0.15,
    'fear': 0.15,
    'authority': 0.10,
    'reward': 0.10,
}

FEATURE_WEIGHTS = {
    'impersonated_entity': 0.10,
    'victim_action': 0.10,
    'urgency_level': 0.05,
    'amount_mentioned': 0.05,
}

META_WEIGHTS = {
    'meta_victim_story': 0.10,
    'meta_fraud_question': 0.05,
}


class FraudScoringAgent(BaseAgent):
    def __init__(self):
        super().__init__("FraudScoringAgent")

    def run(self, state: dict) -> AgentResult:
        self.log("Computing rule-based fraud score...")

        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        active_features = extraction.get('active_key_features', {})
        tactic_values = extraction.get('tactic_values', {})

        label_score = self._score_labels(active_labels)
        tactic_score = self._score_tactics(active_tactics, tactic_values)
        feature_score = self._score_features(active_features)
        meta_score = self._score_meta(active_labels)
        synergy = self._synergy_bonus(active_labels, active_tactics, active_features)

        fraud_score = label_score + tactic_score + feature_score + meta_score + synergy
        fraud_score = min(1.0, max(0.0, fraud_score))

        covered = len(active_labels) + len(active_tactics)
        model_confidence = min(1.0, covered / 5.0)

        risk_features = []
        for l in active_labels:
            w = FRAUD_LABEL_WEIGHTS.get(l, 0) + META_WEIGHTS.get(l, 0)
            risk_features.append({'feature': l, 'weight': w, 'contribution': w})
        for t in active_tactics:
            w = TACTIC_WEIGHTS.get(t, 0)
            risk_features.append({'feature': t, 'weight': w, 'contribution': w * tactic_values.get(t, 1.0)})
        if synergy > 0:
            risk_features.append({'feature': 'signal_synergy', 'weight': synergy, 'contribution': synergy})

        if fraud_score >= 0.75:
            risk_level = 'HIGH'
        elif fraud_score >= 0.50:
            risk_level = 'MEDIUM'
        elif fraud_score >= 0.25:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'

        scoring = {
            'fraud_score': round(fraud_score, 4),
            'model_confidence': round(model_confidence, 4),
            'risk_level': risk_level,
            'label_score': round(label_score, 4),
            'tactic_score': round(tactic_score, 4),
            'feature_score': round(feature_score, 4),
            'meta_score': round(meta_score, 4),
            'synergy_bonus': round(synergy, 4),
            'risk_features': risk_features,
            'rule_breakdown': {
                'label_contribution': label_score,
                'tactic_contribution': tactic_score,
                'feature_contribution': feature_score,
                'meta_contribution': meta_score,
                'synergy_contribution': synergy,
            }
        }

        self.log(f"Fraud score: {fraud_score:.3f} (risk: {risk_level})")
        return AgentResult(
            success=True,
            data=scoring,
            message="Fraud scoring complete",
            metrics={'fraud_score': fraud_score, 'risk_level': risk_level}
        )

    def _score_labels(self, labels: list) -> float:
        return min(1.0, sum(FRAUD_LABEL_WEIGHTS.get(l, 0.0) for l in labels))

    def _score_tactics(self, tactics: list, values: dict) -> float:
        return min(1.0, sum(TACTIC_WEIGHTS.get(t, 0.0) * values.get(t, 1.0) for t in tactics))

    NON_INDICATIVE = {'', 'unknown', 'none', 'no', 'low', 'minimal', 'false', '0', 'null'}

    def _score_features(self, features: dict) -> float:
        score = 0.0
        for feat, val in features.items():
            if val is not None and str(val).lower().strip() not in self.NON_INDICATIVE:
                score += FEATURE_WEIGHTS.get(feat, 0.0)
        return min(1.0, score)

    def _score_meta(self, labels: list) -> float:
        return min(1.0, sum(META_WEIGHTS.get(l, 0.0) for l in labels))

    def _synergy_bonus(self, labels: list, tactics: list, features: dict) -> float:
        bonus = 0.0
        has_labels = len(labels) >= 1
        has_tactics = len(tactics) >= 1
        impersonation = features.get('impersonated_entity', 'unknown')
        has_impersonation = str(impersonation).lower().strip() not in self.NON_INDICATIVE

        if has_labels and has_tactics:
            bonus += 0.10

        if has_labels and has_impersonation:
            bonus += 0.10

        if has_labels and has_tactics and has_impersonation:
            bonus += 0.05

        return min(0.30, bonus)
