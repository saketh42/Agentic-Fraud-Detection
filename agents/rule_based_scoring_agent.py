"""
Rule-Based Scoring Agent - Analyze phase of MAPE-K
Applies transparent weighted rules for explainable fraud scoring.
"""
from .base import BaseAgent, AgentResult


FRAUD_LABEL_WEIGHTS = {
    'credential_phishing': 0.25,
    'social_authority_scam': 0.25,
    'social_urgency_scam': 0.20,
    'commerce_fake_seller': 0.20,
    'transaction_upi_fraud': 0.15,
    'transaction_card_fraud': 0.15,
    'transaction_bank_transfer': 0.20,
    'commerce_nondelivery': 0.15,
    'meta_victim_story': 0.10,
    'meta_fraud_question': 0.05
}

TACTIC_WEIGHTS = {
    'urgency': 0.15,
    'fear': 0.15,
    'authority': 0.10,
    'reward': 0.10
}

FEATURE_WEIGHTS = {
    'impersonated_entity': 0.10,
    'payment_method': 0.05,
    'fraud_channel': 0.05,
    'request_type': 0.05,
    'victim_action': 0.05,
    'urgency_level': 0.10
}


class RuleBasedScoringAgent(BaseAgent):
    """
    Applies transparent weighted rules for explainable fraud scoring.
    
    Provides:
    - fraud_score: Weighted rule score [0, 1]
    - model_confidence: Based on feature coverage
    - risk_features: Breakdown of contributing factors
    - rule_breakdown: Transparent explanation
    """
    
    def __init__(self):
        super().__init__("RuleBasedScoringAgent")
        self.label_weights = FRAUD_LABEL_WEIGHTS
        self.tactic_weights = TACTIC_WEIGHTS
        self.feature_weights = FEATURE_WEIGHTS
    
    def run(self, state: dict) -> AgentResult:
        self.log("Computing rule-based fraud score...")
        
        extraction = state.get('extraction', {})
        active_labels = extraction.get('active_fraud_labels', [])
        active_tactics = extraction.get('active_psychological_tactics', [])
        key_features = extraction.get('active_key_features', {})
        tactic_values = extraction.get('tactic_values', {})
        
        if not active_labels and not active_tactics:
            return AgentResult(
                success=False,
                message="No features to score"
            )
        
        # Calculate weighted scores
        label_score = self._calculate_label_score(active_labels)
        tactic_score = self._calculate_tactic_score(active_tactics, tactic_values)
        feature_score = self._calculate_feature_score(key_features)
        
        # Combined score (weighted average)
        fraud_score = (0.5 * label_score + 0.3 * tactic_score + 0.2 * feature_score)
        fraud_score = min(1.0, max(0.0, fraud_score))
        
        # Confidence based on feature coverage
        covered_features = len(active_labels) + len(active_tactics)
        model_confidence = min(1.0, covered_features / 5.0)
        
        # Risk features breakdown
        risk_features = []
        for label in active_labels:
            weight = self.label_weights.get(label, 0.0)
            risk_features.append({
                'feature': label,
                'weight': weight,
                'contribution': weight * label_score
            })
        
        for tactic in active_tactics:
            weight = self.tactic_weights.get(tactic, 0.0)
            risk_features.append({
                'feature': tactic,
                'weight': weight,
                'contribution': weight * tactic_score
            })
        
        # Risk level
        if fraud_score >= 0.75:
            risk_level = 'HIGH'
        elif fraud_score >= 0.50:
            risk_level = 'MEDIUM'
        elif fraud_score >= 0.25:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        scoring = {
            'fraud_score': fraud_score,
            'model_confidence': model_confidence,
            'risk_level': risk_level,
            'label_score': label_score,
            'tactic_score': tactic_score,
            'feature_score': feature_score,
            'risk_features': risk_features,
            'rule_breakdown': {
                'label_contribution': label_score * 0.5,
                'tactic_contribution': tactic_score * 0.3,
                'feature_contribution': feature_score * 0.2
            }
        }
        
        self.log("Fraud score: " + str(round(fraud_score, 3)) + " (risk: " + risk_level + ")")
        
        return AgentResult(
            success=True,
            data=scoring,
            message="Rule-based scoring complete",
            metrics={
                'fraud_score': fraud_score,
                'model_confidence': model_confidence,
                'risk_level': risk_level
            }
        )
    
    def _calculate_label_score(self, active_labels: list) -> float:
        """Calculate score from active fraud labels"""
        if not active_labels:
            return 0.0
        
        total_weight = 0.0
        for label in active_labels:
            total_weight += self.label_weights.get(label, 0.0)
        
        return min(1.0, total_weight)
    
    def _calculate_tactic_score(self, active_tactics: list, tactic_values: dict) -> float:
        """Calculate score from psychological tactics"""
        if not active_tactics:
            return 0.0
        
        total_weight = 0.0
        for tactic in active_tactics:
            # Use both presence and intensity
            weight = self.tactic_weights.get(tactic, 0.0)
            intensity = tactic_values.get(tactic, 1.0)
            total_weight += weight * intensity
        
        return min(1.0, total_weight)
    
    def _calculate_feature_score(self, key_features: dict) -> float:
        """Calculate score from key features"""
        if not key_features:
            return 0.0
        
        total_weight = 0.0
        for feature, value in key_features.items():
            if value is not None and value != 'unknown':
                total_weight += self.feature_weights.get(feature, 0.0)
        
        return min(1.0, total_weight * 2)