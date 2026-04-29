"""
Feature Extraction Agent - Analyze phase of MAPE-K
Parses transaction data to extract fraud labels, key features, and psychological tactics.
"""
import pandas as pd
import numpy as np
from .base import BaseAgent, AgentResult


FRAUD_LABELS = [
    'transaction_upi_fraud',
    'transaction_card_fraud',
    'transaction_bank_transfer',
    'commerce_nondelivery',
    'commerce_fake_seller',
    'credential_phishing',
    'social_authority_scam',
    'social_urgency_scam',
    'meta_victim_story',
    'meta_fraud_question'
]

KEY_FEATURES = [
    'payment_method',
    'fraud_channel',
    'victim_action',
    'request_type',
    'impersonated_entity',
    'currency',
    'urgency_level'
]

PSYCHOLOGICAL_TACTICS = [
    'urgency',
    'fear',
    'authority',
    'reward'
]

PATTERN_WEIGHTS = {
    'credential_phishing': 0.25,
    'social_authority_scam': 0.25,
    'social_urgency_scam': 0.20,
    'commerce_fake_seller': 0.20,
    'transaction_upi_fraud': 0.15,
    'transaction_card_fraud': 0.15,
    'transaction_bank_transfer': 0.20,
    'commerce_nondelivery': 0.15,
    'meta_victim_story': 0.10,
    'meta_fraud_question': 0.05,
    'urgency': 0.15,
    'fear': 0.15,
    'authority': 0.10,
    'reward': 0.10
}


class FeatureExtractionAgent(BaseAgent):
    """
    Extracts fraud indicators from transaction data.
    
    Produces:
    - active_fraud_labels: List of fraud types detected
    - active_key_features: Behavioral indicators
    - active_psychological_tactics: Manipulation techniques
    - semantic_profile: Combined fraud signature
    - label_count, tactic_count
    """
    
    def __init__(self):
        super().__init__("FeatureExtractionAgent")
        self.fraud_labels = FRAUD_LABELS
        self.key_features = KEY_FEATURES
        self.tactics = PSYCHOLOGICAL_TACTICS
    
    def run(self, state: dict) -> AgentResult:
        self.log("Extracting fraud features from transaction...")
        
        data = state.get('data')
        
        if data is None:
            return AgentResult(
                success=False,
                message="No data provided"
            )
        
        if isinstance(data, pd.DataFrame):
            return self._extract_from_dataframe(data, state)
        else:
            return self._extract_from_dict(data, state)
    
    def _extract_from_dataframe(self, data: pd.DataFrame, state: dict) -> AgentResult:
        """Extract features from DataFrame"""
        
        # Find target column
        target_col = state.get('target_col', 'is_fraud')
        
        # Identify column mappings
        label_cols = [c for c in data.columns if any(label in c.lower() for label in self.fraud_labels)]
        feature_cols = [c for c in data.columns if 'key_features' in c.lower() or any(f in c.lower() for f in self.key_features)]
        tactic_cols = [c for c in data.columns if 'psychological' in c.lower() or any(t in c.lower() for t in self.tactics)]
        
        # Find active labels (where value = 1)
        active_labels = []
        for col in data.columns:
            for label in self.fraud_labels:
                if label in col.lower() and data[col].dtype in ['int64', 'float64', 'int32']:
                    if data[col].sum() > 0:
                        active_labels.append(label)
        
        # Extract key features
        key_features_dict = {}
        for col in data.columns:
            if 'key_features' in col.lower():
                prefix = col.split('.')[-1] if '.' in col else col
                key_features_dict[prefix] = data[col].mode().iloc[0] if len(data[col]) > 0 else None
        
        # Extract psychological tactics
        tactic_values = {}
        for col in data.columns:
            if 'psychological' in col.lower():
                tactic_name = col.split('.')[-1] if '.' in col else col
                tactic_values[tactic_name] = float(data[col].mean()) if data[col].dtype in ['float64', 'int64'] else 0.0
        
        # Active tactics (where value > threshold)
        active_tactics = [t for t, v in tactic_values.items() if v > 0.3]
        
        label_count = len(active_labels)
        tactic_count = len(active_tactics)
        
        # Generate semantic profile
        if label_count >= 3:
            semantic_profile = "high_indicator_multi_fraud"
        elif label_count >= 1 and tactic_count >= 2:
            semantic_profile = "combined_psychological_pressure"
        elif label_count >= 1:
            semantic_profile = "single_indicator_fraud"
        else:
            semantic_profile = "low_indicator_transaction"
        
        # Risk signals
        risk_signals = []
        if 'credential_phishing' in active_labels:
            risk_signals.append('credential_targeted')
        if 'social_authority_scam' in active_labels:
            risk_signals.append('authority_impersonation')
        if 'urgency' in active_tactics:
            risk_signals.append('time_pressure')
        if 'fear' in active_tactics:
            risk_signals.append('emotional_manipulation')
        
        extraction = {
            'active_fraud_labels': list(set(active_labels)),
            'active_key_features': key_features_dict,
            'active_psychological_tactics': active_tactics,
            'semantic_profile': semantic_profile,
            'label_count': label_count,
            'tactic_count': tactic_count,
            'tactic_values': tactic_values,
            'risk_signals': risk_signals,
            'pattern_weights': PATTERN_WEIGHTS
        }
        
        self.log("Extracted " + str(label_count) + " labels, " + str(tactic_count) + " tactics")
        self.log("Profile: " + semantic_profile)
        
        return AgentResult(
            success=True,
            data=extraction,
            message="Feature extraction complete",
            metrics={
                'label_count': label_count,
                'tactic_count': tactic_count,
                'semantic_profile': semantic_profile
            }
        )
    
    def _extract_from_dict(self, data: dict, state: dict) -> AgentResult:
        """Extract features from single transaction dict"""
        
        active_labels = []
        for label in self.fraud_labels:
            if data.get(label, 0) == 1:
                active_labels.append(label)
        
        active_tactics = []
        for tactic in self.tactics:
            if data.get(tactic, 0) > 0.3:
                active_tactics.append(tactic)
        
        key_features_dict = {k: data.get(k) for k in self.key_features if k in data}
        
        label_count = len(active_labels)
        tactic_count = len(active_tactics)
        
        if label_count >= 3:
            semantic_profile = "high_indicator_multi_fraud"
        elif label_count >= 1 and tactic_count >= 2:
            semantic_profile = "combined_psychological_pressure"
        elif label_count >= 1:
            semantic_profile = "single_indicator_fraud"
        else:
            semantic_profile = "low_indicator_transaction"
        
        extraction = {
            'active_fraud_labels': active_labels,
            'active_key_features': key_features_dict,
            'active_psychological_tactics': active_tactics,
            'semantic_profile': semantic_profile,
            'label_count': label_count,
            'tactic_count': tactic_count,
            'pattern_weights': PATTERN_WEIGHTS
        }
        
        return AgentResult(
            success=True,
            data=extraction,
            message="Feature extraction complete"
        )