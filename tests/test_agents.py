import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from agents import (
    MonitorAgent, FeatureExtractionAgent, FraudScoringAgent,
    PatternLearningAgent, AdversarialSimulationAgent,
    PlanningAgent, ExecuteAgent, LLMReasoningAgent
)

SAMPLE_TXN = {
    "transaction_id": "TXN-001-test",
    "is_fraud": None,
    "fraud_type": None,
    "transaction_upi_fraud": 0,
    "transaction_card_fraud": 0,
    "transaction_bank_transfer": 0,
    "commerce_nondelivery": 0,
    "commerce_fake_seller": 0,
    "credential_phishing": 1,
    "social_authority_scam": 1,
    "social_urgency_scam": 0,
    "meta_victim_story": 0,
    "meta_fraud_question": 0,
    "payment_method": "credit_card",
    "fraud_channel": "email",
    "victim_action": "click_link",
    "request_type": "verify_account",
    "impersonated_entity": "bank",
    "amount_mentioned": "yes",
    "currency": "USD",
    "urgency_level": "high",
    "amount_mentioned_value": 5000.0,
    "urgency": 0.8,
    "fear": 0.6,
    "authority": 0.7,
    "reward": 0.0,
}


class TestMonitorAgent:
    def setup_method(self):
        self.agent = MonitorAgent()

    def test_valid_transaction(self):
        result = self.agent.run({"transaction": SAMPLE_TXN})
        assert result.success
        assert 'masked_transaction_id' in result.data
        assert result.data['masked_transaction_id'] == "****test"
        assert result.data['validation_passed']

    def test_missing_id(self):
        result = self.agent.run({"transaction": {"credential_phishing": 1}})
        assert not result.success

    def test_missing_labels_defaulted(self):
        bad = {"transaction_id": "T1"}
        result = self.agent.run({"transaction": bad})
        assert result.success
        assert result.data['validation_passed']
        assert result.data['monitored_transaction']['credential_phishing'] == 0


class TestFeatureExtraction:
    def setup_method(self):
        self.agent = FeatureExtractionAgent()

    def test_extracts_labels(self):
        state = {'monitored_transaction': SAMPLE_TXN}
        result = self.agent.run(state)
        assert result.success
        assert 'credential_phishing' in result.data['active_fraud_labels']
        assert 'social_authority_scam' in result.data['active_fraud_labels']
        assert 'urgency' in result.data['active_psychological_tactics']
        assert result.data['label_count'] == 2

    def test_no_data(self):
        result = self.agent.run({})
        assert not result.success


class TestFraudScoring:
    def setup_method(self):
        self.agent = FraudScoringAgent()

    def test_phishing_score(self):
        extraction = {
            'active_fraud_labels': ['credential_phishing', 'social_authority_scam'],
            'active_psychological_tactics': ['urgency', 'fear', 'authority'],
            'active_key_features': {'impersonated_entity': 'bank', 'victim_action': 'click_link'},
            'tactic_values': {'urgency': 0.8, 'fear': 0.6, 'authority': 0.7, 'reward': 0.0}
        }
        result = self.agent.run({'extraction': extraction})
        assert result.success
        assert result.data['fraud_score'] > 0.3
        assert result.data['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_empty_extraction(self):
        extraction = {
            'active_fraud_labels': [],
            'active_psychological_tactics': [],
            'active_key_features': {},
            'tactic_values': {}
        }
        result = self.agent.run({'extraction': extraction})
        assert result.success
        assert result.data['fraud_score'] == 0.0
        assert result.data['risk_level'] == 'MINIMAL'


class TestLLMReasoning:
    def setup_method(self):
        self.agent = LLMReasoningAgent(mock_mode=True)

    def test_mock_reasoning(self):
        state = {
            'extraction': {
                'active_fraud_labels': ['credential_phishing', 'social_authority_scam'],
                'active_psychological_tactics': ['urgency', 'fear'],
            },
            'scoring': {'fraud_score': 0.7, 'risk_level': 'HIGH'},
            'pattern_learning': {'pattern_name': 'HYBRID_SOCIAL_ENGINEERING', 'is_emerging': False},
            'context': {'evidence': []}
        }
        result = self.agent.run(state)
        assert result.success
        assert result.data['risk_level'] in ('LOW', 'MEDIUM', 'HIGH')
        assert 'reasoning_summary' in result.data
        assert 'fraud_pattern' in result.data
        assert 'adversarial_risk' in result.data


class TestPatternLearning:
    def setup_method(self):
        self.agent = PatternLearningAgent(db_path=":memory:")

    def test_phishing_pattern(self):
        extraction = {
            'active_fraud_labels': ['credential_phishing'],
            'active_psychological_tactics': ['urgency'],
        }
        result = self.agent.run({'extraction': extraction})
        assert result.success
        assert result.data['detected_pattern'] is not None

    def test_no_signals(self):
        extraction = {
            'active_fraud_labels': [],
            'active_psychological_tactics': [],
        }
        result = self.agent.run({'extraction': extraction})
        assert not result.success


class TestAdversarialSimulation:
    def setup_method(self):
        self.agent = AdversarialSimulationAgent()

    def test_generates_variants(self):
        state = {
            'extraction': {
                'active_fraud_labels': ['credential_phishing'],
                'active_psychological_tactics': ['urgency', 'fear'],
            },
            'scoring': {'fraud_score': 0.65}
        }
        result = self.agent.run(state)
        assert result.success
        assert len(result.data['generated_variants']) > 0
        assert result.data['adversarial_risk'] in ('LOW', 'MEDIUM', 'HIGH')

    def test_no_fraud_instructions(self):
        state = {
            'extraction': {'active_fraud_labels': [], 'active_psychological_tactics': []},
            'scoring': {'fraud_score': 0.0}
        }
        result = self.agent.run(state)
        assert result.success
        for v in result.data['generated_variants']:
            assert 'description' not in v or 'scam' not in v.get('description', '').lower()


class TestPlanningAgent:
    def setup_method(self):
        self.agent = PlanningAgent()

    def test_low_risk_allow(self):
        state = {
            'scoring': {'fraud_score': 0.2, 'risk_level': 'LOW'},
            'llm_reasoning': {'risk_level': 'LOW'},
            'pattern_learning': {},
            'adversarial_simulation': {'adversarial_risk': 'LOW'},
            'extraction': {'active_fraud_labels': [], 'active_psychological_tactics': []}
        }
        result = self.agent.run(state)
        assert result.success
        assert 'ALLOW_TRANSACTION' in result.data['actions']

    def test_high_risk_block(self):
        state = {
            'scoring': {'fraud_score': 0.95, 'risk_level': 'HIGH'},
            'llm_reasoning': {'risk_level': 'HIGH'},
            'pattern_learning': {},
            'adversarial_simulation': {'adversarial_risk': 'HIGH'},
            'extraction': {'active_fraud_labels': [], 'active_psychological_tactics': []}
        }
        result = self.agent.run(state)
        assert result.success
        assert 'BLOCK_TRANSACTION' in result.data['actions']
        assert 'TRIGGER_MFA' in result.data['actions']
        assert 'CREATE_MANUAL_REVIEW_CASE' in result.data['actions']
        assert 'TRIGGER_RETRAINING_REVIEW' in result.data['actions']


class TestExecuteAgent:
    def setup_method(self):
        self.agent = ExecuteAgent(db_path=":memory:")

    def test_execute_actions(self):
        state = {
            'plan': {'actions': ['ALLOW_TRANSACTION', 'TRIGGER_MFA']},
            'masked_transaction_id': '****test'
        }
        result = self.agent.run(state)
        assert result.success
        assert len(result.data['execution_results']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
