"""
MAPE-K Pipeline Orchestrator
Runs all agents in sequence matching the specified architecture.
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    MonitorAgent, FeatureExtractionAgent, ContextRetrievalAgent,
    FraudScoringAgent, PatternLearningAgent,
    AdversarialSimulationAgent, PlanningAgent, ExecuteAgent,
    FeedbackAgent, KnowledgeStore
)


class MAPEKPipeline:
    def __init__(self, db_path: str = "knowledge_store.db"):
        self.db_path = db_path
        self.knowledge = KnowledgeStore(db_path)

        self.monitor = MonitorAgent()
        self.extractor = FeatureExtractionAgent()
        self.context = ContextRetrievalAgent(db_path)
        self.scorer = FraudScoringAgent()
        self.pattern_learner = PatternLearningAgent(db_path)
        self.adversarial = AdversarialSimulationAgent()
        self.planner = PlanningAgent()
        self.executor = ExecuteAgent(db_path)
        self.feedback = FeedbackAgent(db_path)

        self.logs = []

    def log(self, msg: str):
        entry = f"[{datetime.now().isoformat()}] {msg}"
        self.logs.append(entry)
        print(entry)

    def process_transaction(self, transaction: dict) -> dict:
        self.log("=" * 60)
        self.log(f"Processing transaction: {transaction.get('transaction_id', 'unknown')}")
        self.log("=" * 60)

        state = {'transaction': transaction}

        # [M] Monitor
        self.log("[M] Monitor Agent")
        result = self.monitor.run(state)
        if not result.success:
            return self._error_response(result.message)
        state.update(result.data)

        # [A] Feature Extraction
        self.log("[A] Feature Extraction Agent")
        result = self.extractor.run(state)
        if not result.success:
            return self._error_response(result.message)
        state['extraction'] = result.data
        state.update(result.data)

        # [A] Context Retrieval
        self.log("[A] Context Retrieval Agent")
        result = self.context.run(state)
        state.update(result.data)

        # [A] Fraud Scoring
        self.log("[A] Fraud Scoring Agent")
        result = self.scorer.run(state)
        if not result.success:
            return self._error_response(result.message)
        state.update({'scoring': result.data})

        # [A] Pattern Learning
        self.log("[A] Pattern Learning Agent")
        result = self.pattern_learner.run(state)
        if result.success:
            state.update({'pattern_learning': result.data})

        # [K] Adversarial Simulation
        self.log("[K] Adversarial Simulation Agent")
        result = self.adversarial.run(state)
        if result.success:
            state.update({'adversarial_simulation': result.data})

        # [P] Planning
        self.log("[P] Planning Agent")
        result = self.planner.run(state)
        if not result.success:
            return self._error_response(result.message)
        state.update({'plan': result.data})

        # [E] Execute
        self.log("[E] Execute Agent")
        result = self.executor.run(state)
        if not result.success:
            return self._error_response(result.message)
        state.update({'execution': result.data})

        # [K] Memory Store
        self.log("[K] Updating Knowledge Store")
        memory_ok = self._store_in_memory(state)

        response = self._build_response(state, memory_ok)
        self.log("=" * 60)
        scoring = state.get('scoring', {})
        fraud_score = scoring.get('fraud_score', 0)
        is_fraud = fraud_score > 0.5
        self.log(f"Transaction {state.get('masked_transaction_id', 'unknown')} complete")
        self.log(f"  Verdict: {'FRAUD' if is_fraud else 'NOT FRAUD'}")
        self.log(f"  Risk: {scoring.get('risk_level', 'UNKNOWN')}")
        self.log(f"  Score: {fraud_score:.3f}")
        self.log(f"  Actions: {state.get('plan', {}).get('actions', [])}")
        self.log("=" * 60)

        return response

    def _store_in_memory(self, state: dict) -> bool:
        try:
            txn_id = state.get('masked_transaction_id', 'unknown')
            extraction = state.get('extraction', {})
            scoring = state.get('scoring', {})
            pattern = state.get('pattern_learning', {})
            adversarial = state.get('adversarial_simulation', {})

            self.knowledge.store_transaction(
                transaction_id=txn_id,
                is_fraud=1 if scoring.get('fraud_score', 0) > 0.5 else 0,
                semantic_profile=extraction.get('semantic_profile', 'unknown'),
                label_count=extraction.get('label_count', 0),
                tactic_count=extraction.get('tactic_count', 0),
                raw_data=transaction if (transaction := state.get('transaction', {})) else {}
            )

            self.knowledge.store_prediction(
                transaction_id=txn_id,
                predicted_prob=scoring.get('fraud_score', 0),
                predicted_label='FRAUD' if scoring.get('fraud_score', 0) > 0.5 else 'CLEAN',
                confidence=scoring.get('model_confidence', 0),
                risk_level=scoring.get('risk_level', 'LOW')
            )

            if pattern.get('detected_pattern'):
                self.knowledge.store_pattern(
                    pattern_name=pattern['detected_pattern'],
                    pattern_type=pattern.get('pattern_type', 'UNKNOWN'),
                    is_emerging=pattern.get('is_emerging_pattern', False)
                )

            if adversarial.get('generated_variants'):
                for v in adversarial.get('generated_variants', []):
                    self.knowledge.store_adversarial_variant(
                        original_id=txn_id,
                        variant_type=v.get('type', 'unknown'),
                        original_score=adversarial.get('original_score', 0),
                        variant_score=adversarial.get('variant_scores', [0])[0] if adversarial.get('variant_scores') else 0,
                        is_robust=adversarial.get('is_robust', True)
                    )

            return True
        except Exception as e:
            self.log(f"Memory store error: {e}")
            return False

    def _build_response(self, state: dict, memory_ok: bool) -> dict:
        scoring = state.get('scoring', {})
        fraud_score = scoring.get('fraud_score', 0)
        is_fraud = fraud_score > 0.5
        return {
            "transaction_id": state.get('masked_transaction_id', 'unknown'),
            "verdict": "FRAUD" if is_fraud else "NOT FRAUD",
            "is_fraud": is_fraud,
            "fraud_score": fraud_score,
            "feature_profile": state.get('extraction', {}),
            "prediction": scoring,
            "pattern_learning": state.get('pattern_learning', {}),
            "adversarial_simulation": state.get('adversarial_simulation', {}),
            "plan": state.get('plan', {}),
            "execution": state.get('execution', {}).get('execution_results', []),
            "memory_update": "SUCCESS" if memory_ok else "FAILED"
        }

    def _error_response(self, message: str) -> dict:
        self.log(f"ERROR: {message}")
        return {"error": message, "status": "failed"}
