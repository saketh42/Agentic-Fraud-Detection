"""
MAPE-K Agentic Fraud Detection API
Real-time connection to MAPE-K Pipeline
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import sys
import os
import json
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

pipeline_state = {
    'model': None,
    'is_trained': False,
    'last_metrics': None,
    'reference_data': None,
    'drift_history': [],
    # Real-time states
    'rows_loaded': 0,
    'features_count': 0,
    'class_distribution': {},
    'psi_score': 0,
    'ks_score': 0,
    'drift_detected': False,
    'synthetic_added': 0,
    'balanced_count': 0,
    'train_samples': 0,
    'test_samples': 0,
    'last_trained': None
}

def load_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'data_binary_only_first_3000.csv')
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_full_pipeline():
    """Run complete MAPE-K pipeline with all agents"""
    print("Running MAPE-K pipeline...")
    
    df = load_data()
    if df is None:
        return None
    
    from agents.ingestion_agent import IngestionAgent
    from agents.drift_agent import DriftAgent
    from agents.balance_agent import BalanceAgent
    from agents.training_agent import TrainingAgent
    from agents.evaluation_agent import EvaluationAgent
    
    ingestion = IngestionAgent()
    drift = DriftAgent(psi_threshold=0.20, ks_threshold=0.05)
    balance = BalanceAgent(target_ratio=0.5, ctgan_epochs=100)
    training = TrainingAgent(model_type='gradient_boosting', adversarial_training=False)
    evaluation = EvaluationAgent()
    
    feature_cols = [c for c in df.columns if c != 'annotation.is_fraud']
    
    state = {
        'data': df,
        'target_col': 'annotation.is_fraud',
        'feature_cols': feature_cols
    }
    
    # 1. INGESTION
    result = ingestion.run(state)
    state.update(result.data)
    state['reference_data'] = df.copy()
    
    rows_loaded = len(df)
    features_count = len(feature_cols)
    class_dist = {
        'fraud': int(df['annotation.is_fraud'].sum()),
        'non_fraud': int((df['annotation.is_fraud'] == 0).sum())
    }
    
    # 2. DRIFT DETECTION
    result = drift.run(state)
    state.update(result.data)
    
    psi_score = state.get('psi_score', 0)
    ks_score = state.get('ks_score', 0)
    drift_detected = state.get('drift_detected', False)
    
    # 3. BALANCE
    result = balance.run(state)
    state.update(result.data)
    
    balanced_data = state.get('balanced_data', df)
    synthetic_added = len(balanced_data) - len(df)
    balanced_count = len(balanced_data)
    
# 4. TRAINING
    result = training.run(state)
    state['model'] = result.data.get('model')
    state['test_features'] = result.data.get('test_features')
    state['test_labels'] = result.data.get('test_labels')
    train_report = result.data.get('training_report', {})
    
    train_samples = train_report.get('train_size', 0)
    test_samples = train_report.get('test_size', 0)
    
    # 5. EVALUATION
    result = evaluation.run(state)
    eval_metrics = result.data.get('evaluation_metrics', {})
    passed = result.data.get('passed', False)
    
    # Update global state with ALL real values
    pipeline_state['model'] = state['model']
    pipeline_state['is_trained'] = True
    pipeline_state['last_metrics'] = eval_metrics
    pipeline_state['reference_data'] = df
    pipeline_state['rows_loaded'] = rows_loaded
    pipeline_state['features_count'] = features_count
    pipeline_state['class_distribution'] = class_dist
    pipeline_state['psi_score'] = psi_score
    pipeline_state['ks_score'] = ks_score
    pipeline_state['drift_detected'] = drift_detected
    pipeline_state['synthetic_added'] = synthetic_added
    pipeline_state['balanced_count'] = balanced_count
    pipeline_state['train_samples'] = train_samples
    pipeline_state['test_samples'] = test_samples
    pipeline_state['last_trained'] = datetime.now().isoformat()
    
    if drift_detected:
        pipeline_state['drift_history'].append({
            'timestamp': datetime.now().isoformat(),
            'psi': psi_score,
            'ks': ks_score
        })
    
    print(f"Pipeline complete. F1: {eval_metrics.get('f1', 0):.4f}, Drift: {drift_detected}")
    
    return {'metrics': eval_metrics, 'passed': passed}

def predict_fraud(transaction_data):
    """Make fraud prediction using trained model"""
    if not pipeline_state['is_trained'] or pipeline_state['model'] is None:
        return None
    
    model = pipeline_state['model']
    feature_cols = pipeline_state['reference_data'].columns.tolist()
    feature_cols = [c for c in feature_cols if c != 'annotation.is_fraud']
    
    features = {}
    
    features['annotation.fraud_labels.transaction_upi_fraud'] = float(transaction_data.get('transaction_upi_fraud', 0))
    features['annotation.fraud_labels.transaction_card_fraud'] = float(transaction_data.get('transaction_card_fraud', 0))
    features['annotation.fraud_labels.transaction_bank_transfer'] = float(transaction_data.get('transaction_bank_transfer', 0))
    features['annotation.fraud_labels.commerce_nondelivery'] = float(transaction_data.get('commerce_nondelivery', 0))
    features['annotation.fraud_labels.commerce_fake_seller'] = float(transaction_data.get('commerce_fake_seller', 0))
    features['annotation.fraud_labels.credential_phishing'] = float(transaction_data.get('credential_phishing', 0))
    features['annotation.fraud_labels.social_authority_scam'] = float(transaction_data.get('social_authority_scam', 0))
    features['annotation.fraud_labels.social_urgency_scam'] = float(transaction_data.get('social_urgency_scam', 0))
    features['annotation.fraud_labels.meta_victim_story'] = float(transaction_data.get('meta_victim_story', 0))
    features['annotation.fraud_labels.meta_fraud_question'] = float(transaction_data.get('meta_fraud_question', 0))
    
    features['annotation.key_features.urgency_level'] = float(transaction_data.get('urgency_level', transaction_data.get('urgency', 0)))
    features['annotation.psychological_tactics.urgency'] = float(transaction_data.get('urgency', 0))
    features['annotation.psychological_tactics.fear'] = float(transaction_data.get('fear', 0))
    features['annotation.psychological_tactics.authority'] = float(transaction_data.get('authority', 0))
    features['annotation.psychological_tactics.reward'] = float(transaction_data.get('reward', 0))
    features['annotation.key_features.amount_normalized'] = float(transaction_data.get('amount_normalized', 0))
    features['annotation.key_features.has_amount'] = float(transaction_data.get('has_amount', 0))
    
    for col in feature_cols:
        if col not in features:
            if col == 'annotation.fraud_type':
                features[col] = transaction_data.get('fraud_type', 'none')
            elif col == 'annotation.key_features.payment_method':
                features[col] = transaction_data.get('payment_method', 'unknown')
            elif col == 'annotation.key_features.fraud_channel':
                features[col] = transaction_data.get('fraud_channel', 'unknown')
            elif col == 'annotation.key_features.victim_action':
                features[col] = transaction_data.get('victim_action', 'none')
            elif col == 'annotation.key_features.request_type':
                features[col] = transaction_data.get('request_type', 'none')
            elif col == 'annotation.key_features.impersonated_entity':
                features[col] = transaction_data.get('impersonated_entity', 'unknown')
            elif col == 'annotation.key_features.amount_mentioned':
                features[col] = transaction_data.get('amount_mentioned', 'unknown')
            elif col == 'annotation.key_features.currency':
                features[col] = transaction_data.get('currency', 'unknown')
            else:
                features[col] = 0
    
    X_raw = {}
    for col in feature_cols:
        val = features.get(col, 0)
        X_raw[col] = val
    
    feature_df = pd.DataFrame([X_raw])
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            feature_df[col] = pd.Categorical(feature_df[col]).codes
    X = feature_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    fraud_prob = model.predict_proba(X)[0][1]
    is_fraud = model.predict(X)[0] == 1
    
    fraud_indicators = sum([
        transaction_data.get('transaction_upi_fraud', 0),
        transaction_data.get('transaction_card_fraud', 0),
        transaction_data.get('transaction_bank_transfer', 0),
        transaction_data.get('commerce_nondelivery', 0),
        transaction_data.get('commerce_fake_seller', 0),
        transaction_data.get('credential_phishing', 0),
        transaction_data.get('social_authority_scam', 0),
        transaction_data.get('social_urgency_scam', 0),
        transaction_data.get('meta_victim_story', 0),
        transaction_data.get('meta_fraud_question', 0)
    ])
    
    psychological_score = (
        transaction_data.get('urgency', 0.5) +
        transaction_data.get('fear', 0.5) +
        transaction_data.get('authority', 0.5) +
        transaction_data.get('reward', 0.5)
    ) / 4
    
    if fraud_prob > 0.7:
        risk_level = "HIGH"
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Pattern detection - match actual UI field names
    if transaction_data.get('transaction_upi_fraud', 0):
        pattern_name = "UPI_FRAUD"
        pattern_type = "TRANSACTION_FRAUD"
    elif transaction_data.get('transaction_card_fraud', 0):
        pattern_name = "CARD_FRAUD"
        pattern_type = "TRANSACTION_FRAUD"
    elif transaction_data.get('transaction_bank_transfer', 0):
        pattern_name = "BANK_TRANSFER_FRAUD"
        pattern_type = "TRANSACTION_FRAUD"
    elif transaction_data.get('credential_phishing', 0):
        pattern_name = "PHISHING"
        pattern_type = "SOCIAL_ENGINEERING"
    elif transaction_data.get('commerce_fake_seller', 0):
        pattern_name = "FAKE_SELLER"
        pattern_type = "COMMERCE_FRAUD"
    elif transaction_data.get('social_urgency_scam', 0):
        pattern_name = "URGENCY_SCAM"
        pattern_type = "SOCIAL_ENGINEERING"
    elif transaction_data.get('social_authority_scam', 0):
        pattern_name = "AUTHORITY_SCAM"
        pattern_type = "SOCIAL_ENGINEERING"
    elif transaction_data.get('commerce_nondelivery', 0):
        pattern_name = "NON_DELIVERY"
        pattern_type = "COMMERCE_FRAUD"
    else:
        pattern_name = "UNKNOWN"
        pattern_type = "UNKNOWN"
    
    return {
        "is_fraud": bool(is_fraud),
        "fraud_probability": float(fraud_prob),
        "risk_level": risk_level,
        "model_confidence": float(round(fraud_prob, 2)),
        "fraud_score": float(fraud_prob),
        "label_score": float(fraud_indicators / 10),
        "tactic_score": float(psychological_score),
        "feature_score": float(fraud_indicators / 10),
        "risk_features": [
            {"feature": "psychological_pressure", "contribution": psychological_score * 0.75},
            {"feature": "fraud_indicators", "contribution": fraud_indicators / 10 * 0.25}
        ],
        "rule_breakdown": {
            "label": fraud_indicators / 10 * 0.5,
            "tactic": psychological_score * 0.3,
            "feature": 0.2
        },
        "pattern_name": pattern_name,
        "pattern_type": pattern_type,
        "pattern_confidence": float(round(fraud_prob, 2))
    }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        "message": "MAPE-K Agentic Fraud Detection API",
        "version": "1.0.0",
        "status": "operational",
        "model_trained": pipeline_state['is_trained']
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": pipeline_state['is_trained'],
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/status')
def status():
    m = pipeline_state.get('last_metrics', {})
    
    def to_python(val):
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val
    
    return jsonify({
        "status": "ok",
        "model_loaded": bool(pipeline_state.get('is_trained', False)),
        "last_trained": pipeline_state.get('last_trained'),
        "pipeline_ready": bool(pipeline_state.get('is_trained', False)),
        "drift_detected": bool(pipeline_state.get('drift_detected', False)),
        "metrics": {
            "accuracy": to_python(m.get('accuracy', 0)),
            "precision": to_python(m.get('precision', 0)),
            "recall": to_python(m.get('recall', 0)),
            "f1": to_python(m.get('f1', 0)),
            "roc_auc": to_python(m.get('roc_auc', 0))
        }
    })

@app.route('/api/metrics')
def metrics():
    m = pipeline_state.get('last_metrics', {})
    
    def to_python(val):
        if hasattr(val, 'item'):  # numpy types
            return val.item()
        return val
    
    return jsonify({
        "accuracy": to_python(m.get('accuracy', 0)),
        "precision": to_python(m.get('precision', 0)),
        "recall": to_python(m.get('recall', 0)),
        "f1": to_python(m.get('f1', 0)),
        "roc_auc": to_python(m.get('roc_auc', 0)),
        "true_negatives": to_python(m.get('true_negatives', 0)),
        "false_positives": to_python(m.get('false_positives', 0)),
        "false_negatives": to_python(m.get('false_negatives', 0)),
        "true_positives": to_python(m.get('true_positives', 0)),
        "fpr": to_python(m.get('fpr', 0)),
        "robustness_curve": m.get('robustness_curve', []),
        "clean_f1": to_python(m.get('clean_f1', 0)),
        "worst_f1": to_python(m.get('worst_f1', 0)),
        "avg_f1": to_python(m.get('avg_f1', 0)),
        "f1_drop": to_python(m.get('f1_drop', 0)),
        "is_robust": to_python(bool(m.get('is_robust', False)))
    })

@app.route('/api/drift')
def drift():
    def to_python(val):
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val
    
    return jsonify({
        "drift_detected": to_python(pipeline_state.get('drift_detected', False)),
        "psi_score": to_python(pipeline_state.get('psi_score', 0)),
        "ks_score": to_python(pipeline_state.get('ks_score', 0)),
        "psi_threshold": 0.20,
        "ks_threshold": 0.05,
        "drift_history": pipeline_state.get('drift_history', [])
    })

@app.route('/api/pipeline/status')
def pipeline_status():
    m = pipeline_state.get('last_metrics', {})
    
    def to_python(val):
        if hasattr(val, 'item'):
            return val.item()
        return val
    
    return jsonify({
        "initialized": bool(pipeline_state.get('is_trained', False)),
        "last_run": pipeline_state.get('last_trained'),
        "drift_detected": bool(pipeline_state.get('drift_detected', False)),
        "ingestion": {
            "status": "completed" if pipeline_state.get('is_trained') else "pending",
            "rows_loaded": to_python(pipeline_state.get('rows_loaded', 0)),
            "features": to_python(pipeline_state.get('features_count', 0)),
            "class_dist": pipeline_state.get('class_distribution', {})
        },
        "drift": {
            "status": "completed" if pipeline_state.get('is_trained') else "pending",
            "psi": to_python(pipeline_state.get('psi_score', 0)),
            "ks": to_python(pipeline_state.get('ks_score', 0)),
            "detected": bool(pipeline_state.get('drift_detected', False))
        },
        "balance": {
            "status": "completed" if pipeline_state.get('is_trained') else "pending",
            "original": to_python(pipeline_state.get('rows_loaded', 0)),
            "balanced": to_python(pipeline_state.get('balanced_count', 0)),
            "synthetic_added": to_python(pipeline_state.get('synthetic_added', 0))
        },
        "training": {
            "status": "completed" if pipeline_state.get('is_trained') else "pending",
            "train_samples": to_python(pipeline_state.get('train_samples', 0)),
            "test_samples": to_python(pipeline_state.get('test_samples', 0))
        },
        "evaluation": {
            "status": "completed" if pipeline_state.get('is_trained') else "pending",
            "f1": to_python(m.get('f1', 0)),
            "roc_auc": to_python(m.get('roc_auc', 0)),
            "precision": to_python(m.get('precision', 0)),
            "recall": to_python(m.get('recall', 0))
        }
    })

@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    try:
        data = request.json or {}
        if not pipeline_state['is_trained']:
            return jsonify({"error": "Model not trained"}), 400
        result = predict_fraud(data)
        if result is None:
            return jsonify({"error": "Prediction failed"}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.json or {}
        if not data or 'transactions' not in data:
            return jsonify({"error": "Expected {'transactions': [...]}"}), 400
        if not pipeline_state['is_trained']:
            return jsonify({"error": "Model not trained"}), 400
        results = []
        for txn in data['transactions']:
            result = predict_fraud(txn)
            results.append(result if result else {"error": "Prediction failed"})
        return jsonify({
            "results": results,
            "total": len(results),
            "fraud_count": sum(1 for r in results if r.get('is_fraud', False))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        if not pipeline_state.get('is_trained') or pipeline_state.get('drift_detected'):
            reason = "Initial run" if not pipeline_state.get('is_trained') else "Drift detected"
            result = run_full_pipeline()
            if result is None:
                return jsonify({"error": "Training failed"}), 500
            return jsonify({
                "success": True,
                "retrained": True,
                "reason": reason,
                "metrics": result['metrics'],
                "drift_detected": pipeline_state.get('drift_detected', False)
            })
        else:
            return jsonify({
                "success": True,
                "retrained": False,
                "message": "Model is current, no drift detected"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/run', methods=['GET'])
def run_pipeline():
    try:
        result = run_full_pipeline()
        return jsonify({
            "success": result is not None,
            "run_id": f"RUN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "metrics": result['metrics'] if result else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/pipeline/history', methods=['GET'])
def pipeline_history():
    return jsonify({
        "runs": [],
        "total_runs": 0
    })

# ============================================================================
# AUTO-TRAIN ON STARTUP
# ============================================================================

print("=" * 60)
print("  MAPE-K Agentic Fraud Detection API")
print("  Running initial pipeline on startup...")
print("=" * 60)

train_result = run_full_pipeline()

if train_result:
    print("✅ Pipeline completed successfully!")
    print(f"   F1 Score: {train_result['metrics'].get('f1', 0):.4f}")
    print(f"   ROC-AUC: {train_result['metrics'].get('roc_auc', 0):.4f}")
    print(f"   Drift Detected: {pipeline_state['drift_detected']}")
else:
    print("⚠️ Pipeline failed - API will return errors")

print("=" * 60)
print("  Server running on http://localhost:8000")
print("=" * 60)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)