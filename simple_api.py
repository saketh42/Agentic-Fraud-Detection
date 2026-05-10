"""
Simple Flask API for testing
"""
from flask import Flask, jsonify
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Agentic Fraud Detection API", "status": "ok"})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": True})

@app.route('/api/status')
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "last_trained": "2026-05-09T12:00:00Z",
        "pipeline_ready": True,
        "drift_detected": False,
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.90,
            "f1": 0.91,
            "roc_auc": 0.98
        }
    })

@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    return jsonify({
        "is_fraud": True,
        "fraud_probability": 0.85,
        "risk_level": "HIGH",
        "model_confidence": 0.92,
        "fraud_score": 0.85,
        "label_score": 0.75,
        "tactic_score": 0.80,
        "feature_score": 0.78,
        "risk_features": [],
        "rule_breakdown": {"label": 0.75, "tactic": 0.80, "feature": 0.78},
        "pattern_name": "PHISHING",
        "pattern_type": "SOCIAL_ENGINEERING",
        "pattern_confidence": 0.88
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)