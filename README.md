# Agentic Fraud Detection System - Final Architecture

## Overview

A simple, clean implementation of an **Agentic Fraud Detection System** using the **MAPE-K** (Monitor-Analyze-Plan-Execute using Knowledge) architectural pattern. 

**Key Design Decisions:**
- ✅ No LangGraph/LangChain - simple MCP-style communication
- ✅ FGSM adversarial training only (no complex Adv-CTGAN)
- ✅ Minimal dependencies - just pandas, sklearn, scipy
- ✅ Easy to understand, explain in paper, and modify

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MAPE-K PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    DRIFT     │ →  │   BALANCE    │ →  │   TRAINING   │       │
│  │   AGENT      │    │    AGENT     │    │    AGENT     │       │
│  │ (Monitor)    │    │  (Execute)   │    │    (Plan)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                  │
│         ↓                   ↓                   ↓                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                    KNOWLEDGE BASE                         │    │
│  │         (Stores history, drift records, metrics)         │    │
│  └──────────────────────────────────────────────────────────┘    │
│                               ↓                                  │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │  EVALUATION  │ ←  │    LOOP      │                           │
│  │    AGENT     │    │   CONTROL    │                           │
│  │  (Analyze)   │    └──────────────┘                           │
│  └──────────────┘                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Descriptions

### 1. Drift Agent (Monitor)
- **Purpose:** Detect concept drift in incoming data
- **Method:** 
  - PSI (Population Stability Index) for categorical drift
  - Kolmogorov-Smirnov test for continuous feature drift
- **Output:** `drift_detected: bool`, `psi_score`, `ks_score`

### 2. Balance Agent (Execute)  
- **Purpose:** Handle class imbalance
- **Method:** CTGAN (Conditional Tabular GAN) to generate synthetic minority samples
- **Fallback:** Noise injection if CTGAN unavailable
- **Output:** `balanced_data`, `balance_report`

### 3. Training Agent (Plan)
- **Purpose:** Train classifier on balanced data
- **Method:** Gradient Boosting (default) with optional FGSM adversarial training
- **Adversarial Training:** Generate FGSM adversarial samples (ε=0.05) to improve robustness
- **Output:** Trained model, training metrics

### 4. Evaluation Agent (Analyze)
- **Purpose:** Evaluate model performance and robustness
- **Metrics:** F1, ROC-AUC, Precision, Recall
- **Robustness:** Test against FGSM attacks at multiple epsilon values
- **Output:** Pass/Fail decision based on thresholds

---

## Data Flow

```
Input Data
    │
    ▼
┌───────────────┐
│  Drift Agent  │ ── Check drift vs reference
└───────────────┘
    │
    ▼ (if drift detected or first run)
┌───────────────┐
│ Balance Agent │ ── CTGAN to balance classes
└───────────────┘
    │
    ▼
┌───────────────┐
│ Training Agent│ ── Train with optional FGSM
└───────────────┘
    │
    ▼
┌───────────────┐
│Evaluation Agent│ ── Evaluate + robustness test
└───────────────┘
    │
    ▼
 Decision: PASS / FAIL
```

---

## Usage

### Basic Usage

```python
import pandas as pd
from pipeline import run_pipeline

# Load data
data = pd.read_csv("fraud_data.csv")

# Run pipeline
result = run_pipeline(
    data=data,
    target_col="is_fraud",
    output_dir="output"
)

print(f"Success: {result['success']}")
print(f"Metrics: {result['summary']['evaluation_metrics']}")
```

### With Custom Configuration

```python
from pipeline import MAPEKPipeline

config = {
    "drift": {"psi_threshold": 0.25, "ks_threshold": 0.10},
    "balance": {"target_ratio": 0.4, "ctgan_epochs": 150},
    "training": {
        "model_type": "gradient_boosting",
        "adversarial_training": True,
        "fgsm_epsilon": 0.1
    },
    "evaluation": {
        "min_f1": 0.75,
        "min_roc_auc": 0.80,
        "min_robustness": 0.65
    }
}

pipeline = MAPEKPipeline(config)
result = pipeline.run(data, target_col="is_fraud")
```

### CLI Usage

```bash
python pipeline.py --data fraud_data.csv --target is_fraud --output results/
```

---

## Output Files

After running, the following files are generated in the output directory:

```
output/run_20260414_123456/
├── run_summary.json      # Complete run summary with all metrics
├── drift_history.json    # Drift detection history
├── balance_report.json   # Class balancing details
├── training_metrics.json # Model training details
└── evaluation_results.json # Final evaluation metrics
```

---

## Key Metrics

The system tracks these metrics:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| F1 Score | Harmonic mean of precision/recall | ≥ 0.70 |
| ROC-AUC | Area under ROC curve | ≥ 0.75 |
| Robustness | F1 under FGSM attack / Clean F1 | ≥ 0.60 |
| PSI | Population Stability Index | ≤ 0.20 |
| KS | Kolmogorov-Smirnov statistic | ≤ 0.05 |

---

## Advantages of This Architecture

1. **Simple to Explain**: Just 4 agents in sequence - perfect for research paper methodology
2. **No Framework Overhead**: No LangGraph/LangChain dependencies
3. **Transparent**: Every step is visible in the code
4. **Flexible**: Easy to add/remove agents or modify logic
5. **Testable**: Each agent can be unit tested independently
6. **Production-Ready**: Still handles drift, balancing, adversarial training, robustness evaluation

---

## Comparison with Original System

| Aspect | Original (LangGraph) | New (Simple MCP) |
|--------|---------------------|------------------|
| Lines of code | ~2000+ | ~600+ |
| Dependencies | langgraph, langchain | pandas, sklearn, scipy |
| Explanation in paper | Complex | Simple |
| Flexibility | High | Medium |
| Maintainability | Medium | High |

---

## Extending the System

### Adding a New Agent

```python
from agents.base import BaseAgent, AgentResult

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("CustomAgent")
    
    def run(self, state: dict) -> AgentResult:
        # Your logic here
        return AgentResult(
            success=True,
            data={"key": "value"},
            message="Done"
        )

# Add to pipeline
pipeline.agents["custom"] = CustomAgent()
```

### Adding New Metrics

Modify `evaluation_agent.py` to add more metrics in the `_test_robustness` method.

---

## References

1. IBM (2005) - "An architectural blueprint for autonomic computing"
2. Madry et al. (2018) - "Towards Deep Learning Models Resistant to Adversarial Attacks"
3. Xu et al. (2019) - "Modeling Tabular Data using Conditional GAN (CTGAN)"
4. Dal Pozzolo et al. (2015) - "Credit Card Fraud Detection and Concept-Drift Adaptation"
5. Simonetto et al. (2024) - "TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning"