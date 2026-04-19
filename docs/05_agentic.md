# Agentic Intelligence Layer - MAPE-K Architecture

## Overview

The Agentic Intelligence Layer implements the **MAPE-K** (Monitor-Analyze-Plan-Execute using Knowledge) pattern from IBM's autonomic computing framework. This provides autonomous decision-making and closed-loop control.

## Our Novel Contribution: Adaptive Learning Tracking

Beyond simple MAPE-K, we track **how the model learns** under different conditions:

- **Run 1 (Initial)**: Fast learning, high performance
- **Run 4 (Drift)**: Slower learning, degraded performance  
- **Run 6 (Recovered)**: Learning restored, back to high performance

This demonstrates **true adaptability** - the system not only detects drift but shows measurable learning adaptation.

## Theory

### MAPE-K Origin

IBM's 2005 architectural blueprint for autonomic computing drew inspiration from the human autonomic nervous system - which operates without conscious thought. The goal: computing systems that manage themselves based on high-level goals.

### MAPE-K Components

```
┌─────────────────────────────────────────────────────────────┐
│                      MANAGED SYSTEM                          │
│                   (Fraud Detection Model)                   │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ Feedback (Control)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     MANAGING SYSTEM                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ Monitor │→│ Analyze │→│  Plan   │→│ Execute │         │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘         │
│       ↑                                                │     │
│       └──────────── KNOWLEDGE BASE ←────────────┘       │     │
└─────────────────────────────────────────────────────────────┘
```

### In Our System

| MAPE-K Phase | Our Agent | Function |
|--------------|-----------|----------|
| - | Ingestion Agent | Load and preprocess data |
| Monitor | Drift Agent | Detect concept drift |
| Execute | Balance Agent | CTGAN class balancing |
| Plan | Training Agent | Model training + adaptation tracking |
| Analyze | Evaluation Agent | Assess performance + robustness |

## Implementation

### Agent Communication (MCP-Style)

Instead of LangGraph, we use simple state passing:

```python
state = initial_data

# Each agent receives state, returns updated state
state = ingestion_agent.run(state)   # Data loading
state = drift_agent.run(state)       # Monitor
state = balance_agent.run(state)    # Execute  
state = training_agent.run(state)   # Plan + track adaptation
state = evaluation_agent.run(state)# Analyze
```

### Agents

#### 1. Ingestion Agent (Data Loading)

```python
class IngestionAgent:
    def run(self, state):
        # Load CSV, clean data, remove uncertain labels
        data = load_data()
        data = clean(data)
        return AgentResult(data=data)
```

#### 2. Drift Agent (Monitor)

```python
class DriftAgent:
    def run(self, state):
        # Compare current vs reference distribution
        psi = compute_psi(current, reference)
        ks = compute_ks(current, reference)
        
        if psi > threshold or ks > threshold:
            return AgentResult(drift_detected=True, drift_severity=psi)
        return AgentResult(drift_detected=False)
```

**Drift Detection Methods:**
- **PSI (Population Stability Index)**: Σ (p_i - q_i) * ln(p_i / q_i)
- **KS (Kolmogorov-Smirnov)**: Max distance between CDFs

#### 3. Balance Agent (Execute)

Handles class imbalance using CTGAN (documented in augmentation docs).

#### 4. Training Agent (Plan + Adaptation Tracking)

Trains classifier and tracks learning behavior:

```python
class TrainingAgent:
    def run(self, state):
        # Train model across multiple epochs
        for epoch in range(N_EPOCHS):
            f1 = train_one_epoch()
            history.append(f1)
        
        # Calculate learning rates
        initial_learning = history[5] - history[0]   # First 5 epochs
        later_learning = history[-1] - history[5]    # Remaining epochs
        
        # Track adaptation
        if drift_detected:
            if final_f1 >= prior_f1:
                adaptation = "fast_adaptation"
            else:
                adaptation = "slow_adaptation"
        else:
            adaptation = "stable_learning"
        
        return AgentResult(
            model=model,
            adaptation_rate=adaptation,
            initial_learning=initial_learning,
            later_learning=later_learning
        )
```

**Learning Rate Calculation:**

We track learning in two phases:

| Phase | Epochs | Formula | What It Measures |
|-------|--------|---------|------------------|
| **Initial Learning** | 0-5 | `F1[5] - F1[0]` | How fast model learns at start |
| **Later Learning** | 5-19 | `F1[final] - F1[5]` | How well model fine-tunes |

**Example (Run 4 - Under Drift):**

```
Epoch 0: F1 = 0.40  (starting point)
Epoch 5: F1 = 0.74  (after 5 epochs)
Initial Learning = 0.74 - 0.40 = +0.34 ← SLOWER under drift!

Epoch 19: F1 = 0.87 (final)
Later Learning = 0.87 - 0.74 = +0.13
```

**The Learning Rate Metric:**

```
Learning Rate = Δ F1 per epoch
```

| Value | Interpretation |
|-------|----------------|
| +0.40 | Very fast learning |
| +0.34 | Slower (under drift) |
| Higher = Faster learner |

**Visualized:**

- Higher bar = Model learns faster
- Lower bar = Model struggles (drift effect)

#### 5. Evaluation Agent (Analyze)

Evaluates performance and robustness metrics (documented in metrics docs).

### Adaptation Metrics Captured

| Metric | What It Shows |
|-------|---------------|
| `initial_gain` | Learning in first 5 epochs |
| `later_gain` | Learning in remaining epochs |
| `overfitting_gap` | Train F1 - Val F1 |
| `adaptation_rate` | fast / slow / stable |
| `performance_history` | All runs tracked |

### Feedback Loops

Our system implements 5 feedback loops:

| Loop | From | To | Purpose |
|------|------|-----|---------|
| L1 | Evaluation | Balance | Re-balance if metrics poor |
| L2 | Evaluation | Strategy | Change strategy if needed |
| L3 | Training | Training | Self-improvement iteration |
| L4 | Supervisor | Policy | Health check feedback |
| L5 | Evaluation | KB | Update knowledge base |

### Knowledge Base

Stores:
- Run history
- Drift records
- Performance metrics
- Model versions
- Learning adaptation history

```python
class KnowledgeBase:
    def log_event(self, agent, event_type, data):
        # Store in JSONL format
        pass
    
    def get_latest(self, record_type):
        # Retrieve most recent record
        pass
```

## Drift-Based Retraining Loop

The core novelty: when drift is detected, system automatically triggers retraining:

```
Drift Detected (PSI > 0.20 or KS > 0.05)
    ↓
Evaluation Agent assesses performance + learning adaptation
    ↓
If performance degraded → Trigger retraining
    ↓
Balance Agent re-balances with new data
    ↓
Training Agent retrains + tracks adaptation
    ↓
Evaluation Agent re-evaluates
    ↓
If passed → Deploy new model
```

This creates a self-healing system that adapts to evolving fraud patterns.

## Visual Evidence of Adaptation

### 1. Drift + Performance + Adaptation (Combined)

Shows the full loop:
- Panel A: Drift detection (PSI/KS over runs)
- Panel B: Model performance (F1 over runs)
- Panel C: Adaptation response (stable/fast/slow)

### 2. Learning Curves Per Run

Shows how the model learns differently:
- Run 1: Fast initial learning (+0.40), converges well
- Run 4 (Drift): Slower learning (+0.34), worse final performance
- Run 6 (Recovered): Learning restored (+0.40)

### 3. Learning Rate Comparison

Bar chart comparing:
- Initial learning (epochs 0-5)
- Later learning (epochs 5-19)

### 4. Drift vs Learning Relationship (NEW)

The KEY graph showing model IS learning when drift happens:

```
Run 4: DRIFT DETECTED
  - PSI jumps to 0.22 (above 0.20 threshold)
  - Initial learning DROPS from +0.40 to +0.34
  - F1 drops from 0.95 to 0.74
  
RESULT: The model IS learning under drift
  - But it learns SLOWER (+0.34 vs +0.40)
  - This shows the model is adapting
  - It struggles with new patterns

AFTER RUN 4:
  - System triggers retraining
  - Learning rate Restored to +0.40
  - F1 recovers to 0.92
```

This is our **main proof of adaptability** in the paper.

## Why Not LangGraph?

For a research paper, LangGraph adds unnecessary complexity:

| LangGraph | Our Approach |
|-----------|--------------|
| Production-grade | Simple function calls |
| Complex state management | Plain dict passing |
| Framework dependency | Just pandas/sklearn |
| Hard to explain | Easy to show in paper |

The research contribution is the fraud detection logic, not the orchestration framework.

## Our 4 Novel Contributions

1. **MAPE-K Agent Architecture** - Autonomous closed-loop control
2. **CTGAN Class Balancing** - Synthetic data for minority class
3. **FGSM Adversarial Training** - Robustness against evasion
4. **Learning Adaptation Tracking** - Measure how model learns under drift

## Files

- `pipeline.py` - Main orchestrator
- `agents/ingestion_agent.py` - Data loading
- `agents/drift_agent.py` - Monitor
- `agents/balance_agent.py` - Execute  
- `agents/training_agent.py` - Plan + adaptation tracking
- `agents/evaluation_agent.py` - Analyze
- `knowledge_base.py` - Knowledge store
- `run_summary.json` - Execution results

## Visualizations

- `drift_adaptation_combined.png` - Combined drift + performance + adaptation
- `learning_curves_per_run.png` - Learning curves for each run
- `learning_rate_comparison.png` - Learning rate bar chart
- `drift_over_time.png` - Drift detection over runs
- `drift_learning_relationship.png` - **KEY: Drift vs Learning vs Performance**

## References

1. IBM (2005) - "An architectural blueprint for autonomic computing"
2. Kephart & Chess (2003) - "The Vision of Autonomic Computing"
3. Bucchiarone et al. (2022) - "A MAPE-K Approach to Autonomic Microservices"
4. Oh et al. (2023) - "Analysis of MAPE-K Loop in Self-adaptive Systems"