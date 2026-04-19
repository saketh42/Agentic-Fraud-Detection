# Metrics and Evaluation

## Overview

The system tracks multiple metrics to evaluate model performance, data quality, and system robustness.

---

## 1. Classification Metrics

### Standard Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correct predictions |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many actual |
| **Recall** | TP / (TP + FN) | Of actual positives, how many captured |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision/recall |
| **ROC-AUC** | Area under ROC curve | Trade-off between TPR and FPR |

### In Our Results

From latest run:
- Accuracy: 99.11%
- Precision: 99.11%
- Recall: 99.11%
- F1: 0.9911
- ROC-AUC: 0.9966

### False Positive Rate

```
FPR = FP / (FP + TN)
```

Our FPR: 0.0779 (7.79%)

---

## 2. Drift Detection Metrics

### Population Stability Index (PSI)

```
PSI = Σ (p_i - q_i) × ln(p_i / q_i)
```

Where:
- p_i = proportion in current distribution (bucket i)
- q_i = proportion in reference distribution (bucket i)

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10 | No significant change |
| 0.10 - 0.20 | Slight change - monitor |
| > 0.20 | Significant change - action needed |

**Our threshold:** 0.20

### Kolmogorov-Smirnov (KS) Statistic

```
KS = max |F_current(x) - F_reference(x)|
```

Maximum distance between cumulative distribution functions.

| KS Value | Interpretation |
|----------|----------------|
| < 0.05 | No significant drift |
| 0.05 - 0.10 | Moderate drift |
| > 0.10 | Significant drift |

**Our threshold:** 0.05

---

## 3. Data Quality Metrics

### Jensen-Shannon Divergence (JSD)

```
JSD = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
where M = 0.5 × (P + Q)
```

Measures distance between real and synthetic distributions.

- JSD = 0: Identical distributions
- JSD = 1: Completely different
- **Our threshold:** < 0.20

### Chi-Squared Test

For categorical features - tests if categorical distribution matches between real and synthetic.

- **Our result:** 100% pass rate on categorical features

### Kolmogorov-Smirnov (Numerical)

For numerical features - tests if numerical distribution matches.

- **Our result:** 97.1% pass rate

---

## 4. Robustness Metrics

### Clean vs Attacked F1

Test F1 at multiple epsilon values:
```python
epsilons = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
```

Our robustness curve:
| Epsilon | F1 Score |
|---------|----------|
| 0.0 (clean) | 0.9911 |
| 0.05 | 0.9857 |
| 0.10 | 0.9826 |
| 0.20 | 0.9842 |
| 0.30 | 0.9819 |

### Robustness Score

```
Robustness = Average(F1 under attack) / Clean F1
```

- **Our result:** 0.9834 (98.34%)
- **Threshold:** ≥ 0.60 (60%)

### F1 Drop

```
F1 Drop = Clean F1 - Worst F1 (across epsilons)
```

- **Our result:** 0.0108 (1.08%)

### Attack Success Rate (ASR)

```
ASR = (Clean F1 - Attacked F1) / Clean F1
```

- **Our result:** 1.09%

---

## 5. TabularBench Metrics

A more rigorous robustness benchmark using Constrained Adaptive Attack (CAA).

### Standard vs Robust Accuracy

| Metric | Value |
|--------|-------|
| Standard Accuracy | 99.35% |
| Robust Accuracy (CAA) | 94.78% |
| Accuracy Drop | 4.57% |
| Attack Success Rate | 5.79% |
| Is Robust | True |

### Interpretation

- **Standard Accuracy**: Performance on clean test data
- **Robust Accuracy**: Performance under CAA attack
- **Accuracy Drop**: How much performance degrades under attack
- **ASR**: Percentage of originally correct predictions now wrong
- **Is Robust**: True if Robust Accuracy > 90% OR Drop < 10%

---

## 6. Model Training Metrics

### Training Details

| Metric | Value |
|--------|-------|
| Model Type | XGBoost |
| Total Training Samples | 6,798 |
| Fraud Samples | 5,394 |
| Non-Fraud Samples | 1,404 |
| Adversarial Samples Added | 3,399 |
| FGSM Epsilon (training) | 0.05 |

### Class Balance

| Stage | Fraud | Non-Fraud | Ratio |
|-------|-------|-----------|-------|
| Original | 2,697 | 351 | 7.68:1 |
| After Balancing | 2,697 | 702 | 3.84:1 |

---

## Summary Table

| Category | Key Metric | Our Value | Threshold |
|----------|------------|-----------|-----------|
| Classification | F1 | 0.9911 | ≥ 0.70 |
| Classification | ROC-AUC | 0.9966 | ≥ 0.75 |
| Drift | PSI | (from run) | ≤ 0.20 |
| Drift | KS | (from run) | ≤ 0.05 |
| Data Quality | JSD | 0.1291 | < 0.20 |
| Robustness | Robustness Score | 0.9834 | ≥ 0.60 |
| TabularBench | Is Robust | True | True |

---

## Files

- `run_summary.json` - Complete run metrics
- `evaluation_agent.py` - Metric computation
- Various `.png` plots - Visualizations

---

## References

1. Dal Pozzolo et al. (2015) - Credit card fraud and concept drift
2. Simonetto et al. (2024) - TabularBench benchmark
3. Melo et al. (2023) - Adversarial training for tabular data