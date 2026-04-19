# Analysis Insights Summary

This document summarizes key findings from the experimental analysis of the agentic fraud detection system.

---

## 1. TabularBench Robustness — Accuracy Trend + Attack Impact

**What it says:**
- Standard accuracy remains very high (~99.9 → 99.3%)
- Robust accuracy (FGSM) drops significantly (~99.7 → 94.8%) under drift
- Attack Success Rate (ASR) jumps from 0.3% → 5.8%
- Accuracy drop increases from 0.22 pp → 4.57 pp

**Interpretation:**
- Model is clean-data strong but adversarially fragile under drift
- Drift amplifies vulnerability → attacks become ~20x more effective
- **Key takeaway:** Drift + adversarial noise = non-linear degradation

---

## 2. Evaluation Metrics Comparison (Baseline vs Drift)

**What it says:**
- Precision increases (0.94 → 0.99)
- F1, Recall, ROC-AUC all improve slightly

**Interpretation:**
- Drift dataset likely becomes: Easier to classify OR more separable
- BUT this contradicts robustness results
- **Key takeaway:** Better metrics ≠ safer model (Model is more confident, not more robust)

---

## 3. False Positive Rate (FPR)

**What it says:**
- FPR drops drastically (~0.53 → 0.08)

**Interpretation:**
- Model becomes more conservative
- Fewer normal transactions flagged as fraud
- **Trade-off:** Likely higher false negatives risk (missed fraud)

---

## 4. Robustness Criteria Comparison

**What it says:**
- Baseline wins on: Robustness score (~0.9968), Robust accuracy, Lower accuracy drop
- Drift performs worse across all robustness metrics

**Interpretation:**
- Drift breaks generalization under perturbations
- **Key takeaway:** Baseline model is structurally more stable

---

## 5. Feature Distribution (Box Plot)

**What it says:**
- Fraud transactions show: Extreme outliers in amount
- Non-fraud is tightly distributed

**Interpretation:**
- Model likely relies heavily on: Amount-based heuristics
- **Risk:** Easily exploitable via: Adversarial scaling / normalization tricks

---

## 6. Confusion Matrices (Synthetic Data Scaling)

**What it says:**
- Increasing synthetic data: False negatives drop (47 → 33), True positives increase (1040 → 1054), False positives stay constant (6)

**Interpretation:**
- CTGAN helps: Recall improves, Minority class learning improves
- **Key takeaway:** Synthetic data improves fraud detection sensitivity

---

## 7. PCA + t-SNE Distributions

**What it says:**
- PCA: heavy overlap between classes
- t-SNE: clearer clustering of fraud points

**Interpretation:**
- Data is: Not linearly separable, But non-linear structure exists
- **Implication:** Justifies using: XGBoost / deep models over linear models

---

## 8. Drift Detection Over Time

**What it says:**
- PSI crosses threshold (~0.2) at run 4
- KS also increases (~0.09 peak)

**Interpretation:**
- Significant drift detected mid-way, Then slight stabilization
- **Key takeaway:** Model should trigger retraining around run 3–4

---

## 9. Performance vs Synthetic Data Size

**What it says:**
- Accuracy, Recall, F1 improve with more synthetic data
- ROC-AUC peaks at +700 samples, drops at +1000

**Interpretation:**
- Too much synthetic data: Introduces noise / distribution distortion
- **Optimal point:** ~700 synthetic samples

---

## 10. Model Performance: Clean vs Robust

**What it says:**
- Baseline: huge drop under attack (0.95 → 0.65)
- CTGAN: better (0.98 → 0.72)
- CTGAN + FGSM: best robustness (0.99 → 0.98)

**Interpretation:**
- Adversarial training works extremely well
- **Key takeaway:** FGSM training gives near immunity to perturbations

---

## 11. ROC Curves

**What it says:**
- All models perform very well (AUC ~0.98+)
- Best: +700 synthetic (~0.9857)

**Interpretation:**
- Strong classifier overall, Minor differences in ranking ability

---

## 12. FGSM Robustness vs Epsilon

**What it says:**
- F1 remains high across increasing epsilon
- Drift model slightly more stable than baseline

**Interpretation:**
- Model has: good gradient resistance, especially after training improvements

---

## Summary of Key Takeaways

| Category | Key Finding |
|----------|-------------|
| **Drift Impact** | Drift makes model 20x more vulnerable to attacks |
| **Metric paradox** | Better accuracy doesn't mean better robustness |
| **CTGAN Value** | 700 synthetic samples is optimal for balance |
| **Adversarial Training** | FGSM training provides near-immunity to perturbations |
| **Retraining Trigger** | Should retrain when PSI crosses 0.2 (around run 3-4) |
| **Model Stability** | Baseline model is structurally more stable than drift model |
| **Feature Risk** | Amount-based features are easily exploitable |