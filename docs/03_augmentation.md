# CTGAN Data Augmentation

## Overview

CTGAN (Conditional Tabular Generative Adversarial Network) addresses the extreme class imbalance in fraud detection by generating synthetic minority class samples.

## Theory

### Why CTGAN?

Fraud detection datasets are typically highly imbalanced:
- Fraud: ~1-5% of transactions
- Non-fraud: ~95-99%

Traditional oversampling (SMOTE) creates synthetic samples by interpolating between existing points - can lead to overfitting.

CTGAN learns the underlying distribution of the minority class and generates novel, realistic samples that:
1. Match statistical properties of real data
2. Add diversity (not just interpolation)
3. Preserve feature correlations

### CTGAN Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CTGAN                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Generator ──────────────► Fake Samples                │
│      ↑                         ↑                         │
│      │                         │                        │
│   Noise                   Discriminator                  │
│   Input                   (Real vs Fake)                │
│      │                         │                        │
│      └──────── Backprop ───────┘                        │
│                                                          │
│   Condition: Class label (fraud/non-fraud)              │
│   Mode-Specific Normalization for mixed data types      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Conditional Generator**
   - Takes noise + class label as input
   - Generates samples for specified class

2. **Mode-Specific Normalization**
   - Handles non-Gaussian, multimodal distributions
   - Different normalization per mode

3. **Training-by-Sampling**
   - Addresses severe class imbalance during training
   - Samples more minority class during training

## Implementation

### SDV (Synthetic Data Vault)

We use the SDV library implementation:
```python
from ctgan import CTGAN
from sdv.metadata import SingleTableMetadata

# Define metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Train CTGAN
ctgan = CTGAN(epochs=100)
ctgan.fit(data, discrete_columns=['is_fraud'])

# Generate synthetic samples
synthetic = ctgan.generate(num_rows=500)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| epochs | 100 | Training iterations |
| batch_size | 16 | Samples per batch |
| generator_dim | (64, 64) | Generator layers |
| discriminator_dim | (64, 64) | Discriminator layers |
| learning_rate | 2e-4 | Optimizer learning rate |
| PAC | 1 | Partial aggregation |

## Quality Validation

### Jensen-Shannon Divergence (JSD)

Measures distance between real and synthetic distributions:
- JSD = 0: Identical distributions
- JSD = 1: Completely different
- Threshold: JSD < 0.20 acceptable

### Chi-Squared Test

For categorical features - ensures categorical distributions match.

### Kolmogorov-Smirnov Test

For numerical features - ensures numerical distributions match.

## Results in Our System

```
Original dataset:
  - Fraud: 2,849 (95%)
  - Non-fraud: 151 (5%)
  - Ratio: 18.87:1

After CTGAN augmentation:
  - Fraud: 2,849
  - Non-fraud: 944 (after 793 synthetic)
  - Ratio: 3.02:1
```

### Quality Metrics (from run)
- JSD: 0.1291 (< 0.20 threshold ✓)
- Statistical Fidelity: 98.5%
- Chi-Squared pass rate (categorical): 100%
- KS test pass rate (numerical): 97.1%

## Files

- `balance_agent.py` - Implementation
- `synthetic_*.csv` - Generated synthetic samples
- `ctgan_evaluation_metrics.json` - Quality metrics

## References

1. Xu et al. (2019) - \"Modeling Tabular Data using Conditional GAN\"
2. SDV Library: https://sdv.dev/
3. CTCN (2023) - CTGAN for fraud detection in PeerJ