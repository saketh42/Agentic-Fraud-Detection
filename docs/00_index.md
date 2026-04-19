# Final System Documentation

## Overview

This folder contains complete documentation for the Agentic Fraud Detection System.

## Documentation Index

| # | File | Description |
|---|------|-------------|
| 01 | [01_scraping.md](01_scraping.md) | Reddit data collection |
| 02 | [02_annotation.md](02_annotation.md) | LLM-based fraud labeling |
| 03 | [03_augmentation.md](03_augmentation.md) | CTGAN synthetic data generation |
| 04 | [04_adversarial.md](04_adversarial.md) | FGSM adversarial training |
| 05 | [05_agentic.md](05_agentic.md) | MAPE-K agent architecture |
| 06 | [06_metrics.md](06_metrics.md) | Evaluation metrics and theory |

## Quick Reference

### System Pipeline

```
Reddit Scraping → LLM Annotation → CTGAN Balancing → FGSM Training → Evaluation
     (docs 1-2)        (doc 3)           (doc 4)              (doc 5)
```

### Key Thresholds

| Metric | Threshold |
|--------|-----------|
| PSI Drift | 0.20 |
| KS Drift | 0.05 |
| JSD Quality | < 0.20 |
| F1 Score | ≥ 0.70 |
| ROC-AUC | ≥ 0.75 |
| Robustness | ≥ 0.60 |

### Key Results

- **F1 Score**: 0.9911
- **ROC-AUC**: 0.9966  
- **Robustness**: 98.34%
- **Drift detected**: Yes → Retrained → Passed

## Architecture Diagram

See `output/architecture.png` in the output folder.

## Code Structure

```
final-system/
├── pipeline.py          # Main orchestrator
├── agents/              # MAPE-K agents
│   ├── base.py
│   ├── drift_agent.py
│   ├── balance_agent.py
│   ├── training_agent.py
│   └── evaluation_agent.py
├── output/              # Generated outputs
│   └── plots/           # Visualizations
└── docs/                # This documentation
```

## Running the System

```python
from pipeline import run_pipeline

result = run_pipeline(
    data_path='data/data_binary_only_first_3000.csv',
    target_col='is_fraud',
    output_dir='output'
)
```

Or use CLI:

```bash
python pipeline.py --data data/data_binary_only_first_3000.csv --target is_fraud
```

## References

See individual documentation files for detailed references.