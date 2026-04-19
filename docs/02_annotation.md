# LLM Annotation System

## Overview

Large Language Models (LLMs) are used to annotate scraped Reddit posts with fraud labels and structured features. This replaces manual labeling at scale.

## Implementation

### Model Used
- **Ollama** with **llama3** (local LLM)
- Fallback to other models if llama3 unavailable

### Prompt Engineering

The LLM is prompted as an expert fraud analyst to extract:

1. **Fraud Classification**
   - `is_fraud`: 1 (fraud), 0 (non-fraud), -1 (uncertain)

2. **Fraud Type Categories**
   - Cryptocurrency scam
   - Investment fraud
   - Romance scam
   - Phishing
   - Tech support scam
   - Lottery/prize scam
   - Other

3. **Payment Methods**
   - Bank transfer
   - UPI
   - Cryptocurrency
   - Gift cards
   - Wire transfer
   - Other

4. **Psychological Tactics**
   - Urgency (time pressure)
   - Fear (threats)
   - Authority (impersonation)
   - Reward (promises)
   - Scarcity (limited time)

5. **Monetary Values**
   - Amount lost (if mentioned)
   - Currency

## Annotation Output

```json
{
  \"is_fraud\": 1,
  \"fraud_type\": \"investment_fraud\",
  \"payment_method\": \"cryptocurrency\",
  \"psychological_tactics\": {
    \"urgency\": 0.8,
    \"fear\": 0.9,
    \"authority\": 0.6,
    \"reward\": 0.7
  },
  \"amount_lost\": 5000,
  \"currency\": \"USD\",
  \"confidence\": 0.85
}
```

## Validation

- Uncertain labels (-1) are filtered out
- Confidence scores above 0.7 are accepted
- Manual review for confidence < 0.7

## Files

- `label.py` - Main labeling script
- `post_annotation_cache.json` - Cached annotations
- `outputs/annotations_*.csv` - Final labeled datasets

## References

1. Few-shot prompting for structured extraction
2. Chain-of-thought reasoning for fraud detection
3. Llama3: https://ollama.ai/