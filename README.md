# MAPE-K Agentic Fraud Detection System

Multi-Agent MAPE-K (Monitor-Analyze-Plan-Execute-Knowledge) Financial Fraud Detection and Pattern Learning System.

## Architecture

```
Transaction Input
  → [M] Monitor Agent           (validates, masks IDs)
  → [A] Feature Extraction      (parses labels/tactics/profiles)
  → [A] Context Retrieval       (queries KnowledgeStore)
  → [A] Fraud Scoring           (rule-based weights)
  → [A] LLM Reasoning           (mock or Ollama)
  → [K] Pattern Learning        (spec pattern rules)
  → [K] Adversarial Simulation  (safe boolean toggles)
  → [P] Planning Agent          (action rules)
  → [E] Execute Agent           (mock tool layer)
  → Feedback Agent              (human feedback)
  → [K] Knowledge Store         (SQLite, 7 tables)
```

## Quick Start

```bash
# Install deps
pip install -r requirements.txt

# Run the API
python run.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/transaction/process` | Process a transaction through the full MAPE-K pipeline |
| POST | `/feedback` | Submit human feedback |
| GET | `/transaction/{id}` | Get transaction details |
| GET | `/patterns` | All learned fraud patterns |
| GET | `/adversarial/{id}` | Adversarial variants for a transaction |
| GET | `/analytics` | System analytics |

## Example Request

```bash
curl -X POST http://localhost:8000/transaction/process \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-001",
    "credential_phishing": 1,
    "social_authority_scam": 1,
    "urgency": 0.8,
    "fear": 0.6,
    "authority": 0.7,
    "impersonated_entity": "bank",
    "victim_action": "click_link",
    "urgency_level": "high"
  }'
```

## LLM Reasoning

The system supports two modes:
1. **Mock mode** (default) — rule-based reasoning, no API key needed
2. **Ollama mode** — uses local Llama3 via Ollama

Set `llm_mock_mode=False` in `api/server.py` to enable Ollama.

## Tests

```bash
python -m pytest tests/ -v
```

## MAPE-K Mapping

- **M** = Monitor Agent
- **A** = Feature Extraction + Context + Scoring + LLM Reasoning
- **P** = Planning Agent
- **E** = Execute Agent (tool layer)
- **K** = Knowledge Store + Pattern Learning + Adversarial Knowledge

## Agents

| Agent | MAPE-K | Purpose |
|-------|--------|---------|
| MonitorAgent | M | Schema validation, ID masking |
| FeatureExtractionAgent | A | Parse labels/tactics/profiles |
| ContextRetrievalAgent | A | Query historical knowledge |
| FraudScoringAgent | A | Rule-based scoring with spec weights |
| LLMReasoningAgent | A | Contextual reasoning (mock/Ollama) |
| PatternLearningAgent | K | Match signals to known patterns |
| AdversarialSimulationAgent | K | Safe boolean feature toggles |
| PlanningAgent | P | Action selection from rules |
| ExecuteAgent | E | Mock tool execution layer |
| FeedbackAgent | - | Human feedback processing |
| KnowledgeStore | K | SQLite persistent memory |
