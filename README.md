# Video Streaming Platform — Recommendations

Agentic AI-powered video recommendation engine using LangGraph.

## Architecture

```
User Request → FastAPI → LangGraph Agent
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
               Retrieve     Rank     Filter
               (ES +       (LLM)    (Business
               pgvector)            rules)
                    │
                    ▼
              Ranked Recommendations
```

### Agent Nodes
1. **Retrieve** — Fetches candidates from ES (search), pgvector (similar), and watch history (trending)
2. **Rank** — Uses LLM (Ollama locally, Bedrock in prod) to score relevance
3. **Filter** — Removes watched videos, enforces min score, applies limit

### LLM Providers
| Provider | Use Case | Config |
|----------|----------|--------|
| Ollama | Local development | `LLM_PROVIDER=ollama` |
| AWS Bedrock | Production (AWS) | `LLM_PROVIDER=bedrock` |

## Development

### Prerequisites
- Python 3.11+
- Ollama running locally (`ollama serve`)
- pgvector and ES (via `../videostreamingplatform-infra/make up`)

### Setup
```bash
pip install -r requirements.txt
make dev  # Start FastAPI dev server
```

### Run tests
```bash
make test
```

### API
```bash
# Get recommendations
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-1", "query": "python tutorial", "limit": 5}'

# Health check
curl http://localhost:8000/health
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_PROVIDER | ollama | LLM provider (ollama/bedrock) |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server URL |
| OLLAMA_MODEL | llama3.1 | Ollama model name |
| BEDROCK_MODEL_ID | anthropic.claude-3-sonnet-* | Bedrock model ID |
| PGVECTOR_URL | postgresql://... | pgvector connection string |
| ELASTICSEARCH_URL | http://localhost:9200 | ES URL |
| MAX_RECOMMENDATIONS | 10 | Max results per request |
