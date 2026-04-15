# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev server (hot-reload)
make dev          # uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Tests
pytest -v                               # all tests
pytest -v tests/test_agent.py           # single file
pytest -v tests/test_rank.py::TestRankCandidates::test_rank_success  # single test

# Lint (ruff, line-length=120, rules E/F/I/W)
make lint

# Embedding batch job (requires LLM + ES + pgvector running)
make embed        # python -m src.embeddings.embed_videos

# Local stack with Ollama
make up           # docker compose up -d (API + Ollama; needs infra network)
make down
```

`asyncio_mode = "auto"` is set in `pyproject.toml` — all `async def test_*` functions run automatically without `@pytest.mark.asyncio`, but existing tests use the decorator anyway (both work).

## Architecture

### Request Flow

```
POST /api/v1/recommend
  → src/api/routes/recommend.py
  → src/agent/graph.py::get_recommendations()
  → LangGraph: retrieve → rank → filter
  → returns list[dict]
```

The compiled graph is a **module-level singleton** (`recommendation_graph = build_graph()`) instantiated at import time.

### LangGraph Agent (3 nodes, linear graph)

**State** (`src/agent/state.py`): `AgentState` dataclass flows through all nodes. Key fields:
- Input: `user_id`, `query` (optional), `limit`
- Built up: `watch_history` (list of video IDs), `candidates` (list of `VideoCandidate`), `ranked_results` (list of dicts from LLM)

**Node 1 — retrieve** (`src/agent/nodes/retrieve.py`):
- Fetches `watch_history` from pgvector (`tools/user_history.py` — queries `watch_history` table)
- If `query` present: ES multi-match search on `title^2, description` (`tools/search_videos.py`)
- Always: trending videos from pgvector (`tools/trending.py` — counts `watch_history` rows in last 24h)
- Deduplicates by `video_id`. Each failure is caught individually (partial results are fine).
- **Note**: `EmbeddingStore.find_similar()` exists but is not yet wired into retrieve.

**Node 2 — rank** (`src/agent/nodes/rank.py`):
- Builds a prompt with watch history (last 20 IDs) + candidates, asks LLM to return JSON array of `{video_id, score, reason}`.
- Falls back to source-based scoring on `JSONDecodeError` (search=0.8, trending=0.5) or any other LLM exception (all=0.5).

**Node 3 — filter** (`src/agent/nodes/filter.py`):
- Removes already-watched videos **unless** `state.query` is set (explicit search bypasses watch filter).
- Drops items with `score < 0.1`.
- Truncates to `state.limit`.

### LLM Provider (`src/llm/`)

`get_llm_provider()` in `provider.py` returns a **module-level singleton** selected by `LLM_PROVIDER` env var:

| Provider | Class | Use Case |
|----------|-------|----------|
| `ollama` (default) | `OllamaProvider` | Local dev; calls `POST /api/chat` and `POST /api/embeddings` |
| `bedrock` | `BedrockProvider` | Production; uses `bedrock.converse()` for text, `amazon.titan-embed-text-v2:0` for embeddings |

`BedrockProvider` wraps synchronous boto3 calls in `asyncio.get_event_loop().run_in_executor()`.

**Test pattern**: Reset the singleton between tests: `import src.llm.provider as mod; mod._provider_instance = None`

### pgvector Schema (`src/embeddings/store.py`)

`EmbeddingStore.initialize()` creates two tables on startup:

```sql
video_embeddings (video_id PK, title, description, embedding vector(1536), updated_at)
watch_history    (id SERIAL, user_id, video_id, event_type, watched_at)
```

The `watch_history` table is queried by both `tools/user_history.py` (per-user lookup) and `tools/trending.py` (aggregation). It is populated externally (from the analytics pipeline or the data service).

### Embedding Batch Job (`src/embeddings/embed_videos.py`)

Run `make embed` to scroll all videos from ES, generate embeddings via LLM, and upsert into `video_embeddings`. This must be re-run whenever the video catalog changes significantly.

### Environment Variables

All config lives in `src/config.py` as a module-level `config = Config()` singleton:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `bedrock` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1` | Model name for both generation and embedding |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | Bedrock model |
| `AWS_REGION` | `us-east-1` | Bedrock/AWS region |
| `PGVECTOR_URL` | `postgresql://recouser:recopass@localhost:5432/recommendations` | pgvector DSN |
| `EMBEDDING_DIMENSION` | `1536` | Must match the model's output dimension |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | ES for video search |
| `ES_VIDEO_INDEX` | `videos` | ES index name |
| `MAX_RECOMMENDATIONS` | `10` | Default limit |

### Local Dev Stack

`make up` starts the API container + Ollama and joins the external `videostreamingplatform-infra` Docker network (for ES and pgvector). The infra network must already exist:

```bash
cd ../videostreamingplatform-infra && make up   # start shared infra first
make up                                          # then start this service
```

### K8s

`k8s/` deploys into the `recommendations` namespace. `k8s/configmap.yaml` holds non-secret env vars; `k8s/secret.yaml` holds `PGVECTOR_URL`. The service is accessed from `metadataservice` via `GET /recommendations` (proxied through the metadata service's `/recommendations` route).
