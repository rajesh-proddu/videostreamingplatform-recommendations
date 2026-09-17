# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dev server (hot-reload)
make dev          # uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production server (gunicorn + uvicorn workers, config in gunicorn.conf.py)
make run          # gunicorn -c gunicorn.conf.py src.api.main:app

# Tests
pytest -v                               # all tests
pytest -v tests/test_agent.py           # single file
pytest -v tests/test_rank.py::TestRankCandidates::test_rank_success  # single test

# Lint (ruff, line-length=120, rules E/F/I/W)
make lint

# Ranking eval (see "Evals" below)
make eval-offline # no-LLM cases, gated on evals/baseline.json (CI runs this)
make eval         # all cases against the configured LLM_PROVIDER

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
  → src/api/routes/recommend.py        (mints request_id, returned in the response)
  → src/agent/graph.py::get_recommendations(request_id=…)
  → LangGraph: retrieve → {rank | rank_deterministic | popular_fallback} → filter
  → returns list[dict]; impression row written in the background
```

The compiled graph is a **module-level singleton** (`recommendation_graph = build_graph()`) instantiated at import time. `ainvoke` returns the final state as a **dict**, not the `AgentState` dataclass — read fields by key.

### LangGraph Agent (5 nodes, conditional routing)

**State** (`src/agent/state.py`): `AgentState` dataclass flows through all nodes. Key fields:
- Input: `user_id`, `query` (optional), `limit`
- Built up: `watch_history` (video IDs), `watch_history_titles` (aligned titles, query path only), `candidates` (list of `VideoCandidate`), `ranked_results` (list of dicts)
- Recorded for the impression log: `route`, `prompt_version`, `rank_fallback`

**retrieve** (`src/agent/nodes/retrieve.py`) — all sources run concurrently under `asyncio.gather`, each wrapped so one failure degrades rather than fails:
- Watch history from `user_features.recent_watches` (`tools/user_history.py`); on the query path, IDs are resolved to titles from `video_embeddings` (`tools/video_titles.py`, falls back to the ID)
- If `query`: ES multi-match (`tools/search_videos.py`) **and** ANN semantic search (`tools/semantic_search.py`)
- Always: taste-vector ANN (`tools/similar.py`, reads `user_features.user_vec`) and precomputed `trending_videos` (`tools/trending.py`)
- Dedups by `video_id` in source priority search → semantic → similar → trending, then caps at `MAX_RANK_CANDIDATES`.

**Routing** (`graph.py::_route_after_retrieve`): no candidates → `popular_fallback`; no query → `rank_deterministic`; otherwise → `rank`. Only `rank` calls the LLM.

**rank** (`src/agent/nodes/rank.py`): prompt = watch-history titles (last 20) + query + candidates; LLM returns a JSON array of `{video_id, score, reason}`. Falls back to source-based scores on `JSONDecodeError` (search=0.8, else 0.5) or flat 0.5 on any other exception, and records `rank_fallback`. **Bump `PROMPT_VERSION` whenever `RANKING_PROMPT` changes.**

**rank_deterministic** — source-weight scores, no LLM. **popular_fallback** — trending over a 7-day window.

**filter** (`src/agent/nodes/filter.py`): removes already-watched videos unless `query` is set, drops `score < 0.1`, truncates to `limit`.

### Impression log & metrics

`src/agent/impressions.py` writes one `recommendation_impressions` row per served response (request_id, route, prompt_version, model_id, rank_fallback, latency, items with rank/score/source) as a tracked background task — never on the request path; failures are logged and counted. The API lifespan creates the table (`ensure_schema`) and drains pending writes on shutdown.

`src/agent/metrics.py` defines the request-path counters (`recommendation_route_total`, `recommendation_rank_fallback_total`, `recommendation_source_candidates_total`, `recommendation_impression_write_failures_total`). Instruments are created lazily because `src.api.main` imports the graph before `init_observability()` runs.

### Evals (`evals/`)

`python -m evals.run` scores ranking quality (NDCG@10, precision@10, recall@10) over the golden set in `evals/cases/*.json`. Each case pins retrieval to a fixed candidate pool (the tools are patched), so the score measures only ranking and filtering, the part that prompt and model changes affect. Relevance labels are graded 0–3; unlisted videos count as 0.

- **Gate**: `--offline` runs only the no-query cases (`rank_deterministic`, `popular_fallback`) and `--check evals/baseline.json` fails on a mean regression **or** on any single case losing more than 0.05 NDCG. After an intentional change, rerun with `--write-baseline evals/baseline.json` and commit the new baseline.
- **LLM cases** (those with a `query`) need a model and don't run in CI. A run fails if any case falls back from LLM ranking, because those scores don't measure the prompt. Baselines are keyed `full:<provider>:<model>`.
- **The seed set is synthetic** (`"synthetic": true`): 10 placeholder cases over an invented catalog that exercise every route. Replace or extend them with hand-labelled cases drawn from `recommendation_impressions` (real queries and served candidates) before treating scores as a quality signal.

### LLM Provider (`src/llm/`)

`get_llm_provider()` in `provider.py` returns a **module-level singleton** selected by `LLM_PROVIDER` env var:

| Provider | Class | Use Case |
|----------|-------|----------|
| `ollama` (default) | `OllamaProvider` | Local dev; calls `POST /api/chat` and `POST /api/embeddings` |
| `bedrock` | `BedrockProvider` | Production; uses `bedrock.converse()` for text, `amazon.titan-embed-text-v2:0` for embeddings |
| `anthropic` | `AnthropicProvider` | Direct Anthropic API; `generate()` only (no embeddings) |

Ollama and Bedrock generate at `temperature 0` so ranking is reproducible. The Anthropic provider can't be pinned: `claude-opus-4-7`+ rejects sampling params.

`BedrockProvider` wraps synchronous boto3 calls in `asyncio.get_event_loop().run_in_executor()`.

**Test pattern**: Reset the singleton between tests: `import src.llm.provider as mod; mod._provider_instance = None`

### pgvector Schema

`EmbeddingStore.initialize()` (`src/embeddings/store.py`, run by the consumer/batch job) creates:

```sql
video_embeddings (video_id PK, title, description, embedding vector(N), updated_at)
user_features    (user_id PK, user_vec vector(N), recent_watches TEXT[], updated_at)   -- analytics feature-jobs, daily
trending_videos  (rank PK, video_id, title, description, watch_count, updated_at)       -- analytics feature-jobs, hourly
```

The API lifespan creates `recommendation_impressions` (`src/agent/impressions.py`). The legacy `watch_history` table is no longer read or created.

### Embedding Batch Job (`src/embeddings/embed_videos.py`)

Run `make embed` to scroll all videos from ES, generate embeddings via LLM, and upsert into `video_embeddings`. This must be re-run whenever the video catalog changes significantly.

### Environment Variables

All config lives in `src/config.py` as a module-level `config = Config()` singleton:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `bedrock`, or `anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1` | Model name for both generation and embedding |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | Bedrock model |
| `AWS_REGION` | `us-east-1` | Bedrock/AWS region |
| `PGVECTOR_URL` | `postgresql://recouser:recopass@localhost:5432/recommendations` | pgvector DSN |
| `EMBEDDING_DIMENSION` | `1536` | Must match the model's output dimension |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | ES for video search |
| `ES_VIDEO_INDEX` | `videos` | ES index name |
| `MAX_RECOMMENDATIONS` | `10` | Default limit |
| `MAX_RANK_CANDIDATES` | `40` | Cap on deduped candidates passed to the ranker |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-opus-4-7` | Anthropic provider |

### Local Dev Stack

`make up` starts the API container + Ollama and joins the external `videostreamingplatform-infra` Docker network (for ES and pgvector). The infra network must already exist:

```bash
cd ../videostreamingplatform-infra && make up   # start shared infra first
make up                                          # then start this service
```

### K8s

`k8s/` deploys into the `recommendations` namespace. `k8s/configmap.yaml` holds non-secret env vars; `k8s/secret.yaml` holds `PGVECTOR_URL`. The service is accessed from `metadataservice` via `GET /recommendations` (proxied through the metadata service's `/recommendations` route).
