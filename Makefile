.PHONY: dev run lint test eval eval-offline build help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Run development server (uvicorn, hot-reload)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run: ## Run production server (gunicorn + uvicorn workers)
	gunicorn -c gunicorn.conf.py src.api.main:app

lint: ## Run ruff linter
	ruff check .

test: ## Run tests
	pytest -v

eval-offline: ## Ranking eval, no-LLM cases, gated on evals/baseline.json (what CI runs)
	python -m evals.run --offline --check evals/baseline.json

eval: ## Ranking eval, all cases, using LLM_PROVIDER (fails if any case falls back)
	python -m evals.run

build: ## Build Docker image
	docker build -t videostreamingplatform-recommendations:latest .

up: ## Start local dev stack (API + Ollama)
	docker compose up -d

down: ## Stop local dev stack
	docker compose down

embed: ## Run embedding batch job
	python -m src.embeddings.embed_videos

consume-embeddings: ## Run the Kafka → pgvector embeddings consumer
	python -m src.consumers.embeddings_consumer
