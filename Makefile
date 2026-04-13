.PHONY: dev lint test build help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Run development server
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

lint: ## Run ruff linter
	ruff check .

test: ## Run tests
	pytest -v

build: ## Build Docker image
	docker build -t videostreamingplatform-recommendations:latest .

up: ## Start local dev stack (API + Ollama)
	docker compose up -d

down: ## Stop local dev stack
	docker compose down

embed: ## Run embedding batch job
	python -m src.embeddings.embed_videos
