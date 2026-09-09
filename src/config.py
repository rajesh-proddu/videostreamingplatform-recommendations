"""Environment-based configuration."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1"))
    bedrock_region: str = field(default_factory=lambda: os.getenv("AWS_REGION", "us-east-1"))
    bedrock_model_id: str = field(
        default_factory=lambda: os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
    )
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"))

    # pgvector
    pgvector_url: str = field(default_factory=lambda: os.getenv("PGVECTOR_URL", "postgresql://recouser:recopass@localhost:5432/recommendations"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536")))

    # Embedding fan-out cap for batch jobs / consumers. Ollama is single-process
    # CPU-bound; Bedrock has per-region throughput limits. 8 is a safe default
    # across both providers; bump for hosted high-throughput endpoints.
    max_concurrent_embeds: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_EMBEDS", "8")))

    # Embeddings consumer
    kafka_video_topic: str = field(
        default_factory=lambda: os.getenv("KAFKA_VIDEO_TOPIC", "video-events")
    )
    kafka_embeddings_group_id: str = field(
        default_factory=lambda: os.getenv("KAFKA_GROUP_ID", "embeddings-consumer")
    )
    kafka_embeddings_dlq_topic: str = field(
        default_factory=lambda: os.getenv(
            "KAFKA_DLQ_TOPIC", "video-events-embeddings-dlq"
        )
    )
    embeddings_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDINGS_BATCH_SIZE", "32"))
    )
    embeddings_idle_flush_seconds: float = field(
        default_factory=lambda: float(
            os.getenv("EMBEDDINGS_IDLE_FLUSH_SECONDS", "5.0")
        )
    )

    # Elasticsearch
    elasticsearch_url: str = field(default_factory=lambda: os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    es_video_index: str = field(default_factory=lambda: os.getenv("ES_VIDEO_INDEX", "videos"))

    # Kafka
    kafka_brokers: str = field(default_factory=lambda: os.getenv("KAFKA_BROKERS", "localhost:9092"))
    kafka_watch_topic: str = field(default_factory=lambda: os.getenv("KAFKA_WATCH_TOPIC", "watch-events"))

    # API
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    max_recommendations: int = field(default_factory=lambda: int(os.getenv("MAX_RECOMMENDATIONS", "10")))

    # Cap on deduped candidates handed to rank_candidates. Keeps the LLM ranking
    # prompt bounded regardless of how many sources contribute results.
    max_rank_candidates: int = field(default_factory=lambda: int(os.getenv("MAX_RANK_CANDIDATES", "40")))


config = Config()
