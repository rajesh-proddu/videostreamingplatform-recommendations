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

    # pgvector
    pgvector_url: str = field(default_factory=lambda: os.getenv("PGVECTOR_URL", "postgresql://recouser:recopass@localhost:5432/recommendations"))
    embedding_dimension: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1536")))

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


config = Config()
