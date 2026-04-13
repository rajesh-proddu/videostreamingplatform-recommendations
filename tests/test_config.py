"""Tests for Config dataclass."""

from src.config import Config


class TestConfigDefaults:
    def test_defaults(self):
        cfg = Config()
        assert cfg.llm_provider == "ollama"
        assert cfg.api_port == 8000
        assert cfg.api_host == "0.0.0.0"
        assert cfg.ollama_base_url == "http://localhost:11434"
        assert cfg.ollama_model == "llama3.1"
        assert cfg.bedrock_region == "us-east-1"
        assert cfg.es_video_index == "videos"
        assert cfg.max_recommendations == 10
        assert cfg.embedding_dimension == 1536

    def test_pgvector_url_default(self):
        cfg = Config()
        assert cfg.pgvector_url == "postgresql://recouser:recopass@localhost:5432/recommendations"


class TestConfigEnvOverrides:
    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")
        monkeypatch.setenv("API_PORT", "9090")
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        monkeypatch.setenv("MAX_RECOMMENDATIONS", "25")

        cfg = Config()
        assert cfg.llm_provider == "bedrock"
        assert cfg.api_port == 9090
        assert cfg.ollama_model == "mistral"
        assert cfg.max_recommendations == 25
