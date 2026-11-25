import os
import pytest


def test_settings_loads_defaults():
    from config import Settings

    s = Settings()
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 100
    assert s.embedding_model == "nomic-embed-text"
    assert s.chat_model == "granite4.1:3b"
    assert s.judge_model == "granite4.1:3b"
    assert s.judge_temperature == 0.2
    assert s.top_k == 7
    assert s.temperature == 0.2
    assert s.mlflow_experiment_name == "rag-pipeline"


def test_settings_types():
    from config import Settings

    s = Settings()
    assert isinstance(s.chunk_size, int)
    assert isinstance(s.chunk_overlap, int)
    assert isinstance(s.top_k, int)
    assert isinstance(s.temperature, float)
    assert isinstance(s.judge_temperature, float)
    assert isinstance(s.max_tokens, int)
    assert isinstance(s.mlflow_tracking_uri, str)
    assert isinstance(s.chroma_dir, str)
    assert isinstance(s.data_dir, str)


def test_settings_env_override(monkeypatch):
    from config import Settings

    monkeypatch.setenv("CHUNK_SIZE", "1000")
    monkeypatch.setenv("CHAT_MODEL", "llama3.1:8b")
    s = Settings()
    assert s.chunk_size == 1000
    assert s.chat_model == "llama3.1:8b"


def test_settings_singleton():
    from config import settings

    assert settings.chunk_size == 1000
    assert settings.embedding_model == "nomic-embed-text"


def test_endpoint_settings():
    from config import settings

    assert settings.chat_base_url == "http://localhost:11434/v1"
    assert settings.embeddings_base_url == "http://localhost:11434/v1"
    assert settings.judge_base_url is None
    assert settings.effective_judge_base_url == settings.chat_base_url
    assert settings.chat_api_key
    assert settings.embeddings_api_key
    assert settings.effective_judge_api_key


def test_mlflow_probe_settings():
    """Verify the offline-probe settings have sensible defaults."""
    from config import settings

    assert settings.mlflow_probe_timeout == 3.0, "probe timeout must be short enough to avoid stalls"
    assert settings.mlflow_probe_cooldown == 600.0, "cooldown allows recovery within 10 minutes"
    assert isinstance(settings.mlflow_probe_timeout, float)
    assert isinstance(settings.mlflow_probe_cooldown, float)


def test_configure_judge_sets_env(monkeypatch):
    from config import configure_judge, settings

    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    configure_judge()

    assert os.environ["OPENAI_API_BASE"] == settings.effective_judge_base_url
    assert os.environ["OPENAI_API_KEY"] == settings.effective_judge_api_key
