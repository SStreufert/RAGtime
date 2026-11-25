import os

from pydantic_settings import BaseSettings
from pydantic import SecretStr


class Settings(BaseSettings):
    # OpenAI-compatible endpoints (Ollama / LM Studio / anything with a /v1 API).
    # Base URLs must include the /v1 suffix.
    chat_base_url: str = "http://localhost:11434/v1"
    embeddings_base_url: str = "http://localhost:11434/v1"
    judge_base_url: str | None = None
    # Local backends ignore the key, but the OpenAI client rejects an empty one.
    # SecretStr is required by langchain_openai's OpenAIEmbeddings/ChatOpenAI.
    chat_api_key: SecretStr = SecretStr("lm-studio")
    embeddings_api_key: SecretStr = SecretStr("lm-studio")
    judge_api_key: SecretStr | None = None

    # Launch-time endpoint availability check (healthcheck.py); set to 1 to bypass.
    skip_endpoint_check: bool = False

    embedding_model: str = "nomic-embed-text"
    chat_model: str = "granite4.1:3b"
    judge_model: str = "granite4.1:3b"
    judge_temperature: float = 0.2

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 7

    # Generation
    temperature: float = 0.2
    max_tokens: int = 4096

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rag-pipeline"
    scorer_timeout: float = 0.0

    # MLflow backend reachability probe (tracker.py). With an HTTP tracking URI
    # and no server running, MLflow's default retry policy (7 retries, backoff 2)
    # blocks ~4.4 minutes per call; tracker probes reachability with one short
    # request instead and caches the failure for the cooldown window.
    mlflow_probe_timeout: float = 3.0
    mlflow_probe_cooldown: float = 600.0

    # Paths
    chroma_dir: str = "./chroma_db"
    data_dir: str = "./data"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def effective_judge_base_url(self) -> str:
        return self.judge_base_url or self.chat_base_url

    @property
    def effective_judge_api_key(self) -> str:
        key = self.judge_api_key or self.chat_api_key
        return key.get_secret_value() if isinstance(key, SecretStr) else key


settings = Settings()


def configure_judge() -> None:
    """Point MLflow's `openai:/` scorers at the judge endpoint.

    MLflow reads these from the process environment (mlflow/metrics/genai/model_utils.py),
    so this is the single place that sets them.
    """
    os.environ["OPENAI_API_BASE"] = settings.effective_judge_base_url
    os.environ["OPENAI_API_KEY"] = settings.effective_judge_api_key
