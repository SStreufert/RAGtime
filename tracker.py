import hashlib
import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import os
os.environ["MLFLOW_TRACING_ENABLED"] = "true"

# Check for tracking disable via env var (set by run_app.py wrapper)
# Must be done BEFORE importing mlflow
_TRACKING_DISABLED = os.environ.get("RAG_DISABLE_TRACKING") == "1"

from config import settings


def disable_tracking() -> None:
    """Disable MLflow tracking for the current process."""
    global _TRACKING_DISABLED
    _TRACKING_DISABLED = True


def is_tracking_disabled() -> bool:
    """Check if MLflow tracking has been explicitly disabled."""
    return _TRACKING_DISABLED


# MLflow is imported and tracking URI is set at module level (fast, local state).
# Experiment setup and tracing are deferred to _ensure_initialized() to avoid
# network calls at import time when the tracking URI is an HTTP endpoint
# with no server running (MLflow defaults: 7 retries × 120s timeout = stall).
if _TRACKING_DISABLED:
    MLFLOW_AVAILABLE = False
    _mlflow = None
    print("MLflow tracking disabled via RAG_DISABLE_TRACKING=1")
else:
    try:
        import mlflow as _mlflow

        MLFLOW_AVAILABLE = True
        _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    except ImportError:
        MLFLOW_AVAILABLE = False
        _mlflow = None
        warnings.warn(
            "MLflow not installed. Tracking disabled. pip install mlflow to enable."
        )

_MLFLOW_INITIALIZED = False
# Monotonic deadline until which the tracking backend is treated as unreachable
# (negative cache). 0.0 means no probe failure recorded.
_tracking_unreachable_until = 0.0


def _probe_tracking_backend(uri: str, timeout: float) -> bool:
    """Single cheap reachability probe for http(s) tracking URIs.

    MLflow's default retry policy (7 retries, exponential backoff) turns one
    call to a dead server into a ~4.4 minute stall. One short request with no
    retries decides reachability instead; ANY HTTP response (even 4xx/5xx)
    proves a server is listening. Non-http(s) URIs (sqlite, file) are local —
    nothing to probe.
    """
    if not uri.startswith(("http://", "https://")):
        return True
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    url = (
        uri.rstrip("/")
        + "/api/2.0/mlflow/experiments/get-by-name?experiment_name=__probe__"
    )
    try:
        with urlopen(Request(url, method="GET"), timeout=timeout):
            return True
    except HTTPError:
        # The server answered (auth required, unknown route, ...) — reachable.
        return True
    except (URLError, OSError):
        # Refused / timed out / unresolvable — treat as down.
        return False


def _ensure_initialized() -> bool:
    """Lazily initialize MLflow experiment + tracing on first use.

    Returns True if MLflow is available and initialization succeeded.
    Safe to call multiple times; subsequent calls are no-ops.

    While the tracking backend is down, one short probe is made per cooldown
    window and the failure is cached — callers get an immediate False instead
    of MLflow's multi-minute retry storm on every single call. Starting the
    server recovers tracking automatically once the cooldown expires.
    """
    global _MLFLOW_INITIALIZED, _tracking_unreachable_until
    if not MLFLOW_AVAILABLE or _TRACKING_DISABLED or _mlflow is None:
        return False
    if _MLFLOW_INITIALIZED:
        return True

    now = time.monotonic()
    if now < _tracking_unreachable_until:
        return False
    if not _probe_tracking_backend(
        _mlflow.get_tracking_uri(), settings.mlflow_probe_timeout
    ):
        _tracking_unreachable_until = now + settings.mlflow_probe_cooldown
        warnings.warn(
            f"MLflow tracking backend at {_mlflow.get_tracking_uri()} is unreachable; "
            f"continuing without tracking (re-probing in {settings.mlflow_probe_cooldown:.0f}s). "
            "To log traces, start the tracking server, e.g.: "
            "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 "
            "(or fix MLFLOW_TRACKING_URI in .env)."
        )
        return False

    try:
        _mlflow.set_experiment(settings.mlflow_experiment_name)
        _mlflow.tracing.enable()
        # Debug: confirm tracing is enabled (only on first initialization)
        print(f"MLflow version: {_mlflow.__version__}")
        print(f"MLflow tracking URI: {_mlflow.get_tracking_uri()}")
        exp = _mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        print(f"MLflow experiment name: {exp.name if exp else 'Not found'} (ID: {exp.experiment_id if exp else 'N/A'})")
        print("MLflow tracing.enable() called")
        if hasattr(_mlflow.tracing, 'is_tracing_enabled'):
            print(f"MLflow tracing is_tracing_enabled: {_mlflow.tracing.is_tracing_enabled()}")
        else:
            print("MLflow tracing is_tracing_enabled attribute not found")
        _MLFLOW_INITIALIZED = True
        return True
    except Exception:  # noqa: BLE001 — initialization failure shouldn't crash the app
        # Backend answered the probe but init failed anyway (e.g. it died in
        # between): cache the failure too, so callers don't re-pay MLflow's
        # multi-minute retry storm on every subsequent call.
        _tracking_unreachable_until = time.monotonic() + settings.mlflow_probe_cooldown
        return False


def is_available() -> bool:
    """Check if MLflow tracking is available, triggering lazy init if needed.

    Returns the actual initialization result: False when the backend is
    unreachable, not just when mlflow is uninstalled or tracking is disabled.
    """
    if not MLFLOW_AVAILABLE or _TRACKING_DISABLED:
        return False
    return _ensure_initialized()


def _hash_prompt(template: str) -> str:
    return hashlib.sha256(template.strip().encode()).hexdigest()[:12]


def eval_run_params() -> dict[str, Any]:
    """Param set for evaluation parent runs (ingestion + judge + generation).

    Single source of truth so eval scripts don't omit generation params
    (chat_model/temperature/top_k) that the MLflow UI displays as columns.
    """
    return {
        "chat_model": settings.chat_model,
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "judge_model": settings.judge_model,
        "judge_temperature": settings.judge_temperature,
        "max_tokens": settings.max_tokens,
    }


@contextmanager
def start_run(run_name: str | None = None, nested: bool = False):
    if not MLFLOW_AVAILABLE or _TRACKING_DISABLED:
        yield None
        return
    if not _ensure_initialized():
        yield None
        return

    with _mlflow.start_run(run_name=run_name, nested=nested) as run:
        yield run


def log_ingestion(
    docs: list,
    chunks: list,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    ingestion_time: float,
    embedding_dim: int | None = None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    if not MLFLOW_AVAILABLE or _TRACKING_DISABLED:
        return
    if not _ensure_initialized():
        return

    chars = [len(d.page_content) for d in docs]
    chunk_sizes = [len(c.page_content) for c in chunks]

    _mlflow.log_params(
        {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": embedding_model,
            "docs_count": len(docs),
            **(extra_params or {}),
        }
    )

    _mlflow.log_metrics(
        {
            "total_pages": len(docs),
            "total_chars": sum(chars),
            "avg_chars_per_page": sum(chars) / max(len(chars), 1),
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / max(len(chunk_sizes), 1),
            "ingestion_time_sec": round(ingestion_time, 3),
            "embedding_dimension": embedding_dim or 0,
        }
    )

    config_snapshot = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "docs_count": len(docs),
        "total_chunks": len(chunks),
    }
    _mlflow.log_dict(config_snapshot, "config_snapshot.json")

    samples = [
        {
            "index": i,
            "text": c.page_content[:500],
            "source": c.metadata.get("source", ""),
            "page": c.metadata.get("page"),
        }
        for i, c in enumerate(chunks[:10])
    ]
    _mlflow.log_dict(samples, "chunk_samples.json")


def log_query(
    question: str,
    answer: str,
    docs: list,
    used_sources: list[int] | None = None,
    retrieval_score: float | None = None,
    retrieval_latency_ms: float = 0,
    generation_latency_ms: float = 0,
    prompt_template: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> None:
    if not MLFLOW_AVAILABLE or _TRACKING_DISABLED:
        return
    if not _ensure_initialized():
        return

    total_latency_ms = retrieval_latency_ms + generation_latency_ms
    context_text = "\n".join(d.page_content for d in docs)
    context_tokens = len(context_text) // 4
    answer_tokens = len(answer) // 4

    _mlflow.log_params(
        {
            "chat_model": settings.chat_model,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "prompt_template_hash": _hash_prompt(prompt_template or ""),
            # Ingestion parameters for correlation with query performance
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            # Judge parameters (affect evaluation/scoring)
            "judge_model": settings.judge_model,
            "judge_temperature": settings.judge_temperature,
            # Generation parameter
            "max_tokens": settings.max_tokens,
            **(extra_params or {}),
        }
    )

    _mlflow.log_metrics(
        {
            "retrieval_score": retrieval_score or 0,
            "retrieval_latency_ms": round(retrieval_latency_ms, 1),
            "generation_latency_ms": round(generation_latency_ms, 1),
            "total_latency_ms": round(total_latency_ms, 1),
            "context_tokens": context_tokens,
            "answer_tokens": answer_tokens,
        }
    )

    query_log = {
        "question": question,
        "answer": answer,
        "sources_used": used_sources,
        "sources": [
            {
                "source": d.metadata.get("source", ""),
                "page": d.metadata.get("page"),
                "text": d.page_content[:300],
            }
            for d in docs
        ],
        "retrieval_score": retrieval_score,
    }
    _mlflow.log_dict(query_log, "query_log.json")

    if prompt_template:
        _mlflow.log_text(prompt_template, "prompt_template.txt")
