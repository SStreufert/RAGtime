import tempfile
import os
import pytest
from unittest.mock import patch

import tracker


def test_tracker_availability():
    assert tracker.is_available() is True


def test_hash_prompt():
    h1 = tracker._hash_prompt("Hello {context}")
    h2 = tracker._hash_prompt("Hello {context}")
    h3 = tracker._hash_prompt("Different {context}")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12


def test_start_run_skips_when_unavailable(tmp_path):
    """Verify start_run yields None when tracking is disabled."""
    original = tracker.MLFLOW_AVAILABLE
    tracker.MLFLOW_AVAILABLE = False
    try:
        with tracker.start_run(run_name="test-run") as run:
            assert run is None
    finally:
        tracker.MLFLOW_AVAILABLE = original


def test_log_ingestion_skips_when_unavailable():
    """Verify log_ingestion is a no-op when tracking is disabled."""
    original = tracker.MLFLOW_AVAILABLE
    tracker.MLFLOW_AVAILABLE = False
    try:
        # Should not raise
        tracker.log_ingestion(
            docs=[], chunks=[], chunk_size=500, chunk_overlap=100,
            embedding_model="test", ingestion_time=0,
        )
    finally:
        tracker.MLFLOW_AVAILABLE = original


def test_log_query_skips_when_unavailable():
    """Verify log_query is a no-op when tracking is disabled."""
    original = tracker.MLFLOW_AVAILABLE
    tracker.MLFLOW_AVAILABLE = False
    try:
        # Should not raise
        tracker.log_query(
            question="test", answer="test", docs=[],
        )
    finally:
        tracker.MLFLOW_AVAILABLE = original


def test_start_run_creates_run():
    import mlflow

    with tracker.start_run(run_name="test-run") as run:
        assert run is not None
        mlflow.log_param("test_param", "value")
        mlflow.log_metric("test_metric", 1.0)


def test_log_ingestion():
    from langchain_core.documents import Document
    import mlflow

    docs = [
        Document(page_content="page 1 text", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="page 2 text", metadata={"source": "test.pdf", "page": 1}),
    ]
    chunks = [
        Document(page_content="chunk 1", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="chunk 2", metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="chunk 3", metadata={"source": "test.pdf", "page": 1}),
    ]

    with tracker.start_run(run_name="test-ingestion"):
        tracker.log_ingestion(
            docs=docs,
            chunks=chunks,
            chunk_size=500,
            chunk_overlap=100,
            embedding_model="nomic-embed-text",
            ingestion_time=1.23,
        )
        run = mlflow.active_run()
        assert run is not None


def test_log_query():
    from langchain_core.documents import Document
    import mlflow

    docs = [
        Document(page_content="context about AI", metadata={"source": "test.pdf", "page": 0}),
    ]

    with tracker.start_run(run_name="test-query"):
        tracker.log_query(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            docs=docs,
            retrieval_score=0.85,
            retrieval_latency_ms=120.5,
            generation_latency_ms=1500.3,
            prompt_template="Answer: {question}",
        )
        run = mlflow.active_run()
        assert run is not None


def test_eval_run_params_include_generation_config():
    """Regression: eval runs must log chat_model/temperature/top_k.

    Without these params the MLflow evaluation-runs page shows empty
    chat_model/temperature/top_k columns and the run Overview omits them.
    """
    from config import settings

    params = tracker.eval_run_params()
    assert params["chat_model"] == settings.chat_model
    assert params["temperature"] == settings.temperature
    assert params["top_k"] == settings.top_k


def test_eval_run_params_full_set():
    from config import settings

    assert tracker.eval_run_params() == {
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
