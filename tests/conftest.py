import os

import pytest

# Diagnostic scripts with module-level live LLM calls — runnable standalone,
# but importing them (pytest collection) would execute a real query.
collect_ignore = [
    "final_test.py",
    "test_standard_attrs.py",
    "test_trace.py",
    "test_trace_exp.py",
    "test_trace_tokens.py",
    "test_ui_improved.py",
]


def pytest_configure(config):
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@pytest.fixture(autouse=True)
def _judge_env():
    """Point MLflow's openai:/ judges at the configured endpoint for every test."""
    from config import configure_judge

    configure_judge()


@pytest.fixture(autouse=True)
def mlflow_local_store(tmp_path):
    """Redirect mlflow's fluent tracking URI to a throwaway sqlite store.

    tracker.py calls mlflow.set_tracking_uri() at import (live URI from .env),
    and fluent state is process-global — without this, any test that starts a
    run writes to the live server. Restores URI and active experiment after.
    """
    import mlflow

    from config import settings

    prev_uri = mlflow.get_tracking_uri()
    prev_experiment = settings.mlflow_experiment_name
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    mlflow.set_experiment("pytest-local")
    try:
        yield
    finally:
        mlflow.set_tracking_uri(prev_uri)
        # Restoring the active experiment is a fluent call that hits the
        # tracking backend; against a remote server that is down it storms
        # MLflow's retry policy (~4.4 min per test teardown). Local stores
        # restore cheaply; for http(s) URIs skip it — each fixture run sets
        # its own experiment anyway.
        if not prev_uri.startswith(("http://", "https://")):
            mlflow.set_experiment(prev_experiment)
