"""Regression tests: entry points must stay responsive when the MLflow backend is down.

Reproduced 2026-11 against a dead ``http://127.0.0.1:9`` backend (hermetic stub
model servers, real rag.answer path): ``mlflow.set_experiment()`` blocked 264.4 s
(MLflow default retry policy: 7 retries x backoff 2), ``tracker.is_available()``
discarded that failure and returned True anyway, so each Streamlit query paid the
retry storm TWICE (~525 s) and every later interaction re-paid it (no failure
caching). The async trace exporter stormed independently per queued trace.

The fix under test: a cheap reachability probe (single request, short timeout)
plus a negative-cache cooldown in ``tracker._ensure_initialized()``, span gating
in ``rag.answer()`` (eval's ``track=False`` path still traces), and a fast clear
exit for eval surfaces.
"""

import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import typer

import tracker

# Loopback discard port: nothing listens, connection refused instantly.
DEAD_URI = "http://127.0.0.1:9"


@pytest.fixture
def dead_backend():
    """Point mlflow at a dead HTTP backend and reset tracker init state."""
    import mlflow

    prev_uri = mlflow.get_tracking_uri()
    prev_initialized = tracker._MLFLOW_INITIALIZED
    prev_unreachable = getattr(tracker, "_tracking_unreachable_until", 0.0)
    mlflow.set_tracking_uri(DEAD_URI)
    tracker._MLFLOW_INITIALIZED = False
    if hasattr(tracker, "_tracking_unreachable_until"):
        tracker._tracking_unreachable_until = 0.0
    yield mlflow
    mlflow.set_tracking_uri(prev_uri)
    tracker._MLFLOW_INITIALIZED = prev_initialized
    if hasattr(tracker, "_tracking_unreachable_until"):
        tracker._tracking_unreachable_until = prev_unreachable


def _fake_store():
    from langchain_core.documents import Document

    doc = Document(
        page_content="Swan Lagoon is a must see destination.",
        metadata={"source": "guide.pdf", "page": 0},
    )
    return SimpleNamespace(
        similarity_search_with_relevance_scores=lambda question, k: [(doc, 0.9)]
    )


def _canned_generate(chain, docs, question):
    return (
        "Swan Lagoon [1].\n\nUSED_SOURCES:\n[1]",
        5.0,
        {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def _fake_span_factory(calls):
    """Fake mlflow fluent start_span: records names, no export machinery.

    Real spans drag MLflow's async trace exporter into the test, which binds
    whatever tracking URI it first sees and can retry-storm in the background
    (observed: ~4.5 min per test in teardown). Offline tests must not export.
    """

    @contextmanager
    def fake_start_span(name=None, span_type=None, **kwargs):
        calls.append(name)
        yield SimpleNamespace(set_attribute=lambda *a, **k: None)

    return fake_start_span


def _install_fake_spans(monkeypatch, calls):
    import mlflow.tracing.fluent as fluent

    monkeypatch.setattr(fluent, "start_span", _fake_span_factory(calls))


class TestTrackerBackendDown:
    def test_is_available_fails_fast_when_backend_down(self, dead_backend):
        """First call must probe once (fast) and return False; the failure must
        be cached so later calls are instant — not another multi-minute storm."""
        t0 = time.perf_counter()
        assert tracker.is_available() is False
        assert time.perf_counter() - t0 < 15, (
            "first call blocked — MLflow retry storm not bypassed"
        )

        t0 = time.perf_counter()
        assert tracker.is_available() is False
        assert time.perf_counter() - t0 < 1, "failure not cached — every call re-storms"


class TestAnswerGating:
    def test_answer_skips_traces_when_backend_down(self, monkeypatch):
        """With tracking unavailable, answer() must not create MLflow spans
        (nothing for the background exporter to storm on) and must still return
        the answer with citations."""
        import rag

        calls = []
        _install_fake_spans(monkeypatch, calls)
        monkeypatch.setattr(tracker, "is_available", lambda: False)
        monkeypatch.setattr(rag, "get_store", lambda: _fake_store())
        monkeypatch.setattr(rag, "_generate_text", _canned_generate)

        text, docs, used = rag.answer("What is a must see?", k=3)

        assert "Swan Lagoon" in text
        assert used == [1]
        assert len(docs) == 1
        assert calls == [], f"spans created while tracking unavailable: {calls}"

    def test_answer_still_traces_for_eval_when_track_false(self, monkeypatch):
        """track=False is the eval path: scorers read these traces, so spans
        must still be created even though per-query tracking is off."""
        import rag

        calls = []
        _install_fake_spans(monkeypatch, calls)
        monkeypatch.setattr(rag, "get_store", lambda: _fake_store())
        monkeypatch.setattr(rag, "_generate_text", _canned_generate)

        rag.answer("What is a must see?", k=3, track=False)

        assert calls == ["rag_query", "retrieval", "generation"]


class TestEvalBackendDown:
    def test_execute_eval_fails_fast_when_backend_down(self, monkeypatch):
        """eval requires MLflow: exit with a clear error immediately instead of
        storming inside mlflow.start_run."""
        import pandas as pd

        import eval_run

        frame = pd.DataFrame(
            [
                {
                    "inputs": {"question": "What is RAG?"},
                    "expectations": {"expected_response": "Retrieval-Augmented Generation."},
                }
            ]
        )
        monkeypatch.setattr(tracker, "is_available", lambda: False)

        with pytest.raises(typer.Exit):
            eval_run.execute_eval(
                frame,
                run_name="pytest-outage",
                predict_fn=lambda question: "an answer",
                scorers=[],
            )
