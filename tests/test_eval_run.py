"""Tests for the eval_run module's interface.

The interface is the test surface: presets, dataset validation, failure
reporting, scorer construction, and one execute_eval round-trip with an
injected predict function and scorer so no judge is needed.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import typer

import eval_run

GOOD_ROW = {
    "inputs": {"question": "What is RAG?"},
    "expectations": {"expected_response": "Retrieval-Augmented Generation."},
}


def _frame(*rows):
    return pd.DataFrame(list(rows))


def _cli_text(result) -> str:
    parts = [getattr(result, "output", "") or ""]
    stderr = getattr(result, "stderr", None)
    if isinstance(stderr, str):
        parts.append(stderr)
    if result.exception is not None:
        parts.append(str(result.exception))
    return "\n".join(parts)


class TestPresets:
    def test_swan_lagoon_preset(self):
        preset = eval_run.DATASET_PRESETS["swan-lagoon"]
        assert preset.run_name == "eval-swan-lagoon"

    def test_my_documents_preset(self):
        preset = eval_run.DATASET_PRESETS["my-documents"]
        assert preset.run_name == "eval-my-documents"

    def test_presets_carry_no_hardcoded_dataset_ids(self):
        """Dataset IDs differ per tracking store; presets hold names only."""
        assert all(
            not hasattr(preset, "dataset_id")
            for preset in eval_run.DATASET_PRESETS.values()
        )

    def test_unknown_preset_rejected_before_any_io(self):
        from typer.testing import CliRunner

        result = CliRunner().invoke(eval_run.app, ["nonsense"])
        assert result.exit_code != 0
        assert "unknown preset" in _cli_text(result)


class TestResolveDataset:
    def test_finds_dataset_by_name(self, monkeypatch):
        found = SimpleNamespace(dataset_id="d-123", name="swan-lagoon")
        seen = {}

        def fake_search(filter_string):
            seen["filter"] = filter_string
            return [found]

        monkeypatch.setattr(eval_run, "search_datasets", fake_search)
        assert eval_run.resolve_dataset("swan-lagoon") is found
        assert seen["filter"] == "name = 'swan-lagoon'"

    def test_missing_dataset_exits_with_import_hint(self, monkeypatch):
        monkeypatch.setattr(eval_run, "search_datasets", lambda filter_string: [])
        with pytest.raises(typer.Exit) as exc:
            eval_run.resolve_dataset("swan-lagoon")
        assert exc.value.exit_code == 1


class TestValidation:
    def test_good_rows_pass(self):
        assert eval_run._validation_errors(_frame(GOOD_ROW)) == []

    def test_missing_inputs(self):
        errors = eval_run._validation_errors(_frame({}))
        assert len(errors) == 1
        assert "inputs is missing" in errors[0]

    def test_inputs_must_be_exactly_question(self):
        row = {"inputs": {"question": "q", "extra": "x"}, "expectations": {}}
        errors = eval_run._validation_errors(_frame(row))
        assert any("exactly" in e for e in errors)

    def test_empty_question(self):
        row = {"inputs": {"question": "   "}, "expectations": {}}
        errors = eval_run._validation_errors(_frame(row))
        assert any("non-empty string" in e for e in errors)

    def test_expectations_required_by_default(self):
        errors = eval_run._validation_errors(_frame({"inputs": {"question": "q"}}))
        assert any("expectations" in e for e in errors)

    def test_expectations_optional_for_ad_hoc_questions(self):
        row = {"inputs": {"question": "q"}}
        assert (
            eval_run._validation_errors(
                _frame(row), require_expectations=False
            )
            == []
        )

    def test_validate_or_exit_passes_good_rows(self):
        assert eval_run.validate_or_exit(_frame(GOOD_ROW)) is None

    def test_validate_or_exit_exits_on_bad_rows(self, capsys):
        with pytest.raises(typer.Exit) as exc:
            eval_run.validate_or_exit(_frame({}))
        assert exc.value.exit_code == 1
        assert "Dataset validation failed" in capsys.readouterr().out


class TestFailureReporting:
    def test_failed_rows_reports_error_columns(self):
        df = _frame(
            {"inputs": {"question": "q1"}, "error_message": "boom"},
            {"inputs": {"question": "q2"}, "scorer/error_message": ""},
        )
        failures = eval_run._failed_rows(df)
        assert len(failures) == 1
        assert "q1" in failures[0]
        assert "boom" in failures[0]

    def test_failed_rows_clean_frame(self):
        df = _frame({"inputs": {"question": "ok"}, "error_message": ""})
        assert eval_run._failed_rows(df) == []

    def test_check_failures_exits_on_failure(self, capsys):
        df = _frame({"inputs": {"question": "q1"}, "error_message": "boom"})
        with pytest.raises(typer.Exit) as exc:
            eval_run.check_failures(df)
        assert exc.value.exit_code == 1
        assert "row(s) failed" in capsys.readouterr().out

    def test_check_failures_passes_on_clean_and_none(self):
        assert eval_run.check_failures(None) is None
        clean = _frame({"inputs": {"question": "ok"}, "error_message": ""})
        assert eval_run.check_failures(clean) is None


class TestScorers:
    def test_build_scorers_five_with_retrieval_sufficiency(self):
        from mlflow.genai.scorers import RetrievalRelevance, RetrievalSufficiency

        scorers = eval_run.build_scorers()
        assert len(scorers) == 5
        assert any(isinstance(s, RetrievalRelevance) for s in scorers)
        assert any(isinstance(s, RetrievalSufficiency) for s in scorers)

    def test_build_scorers_temperature_only_and_timeout(self):
        from config import settings

        for scorer in eval_run.build_scorers():
            assert scorer.inference_params == {
                "temperature": settings.judge_temperature
            }
            assert scorer.timeout == settings.scorer_timeout


class TestExecuteEval:
    DATA = _frame(GOOD_ROW)

    @staticmethod
    def _constant_scorer():
        from mlflow.genai.scorers import scorer

        @scorer
        def constant_score(outputs) -> float:
            return 1.0

        return constant_score

    def test_logs_run_params_metrics_without_reconcile(self):
        import mlflow

        results = eval_run.execute_eval(
            self.DATA,
            run_name="pytest-eval-run",
            reconcile=False,
            predict_fn=lambda question: "an answer",
            scorers=[self._constant_scorer()],
        )

        assert results.run_id
        assert any("constant_score" in name for name in results.metrics)

        run = mlflow.get_run(results.run_id)
        assert "chat_model" in run.data.params
        assert run.data.tags.get("reconciled") != "true"

    def test_reconcile_flag_marks_run(self):
        import mlflow

        results = eval_run.execute_eval(
            self.DATA,
            run_name="pytest-eval-run-reconciled",
            reconcile=True,
            predict_fn=lambda question: "an answer",
            scorers=[self._constant_scorer()],
        )

        run = mlflow.get_run(results.run_id)
        assert run.data.tags.get("reconciled") == "true"


class TestRagPredictFnGuard:
    """An empty answer is a generation failure, not a gradeable response.

    Run cf181f34: a reasoning model burned the whole max_tokens budget on
    reasoning_content, 23/25 answers were "", and the judges vacuously passed
    the hollow traces. The guard fails the row instead.
    """

    def test_empty_answer_raises(self, monkeypatch):
        import rag

        monkeypatch.setattr(rag, "answer", lambda q, k=3, track=True: ("", [], []))
        with pytest.raises(ValueError, match="empty answer"):
            eval_run.rag_predict_fn("What is RAG?")

    def test_whitespace_answer_raises(self, monkeypatch):
        import rag

        monkeypatch.setattr(rag, "answer", lambda q, k=3, track=True: ("  \n", [], []))
        with pytest.raises(ValueError, match="empty answer"):
            eval_run.rag_predict_fn("What is RAG?")

    def test_nonempty_answer_passes_through(self, monkeypatch):
        import rag

        monkeypatch.setattr(
            rag, "answer", lambda q, k=3, track=True: ("RAG stands for X.", [], [])
        )
        assert eval_run.rag_predict_fn("What is RAG?") == "RAG stands for X."


def test_rag_predict_fn_creates_retriever_span():
    """Regression test: rag_predict_fn's trace must contain a RETRIEVER span.

    The RetrievalGroundedness scorer requires at least one span with
    span_type == 'RETRIEVER' in the trace. Without it, scoring fails with:
    'No retrieval context found in the trace.'
    """
    import json
    import time

    import mlflow

    from eval_run import rag_predict_fn

    with mlflow.start_run() as run:
        result = rag_predict_fn("What is the capital of France?")

    assert isinstance(result, str)
    assert len(result) > 0

    # Trace export is async — wait for the queue to flush, filtering by run id
    # to avoid picking up traces from subsequent tests
    run_id = run.info.run_id
    trace_data = None
    for _ in range(30):
        time.sleep(1)
        traces = mlflow.search_traces(
            filter_string=f"run_id = '{run_id}'",
            max_results=1,
        )
        if len(traces) > 0:
            trace_data = json.loads(traces.iloc[0]["trace"])
            break
    assert trace_data is not None, f"No trace found for run {run_id} after 30s"
    spans = trace_data["data"]["spans"]
    # mlflow.spanType values are JSON-serialized (e.g. '"RETRIEVER"'), so parse them
    span_types = [
        json.loads(s["attributes"].get("mlflow.spanType", '""')) for s in spans
    ]

    assert "RETRIEVER" in span_types, (
        f"Expected RETRIEVER span type in trace, got: {span_types}. "
        "RetrievalGroundedness scorer will fail without it."
    )
