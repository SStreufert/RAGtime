"""Run an MLflow GenAI evaluation over a registered dataset preset.

One module owns the eval-run body: validate golden questions, build the judge
scorers, answer every golden question through the RAG pipeline, log one MLflow
run, and reconcile judge verdicts against their rationales.

Usage:
    python eval_run.py swan-lagoon            # full dataset (linked run)
    python eval_run.py my-documents --limit 2 # smoke test (anonymous snapshot)

Exits non-zero if any row failed or any scorer errored.
"""

import os
from dataclasses import dataclass
from typing import Callable

# Cap MLflow's eval fan-out. Defaults ("auto" rate limit) derive ~40 predict /
# 200 score pool workers, and each scored golden question runs its 5 scorers
# concurrently — dozens of simultaneous judge calls. A local single-model
# backend serializes them; queue waits exceed MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
# (300), clients abort and retry, and the run stalls at 0% with
# "fetch failed" scorer errors. With these caps at most 3 rows x 1 scorer hit
# the judge at once.
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "3")
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")
# rag.answer always emits spans, so mlflow's dry-run trace check (and its
# predict_fn auto-wrap) would be one wasted extra prediction per eval.
os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")

import mlflow
import mlflow.genai  # noqa: F401  (explicit import for Pylance)
import typer
from mlflow.genai.datasets import search_datasets
from mlflow.genai.scorers import (
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    RetrievalSufficiency,
)

from config import configure_judge, settings
from eval_consistency import maybe_reconcile_run
from golden_questions import validation_errors
from healthcheck import ensure_or_exit
import tracker

app = typer.Typer(
    add_completion=False, help="Run an eval over a registered dataset preset"
)

RECONCILE_OPTION = typer.Option(
    True,
    "--reconcile/--no-reconcile",
    help="Reconcile judge verdicts with rationale text after scoring",
)


@dataclass(frozen=True)
class DatasetPreset:
    """A name standing for one registered dataset and its eval run name.

    The dataset is looked up in MLflow by name — no hardcoded dataset IDs,
    which differ per tracking store.
    """

    run_name: str


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "swan-lagoon": DatasetPreset(
        run_name="eval-swan-lagoon",
    ),
    "my-documents": DatasetPreset(
        run_name="eval-my-documents",
    ),
}


def resolve_dataset(preset_name: str):
    """Resolve a dataset by searching MLflow for a dataset with that name.

    Raises typer.Exit(1) if the dataset is not registered, with a hint on
    how to import it.
    """
    results = search_datasets(filter_string=f"name = '{preset_name}'")
    if not results:
        typer.echo(
            f"Dataset {preset_name!r} not found in MLflow. "
            f"Import it first: python scripts/import_dataset.py {preset_name}",
            err=True,
        )
        raise typer.Exit(1)
    return results[0]


def build_scorers() -> list:
    """The five judge scorers every eval run grades golden questions with."""
    judge_model = f"openai:/{settings.judge_model}"
    inference_params = {"temperature": settings.judge_temperature}
    scorers = [
        RetrievalGroundedness(model=judge_model, inference_params=inference_params),
        RetrievalRelevance(model=judge_model, inference_params=inference_params),
        RelevanceToQuery(model=judge_model, inference_params=inference_params),
        Correctness(model=judge_model, inference_params=inference_params),
        RetrievalSufficiency(model=judge_model, inference_params=inference_params),
    ]
    for s in scorers:
        s.timeout = settings.scorer_timeout
    return scorers


def rag_predict_fn(question: str) -> str:
    """predict_fn for mlflow.genai.evaluate() — answers via the RAG pipeline.

    Scorers read the trace, and spans are always emitted; nested per-question
    query runs are not wanted inside an eval run, so tracking is off.
    """
    from rag import answer

    answer_text, _docs, _used = answer(question, k=settings.top_k, track=False)
    if not answer_text or not answer_text.strip():
        # An empty answer is a generation failure (e.g. a reasoning model
        # burning the whole max_tokens budget on reasoning_content), not a
        # gradeable response: fail the row so scorers cannot vacuously pass
        # it and check_failures exits non-zero.
        raise ValueError(
            "RAG pipeline returned an empty answer — generation likely hit "
            "max_tokens with reasoning-only output (raise MAX_TOKENS or use "
            "a non-reasoning chat model)"
        )
    return answer_text


def _validation_errors(df, *, require_expectations: bool = True) -> list[str]:
    """Per-row dataset problems citing record ids. Pure; no I/O.

    Thin DataFrame wrapper over ``golden_questions.validation_errors`` — the
    same contract the import step enforces, so a dataset that imports is one
    that evaluates. Row-by-row conversion keeps zero-column frames visible
    (``to_dict("records")`` would silently drop them).
    """
    return validation_errors(
        [row.to_dict() for _, row in df.iterrows()],
        require_expectations=require_expectations,
    )


def validate_or_exit(df, *, require_expectations: bool = True) -> None:
    """Refuse to evaluate malformed golden questions; exit non-zero."""
    errors = _validation_errors(df, require_expectations=require_expectations)
    if errors:
        print(f"Dataset validation failed ({len(errors)} problem(s)):")
        for e in errors:
            print(f"  ! {e}")
        print("Fix the records and retry.")
        raise typer.Exit(1)


def _failed_rows(result_df) -> list[str]:
    """Summaries of golden questions with predict errors or scorer errors."""
    failures = []
    error_cols = [
        c
        for c in result_df.columns
        if c == "error_message" or c.endswith("/error_message")
    ]
    for _, row in result_df.iterrows():
        row_errors = []
        for col in error_cols:
            value = row.get(col)
            if value is None or str(value) == "nan" or str(value).strip() == "":
                continue
            row_errors.append(f"{col}: {value}")
        if row_errors:
            question = (row.get("inputs") or {}).get("question", "?")
            failures.append(f"{question!r} -> {'; '.join(row_errors)}")
    return failures


def check_failures(result_df) -> None:
    """Print per-row failures and exit non-zero if any occurred."""
    failures = _failed_rows(result_df) if result_df is not None else []
    if failures:
        print(f"\n{len(failures)} row(s) failed:")
        for f in failures:
            print(f"  ! {f}")
        raise typer.Exit(1)


def execute_eval(
    data,
    *,
    run_name: str,
    reconcile: bool = True,
    predict_fn: Callable[[str], str] = rag_predict_fn,
    scorers: list | None = None,
):
    """Log one eval run over ``data``; return the mlflow evaluate() results."""
    configure_judge()
    # Eval cannot run without the tracking backend; fail fast with a clear
    # message instead of MLflow's multi-minute retry storm inside start_run.
    if not tracker.is_available():
        typer.echo(
            "MLflow tracking backend is unreachable; evaluation requires it. "
            "Start the tracking server, e.g.: "
            "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 "
            f"(tracking URI: {mlflow.get_tracking_uri()})",
            err=True,
        )
        raise typer.Exit(1)
    if scorers is None:
        scorers = build_scorers()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(tracker.eval_run_params())
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        maybe_reconcile_run(
            results.run_id, enabled=reconcile, metrics=results.metrics
        )
    return results


@app.command()
def main(
    preset: str = typer.Argument(..., help="Dataset preset name"),
    limit: int = typer.Option(
        None, min=1, help="Smoke test: evaluate only the first N golden questions"
    ),
    reconcile: bool = RECONCILE_OPTION,
):
    """Evaluate the RAG pipeline against a registered dataset preset."""
    try:
        spec = DATASET_PRESETS[preset]
    except KeyError:
        known = ", ".join(sorted(DATASET_PRESETS))
        raise typer.BadParameter(f"unknown preset {preset!r} (known: {known})")

    dataset = resolve_dataset(preset)
    df = dataset.to_df()
    print(f"Loaded dataset {dataset.name!r} ({len(df)} records)")

    validate_or_exit(df)

    # Local dataset problems first, then the three endpoints (chat+embeddings
    # via rag_predict_fn, judge via the MLflow scorers).
    ensure_or_exit(("chat", "embeddings", "judge"))

    if limit:
        data = df.head(limit)
        print(f"Smoke mode: evaluating first {limit} records (anonymous snapshot)")
    else:
        data = dataset  # run gets linked to the registered dataset

    print(f"Judge: {settings.judge_model} @ temp={settings.judge_temperature}")
    results = execute_eval(data, run_name=spec.run_name, reconcile=reconcile)

    print(f"\nRun: {results.run_id} ({spec.run_name})")
    print("Metrics:")
    for metric, avg in sorted(results.metrics.items()):
        print(f"  {metric}: {avg:.4f}")

    check_failures(results.result_df)

    exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if exp:
        print(
            f"\nUI: http://127.0.0.1:5000/#/experiments/{exp.experiment_id}"
            f"/runs/{results.run_id}"
        )


if __name__ == "__main__":
    app()
