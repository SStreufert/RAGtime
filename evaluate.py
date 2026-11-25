import json
from pathlib import Path

import mlflow
import mlflow.genai  # noqa: F401  (explicit import for Pylance)
import pandas as pd
import typer

from config import settings
from eval_run import (
    RECONCILE_OPTION,
    check_failures,
    execute_eval,
    validate_or_exit,
)
from healthcheck import ensure_or_exit

# Module-level MLflow setup is deferred to first command execution to avoid
# requiring a live MLflow server during pytest collection.
_EXPERIMENT_SET = False


def _ensure_experiment():
    global _EXPERIMENT_SET
    if not _EXPERIMENT_SET:
        mlflow.set_experiment(settings.mlflow_experiment_name)
        _EXPERIMENT_SET = True


app = typer.Typer(help="RAG evaluation with MLflow GenAI scorers")


def _check_deps():
    try:
        import mlflow.genai  # noqa: F401
    except ImportError:
        typer.echo(
            "mlflow not installed or version too old. Install with: pip install mlflow>=2.17",
            err=True,
        )
        raise typer.Exit(1)


def _load_eval_dataset(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        typer.echo(f"Evaluation dataset not found: {p}", err=True)
        raise typer.Exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to evaluate"),
    file: str = typer.Option(..., help="Path to MLflow evaluation dataset JSON"),
    reconcile: bool = RECONCILE_OPTION,
):
    """Evaluate a single question against the RAG pipeline."""
    _check_deps()
    _ensure_experiment()

    dataset = _load_eval_dataset(file)

    row = next(
        (item for item in dataset if item["inputs"]["question"] == question),
        None,
    )
    in_dataset = row is not None
    if row is None:
        row = {"inputs": {"question": question}}
    else:
        # Only keep inputs and expectations to avoid confusing the scorers
        row = {
            "inputs": row.get("inputs", {}),
            "expectations": row.get("expectations", {}),
        }

    frame = pd.DataFrame([row])
    validate_or_exit(frame, require_expectations=in_dataset)
    ensure_or_exit(("chat", "embeddings", "judge"))

    results = execute_eval(
        frame,
        run_name=f"query-{hash(question) % 10000:04d}",
        reconcile=reconcile,
    )

    typer.echo(f"\nQuestion: {question}")
    typer.echo(f"\nScores:")
    for metric, avg in results.metrics.items():
        typer.echo(f"  {metric}: {avg:.4f}")

    check_failures(results.result_df)


@app.command()
def batch(
    file: str = typer.Option(..., help="Path to MLflow evaluation dataset JSON"),
    reconcile: bool = RECONCILE_OPTION,
):
    """Batch-evaluate all questions and log aggregate scores."""
    _check_deps()
    _ensure_experiment()

    dataset = _load_eval_dataset(file)

    # Filter to only include inputs and expectations
    filtered_dataset = [
        {"inputs": item.get("inputs", {}), "expectations": item.get("expectations", {})}
        for item in dataset
    ]

    typer.echo(f"Evaluating {len(filtered_dataset)} questions...")
    frame = pd.DataFrame(filtered_dataset)
    validate_or_exit(frame)
    ensure_or_exit(("chat", "embeddings", "judge"))

    results = execute_eval(frame, run_name="batch-evaluation", reconcile=reconcile)

    typer.echo(f"\nAggregate scores ({len(dataset)} questions):")
    for metric, avg in results.metrics.items():
        typer.echo(f"  {metric}: {avg:.4f}")

    check_failures(results.result_df)


@app.command()
def questions(
    file: str = typer.Option(..., help="Path to MLflow evaluation dataset JSON"),
):
    """List available evaluation questions."""
    dataset = _load_eval_dataset(file)
    for i, item in enumerate(dataset):
        q = item["inputs"]["question"]
        cat = item.get("tags", {}).get("category", "")
        typer.echo(f"{i + 1}. [{cat}] {q}")


if __name__ == "__main__":
    app()
