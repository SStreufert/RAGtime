"""Import a golden-question JSON file as an MLflow GenAI dataset.

The JSON is validated first (structure + ``REPLACE_ME`` placeholders), so an
import can never succeed with a file the eval would reject. Idempotent by
default: an existing registered dataset is left untouched. Re-import edited
questions with ``--force``, which deletes and recreates the dataset from the
current JSON (prior eval runs keep their data but lose the dataset link).

Usage:
    python scripts/import_dataset.py                # your template (my-documents)
    python scripts/import_dataset.py swan-lagoon    # the shipped example
    python scripts/import_dataset.py --force        # replace after editing questions
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import typer  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.genai.datasets import (  # noqa: E402
    create_dataset,
    delete_dataset,
    search_datasets,
)

from config import settings  # noqa: E402
from golden_questions import (  # noqa: E402
    DATASET_FILES,
    PLACEHOLDER_MARKER,
    placeholder_errors,
    validation_errors,
)

app = typer.Typer(
    add_completion=False,
    help="Import a golden-question JSON file as an MLflow GenAI dataset",
)

DEFAULT_EXPERIMENT_ID = "0"


def _load_records(json_path: Path) -> list:
    """Read the JSON file and sanity-check that it is a non-empty record array."""
    try:
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
    except FileNotFoundError:
        typer.echo(f"Dataset JSON not found: {json_path}", err=True)
        raise typer.Exit(1)
    except json.JSONDecodeError as exc:
        typer.echo(f"{json_path} is not valid JSON: {exc}", err=True)
        raise typer.Exit(1)

    if not isinstance(records, list) or not records:
        typer.echo(
            f"{json_path} must contain a non-empty JSON array of records:\n"
            '[{"inputs": {"question": ...}, '
            '"expectations": {"expected_response": ...}}]',
            err=True,
        )
        raise typer.Exit(1)
    return records


def _check_records(records: list, json_path: Path) -> None:
    """Refuse placeholder or malformed files before touching MLflow."""
    placeholders = placeholder_errors(records)
    if placeholders:
        typer.echo(
            f"Your golden questions are still a template. Edit {json_path} and "
            f"replace every '{PLACEHOLDER_MARKER}' with a real question and the "
            "fact a correct answer must support:",
            err=True,
        )
        for p in placeholders:
            typer.echo(f"  ! {p}", err=True)
        raise typer.Exit(1)

    errors = validation_errors(records)
    if errors:
        typer.echo(f"Dataset validation failed ({len(errors)} problem(s)):", err=True)
        for e in errors:
            typer.echo(f"  ! {e}", err=True)
        typer.echo("Fix the records and retry.", err=True)
        raise typer.Exit(1)


def _echo_tracking_hint(exc: MlflowException) -> None:
    """Translate MLflow backend failures into the fix, not the stack trace."""
    typer.echo(
        f"MLflow error: {exc}\n"
        "GenAI datasets need a SQL-backed MLflow server. Start one with:\n"
        "  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000\n"
        f"(current tracking URI: {settings.mlflow_tracking_uri}; "
        "override with --tracking-uri)",
        err=True,
    )


@app.command()
def main(
    preset: str = typer.Argument(
        "my-documents",
        help="Dataset preset to import (known: " + ", ".join(sorted(DATASET_FILES)) + ")",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Replace an existing registered dataset with the current JSON "
            "contents (prior eval runs keep their data but lose the dataset link)"
        ),
    ),
    tracking_uri: str = typer.Option(
        None,
        "--tracking-uri",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI from .env)",
    ),
) -> None:
    if preset not in DATASET_FILES:
        known = ", ".join(sorted(DATASET_FILES))
        raise typer.BadParameter(f"unknown preset {preset!r} (known: {known})")

    json_path = DATASET_FILES[preset]
    records = _load_records(json_path)
    _check_records(records, json_path)

    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)

    try:
        existing = search_datasets(filter_string=f"name = '{preset}'")
    except MlflowException as exc:
        _echo_tracking_hint(exc)
        raise typer.Exit(1)

    if existing:
        if not force:
            print(
                f"Using existing dataset: {existing[0].dataset_id} ({preset}). "
                "Re-run with --force to replace it from the JSON file."
            )
            return
        # NB: delete_dataset's first positional parameter is `name` (Databricks
        # only) — the dataset_id must be passed as a keyword.
        delete_dataset(dataset_id=existing[0].dataset_id)
        print(
            f"Deleted existing dataset {existing[0].dataset_id} ({preset}) — "
            "prior eval runs keep their data but lose the dataset link."
        )

    try:
        ds = create_dataset(name=preset, experiment_id=[DEFAULT_EXPERIMENT_ID])
        ds.merge_records(records)
    except MlflowException as exc:
        _echo_tracking_hint(exc)
        raise typer.Exit(1)
    print(f"Created dataset: {ds.dataset_id} ({preset}, {len(records)} records)")


if __name__ == "__main__":
    app()
