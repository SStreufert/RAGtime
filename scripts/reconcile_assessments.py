"""Retroactively reconcile judge verdicts with rationale text for existing runs.

Scans one or more evaluation runs, flips categorical yes/no assessments whose
rationale unambiguously concludes the opposite, flags hedged rationales as
uncertain (metadata only, value untouched), re-logs corrected metrics under
their original names, and tags each processed run as reconciled.

Usage:
    python scripts/reconcile_assessments.py --run-id <run_id>            # apply
    python scripts/reconcile_assessments.py --run-id <run_id> --dry-run  # preview
    python scripts/reconcile_assessments.py --all                       # every run in the experiment

Exits non-zero if any write-back failed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from config import settings  # noqa: E402
from eval_consistency import format_report, reconcile_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile judge verdicts with rationale text in existing runs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="MLflow run ID to reconcile")
    group.add_argument(
        "--all",
        action="store_true",
        help="Reconcile every run in the experiment (explicit opt-in)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned corrections without writing anything",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    if args.run_id:
        run_ids = [args.run_id]
    else:
        exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        if exp is None:
            print(f"Experiment {settings.mlflow_experiment_name!r} not found")
            return 1
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id], max_results=1000
        )
        run_ids = [r.info.run_id for r in runs]
        print(f"Reconciling {len(run_ids)} run(s) in experiment {exp.experiment_id}")

    had_errors = False
    for run_id in run_ids:
        report = reconcile_run(run_id, dry_run=args.dry_run)
        if report.traces_checked == 0 and not report.actions:
            print(f"Run {run_id}: no traced eval rows, skipped")
            continue
        print(format_report(report))
        if report.errors:
            had_errors = True
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
