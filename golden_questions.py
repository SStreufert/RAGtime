"""Golden question registry and validation.

This module defines the dataset presets, their JSON file paths, and validation
logic shared by the import CLI, eval runs, and ad-hoc evaluation.
"""

import json
from pathlib import Path
from typing import Any


# --- Preset registry ---------------------------------------------------------

# Mapping from preset name to the golden-question JSON file on disk.
# These are the only presets the CLI accepts; new ones must be added here.
DATASET_FILES: dict[str, str] = {
    "swan-lagoon": "data/swan_lagoon_golden_questions.json",
    "my-documents": "data/my_documents_golden_questions.json",
}


# --- Placeholder detection ---------------------------------------------------

# Sentinel string that marks a record as an unedited template.
# The import CLI refuses files containing this marker.
PLACEHOLDER_MARKER = "REPLACE_ME"


def placeholder_errors(records: list[dict[str, Any]]) -> list[str]:
    """Return a list of error strings for records still containing placeholders.

    Each error identifies the record index and the field containing the marker.
    """
    errors = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        inputs = record.get("inputs", {})
        expectations = record.get("expectations", {})
        question = inputs.get("question", "") if isinstance(inputs, dict) else ""
        expected = (
            expectations.get("expected_response", "") if isinstance(expectations, dict) else ""
        )
        if isinstance(question, str) and PLACEHOLDER_MARKER in question:
            errors.append(f"record {i}: inputs.question contains '{PLACEHOLDER_MARKER}'")
        if isinstance(expected, str) and PLACEHOLDER_MARKER in expected:
            errors.append(f"record {i}: expectations.expected_response contains '{PLACEHOLDER_MARKER}'")
    return errors


# --- Structural validation ---------------------------------------------------

def validation_errors(
    records: list[dict[str, Any]], *, require_expectations: bool = True
) -> list[str]:
    """Return a list of structural validation errors for the records.

    Checks:
    - Each record is a dict
    - Each record has 'inputs' dict with 'question' (non-empty string)
    - If require_expectations: each record has 'expectations' dict with
      'expected_response' (non-empty string)
    """
    errors = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {i}: must be a dict")
            continue

        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            errors.append(f"record {i}: inputs must be a dict")
        else:
            question = inputs.get("question")
            if not isinstance(question, str) or not question.strip():
                errors.append(f"record {i}: inputs.question must be a non-empty string")

        if require_expectations:
            expectations = record.get("expectations")
            if not isinstance(expectations, dict):
                errors.append(f"record {i}: expectations must be a dict")
            else:
                expected = expectations.get("expected_response")
                if not isinstance(expected, str) or not expected.strip():
                    errors.append(
                        f"record {i}: expectations.expected_response must be a non-empty string"
                    )
    return errors


# --- Load helpers ------------------------------------------------------------

def load_records(json_path: str | Path) -> list[dict[str, Any]]:
    """Load and return records from a golden-question JSON file."""
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_dataset_path(preset: str) -> Path:
    """Return the Path to the JSON file for a given preset."""
    if preset not in DATASET_FILES:
        known = ", ".join(sorted(DATASET_FILES))
        raise ValueError(f"unknown preset {preset!r} (known: {known})")
    return Path(DATASET_FILES[preset])