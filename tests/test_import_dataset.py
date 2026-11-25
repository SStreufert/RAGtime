"""Tests for the golden-question registry and the import CLI.

The registry test is the regression for a real failure mode: the ai-primer
preset once pointed at a deleted JSON file, and nothing failed until an
import crashed. Local checks here never touch MLflow — the CLI tests
monkeypatch the dataset API functions.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from golden_questions import (
    DATASET_FILES,
    PLACEHOLDER_MARKER,
    placeholder_errors,
    validation_errors,
)


class TestRegistry:
    def test_registered_dataset_files_exist(self):
        """Every preset's JSON must exist on disk."""
        for preset, path in DATASET_FILES.items():
            assert Path(path).exists(), f"{preset} -> {path} not found"

    def test_my_documents_template_is_a_fresh_placeholder(self):
        """The committed template is structurally valid but unmistakably
        unfilled, so importing it without editing is refused."""
        records = json.loads(
            Path(DATASET_FILES["my-documents"]).read_text(encoding="utf-8")
        )
        assert records, "template must ship with placeholder records"
        assert validation_errors(records) == [], (
            "placeholders must be structurally valid"
        )
        assert placeholder_errors(records), (
            "placeholders must carry the sentinel so import refuses them"
        )


class TestValidation:
    def test_placeholder_detected_in_question_and_expectation(self):
        records = [
            {
                "inputs": {"question": f"{PLACEHOLDER_MARKER}: q"},
                "expectations": {"expected_response": f"{PLACEHOLDER_MARKER}: e"},
            }
        ]
        errors = placeholder_errors(records)
        assert len(errors) == 2
        assert all(PLACEHOLDER_MARKER in e for e in errors)

    def test_real_records_pass_placeholder_check(self):
        records = [
            {
                "inputs": {"question": "What is RAG?"},
                "expectations": {
                    "expected_response": "Retrieval-Augmented Generation."
                },
            }
        ]
        assert placeholder_errors(records) == []

    def test_missing_expectations_flagged(self):
        records = [{"inputs": {"question": "q"}}]
        errors = validation_errors(records)
        assert any("expectations" in e for e in errors)

    def test_non_dict_record_flagged(self):
        errors = validation_errors(["not a record"])
        assert len(errors) == 1
        assert "record 0" in errors[0]


class TestImportCLI:
    def test_unknown_preset_rejected(self):
        from scripts.import_dataset import app

        result = CliRunner().invoke(app, ["nonsense"])
        assert result.exit_code != 0

    def test_placeholder_template_refused(self):
        """Importing the committed template as-is must exit non-zero."""
        from scripts.import_dataset import app

        result = CliRunner().invoke(app, ["my-documents"])
        assert result.exit_code != 0
        assert PLACEHOLDER_MARKER in result.output

    def test_existing_dataset_skipped_without_force(self, monkeypatch):
        from scripts import import_dataset as imp

        monkeypatch.setattr(
            imp,
            "search_datasets",
            lambda filter_string: [SimpleNamespace(dataset_id="d-1")],
        )
        created = []
        monkeypatch.setattr(
            imp, "create_dataset", lambda **kw: created.append(kw)
        )
        result = CliRunner().invoke(imp.app, ["swan-lagoon"])
        assert result.exit_code == 0, result.output
        assert created == []
        assert "--force" in result.output

    def test_force_deletes_and_recreates(self, monkeypatch):
        from scripts import import_dataset as imp

        monkeypatch.setattr(
            imp,
            "search_datasets",
            lambda filter_string: [SimpleNamespace(dataset_id="d-1")],
        )
        deleted = []
        monkeypatch.setattr(
            imp, "delete_dataset", lambda dataset_id: deleted.append(dataset_id)
        )

        class FakeDataset:
            dataset_id = "d-2"
            merged = 0

            def merge_records(self, records):
                self.merged = len(records)

        fake = FakeDataset()

        def fake_create(*, name, experiment_id):
            return fake

        monkeypatch.setattr(imp, "create_dataset", fake_create)
        result = CliRunner().invoke(imp.app, ["swan-lagoon", "--force"])
        assert result.exit_code == 0, result.output
        assert deleted == ["d-1"]
        assert fake.dataset_id == "d-2"
        assert fake.merged > 0
