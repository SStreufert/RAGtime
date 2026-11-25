import json
from pathlib import Path

DATASET_FILE = Path("data/my_documents_golden_questions.json")


def test_eval_dataset_exist():
    assert DATASET_FILE.exists(), f"{DATASET_FILE} not found"


def test_eval_dataset_format():
    data = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
    for item in data:
        assert "inputs" in item
        assert "question" in item["inputs"]
        assert isinstance(item["inputs"]["question"], str)
        if "expectations" in item:
            assert "expected_response" in item["expectations"]


def test_evaluate_imports():
    """Verify evaluate module can be imported."""
    import evaluate

    assert hasattr(evaluate, "app")


def test_evaluate_questions_command():
    """Verify the questions CLI command lists them.

    Content-agnostic on purpose: the file is the user-owned template, so it
    holds REPLACE_ME placeholders until personalized.
    """
    from typer.testing import CliRunner
    from evaluate import app

    runner = CliRunner()
    result = runner.invoke(app, ["questions", "--file", str(DATASET_FILE)])
    assert result.exit_code == 0, result.output
    assert result.output.strip(), "expected at least one question listed"


def test_evaluate_requires_file():
    """--file is required; there is no default dataset path anymore."""
    from typer.testing import CliRunner
    from evaluate import app

    runner = CliRunner()
    result = runner.invoke(app, ["questions"])
    assert result.exit_code != 0


def test_query_validates_before_evaluating(tmp_path):
    """A golden question from the file without expectations never reaches
    the judge: shared validation exits first."""
    from typer.testing import CliRunner
    from evaluate import app

    bad = tmp_path / "bad_dataset.json"
    bad.write_text(
        json.dumps([{"inputs": {"question": "What is RAG?"}}]),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app, ["query", "What is RAG?", "--file", str(bad)]
    )
    assert result.exit_code == 1
    assert "Dataset validation failed" in result.output
