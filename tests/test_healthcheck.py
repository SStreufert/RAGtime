import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from openai import APIConnectionError, APIStatusError
from typer.testing import CliRunner

import evaluate as evaluate_module
import eval_run
import healthcheck
from healthcheck import (
    Endpoint,
    all_ok,
    ensure_or_exit,
    probe_endpoint,
    probe_launch,
    report_text,
    run_check,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    healthcheck._success_cache.clear()
    yield
    healthcheck._success_cache.clear()


def _endpoint(role="chat", model="llama3.2", base_url="http://host:1234/v1"):
    return Endpoint(
        role=role,
        base_url=base_url,
        api_key="key",
        model=model,
        base_url_env="CHAT_BASE_URL",
        model_env="CHAT_MODEL",
    )


def _ok_result(role="chat"):
    return healthcheck.ProbeResult(
        role=role, base_url="http://h/v1", model="m", ok=True, detail="ok"
    )


def _fail_result(role="chat"):
    return healthcheck.ProbeResult(
        role=role, base_url="http://h/v1", model="m", ok=False, detail="boom"
    )


def _fake_openai(monkeypatch, *, ids=(), exc=None):
    """Replace the openai client with one whose models.list() does the given thing."""
    captured = {}

    class FakeModels:
        def list(self):
            if exc is not None:
                raise exc
            return SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids])

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr(healthcheck, "OpenAI", FakeClient)
    return captured


# --- probe_endpoint ---------------------------------------------------------


def test_probe_success(monkeypatch):
    captured = _fake_openai(monkeypatch, ids=["llama3.2", "nomic-embed-text"])
    result = probe_endpoint(_endpoint())
    assert result.ok and not result.degraded
    assert captured["timeout"] == healthcheck.PROBE_TIMEOUT_S
    assert captured["max_retries"] == 0


def test_probe_model_not_served_fails_with_available_ids(monkeypatch):
    _fake_openai(monkeypatch, ids=["other-model", "another-one"])
    result = probe_endpoint(_endpoint(model="llama3.2"))
    assert not result.ok
    assert "model 'llama3.2' not served" in result.detail
    assert "other-model, another-one" in result.detail
    assert "CHAT_MODEL" in result.detail  # remediation hint


def test_probe_latest_tag_matches_bare_model(monkeypatch):
    """Ollama lists `nomic-embed-text:latest` for the configured bare name."""
    _fake_openai(monkeypatch, ids=["nomic-embed-text:latest"])
    assert probe_endpoint(_endpoint(model="nomic-embed-text")).ok
    _fake_openai(monkeypatch, ids=["qwen3.5:4b"])
    assert probe_endpoint(_endpoint(model="qwen3.5:4b:latest")).ok


def test_probe_unreachable_fails_with_base_url_hint(monkeypatch):
    request = httpx.Request("GET", "http://host:1234/v1/models")
    exc = APIConnectionError(request=request)
    _fake_openai(monkeypatch, exc=exc)
    result = probe_endpoint(_endpoint())
    assert not result.ok
    assert "unreachable" in result.detail
    assert "CHAT_BASE_URL" in result.detail


def test_probe_models_endpoint_unsupported_degrades_to_pass(monkeypatch):
    request = httpx.Request("GET", "http://host:1234/v1/models")
    exc = APIStatusError(
        "not found", response=httpx.Response(404, request=request), body=None
    )
    _fake_openai(monkeypatch, exc=exc)
    result = probe_endpoint(_endpoint())
    assert result.ok
    assert result.degraded
    assert "HTTP 404" in result.detail


def test_probe_http_error_still_fails(monkeypatch):
    request = httpx.Request("GET", "http://host:1234/v1/models")
    exc = APIStatusError(
        "unauthorized",
        response=httpx.Response(401, request=request),
        body=None,
    )
    _fake_openai(monkeypatch, exc=exc)
    result = probe_endpoint(_endpoint())
    assert not result.ok
    assert "HTTP 401" in result.detail


# --- run_check / report -----------------------------------------------------


def test_run_check_probes_every_requested_role(monkeypatch):
    seen = []
    monkeypatch.setattr(
        healthcheck, "probe_endpoint", lambda ep: seen.append(ep.role) or _ok_result(ep.role)
    )
    results = run_check(("chat", "embeddings", "judge"))
    assert sorted(seen) == ["chat", "embeddings", "judge"]
    assert all_ok(results)


def test_report_text_success_and_failure(capsys):
    ok = [_ok_result("chat")]
    assert report_text(ok).startswith("endpoints ok: chat=")

    failed = [_ok_result("chat"), _fail_result("judge")]
    text = report_text(failed)
    assert "endpoint check failed (1 problem(s))" in text
    assert "judge" in text and "boom" in text
    assert "chat" not in text.split("\n", 1)[1]

    assert report_text([]) == "endpoint check skipped (SKIP_ENDPOINT_CHECK set)"


# --- probe_launch caching / skip --------------------------------------------


def test_probe_launch_caches_successes(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        healthcheck, "run_check", lambda roles: calls.append(roles) or [_ok_result()]
    )
    first = probe_launch(("chat", "embeddings"))
    second = probe_launch(("chat", "embeddings"))
    assert calls == [("chat", "embeddings")]
    assert first is second
    assert "endpoints ok" in capsys.readouterr().out


def test_probe_launch_reprobes_failures(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        healthcheck, "run_check", lambda roles: calls.append(roles) or [_fail_result()]
    )
    probe_launch(("chat",))
    probe_launch(("chat",))
    assert len(calls) == 2
    assert "endpoint check failed" in capsys.readouterr().err


def test_skip_flag_skips_probe(monkeypatch):
    monkeypatch.setattr(healthcheck.settings, "skip_endpoint_check", True)
    monkeypatch.setattr(
        healthcheck, "run_check", lambda roles: pytest.fail("should not probe")
    )
    results = probe_launch(("chat", "embeddings"))
    assert results == []
    assert all_ok(results)


def test_ensure_or_exit_fails_nonzero(monkeypatch):
    monkeypatch.setattr(healthcheck, "run_check", lambda roles: [_fail_result()])
    with pytest.raises(SystemExit) as excinfo:
        ensure_or_exit(("chat",))
    assert excinfo.value.code == 1


def test_ensure_or_exit_passes_through_on_success(monkeypatch):
    expected = [_ok_result()]
    monkeypatch.setattr(healthcheck, "run_check", lambda roles: expected)
    assert ensure_or_exit(("chat",)) is expected


# --- standalone -------------------------------------------------------------


def test_standalone_green(monkeypatch, capsys):
    monkeypatch.setattr(healthcheck, "run_check", lambda roles: [_ok_result()])
    with pytest.raises(SystemExit) as excinfo:
        healthcheck.main()
    assert excinfo.value.code == 0
    assert "GREEN" in capsys.readouterr().out


def test_standalone_red(monkeypatch, capsys):
    monkeypatch.setattr(healthcheck, "run_check", lambda roles: [_fail_result()])
    with pytest.raises(SystemExit) as excinfo:
        healthcheck.main()
    assert excinfo.value.code == 1
    assert "RED" in capsys.readouterr().err


def test_standalone_ignores_skip_flag(monkeypatch):
    monkeypatch.setattr(healthcheck.settings, "skip_endpoint_check", True)
    seen = []
    monkeypatch.setattr(
        healthcheck, "run_check", lambda roles: seen.append(roles) or [_ok_result()]
    )
    with pytest.raises(SystemExit):
        healthcheck.main()
    assert seen  # probe happened despite the skip flag


# --- surface wiring ---------------------------------------------------------


def test_eval_run_checks_endpoints_after_dataset_validation(monkeypatch):
    """Malformed datasets fail first (instant/local) — endpoints never probed."""
    seen = []

    class FakeDataset:
        name = "fake"

        def to_df(self):
            return pd.DataFrame([{"inputs": {"question": ""}}])

    monkeypatch.setattr(
        eval_run, "search_datasets", lambda filter_string: [FakeDataset()]
    )
    monkeypatch.setattr(
        eval_run, "ensure_or_exit", lambda roles: seen.append(roles)
    )
    runner = CliRunner()
    result = runner.invoke(eval_run.app, ["swan-lagoon"])
    assert result.exit_code == 1
    assert seen == []  # validate_or_exit fired before the probe


def test_eval_run_probes_all_three_roles(monkeypatch):
    class FakeDataset:
        name = "fake"

        def to_df(self):
            return pd.DataFrame(
                [
                    {
                        "inputs": {"question": "q?"},
                        "expectations": {"expected_response": "a"},
                    }
                ]
            )

    monkeypatch.setattr(
        eval_run, "search_datasets", lambda filter_string: [FakeDataset()]
    )
    seen = []

    def fake_ensure(roles):
        seen.append(roles)
        raise SystemExit(0)

    monkeypatch.setattr(eval_run, "ensure_or_exit", fake_ensure)
    runner = CliRunner()
    result = runner.invoke(eval_run.app, ["swan-lagoon"])
    assert result.exit_code == 0
    assert seen == [("chat", "embeddings", "judge")]


def test_evaluate_questions_skips_endpoint_check(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps([{"inputs": {"question": "q?"}, "expectations": {"expected_response": "a"}}]),
        encoding="utf-8",
    )

    def forbidden(roles):
        raise AssertionError("questions must not probe endpoints")

    monkeypatch.setattr(evaluate_module, "ensure_or_exit", forbidden)
    runner = CliRunner()
    result = runner.invoke(evaluate_module.app, ["questions", "--file", str(dataset)])
    assert result.exit_code == 0
    assert "q?" in result.output
