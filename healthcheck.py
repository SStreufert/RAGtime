"""Launch-time availability check for the configured LLM endpoints.

Probes each role's OpenAI-compatible base URL via the openai client's
``models.list()`` (works on both Ollama and LM Studio), verifies the
configured model is actually served, and reports per-role results.

Launch surfaces call ``probe_launch()`` (successes cached per process,
failures re-probed) and fail fast on problems; ``python healthcheck.py``
probes all three roles standalone with the GREEN/RED + exit 0/1 convention
from scripts/probe_eval_run_params.py.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError

from config import settings

PROBE_TIMEOUT_S = 5.0
ALL_ROLES = ("chat", "embeddings", "judge")

_REACHABLE_HINT = "Start the server (`ollama serve` / LM Studio) or fix {base_env} in .env."
_MODEL_HINT = "Load/pull the model or fix {model_env} in .env."


@dataclass(frozen=True)
class Endpoint:
    role: str
    base_url: str
    api_key: str
    model: str
    base_url_env: str
    model_env: str


@dataclass(frozen=True)
class ProbeResult:
    role: str
    base_url: str
    model: str
    ok: bool
    detail: str
    degraded: bool = False


def endpoints_for(roles) -> list[Endpoint]:
    known = {
        "chat": Endpoint(
            role="chat",
            base_url=settings.chat_base_url,
            api_key=settings.chat_api_key.get_secret_value(),
            model=settings.chat_model,
            base_url_env="CHAT_BASE_URL",
            model_env="CHAT_MODEL",
        ),
        "embeddings": Endpoint(
            role="embeddings",
            base_url=settings.embeddings_base_url,
            api_key=settings.embeddings_api_key.get_secret_value(),
            model=settings.embedding_model,
            base_url_env="EMBEDDINGS_BASE_URL",
            model_env="EMBEDDING_MODEL",
        ),
        "judge": Endpoint(
            role="judge",
            base_url=settings.effective_judge_base_url,
            api_key=settings.effective_judge_api_key,
            model=settings.judge_model,
            # Judge falls back to the chat endpoint when JUDGE_BASE_URL is unset.
            base_url_env="JUDGE_BASE_URL" if settings.judge_base_url else "CHAT_BASE_URL",
            model_env="JUDGE_MODEL",
        ),
    }
    return [known[role] for role in roles]


def _failed(ep: Endpoint, detail: str, hint: str) -> ProbeResult:
    # ASCII separator: em dashes mangle in Windows console output.
    return ProbeResult(
        role=ep.role,
        base_url=ep.base_url,
        model=ep.model,
        ok=False,
        detail=f"{detail}. "
        f"{hint.format(base_env=ep.base_url_env, model_env=ep.model_env)}",
    )


def _model_served(configured: str, listed: list[str]) -> bool:
    """Exact match, ignoring Ollama's `:latest` tag (`nomic-embed-text` vs
    `nomic-embed-text:latest` — the server does serve the bare name)."""

    def norm(model_id: str) -> str:
        return model_id[: -len(":latest")] if model_id.endswith(":latest") else model_id

    target = norm(configured)
    return any(norm(model_id) == target for model_id in listed)


def probe_endpoint(ep: Endpoint) -> ProbeResult:
    try:
        client = OpenAI(
            base_url=ep.base_url,
            api_key=ep.api_key,
            timeout=PROBE_TIMEOUT_S,
            max_retries=0,
        )
        listing = client.models.list()
    except APIConnectionError as exc:
        cause = exc.__cause__
        detail = f"unreachable ({type(exc).__name__}: {cause or exc})"
        return _failed(ep, detail, _REACHABLE_HINT)
    except APIStatusError as exc:
        # 404/405: the server answered but does not implement /models.
        # Reachability was the real signal — warn (via detail) and pass.
        if exc.status_code in (404, 405):
            return ProbeResult(
                role=ep.role,
                base_url=ep.base_url,
                model=ep.model,
                ok=True,
                degraded=True,
                detail=(
                    f"reachable but {ep.base_url}/models unsupported "
                    f"(HTTP {exc.status_code}); model presence not verified"
                ),
            )
        return _failed(
            ep,
            f"HTTP {exc.status_code}: {exc.message}",
            _REACHABLE_HINT,
        )
    except OpenAIError as exc:
        return _failed(ep, f"{type(exc).__name__}: {exc}", _REACHABLE_HINT)
    except Exception as exc:  # noqa: BLE001 — a launch check must not crash
        return _failed(ep, f"{type(exc).__name__}: {exc}", _REACHABLE_HINT)

    served = [model.id for model in getattr(listing, "data", [])]
    if not _model_served(ep.model, served):
        available = ", ".join(served[:8]) or "(none)"
        return _failed(
            ep,
            f"model {ep.model!r} not served; available: {available}",
            _MODEL_HINT,
        )
    return ProbeResult(
        role=ep.role,
        base_url=ep.base_url,
        model=ep.model,
        ok=True,
        detail="ok",
    )


def run_check(roles) -> list[ProbeResult]:
    """Probe every requested role in parallel; never raises."""
    endpoints = endpoints_for(roles)
    if not endpoints:
        return []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
        return list(pool.map(probe_endpoint, endpoints))


def all_ok(results: list[ProbeResult]) -> bool:
    return all(result.ok for result in results)


def report_text(results: list[ProbeResult]) -> str:
    if not results:
        return "endpoint check skipped (SKIP_ENDPOINT_CHECK set)"
    if all_ok(results):
        line = "endpoints ok: " + ", ".join(
            f"{r.role}={r.base_url} ({r.model})" for r in results
        )
        degraded = [r for r in results if r.degraded]
        if degraded:
            line += " | degraded: " + "; ".join(
                f"{r.role}: {r.detail}" for r in degraded
            )
        return line
    failed = [r for r in results if not r.ok]
    lines = [f"endpoint check failed ({len(failed)} problem(s)):"]
    lines += [
        f"  ! {r.role:11s} {r.base_url} -> {r.detail}" for r in failed
    ]
    return "\n".join(lines)


_success_cache: dict[tuple[str, ...], list[ProbeResult]] = {}


def probe_launch(roles) -> list[ProbeResult]:
    """Probe for launch: successes (and skips) cached per process, failures re-probed.

    Prints the report — success to stdout, failure to stderr — on every fresh
    probe, so a fixed-and-retried failure leaves a visible trail in the terminal.
    """
    key = tuple(roles)
    cached = _success_cache.get(key)
    if cached is not None:
        return cached
    results = [] if settings.skip_endpoint_check else run_check(key)
    print(report_text(results), file=sys.stdout if all_ok(results) else sys.stderr)
    if all_ok(results):
        _success_cache[key] = results
    return results


def ensure_or_exit(roles, exit_code: int = 1) -> list[ProbeResult]:
    """For CLI surfaces: probe, report, and exit non-zero if any role failed."""
    results = probe_launch(roles)
    if not all_ok(results):
        raise SystemExit(exit_code)
    return results


def main() -> None:
    """Standalone probe of all three roles. Green/red output; exit 0/1.

    Deliberately ignores SKIP_ENDPOINT_CHECK: bypassing a check you explicitly
    invoked is nonsense.
    """
    results = run_check(ALL_ROLES)
    if all_ok(results):
        print(f"GREEN: {report_text(results)}")
        raise SystemExit(0)
    print(f"RED: {report_text(results)}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
