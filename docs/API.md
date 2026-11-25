# REST API (`api.py`)

## Introduction
The REST API (`api.py`) lets other applications query the pipeline programmatically. This document covers setup, core concepts, the API reference, and usage examples. For the broader project, refer to the [README.md](README.md).

## Getting Started
To get started with the REST API, follow these steps:

1. **Run the API**:
   ```bash
   uvicorn api:app --reload
   ```
   On startup, the app runs `healthcheck.probe_launch(('chat', 'embeddings'))` via the FastAPI lifespan hook. If either endpoint is unreachable or doesn't serve its configured model, the app **refuses to start** with a per-role error message, same fail-fast behavior as the CLI.

2. **Verify System Health**:
   ```bash
   curl http://localhost:8000/health
   ```
   This command checks the health of the API endpoints. The output looks like the following. You are looking for `"ok": true` throughout. 

   ```json
   {
    "ok": true,
    "endpoints": [
      {
        "role": "chat",
        "base_url": "http://localhost:11434/v1",
        "model": "granite4.1:3b",
        "ok": true,
        "degraded": false,
        "detail": "granite4.1:3b served /v1/models"
      },
      {
        "role": "embeddings",
        "base_url": "http://localhost:11434/v1",
        "model": "nomic-embed-text",
        "ok": true,
        "degraded": false,
        "detail": "nomic-embed-text served /v1/models"
      }
    ]
   }
   ```
3. **Proceed to Usage Examples** (below)  

## Core Concepts
The REST API runs the same 'Retrieve → Generate → Trace' pipeline as the CLI and Streamlit UI. Every request is traced in MLflow. For pipeline architecture details, refer to [ARCHITECTURE.md](ARCHITECTURE.md).

## API Reference
This section documents the available endpoints:

### `GET /health`

Live endpoint status. Runs a fresh probe per call (not cached).

**Response** `200 OK` (healthy):

> *See the Health output above*

 or `503 Service Unavailable` (degraded):

``` json
{
  "ok": false,
  "endpoints": [
    {
      "role": "chat",
      "base_url": "http://localhost:11434/v1",
      "model": "granite4.1:3b",
      "ok": false,
      "degraded": false,
      "detail": "Chat endpoint unreachable"
    },
    {
      "role": "embeddings",
      "base_url": "http://localhost:11434/v1",
      "model": "nomic-embed-text",
      "ok": true,
      "degraded": false,
      "detail": "nomic-embed-text served /v1/models"
    }
  ]
}
```

If `degraded: true`, the endpoint answered `/v1/models` but did not serve the configured model; the app is up but that role is misconfigured.

``` json
{
  "ok": false,
  "endpoints": [
    {
      "role": "chat",
      "base_url": "http://localhost:11434/v1",
      "model": "granite4.1:3b",
      "ok": true,
      "degraded": true,
      "detail": "Chat endpoint answered /v1/models but did not serve the configured model"
    },
    {
      "role": "embeddings",
      "base_url": "http://localhost:11434/v1",
      "model": "nomic-embed-text",
      "ok": true,
      "degraded": false,
      "detail": "nomic-embed-text served /v1/models"
    }
  ]
}
```

### `POST /ask`

Ask a question through the full retrieve → generate → trace pipeline.

**Request body** (`application/json`):

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `question` | `string` |  | required |
| `k` | `integer` | `7` | `>= 1` |

*Note: Override 'k' via the request body (below) or set `TOP_K` in `.env` to change globally.*

```json
{ "question": "What destination is a Must See?", "k": 3 }
```

**Response** `200 OK`:

```json
{
  "answer": "The Swan Lagoon visitor centre is a Must See destination... [p. 3]",
  "sources_used": 3,
  "sources": [
    { "index": 1, "source": "swan-lagoon-guide.pdf", "page": 3 },
    { "index": 2, "source": "business-directory.pdf", "page": 7 },
    { "index": 3, "source": "service-guide.pdf", "page": 12 }
  ]
}
```

- `answer`: generated text with inline `[p. N]` page citations.
- `sources_used`: number of chunks actually used in generation (may be less than `k` if retrieval returned fewer).
- `sources`: the retrieved chunks, one entry each, with the originating PDF filename and page number.

**Error responses**:

| Status | Condition |
|--------|-----------|
| `422` | Missing `question` or `k` out of range (FastAPI validation) |
| `503` | Chat or embeddings endpoint unreachable (from `/health` logic) |
| `500` | Generation failed, e.g. the chat model returned an empty answer (`rag_predict_fn` raises `ValueError` rather than returning `""`) |

## Usage Examples
Common tasks:

### Verifying System Health

```bash
# Health check
curl http://localhost:8000/health
```

### Querying the Knowledge Base

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the opening hours of the visitor centre?", "k": 3}'
```

### Larger Retrieval Context

```bash
# Larger retrieval context
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What services are available?", "k": 7}'
```

### Windows (PowerShell)

`curl` in PowerShell is an alias for `Invoke-WebRequest`, which does not accept curl's `-H` / `-d` flags. Use `Invoke-RestMethod` or `curl.exe` if the real curl is installed.

```powershell
# Health check
Invoke-RestMethod http://localhost:8000/health

# Ask a question
$body = @{ question = "What are the opening hours of the visitor centre?"; k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body $body

# Larger retrieval context
$body = @{ question = "What services are available?"; k = 7 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body $body
```

**curl.exe (if installed separately, not the PowerShell alias):**

```powershell
curl.exe -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  -d '{"question": "What are the opening hours of the visitor centre?", "k": 3}'
```

## OpenAPI Docs
FastAPI auto-generates interactive docs at:

- `/docs`: Swagger UI
- `/redoc`: ReDoc

Point your browser at `http://localhost:8000/docs` to explore and try endpoints live.

## Relationship to Other Entry Points

| Entry point | Same pipeline? | Difference |
|-------------|----------------|------------|
| `python rag.py` | yes | CLI, prints answer to stdout |
| `python run_app.py` | yes | Chat UI, renders citations |
| `uvicorn api:app` | yes | REST, machine-to-machine |
| `eval_run.py` / `evaluate.py` | yes | Batch eval + scoring, not a server |

All four call `rag.answer()`, so behavior is identical; only the transport differs.

---

## Deep Dives

For further information on this project, see the documentation [README.md](README.md).

**Vocabulary**: See [`../GLOSSARY.md`](../GLOSSARY.md) for the controlled terminology used across this project (corpus, dataset, golden question, judge, scorer, assessment, verdict, rationale, reconciliation, endpoint check).
