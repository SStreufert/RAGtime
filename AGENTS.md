# AGENTS.md: Local RAG with MLflow Evaluation

This is a **local RAG pipeline** that treats LLM quality as an experiment. It ingests PDFs, answers questions via any OpenAI-compatible server (LM Studio / Ollama), and uses MLflow GenAI to grade every answer with five LLM-judge scorers, plus a post-scoring **reconciliation** pass that corrects judge verdicts contradicting their own rationale.

---

## Quick Commands

```bash
# Setup
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env

# Start model server (LM Studio: http://localhost:1234/v1  |  Ollama: http://localhost:11434/v1)

# Ingest PDFs from data/ → ChromaDB (logs MLflow ingestion run)
python ingest.py

# Ask a question (CLI)
python rag.py

# Streamlit UI
python run_app.py

# FastAPI
uvicorn api:app --reload

# Full eval run (registered dataset, linked in MLflow)
python eval_run.py swan-lagoon

# Import the golden-question template as a dataset (one-time; --force to refresh)
python scripts/import_dataset.py

# Smoke test (2 rows, anonymous snapshot)
python eval_run.py swan-lagoon --limit 2

# Ad-hoc eval against JSON file
python evaluate.py batch --file data/swan_lagoon_golden_questions.json

# Standalone endpoint health probe (GREEN/RED, exit 0/1)
python healthcheck.py

# Retroactive reconciliation
python scripts/reconcile_assessments.py --run-id <id> [--dry-run | --all]

# Tests (most offline; see Rough Edges)
python -m pytest tests/ -v
```

---

## Architecture Highlights

| File | Responsibility |
|------|----------------|
| `config.py` | Centralized pydantic-settings; loads `.env`; exposes `settings` singleton + `configure_judge()` |
| `tracker.py` | MLflow wrapper (soft import); `log_ingestion`, `log_query`, `eval_run_params()` |
| `healthcheck.py` | Launch-time endpoint probe: verifies each role answers `/v1/models` **and** serves its model |
| `ingest.py` | PDF → chunk → embed → Chroma; logs ingestion run |
| `rag.py` | Query pipeline: retrieve → generate → trace (spans: `retrieval` + `generation`) |
| `eval_run.py` | Eval runs over **registered dataset presets** (Typer CLI); caps MLflow workers |
| `evaluate.py` | Ad-hoc eval over **JSON files** (Typer CLI); single question / batch / list |
| `eval_consistency.py` | Verdict/rationale reconciliation engine (negation-aware, rightmost-match) |
| `app.py` / `api.py` | Streamlit UI / FastAPI REST: both probe endpoints on launch |

---

## Configuration (config.py → .env)

Three **independent model roles**: can live on different hosts:

| Role | Base URL | Model | API Key |
|------|----------|-------|---------|
| Embeddings | `EMBEDDINGS_BASE_URL` | `EMBEDDING_MODEL` | `EMBEDDINGS_API_KEY` |
| Chat | `CHAT_BASE_URL` | `CHAT_MODEL` | `CHAT_API_KEY` |
| Judge | `JUDGE_BASE_URL` (falls back to `CHAT_BASE_URL`) | `JUDGE_MODEL` | `JUDGE_API_KEY` (falls back to `CHAT_API_KEY`) |

**Critical defaults** (code wins over `.env.example`, your `.env` wins over both):

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 5
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"   # or http://localhost:5000 for server mode
CHROMA_DIR = "./chroma_db"
DATA_DIR = "./data"
SKIP_ENDPOINT_CHECK = false
```

**Endpoint check**: every launch calls `healthcheck.probe_launch(roles)`; successes cached per process, failures re-probed. `SKIP_ENDPOINT_CHECK=1` bypasses (CI / offline edits).

---

## ChromaDB Coupling ⚠️

The vector store is **bound to a specific `(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)` combination**. `Chroma.from_documents` **appends** into an existing folder; it does **not** replace.

**When any of those three settings change:**
```powershell
Remove-Item -Recurse .\chroma_db   # Windows
# rm -rf ./chroma_db               # macOS/Linux
python ingest.py
```
The folders `chroma_db 500-100`, `chroma_db 750-100`, `chroma_db 1000-100` at repo root are this archive-by-rename workflow in practice.

---

## MLflow Worker Caps (eval_run.py)

MLflow defaults (~40 predict / ~200 scorer workers) **flood a single local judge**; queue waits exceed the 300 s gateway timeout, clients retry, run stalls at 0%.

Capped at import time in `eval_run.py`:
```python
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "3")       # predict
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1") # score
os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")
```
At most 3 rows × 1 scorer hit the judge concurrently.

---

## Evaluation & Reconciliation

**Five scorers** (all `JUDGE_MODEL` @ `JUDGE_TEMPERATURE=0.0`):
- `RetrievalRelevance`: is each chunk relevant to the request?
- `RetrievalGroundedness`: is the answer grounded in retrieved context?
- `RelevanceToQuery`: does the answer address the question?
- `Correctness`: does the answer support expected facts?
- `RetrievalSufficiency`: is retrieved context sufficient to answer?

**Reconciliation** (`eval_consistency.py`): runs automatically after every eval (`--no-reconcile` to skip).
- Classifies rationale **conclusion** (last 1–2 sentences, negation-aware rewrites)
- Flips verdict when rationale unambiguously contradicts it
- Flags hedged rationales (`partially`, `not entirely`, `to some extent`) as **uncertain**; never auto-flipped
- Re-logs corrected metrics under original names; tags run with `reconciled`, `reconcile_corrected=N`, `reconcile_uncertain=N`
- Retroactive: `scripts/reconcile_assessments.py --run-id <id> [--dry-run | --all]`
- Regression tests in `tests/test_eval_consistency.py` use **verbatim rationales from real traces**

---

## Testing

```bash
python -m pytest tests/ -v
```

**Most tests are offline**: they redirect MLflow to a throwaway SQLite store (`conftest.py` `mlflow_local_store` fixture) and point the judge env via `configure_judge()`.

**Rough edges (require live services):**
- `test_config.py` asserts specific `.env` values; fails if your `.env` differs
- `test_rag_predict_fn_creates_retriever_span` needs running model server + indexed Chroma
- `tracker.py` prints debug lines to stdout on import

---

## Project Structure

```
├── config.py               # Settings (pydantic-settings), .env loading, judge env routing
├── tracker.py              # MLflow run/param/metric/artifact logging
├── ingest.py               # PDF -> chunk -> embed -> Chroma
├── rag.py                  # Query pipeline: retrieve, generate, trace
├── eval_run.py             # Eval runs over dataset presets (typer CLI)
├── evaluate.py             # Eval runs over JSON dataset files (typer CLI)
├── eval_consistency.py     # Verdict/rationale reconciliation
├── healthcheck.py          # Launch-time endpoint availability check (+ standalone CLI)
├── app.py / api.py         # Streamlit UI / FastAPI REST endpoint
├── scripts/                # Retroactive reconcile, dataset repair, MLflow probes
├── data/                   # Source PDFs + evaluation dataset JSONs
├── docs/
│   ├── EVALUATION.md       # From PDFs to scored eval runs: golden questions, dataset registration, MLflow workflow
│   ├── MLFLOW.md           # MLflow logging reference: params, metrics, artifacts, worker caps
│   ├── RECONCILIATION.md   # Reconciliation mechanism, guarantees, future work
│   └── images/
├── tests/                  # pytest suite
├── GLOSSARY.md             # Domain vocabulary
└── SPEC-mlflow-integration.md   # Design history (pre-implementation spec)
```

---

## Key Conventions for Agents

1. **Config is single-source**: never hardcode model names, chunk params, or URLs; import from `config.settings`
2. **MLflow is soft dependency**: `tracker.is_available()` guards all tracking calls; ingestion/query work without it
3. **Endpoint check is mandatory at launch**: `healthcheck.ensure_or_exit(("chat", "embeddings"))` (or `("chat", "embeddings", "judge")` for eval); CLI exits non-zero on failure
4. **Traces are first-class**: `rag.answer()` always emits `rag_query` trace with `retrieval` + `generation` spans; scorers grade these
5. **Eval runs share param set**: `tracker.eval_run_params()` ensures runs compare as columns in MLflow UI
6. **Chroma store is process-wide singleton** (`rag.get_store()`); `PersistentClient` is not thread-safe to construct concurrently
7. **Empty answer = generation failure**: `rag_predict_fn` raises `ValueError` rather than returning `""` (prevents vacuous scorer passes)

---

## Common Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `chroma_db` grows but quality doesn't change | Re-ingested without clearing after chunk/embedding change | Delete `chroma_dir` before `ingest.py` |
| Eval stalls at 0% "fetch failed" | MLflow worker defaults flood local judge | Caps are in `eval_run.py`; don't remove them |
| Judge verdict contradicts rationale | Small local model | Reconciliation auto-corrects; larger `JUDGE_MODEL` reduces rate |
| `test_config.py` fails | Your `.env` differs from test expectations | Tests assert specific defaults; override via `monkeypatch` in new tests |
| `ModuleNotFoundError: mlflow` | Not installed | `pip install -r requirements.txt`; tracker warns but ingestion works |

---

## References

- **Evaluation workflow**: `docs/EVALUATION.md`: from PDFs to scored eval runs, dataset registration, MLflow UI
- **MLflow logging reference**: `docs/MLFLOW.md`: params, metrics, artifacts, worker caps
- **Reconciliation mechanism**: `docs/RECONCILIATION.md`: verdict/rationale corrections, guarantees
- **Domain vocabulary**: `GLOSSARY.md`