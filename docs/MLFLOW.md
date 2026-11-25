# MLflow Logging Reference

Every stage of this pipeline writes to MLflow: ingestion, each answer, and each evaluation. For the user-facing experiment workflow, see [EVALUATION.md](EVALUATION.md).

---

## Backends

`MLFLOW_TRACKING_URI` (in `.env`) accepts any of:

| Value | Use Case | Notes |
|-------|----------|-------|
| `sqlite:///mlflow.db` | Local development | Survives `mlflow ui` restarts; use for local development with ad-hoc eval runs |
| `http://localhost:5000` | MLflow server | **Required** for the GenAI Evaluation UI, registered datasets, and the Quality tab; the server itself must run a SQL backend (`--backend-store-uri sqlite:///mlflow.db`) |

For local development, start the server with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

View results with `mlflow ui` (SQLite) or open the server URL when pointing at `http://localhost:5000`. Experiments are compared side-by-side as columns in the MLflow UI because every eval run logs the same param set.

---

## Ingestion Runs

```bash
python ingest.py
```

| Category | Logged Items |
|----------|--------------|
| **Params** | `chunk_size`, `chunk_overlap`, `embedding_model`, `docs_count` |
| **Metrics** | `total_pages`, `total_chars`, `avg_chars_per_page`, `total_chunks`, `avg_chunk_size`, `ingestion_time_sec`, `embedding_dimension` |
| **Artifacts** | `config_snapshot.json`, `chunk_samples.json` (first 10 chunks: the quickest way to eyeball whether a chunking change did what you intended) |

---

## Query Runs

One run per tracked answer from the CLI (`rag.py`), Streamlit (`app.py`), or FastAPI (`api.py`); nested when a parent run is active.

| Category | Logged Items |
|----------|--------------|
| **Params** | `chat_model`, `temperature`, `top_k`, `chunk_size`, `chunk_overlap`, `embedding_model`, `judge_model`, `judge_temperature`, `max_tokens`, `prompt_template_hash` |
| **Metrics** | `retrieval_score`, retrieval/generation/total latency (ms), `context_tokens` / `answer_tokens` (estimated as `len(text)//4`) |
| **Artifacts** | `query_log.json`, `prompt_template.txt` |

---

## Traces

Every answer writes a `rag_query` trace with two spans:

| Span | Details |
|------|---------|
| **`retrieval`** | Retrieved chunks, `retrieval_score` |
| **`generation`** | Token usage, latency, model/provider |

Scorers grade these traces; they appear under **Traces** in the MLflow UI.

---

## Eval Runs

```bash
python eval_run.py swan-lagoon
python evaluate.py batch --file data/my_documents_golden_questions.json
```

| Category | Logged Items |
|----------|--------------|
| **Params** | The shared eval param set (`chat_model`, `temperature`, `top_k`, `chunk_size`, `chunk_overlap`, `embedding_model`, `judge_model`, `judge_temperature`, `max_tokens`): this is what makes runs comparable as columns |
| **Metrics** | Per-scorer means: `retrieval_groundedness/mean`, `retrieval_relevance/mean`, `retrieval_relevance/precision/mean`, `relevance_to_query/mean`, `correctness/mean`, `retrieval_sufficiency/mean`; plus corrected means re-logged after [reconciliation](RECONCILIATION.md) |
| **Tags** | `reconciled`, `reconcile_corrected=N`, `reconcile_uncertain=N` |
| **Dataset link** | Full runs link to the registered dataset; `--limit` smoke runs evaluate an anonymous snapshot instead |

See [EVALUATION.md#step-5---interpreting-results](EVALUATION.md#step-5---interpreting-results) for how to interpret these in the MLflow UI.

---

## Worker Caps

Eval runs cap MLflow's worker fan-out at import time in `eval_run.py`:

```python
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "3")          # 3 predict workers
os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS", "1")    # 1 scorer at a time
os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")
```

MLflow's defaults (~40 predict / ~200 score workers) flood a single local judge model: queue waits exceed the 300 s gateway timeout, clients abort and retry, and the run stalls at 0% with "fetch failed" scorer errors. With the caps, at most 3 rows × 1 scorer hit the judge at once.

---

## Retroactive Reconciliation

For runs that already exist (see [RECONCILIATION.md](RECONCILIATION.md) for the mechanism):

```bash
python scripts/reconcile_assessments.py --run-id <run_id>             # apply
python scripts/reconcile_assessments.py --run-id <run_id> --dry-run    # preview
python scripts/reconcile_assessments.py --all                          # every run in the experiment
```

---

## Probe Scripts

`scripts/` holds live checks that the logging actually works against a running server:

| Script | Purpose |
|--------|---------|
| `probe_eval_run_params.py` | Verifies `chat_model` / `temperature` / `top_k` land in run params via the same REST API the UI reads |
| `quality_tab_loop.py` | Live probe of the `POST /ajax-api/3.0/mlflow/traces/metrics` endpoint backing the UI Quality tab |

These check MLflow, not the models. For probing the configured LLM endpoints themselves, use the launch-time endpoint check (`healthcheck.py`, run as `python healthcheck.py`).

---

## Dependency

MLflow is required (pinned in `requirements.txt`): every query writes traces and every eval runs through it. `tracker.py` guards the import and warns if MLflow is missing; ingestion completes without it, but querying and evaluation will not.

---

## Deep Dives

For further information on this project, see the documentation [`README.md`](README.md).

**Vocabulary**: See [`../GLOSSARY.md`](../GLOSSARY.md) for the controlled terminology used across this project (corpus, dataset, golden question, judge, scorer, assessment, verdict, rationale, reconciliation, endpoint check).