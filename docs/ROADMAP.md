# Roadmap

**Consolidated, production-first roadmap.** Every item follows the template: **Problem → Why it matters → Proposed solution → How it solves it → Success metric**. Items are ordered by production impact for the end user.

---

## Template

> **Problem**: <what is broken or missing>
>
> **Why it matters**: <impact on end users or production operations>
>
> **Proposed solution**: <concrete approach>
>
> **How it solves it**: <mechanism connecting solution to problem>
>
> **Success metric**: <measurable, verifiable outcome>

---

## 1. Multi-Format Ingestion (Markdown, HTML, Text, Word)

**Problem**: `ingest.py` only processes PDFs via `PyPDFLoader`. Real-world corpora include Markdown docs, HTML exports, Confluence/Notion dumps, and Word documents.

**Why it matters**: Without multi-format support, users must manually convert everything to PDF before ingestion, a friction point that blocks adoption and makes the pipeline unusable for heterogeneous knowledge bases.

**Proposed solution**: Extend `ingest.py` with format detection by file extension. Use LangChain loaders: `UnstructuredMarkdownLoader` (`.md`), `UnstructuredHTMLLoader` (`.html`), `TextLoader` (`.txt`), `UnstructuredWordDocumentLoader` (`.docx`). Preserve metadata (source path, headings as chunk metadata). Keep PDF pipeline unchanged.

**How it solves it**: Each loader extracts text + structure appropriate to its format. Heading-aware splitting (for Markdown/HTML) produces semantically coherent chunks. Single ingestion entry point handles all formats transparently.

**Success metric**: Ingest a mixed corpus (4 PDFs + 3 Markdown + 2 HTML + 1 Word) in one `python ingest.py` run; eval run over golden questions spanning all formats achieves correctness ≥ 0.80.

---

## 2. Cross-Encoder Reranker for Retrieval Quality

**Problem**: Retrieval relevance caps at ~0.30 (RESULTS.md line 92), indicating many retrieved chunks are distractors. The bi-encoder embedding model retrieves broadly but cannot precisely rank relevance.

**Why it matters**: Low retrieval relevance pollutes the generation context, reducing correctness and groundedness. A reranker prunes distractors before the LLM sees them, directly improving answer quality.

**Proposed solution**: Add a cross-encoder reranker (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) as a post-retrieval step in `rag.py`. Retrieve `k=20` with bi-encoder, rerank top 20, pass top `TOP_K` (default 7) to generation. Make reranker model configurable via `RERANKER_MODEL` in `.env`; disable by leaving unset.

**How it solves it**: Cross-encoders attend to query+document jointly, providing far better relevance discrimination than bi-encoder cosine similarity. The two-stage retrieve→rerank pattern is standard in production RAG.

**Success metric**: Retrieval relevance (RetrievalRelevance scorer) improves from ~0.30 to ≥ 0.50; correctness improves by ≥ 0.05; generation latency increase < 500ms per query on CPU.

---

## 3. FAQ-to-Golden-Questions Pipeline

**Problem**: No mechanism exists to capture frequently asked questions from production usage and promote them into the golden-question dataset for regression testing.

**Why it matters**: Production questions reveal real user intent and failure modes that authored golden questions miss. Without a feedback loop, the eval dataset drifts from reality.

**Proposed solution**: 
1. Add a `faq_capture.py` script (or extend `api.py` `/ask` endpoint) to log questions + answers + user feedback (thumbs up/down) to a `data/faq_log.jsonl`.
2. Add a `scripts/promote_faq.py` CLI that: reads the FAQ log, filters by frequency + negative feedback, presents candidates for review, and on approval appends them to `data/my_documents_golden_questions.json` with `expected_response` drafted from the best answer.
3. Document the workflow in `docs/EVALUATION.md`.

**How it solves it**: Creates a closed loop: production traffic → logged FAQs → human curation → golden dataset → eval regression → confidence in changes.

**Success metric**: At least 5 promoted FAQs per quarter; promoted questions catch at least 1 regression per release cycle (measured by eval run failure on promoted question before fix, pass after).

---

## 4. Docker Turnkey Deployment

**Problem**: Running the full stack (Ollama/LM Studio + MLflow server + Chroma + Streamlit/FastAPI) requires manual service orchestration, environment setup, and port management.

**Why it matters**: A single `docker compose up` that brings up the entire pipeline (model server, MLflow, vector store, UI, API) lowers the barrier to evaluation and demo, enables CI integration, and demonstrates production deployment competence.

**Proposed solution**: Create `Dockerfile` (multi-stage: build deps → runtime) and `docker-compose.yml` with services:
- `ollama` (or `lm-studio` if licensed): model server
- `mlflow`: tracking server with SQLite backend
- `chroma`: vector store (or use embedded Chroma in app)
- `api`: FastAPI on port 8000
- `ui`: Streamlit on port 8501
- `ingest`: one-shot ingestion job (runs on startup if `CHROMA_DIR` empty)
Use `.env` for all config; volumes for `chroma_db/`, `mlflow.db`, `data/`.

**How it solves it**: Encapsulates all dependencies, ports, and startup order in one file. New users clone repo, `cp .env.example .env`, `docker compose up`, and have a working RAG + eval stack in minutes.

**Success metric**: Fresh clone → `docker compose up` → `python eval_run.py swan-lagoon --limit 2` passes in < 10 minutes on a machine with 16GB RAM + GPU (or CPU-only with small models).

---

## 5. Reliability & Observability Hardening

**Problem**: The pipeline has launch-time health checks but lacks runtime resilience: no retry/timeout policies on model calls, no circuit breakers, no structured logging, no metrics export, no alerting hooks.

**Why it matters**: In production, model servers become slow/unavailable, network blips occur, and silent failures (empty answers, stale Chroma) degrade UX without visibility.

**Proposed solution**: 
- Add `tenacity` retry policies with exponential backoff to `ChatOpenAI` and `OpenAIEmbeddings` clients in `rag.py` and `ingest.py`.
- Add structured JSON logging (`structlog` or stdlib `logging` with JSON formatter) to all modules; log request_id, latency, token counts, error types.
- Export Prometheus metrics from `api.py` (`/metrics` endpoint): request count, latency histogram, error rate, active connections.
- Add `healthcheck.py` liveness/readiness probes for K8s/Docker health checks (already has standalone CLI; extend to HTTP `/healthz` in `api.py`).

**How it solves it**: Retries mask transient failures. Structured logs enable log aggregation. Metrics enable dashboards and alerting. Health probes enable orchestration.

**Success metric**: 99.9% request success rate under 5% model server error injection (chaos test); p99 latency < 10s; all errors appear in structured logs with request_id correlation.

---

## 6. Synthesis Evaluation Metric (Chunk-Disconnect Detection)

**Problem**: Answers assembled from disjoint chunks can be locally correct per-chunk but globally disconnected, missing synthesis a human would make. The 1000-char chunk increase helped (sufficiency +0.20) but may not solve multi-hop reasoning across non-adjacent chunks.

**Why it matters**: Without measuring synthesis, we cannot know if retrieval improvements (reranker, hierarchical retrieval) actually help complex questions.

**Proposed solution**: Implement a custom MLflow GenAI scorer `SynthesisScorer` that: (1) identifies which chunks the answer cites, (2) checks if cited chunks are non-adjacent in source documents, (3) verifies the answer combines facts from those chunks coherently (via judge prompt). Add to `eval_run.py` scorer list.

**How it solves it**: Directly measures the chunk-disconnect problem. A low synthesis score on multi-hop questions signals need for hierarchical/graph/iterative retrieval.

**Success metric**: Synthesis scorer added to eval runs; on Swan Lagoon multi-hop questions (e.g., "How did Swan Lagoon transition from mining to tourism?"), synthesis score correlates with human expert rating (Spearman ρ ≥ 0.7).

---

## 7. Larger Dedicated Judge Model (Evaluation Fidelity)

**Problem**: Current judge (`nvidia/nemotron-3-nano-4b`) contradicts its own rationale ~6-7% of the time (pipeline reconcile accuracy 0.943; retrieval 0.931 per RESULTS.md). Granite 4.1:3b self-judge achieves 0.970/0.951.

**Why it matters**: Judge contradictions poison metrics. Until the judge is reliable, we cannot trust correctness/groundedness scores to guide pipeline improvements.

**Proposed solution**: Run evals with a larger judge (e.g., `llama3.1:70b` via Ollama, or `gpt-4o-mini` via API) to establish an upper bound on reconciliation correction rate. Configure via `JUDGE_MODEL` + `JUDGE_BASE_URL` in `.env` (separate from chat). Measure correction count per eval run as the metric for judging the judge.

**How it solves it**: A more capable judge produces fewer rationale/verdict contradictions. The correction count becomes a direct measure of judge quality. Once corrections drop below target, we can trust the metrics.

**Success metric**: Pipeline reconciliation corrections < 5% (reconcile accuracy > 0.95) with the larger judge; correction count stabilizes across 5 consecutive eval runs.

---

## 8. Multi-Pass Reconciliation with Convergence Tracking

**Problem**: Reconciliation runs once per eval. Some runs may need multiple passes to converge (flip → new rationale check → flip). The number of passes is a latent signal of judge consistency.

**Why it matters**: A run needing 3 passes vs. 1 pass indicates deeper judge confusion. Tracking passes turns judge reliability into a time-series metric.

**Proposed solution**: Wrap `reconcile_run()` in a loop in `eval_consistency.py`; stop when `corrections_this_pass == 0` or `passes >= 3`. Log per-pass metrics: `reconcile_pass_1_corrections`, `reconcile_pass_2_corrections`, `reconcile_total_passes`, `reconcile_converged` (bool).

**How it solves it**: Iterative reconciliation catches cascading flips. Pass count becomes a continuous judge-reliability metric, not just a binary "reconciled" tag.

**Success metric**: Multi-pass reconciliation implemented; `reconcile_total_passes` logged on every eval run; median passes ≤ 1.5 across 10 runs.

---

## 9. RAGAS Comparison & Potential Migration

**Problem**: Five bespoke MLflow GenAI scorers require custom judge prompt maintenance. RAGAS provides community-standard metrics (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall) with built-in critic LLMs.

**Why it matters**: Standard metrics enable cross-project benchmarking and reduce maintenance burden. If RAGAS correlates well with our scorers, we can migrate.

**Proposed solution**: Add RAGAS scorers alongside current MLflow scorers in a parallel eval run (`eval_run.py` `--scorers ragas` or new preset). Correlate scores per question (Spearman ρ). If ρ ≥ 0.8 for key metrics (Correctness ↔ Faithfulness, RelevanceToQuery ↔ AnswerRelevancy), migrate.

**How it solves it**: Empirical comparison avoids religious wars. Correlation validates (or invalidates) RAGAS as a drop-in replacement.

**Success metric**: Parallel eval run completed; correlation report generated; decision documented (migrate / keep both / reject).

---

## 10. Automatic Chunk Size Optimization (Hyperparameter Sweep)

**Problem**: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` are set manually. The Chroma store is coupled to `(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL)`; changing any requires full re-ingest.

**Why it matters**: Optimal chunking depends on corpus, embedding model, and query distribution. Manual tuning is slow and suboptimal.

**Proposed solution**: Add `scripts/optimize_chunks.py` that: defines a grid (e.g., chunk_size ∈ {500, 750, 1000, 1500}, overlap ∈ {50, 100, 200}, top_k ∈ {5, 7, 10}); for each combo, deletes `chroma_db`, ingests, runs `eval_run.py --limit 10` (smoke), logs params + metrics to MLflow as a hyperparameter sweep. Uses MLflow's nested runs for organization.

**How it solves it**: Automates the re-ingest + eval loop. MLflow UI shows chunk params as columns → easy comparison. Best config promoted to `.env`.

**Success metric**: Grid search completes overnight; best config improves correctness by ≥ 0.03 over defaults; promoted config documented in `.env.example`.

---

## 11. Scalability: Larger Corpora & Concurrent Users

**Problem**: Current corpus is 4 PDFs (~360KB total). Chroma is in-process; no connection pooling, no query caching, no horizontal scaling story.

**Why it matters**: Production corpora are 100MB–10GB+. Concurrent users will hit GIL/threading limits in the current singleton Chroma store.

**Proposed solution**: 
- Evaluate Chroma persistent client + HTTP server mode (`chroma run --host 0.0.0.0 --port 8000`) for multi-process access.
- Add Redis caching layer for embeddings + frequent query results in `rag.py`.
- Profile memory/latency with 100MB corpus (synthetic or real).
- Document scaling limits and architecture for >1GB corpora (sharding, filtering, hybrid search).

**How it solves it**: HTTP Chroma enables multiple API workers. Caching reduces redundant embedding/compute. Profiling reveals bottlenecks before they hit production.

**Success metric**: 100MB corpus ingested in < 30 min; p99 query latency < 5s at 10 concurrent users (locust test); memory < 4GB.

---

## 12. Conversation Memory (Multi-Turn RAG)

**Problem**: Each query is independent. No session context means follow-up questions ("What about its hours?") fail because "it" has no referent.

**Why it matters**: Real users converse. Single-turn RAG feels like a search box, not an assistant.

**Proposed solution**: Add `session_id` to `rag.answer()` and `api.py` `/ask`. Store last N turns (Q+A+citations) in Chroma collection per session or in Redis. Prepend conversation summary to retrieval query. Expose `session_id` in Streamlit UI and FastAPI.

**How it solves it**: Session-scoped context enables referential follow-ups. Chroma/Redis provides persistence across restarts.

**Success metric**: Multi-turn eval: 5 follow-up questions on Swan Lagoon (e.g., "When is the festival?" → "What time does it start?" → "Is there parking?") all answered correctly with context from prior turns.

---

## 13. Streaming Answers in UI

**Problem**: Streamlit UI blocks until full answer generates. No token streaming → perceived latency = full generation time.

**Why it matters**: Streaming is table stakes for LLM UX. Users expect tokens to appear progressively.

**Proposed solution**: Update `rag.answer()` to support `stream=True` yielding tokens via `ChatOpenAI.stream()`. Update `app.py` to use `st.write_stream()` or `st.empty()` progressive rendering. Update `api.py` `/ask` to support Server-Sent Events (SSE) for streaming JSON.

**How it solves it**: Tokens render as they arrive. Perceived latency drops to first-token time (~500ms vs 5-10s full generation).

**Success metric**: First token latency < 1s; full answer streams smoothly; no UI flicker or buffering artifacts.

---

## Removed / Deferred

| Item | Reason |
|------|--------|
| Judge fine-tuning on reconciliation corrections | Circular: trains judge on its own potentially biased corrections; very high effort |
| Kubernetes deployment manifests | Infrastructure concern; Docker Compose (Item 4) covers turnkey deployment |
| Granite 4.1 as universal judge (fixed judge across chat models) | Covered by Item 7 (larger judge) + RAGAS comparison (Item 9) |
| Token-budget-aware generation for reasoning models | Niche; disable thinking mode (RESULTS.md line 42) is simpler workaround |

---

## Related Documentation

- **Evaluation workflow**: [`EVALUATION.md`](EVALUATION.md): from PDFs to scored eval runs
- **MLflow logging reference**: [`MLFLOW.md`](MLFLOW.md): params, metrics, artifacts, worker caps
- **Reconciliation mechanism**: [`RECONCILIATION.md`](RECONCILIATION.md): verdict/rationale corrections
- **Model comparisons**: [`RESULTS.md`](RESULTS.md): quality tables and judge consistency metrics
- **Architecture**: [`ARCHITECTURE.md`](ARCHITECTURE.md): component diagram, trade-offs
- **Quickstart**: [`QUICKSTART.md`](QUICKSTART.md): end-to-end walkthrough

**Vocabulary**: See [`../GLOSSARY.md`](../GLOSSARY.md) for the controlled terminology used across this project (corpus, dataset, golden question, judge, scorer, assessment, verdict, rationale, reconciliation, endpoint check).