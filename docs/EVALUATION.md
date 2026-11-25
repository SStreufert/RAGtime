# Evaluation: From Your PDFs to a Scored, Comparable Run

An **experiment** is an **eval run**: every **golden question** in a dataset is answered through the RAG pipeline, each trace is graded by five LLM-judge scorers, and verdicts are reconciled against their own rationales. Everything that varies between experiments (chat model, judge model, `TOP_K`, `CHUNK_SIZE`) is read from `.env` and logged on every run, so runs stay comparable as columns in the MLflow UI.

---

## Prerequisites

Before you begin, ensure:

- Three model endpoints are running and serving their configured models:
  - **Embeddings** (`EMBEDDINGS_BASE_URL` / `EMBEDDING_MODEL`)
  - **Chat** (`CHAT_BASE_URL` / `CHAT_MODEL`)
  - **Judge** (`JUDGE_BASE_URL` / `JUDGE_MODEL`, falls back to chat if not set)

- A **MLflow tracking server** is running (required for registered datasets and the Evaluation tab):

  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```

  > **Why a server is required**: MLflow's GenAI datasets only work on a **SQL backend**; the `sqlite:///mlflow.db` command above provides one. Importing datasets, viewing judge rationales in the MLflow UI, and comparing runs side-by-side all need this server on `:5000`. Runs previously logged to a file store won't appear in this UI (`mlflow migrate-filestore` can move them if you need the history).

---

## Step 1: Ingest Your PDFs

Place your PDFs in `data/`, delete or rename `chroma_db` ( ChromaDB is coupled to `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `EMBEDDING_MODEL`), and ingest:

```bash
python ingest.py
```

Bringing your own documents? Remove the example PDFs (`sl_*.pdf`) from `data/` first if you want a corpus of only your documents, or keep them alongside yours if you want the swan-lagoon example to keep working.

For detailed chunking options and troubleshooting, see [QUICKSTART.md#ingesting-documents](QUICKSTART.md#ingesting-documents).

---

## Step 2: Write Golden Questions

Fill in your golden questions in the shipped template `data/my_documents_golden_questions.json` (it comes with `REPLACE_ME` placeholder records). Each question must be a single fact and include an `expected_response` extracted from your source documents. 

We use the Dataset feature in MLFlow to store these golden questions.

### Dataset JSON format

```json
[
  {
    "inputs": { "question": "What are the opening hours of the Swan Lagoon visitor centre?" },
    "expectations": { "expected_response": "The visitor centre is open 9am to 5pm, seven days a week." }
  },
  {
    "inputs": { "question": "What amenities are available at the lagoon?" },
    "expectations": { "expected_response": "Café, deckchairs, boat hire, and a gift shop are available." }
  }
]
```

- `inputs.question`: non-empty string; `inputs` must contain exactly the key `question`.
- `expectations.expected_response`: non-empty string; the facts an acceptable answer must support.

### Good question writing tips

- **One fact per question.** A question with five sub-facts gets one fuzzy score; five questions get five scores you can act on.
- **Write `expected_response` from the source document**, not from memory of what a model said. It is the ground truth the Correctness scorer measures against.
- **Cover the failure modes you care about**:
  - Questions answerable from one chunk
  - Questions that *require* two chunks (expose chunk-boundary disconnects)
  - Questions the corpus cannot answer (should return "I don't know", not hallucinate)
- **Start small**: a 5–10 question dataset you trust beats a 100-question dataset you don't.

The shipped example, `data/swan_lagoon_golden_questions.json` (25 questions over the example corpus), shows a filled-in dataset.

---

## Step 3: Test Your Questions

Before importing your dataset to MLflow, test it to ensure your golden questions work correctly.

### Test questions with ad-hoc evaluation

Evaluate against your template file without registering a dataset:

```bash
python evaluate.py batch --file data/my_documents_golden_questions.json
```

This runs a complete evaluation cycle: each question is answered through the RAG pipeline, all five LLM-judge scorers grade the results, and verdicts are reconciled against their own rationales.

**What you'll see:**
- Number of questions being evaluated
- Aggregate scores for each scorer (e.g., `retrieval_groundedness/mean: 0.8542`)
- Links to the MLflow UI for the run
- Any errors or failures (the command exits non-zero if issues occur)

**Example output:**
```
Evaluating 25 questions...

Run: 9f2a1b3c4d5e4f8a9b0c1d2e3f4a5b6c (batch-evaluation)
Metrics:
  correctness/mean: 0.9200
  retrieval_groundedness/mean: 0.8800
  retrieval_relevance/mean: 0.9400
  retrieval_relevance/precision/mean: 0.8900
  relevance_to_query/mean: 0.9600
  retrieval_sufficiency/mean: 0.8500

UI: http://127.0.0.1:5000/#/experiments/1234567890/runs/9f2a1b3c4d5e4f8a9b0c1d2e3f4a5b6c
```

If the output shows reasonable scores (say, above 0.5 for most metrics), proceed to Step 4. Otherwise, review your questions to make sure they are representative and that the answers can be extracted from the provided text. Edit as needed and retest. 

---

## Step 4: Import to MLflow and Run Full Evaluation

Once your questions pass testing, import the dataset in MLflow and run a full, linked experiment:

1. **Import your dataset (questions) to MLflow** (one-time, idempotent):

   ```bash
   python scripts/import_dataset.py swan-lagoon     # the shipped example
   python scripts/import_dataset.py                 # your documents (default)
   ```

   The import validates your JSON first (structure and `REPLACE_ME` placeholders) and exits non-zero on problems, so an import can never succeed with a file the eval would reject.

   > **Edited your questions since importing?** Re-run with `--force`: it replaces the registered dataset with the current JSON contents. Prior eval runs keep their data but lose the dataset link.

2. **Run the full experiment**:

   ```bash
   python eval_run.py swan-lagoon                   # the shipped example
   python eval_run.py my-documents                  # your documents
   ```

   The run is linked to the registered dataset and comparable with other runs in the MLflow UI. `eval_run.py` looks the dataset up in MLflow **by name**; no dataset IDs to copy anywhere.

---

## Step 5: Interpret the Results

### In the MLflow UI (http://localhost:5000)

- **Evaluation tab**: See Pass/Fail metrics and judge rationales for each question.
- **Traces tab**: Click a `rag_query` trace to see the retrieval/generation lifecycle with retrieved chunks and citations.
- **Run summary**: Per-scorer means appear as metrics; reconciliation tags appear as run tags.

### Run-level metrics (logged automatically)

| Metric | Scorer |
|--------|--------|
| `retrieval_groundedness/mean` | Is the answer grounded in retrieved context? |
| `retrieval_relevance/mean` | Is each retrieved chunk relevant to the request? |
| `retrieval_relevance/precision/mean` | Precision of retrieved chunks |
| `relevance_to_query/mean` | Does the answer address the question? |
| `correctness/mean` | Does the answer support expected facts? |
| `retrieval_sufficiency/mean` | Is retrieved context sufficient to answer? |

### Reconciliation tags (see [RECONCILIATION.md](RECONCILIATION.md))

| Tag | Meaning |
|-----|---------|
| `reconciled` | Run completed reconciliation pass |
| `reconcile_corrected=N` | Verdicts flipped against their rationales |
| `reconcile_uncertain=N` | Hedged rationales not auto-flipped |

### Comparing runs

Because every eval run logs the same param set (`chat_model`, `temperature`, `top_k`, `chunk_size`, `chunk_overlap`, `embedding_model`, `judge_model`, `judge_temperature`, `max_tokens`), runs appear as comparably-columned in the MLflow UI's side-by-side comparison view.

---

## Step 6: Iterate

To compare experimental variables:

1. Change one setting in `.env` (e.g., `TOP_K`, `CHAT_MODEL`, `CHUNK_SIZE`).
2. If `CHUNK_SIZE` or `CHUNK_OVERLAP` changed, delete `chroma_db` before re-ingesting.
3. Re-ingest: `python ingest.py`.
4. Re-run eval: `python eval_run.py my-documents` (or `swan-lagoon`).
5. Compare columns in the MLflow UI.

---

## Reference

### The Five Scorers

All are MLflow GenAI built-ins, powered by `JUDGE_MODEL` at `JUDGE_TEMPERATURE`:

| Scorer | Asks |
|--------|------|
| **Retrieval Relevance** | Is each retrieved chunk relevant to the input request? |
| **Retrieval Groundedness** | Is the answer grounded in the retrieved context? |
| **Relevance to Query** | Does the answer address the question? |
| **Correctness** | Does the answer support the expected facts? |
| **Retrieval Sufficiency** | Is the retrieved context sufficient to answer? |

### Presets (in `eval_run.py`)

| Preset | Dataset | Run name |
|--------|---------|----------|
| `swan-lagoon` | Swan Lagoon (25 records, registered) | `eval-swan-lagoon` |
| `my-documents` | Your documents (template at `data/my_documents_golden_questions.json`) | `eval-my-documents` |

### CLI Commands

```bash
# Ad-hoc (test your questions before importing)
python evaluate.py batch     --file data/my_documents_golden_questions.json  # all questions
python evaluate.py query     "Your question here" --file data/my_documents_golden_questions.json
python evaluate.py questions --file data/my_documents_golden_questions.json  # list questions

# Registered
python eval_run.py my-documents              # full run, linked to dataset (or swan-lagoon)
python eval_run.py my-documents --limit 2    # smoke test: first N rows
python eval_run.py my-documents --no-reconcile  # skip verdict fix-up
```

### Dataset management

Two presets are supported:

```bash
python scripts/import_dataset.py                       # my-documents (your template)
python scripts/import_dataset.py swan-lagoon           # the shipped example
python scripts/import_dataset.py my-documents --force  # replace after editing questions
```

The import validates the JSON first (structure + `REPLACE_ME` placeholders) and exits non-zero on problems. Without `--force`, an existing registered dataset is left untouched.

### Exit Codes

Both `eval_run.py` and `evaluate.py` exit **non-zero** if:
- Dataset validation fails (malformed records)
- Registered dataset not found in MLflow (`eval_run.py` only)
- Endpoint check fails (model server down)
- Any row produces an error
- Any scorer errors

---

## Related Documentation

- **MLflow logging reference**: [`MLFLOW.md`](MLFLOW.md): what gets logged, backend options, worker caps
- **Reconciliation mechanism**: [`RECONCILIATION.md`](RECONCILIATION.md): how verdict/rationale corrections work
- **Full model comparisons**: [`RESULTS.md`](RESULTS.md): quality tables and judge consistency metrics

**Vocabulary**: See [`../GLOSSARY.md`](../GLOSSARY.md) for the controlled terminology used across this project (corpus, dataset, golden question, judge, scorer, assessment, verdict, rationale, reconciliation, endpoint check).