# Local RAG

A local RAG pipeline that answers questions from an ingested PDF corpus, graded by LLM judges and tracked in MLflow.

## Language

**Corpus**:
The ingested PDF documents the pipeline retrieves from.
_Avoid_: knowledge base, docs, dataset

**Dataset**:
A registered collection of golden questions with their expected responses.
_Avoid_: corpus, test set

**Golden question**:
A dataset entry pairing a question with the expected response used as ground truth for judging.
_Avoid_: record, row, test case, sample

**Dataset preset**:
A name that stands for one registered dataset and the run name used when evaluating it.
_Avoid_: experiment, profile, config

**Eval run**:
An MLflow run that answers a dataset's golden questions and records the judges' scores for them.
_Avoid_: experiment, test run, evaluation job

**Judge**:
The LLM that a scorer calls to grade one aspect of an eval run.
_Avoid_: model, oracle

**Scorer**:
A judge-backed check that grades every golden question of an eval run on one aspect, such as correctness or retrieval sufficiency.
_Avoid_: metric, benchmark

**Assessment**:
A judge's verdict together with its rationale, attached to one trace.
_Avoid_: feedback, score

**Verdict**:
The normalized yes or no that a judge returns for an assessment.
_Avoid_: label, prediction

**Rationale**:
The judge's written justification, stored alongside its verdict.
_Avoid_: explanation, reason

**Reconciliation**:
The post-scoring pass that compares each verdict with its rationale and corrects unambiguous contradictions.
_Avoid_: cleanup, repair

**Endpoint check**:
The launch-time probe that verifies each configured OpenAI-compatible endpoint answers and serves its configured model before any work starts (`healthcheck.py`).
_Avoid_: health check, ping, smoke test
