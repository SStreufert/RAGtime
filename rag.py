import sys
import os

# Fix Windows console encoding for MLflow's emoji output (must be before MLflow import)
if sys.platform == "win32":
    # Skip under pytest: replacing the streams corrupts pytest's capture
    # (crashes the reporter with "lost sys.stderr"), and there is no Windows
    # console to fix under capture. CLI/Streamlit surfaces are unaffected.
    if "PYTEST_CURRENT_TEST" not in os.environ:
        # Wrap stdout/stderr to handle UTF-8 gracefully
        try:
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1, errors="replace")
            if hasattr(sys.stderr, "buffer"):
                sys.stderr = open(sys.stderr.fileno(), mode="w", encoding="utf-8", buffering=1, errors="replace")
        except (OSError, ValueError):
            pass  # captured or nonstandard streams: keep the existing ones

import threading
import time
import tracker

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

from config import settings

CHAT_MODEL = settings.chat_model
EMBED_MODEL = settings.embedding_model
CHROMA_DIR = settings.chroma_dir

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _short_source(source: str | None) -> str:
    """Return path relative to project root, or the original if outside."""
    if not source:
        return ""
    try:
        return os.path.relpath(source, _PROJECT_ROOT)
    except ValueError:
        # Different drive on Windows — return as-is
        return source

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer is partly contained, provide the best possible answer based on the context.
If the context does not contain the answer, say that you don't know.

Each numbered passage below is one retrieved document. You MUST cite the passages you use by including [n] markers in your answer. After your answer, you MUST declare exactly which passages you used on a line starting with "USED_SOURCES:".

Context:
{context}

Question:
{question}

Reply in this exact format:

ANSWER:
<your answer with [n] citations where appropriate>

USED_SOURCES:
[1], [3]
"""

PROMPT = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

_store_lock = threading.Lock()
_store = None


def get_store():
    """Return a process-wide shared Chroma store.

    Chroma's PersistentClient is not safe to construct concurrently: parallel
    calls race inside SharedSystemClient's identifier->system registry and
    corrupt its Rust bindings. Build it once under a lock and reuse it.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=OpenAIEmbeddings(
                        model=EMBED_MODEL,
                        base_url=settings.embeddings_base_url,
                        api_key=settings.embeddings_api_key,
                        check_embedding_ctx_length=False,
                    ),
                )
    return _store


def format_docs(docs):
    """Render retrieved docs as numbered passages.

    The index is assigned by retrieval rank (1..k) so the LLM can cite
    [1], [2]... against sources that actually exist. The same numbering is
    what app.py uses to render the trailing "Sources used" block, so the
    numbers in the answer always resolve to real documents.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        label = page + 1 if isinstance(page, int) else "?"
        parts.append(f"[{i}] (page {label}) {doc.page_content}")
    return "\n\n".join(parts)


def parse_used_sources(text: str, k: int) -> list[int]:
    """Extract the indices the LLM declared it used.

    Looks for the "USED_SOURCES:" line first. If the model ignored that
    convention, falls back to scanning the whole answer for [n] references.
    Returns indices in the order they appear, deduplicated, clamped to the
    range that actually exists (1..k).
    """
    import re

    used: list[int] = []
    marker = re.search(r"USED_SOURCES:\s*(.*)", text, re.IGNORECASE)
    if marker:
        used = [int(n) for n in re.findall(r"\[(\d+)\]", marker.group(1))]
    else:
        used = [int(n) for n in re.findall(r"\[(\d+)\]", text)]

    seen = set()
    result = []
    for n in used:
        if 1 <= n <= k and n not in seen:
            seen.add(n)
            result.append(n)
    return result


def strip_citations(text: str) -> str:
    """Remove [n] markers from prose so the displayed answer is clean.

    The trailing Sources used block is rendered separately by app.py from
    the parsed indices, so the body should not carry the markers.
    """
    import re

    cleaned = re.sub(r"\[\d+\]", "", text)
    # Drop the USED_SOURCES section and any trailing whitespace left behind.
    cleaned = re.split(r"\n?\s*USED_SOURCES:", cleaned, flags=re.IGNORECASE)[0]
    # Drop "ANSWER:" prefix if present
    cleaned = re.sub(r"^\s*ANSWER:\s*", "", cleaned, flags=re.IGNORECASE)
    # Drop empty citation markers like "(source: )" or "(source:)"
    cleaned = re.sub(r"\(\s*source\s*:\s*\)", "", cleaned, flags=re.IGNORECASE)
    # Drop malformed citations like "as stated in ." or "as mentioned in  and reinforced in ."
    # Match citation phrases and everything up to the next period or end of string
    cleaned = re.sub(
        r"\b(?:as\s+(?:stated|mentioned|noted)\s+in|per|according\s+to)\b.*?(?=\.|$)",
        "",
        cleaned,
        flags=re.IGNORECASE
    )
    # Drop dangling "in" with empty brackets
    cleaned = re.sub(r"\bin\s+\[\s*\]", "", cleaned, flags=re.IGNORECASE)
    # Drop empty brackets anywhere
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    # Clean up double spaces and trailing commas before periods
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    return cleaned.strip()


def _retrieve_docs(db, question, k):
    """Retrieve top-k chunks; returns (docs, retrieval_ms, top relevance score)."""
    retrieval_start = time.perf_counter()
    doc_score_pairs = db.similarity_search_with_relevance_scores(question, k=k)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    docs = [doc for doc, _ in doc_score_pairs]
    retrieval_score = doc_score_pairs[0][1] if doc_score_pairs else None
    return docs, retrieval_ms, retrieval_score


def _generate_text(chain, docs, question):
    """Run the prompt|llm|parser chain; returns (text, generation_ms, token_usage)."""
    generation_start = time.perf_counter()
    text = chain.invoke({"context": format_docs(docs), "question": question})
    generation_ms = (time.perf_counter() - generation_start) * 1000

    context_text = "\n".join(d.page_content for d in docs)
    context_tokens = len(context_text) // 4
    answer_tokens = len(text) // 4
    token_usage = {
        "input_tokens": context_tokens,
        "output_tokens": answer_tokens,
        "total_tokens": context_tokens + answer_tokens
    }
    return text, generation_ms, token_usage


def answer(question, k=3, track=True):
    # One availability check up front gates BOTH the trace spans and the query
    # logging. With the MLflow backend down, a query costs one short probe
    # instead of a multi-minute retry storm, and no traces are queued for the
    # background exporter to fail on. track=False (eval) always traces: the
    # eval run requires MLflow and its scorers read these spans.
    tracking_on = (not track) or tracker.is_available()

    import json
    from mlflow.tracing.fluent import start_span
    from mlflow.entities import SpanType

    db = get_store()

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=settings.temperature,
        max_completion_tokens=settings.max_tokens,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
    )
    chain = PROMPT | llm | StrOutputParser()

    if tracking_on:
        with start_span(name="rag_query", span_type=SpanType.CHAIN) as root_span:
            root_span.set_attribute("mlflow.spanInputs", json.dumps({
                "question": question,
                "k": k
            }))

            with start_span(name="retrieval", span_type=SpanType.RETRIEVER) as retrieval_span:
                docs, retrieval_ms, retrieval_score = _retrieve_docs(db, question, k)

                retrieval_span.set_attribute("mlflow.spanInputs", json.dumps({
                    "question": question,
                    "k": k
                }))
                retrieval_span.set_attribute("mlflow.spanOutputs", json.dumps([
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ]))
                if retrieval_score is not None:
                    retrieval_span.set_attribute(
                        "retrieval_score", float(retrieval_score)
                    )

            with start_span(name="generation", span_type=SpanType.CHAT_MODEL) as gen_span:
                text, generation_ms, token_usage = _generate_text(chain, docs, question)

                gen_span.set_attribute("mlflow.chat.tokenUsage", token_usage)
                gen_span.set_attribute("mlflow.llm.model", CHAT_MODEL)
                gen_span.set_attribute("mlflow.llm.provider", settings.chat_base_url)
                gen_span.set_attribute("mlflow.message.format", "langchain")
                gen_span.set_attribute("mlflow.spanInputs", json.dumps({
                    "messages": [{"role": "user", "content": format_docs(docs) + "\n\nQuestion: " + question}],
                    "model": CHAT_MODEL,
                    "temperature": settings.temperature
                }))
                gen_span.set_attribute("mlflow.spanOutputs", json.dumps({
                    "predictions": [{"role": "assistant", "content": text}],
                    "generation_latency_ms": generation_ms
                }))

            root_span.set_attribute("mlflow.spanOutputs", json.dumps({
                "answer": text,
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms
            }))
    else:
        docs, retrieval_ms, retrieval_score = _retrieve_docs(db, question, k)
        text, generation_ms, _token_usage = _generate_text(chain, docs, question)

    used_indices = parse_used_sources(text, k)
    clean_text = strip_citations(text)

    # Tracker logging — only when explicitly enabled
    if track and tracking_on:
        with tracker.start_run(run_name="query", nested=True) as run:
            if run:
                tracker.log_query(
                    question=question,
                    answer=text,
                    docs=docs,
                    used_sources=used_indices,
                    retrieval_score=retrieval_score,
                    retrieval_latency_ms=retrieval_ms,
                    generation_latency_ms=generation_ms,
                    prompt_template=PROMPT_TEMPLATE,
                )

    return clean_text, docs, used_indices


if __name__ == "__main__":
    import argparse
    import sys

    # Fix Windows console encoding for MLflow's emoji output
    if sys.platform == "win32":
        import os
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example: python rag.py "What destination is a Must See?"'
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Question to ask (if omitted, runs interactively)"
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable MLflow tracking for this query"
    )
    args = parser.parse_args()

    from healthcheck import ensure_or_exit

    ensure_or_exit(("chat", "embeddings"))

    # If no question provided, run interactively
    if args.question is None:
        print("Enter your question (or 'quit' to exit):")
        try:
            question = input("> ").strip()
            if not question or question.lower() in ("quit", "exit", "q"):
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    else:
        question = args.question

    text, sources, used = answer(question, k=args.top_k, track=not args.no_track)
    print(text)
    print("\n--- sources used ---")
    for i in used:
        doc = sources[i - 1]
        page = doc.metadata.get("page")
        label = page + 1 if isinstance(page, int) else "?"
        print(f"[{i}] {_short_source(doc.metadata.get('source'))} p. {label}")
    print("\n--- all retrieved ---")
    for i, doc in enumerate(sources, start=1):
        page = doc.metadata.get("page")
        label = page + 1 if isinstance(page, int) else "?"
        print(f"[{i}] {_short_source(doc.metadata.get('source'))} p. {label}")
