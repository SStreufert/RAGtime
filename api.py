# api.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import healthcheck
from rag import answer, _short_source


@asynccontextmanager
async def lifespan(_app: FastAPI):
    results = healthcheck.probe_launch(("chat", "embeddings"))
    if not healthcheck.all_ok(results):
        raise RuntimeError(healthcheck.report_text(results))
    yield


app = FastAPI(title="Local RAG API", lifespan=lifespan)


class Query(BaseModel):
    question: str
    k: int = Field(default=3, ge=1)


@app.get("/health")
def health():
    """Live endpoint status — fresh probe per call, 200 healthy / 503 degraded."""
    results = healthcheck.run_check(("chat", "embeddings"))
    return JSONResponse(
        status_code=200 if healthcheck.all_ok(results) else 503,
        content={
            "ok": healthcheck.all_ok(results),
            "endpoints": [
                {
                    "role": r.role,
                    "base_url": r.base_url,
                    "model": r.model,
                    "ok": r.ok,
                    "degraded": r.degraded,
                    "detail": r.detail,
                }
                for r in results
            ],
        },
    )


@app.post("/ask")
def ask(query: Query):
    text, docs, used = answer(query.question, k=query.k)
    return {
        "answer": text,
        "sources_used": used,
        "sources": [
            {
                "index": i,
                "source": _short_source(doc.metadata.get("source")),
                "page": doc.metadata.get("page"),
            }
            for i, doc in enumerate(docs, start=1)
        ],
    }