"""HTTP API and a one-page UI. Run: uvicorn rag.api:app --reload"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.answer import answer_question
from rag.chunking import STRATEGIES
from rag.config import get_settings
from rag.db import connect, init_db
from rag.retrieval import MODES, search


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # create tables on startup so /health and /ask work before the first ingest
    yield


app = FastAPI(title="Manuals RAG", version="1.0", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: str = "section"
    mode: str | None = None


def _validate(q: Query) -> str:
    mode = q.mode or ("hybrid_rerank" if get_settings().reranker != "none" else "hybrid")
    if q.strategy not in STRATEGIES or mode not in MODES:
        raise HTTPException(422, f"strategy in {STRATEGIES}, mode in {MODES}")
    return mode


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    with connect() as conn:
        docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    return {"status": "ok", "documents": docs}


@app.post("/search")
def search_endpoint(q: Query):
    hits, timings = search(q.question, strategy=q.strategy, mode=_validate(q))
    return {"hits": [h.to_dict() for h in hits], "timings": timings}


@app.post("/ask")
def ask(q: Query):
    return answer_question(q.question, strategy=q.strategy, mode=_validate(q)).to_dict()
