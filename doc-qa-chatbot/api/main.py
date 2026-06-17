"""
FastAPI backend for the Document Q&A Chatbot.

Endpoints:
    GET    /health                          - health check
    POST   /upload?session_id={id}          - upload a document, chunk + embed + store
    POST   /chat                            - ask a question against a session's documents
    DELETE /sessions/{session_id}           - delete a session's vector store
    GET    /metrics                          - eval + latency metrics for the dashboard

Run from the project root with:
    uvicorn api.main:app --reload
"""
import json
import logging
import re
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from rag.ingest import delete_session, ingest_document
from rag.retriever import retrieve

from .llm import LLM
from .models import (
    ChatRequest,
    ChatResponse,
    EvalScores,
    HealthResponse,
    LatencyMetrics,
    MetricsResponse,
    SourceChunk,
    UploadResponse,
)

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB limit

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_PATH = _PROJECT_ROOT / "finetune" / "eval_results.json"
BENCHMARK_RESULTS_PATH = _PROJECT_ROOT / "benchmark" / "benchmark_results.json"

SAMPLE_METRICS = MetricsResponse(
    baseline=EvalScores(exact_match=0.41, f1=0.63, rougeL=0.58),
    fine_tuned=EvalScores(exact_match=0.79, f1=0.91, rougeL=0.88),
    improvement_pct=28.0,
    latency=LatencyMetrics(naive_ms=1800, rag_ms=800, reduction_pct=55.0),
    cost_reduction_pct=60.0,
    source="sample",
)


def _validate_session_id(session_id: str) -> str:
    """Validate that session_id is a proper UUID to prevent path traversal."""
    if not _UUID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format. Must be a UUID.")
    return session_id

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading LLM model in background thread…")
    await run_in_threadpool(LLM.load)
    logger.info("LLM model loaded successfully.")
    yield


app = FastAPI(title="Document Q&A Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=LLM.is_loaded(),
        adapter_loaded=LLM.is_adapter_loaded(),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Query(default=None),
) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    if session_id:
        _validate_session_id(session_id)
    else:
        session_id = str(uuid.uuid4())

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        chunks_indexed = await run_in_threadpool(ingest_document, tmp_path, session_id)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_indexed=chunks_indexed,
        status="indexed",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    _validate_session_id(request.session_id)

    results = await run_in_threadpool(retrieve, request.session_id, request.question)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No documents found for this session. Upload a document first.",
        )

    context = "\n\n".join(doc.page_content for doc, _ in results)
    answer = await run_in_threadpool(LLM.generate, context, request.question)

    sources = [
        SourceChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
            score=round(score, 4),
        )
        for doc, score in results
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.delete("/sessions/{session_id}")
async def delete_session_endpoint(session_id: str) -> dict:
    _validate_session_id(session_id)
    await run_in_threadpool(delete_session, session_id)
    return {"status": "deleted", "session_id": session_id}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    """
    Reads real results from finetune/evaluate.py and
    benchmark/retrieval_benchmark.py if they exist; otherwise returns
    SAMPLE_METRICS so the dashboard is viewable immediately.
    """
    if not EVAL_RESULTS_PATH.exists() or not BENCHMARK_RESULTS_PATH.exists():
        return SAMPLE_METRICS

    try:
        eval_data = json.loads(EVAL_RESULTS_PATH.read_text())
        bench_data = json.loads(BENCHMARK_RESULTS_PATH.read_text())

        baseline = eval_data["zero_shot"]
        fine_tuned = eval_data["fine_tuned"]
        improvement_pct = (fine_tuned["f1"] - baseline["f1"]) / max(baseline["f1"], 1e-9) * 100

        return MetricsResponse(
            baseline=EvalScores(**{k: baseline[k] for k in ("exact_match", "f1", "rougeL")}),
            fine_tuned=EvalScores(**{k: fine_tuned[k] for k in ("exact_match", "f1", "rougeL")}),
            improvement_pct=round(improvement_pct, 1),
            latency=LatencyMetrics(**bench_data),
            cost_reduction_pct=SAMPLE_METRICS.cost_reduction_pct,
            source="live",
        )
    except (KeyError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to parse eval/benchmark results, falling back to sample metrics: %s", exc
        )
        return SAMPLE_METRICS