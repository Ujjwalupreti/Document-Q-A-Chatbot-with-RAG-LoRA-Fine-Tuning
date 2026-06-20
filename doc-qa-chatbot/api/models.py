from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    question: str


class SourceChunk(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int
    status: str = "indexed"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter_loaded: bool = False


# ── Model Metrics Dashboard ──

class EvalScores(BaseModel):
    exact_match: float
    f1: float
    rougeL: float


class LatencyMetrics(BaseModel):
    naive_ms: float
    rag_ms: float
    reduction_pct: float


class MetricsResponse(BaseModel):
    """
    Powers the Model Metrics Dashboard view. `source` is "live" when read
    from real eval/benchmark run output, or "sample" when falling back to
    placeholder figures (so the dashboard is viewable before fine-tuning
    has been run).
    """
    baseline: EvalScores
    fine_tuned: EvalScores
    improvement_pct: float
    latency: LatencyMetrics
    cost_reduction_pct: float
    source: str
