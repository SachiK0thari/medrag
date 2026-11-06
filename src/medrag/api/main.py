"""FastAPI service exposing the agent over HTTP."""

from __future__ import annotations

import math

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from medrag import __version__
from medrag.agent.graph import MedRagAgent
from medrag.config import MAX_QUESTION_CHARS
from medrag.models import Answer
from medrag.ratelimit import get_limiter

app = FastAPI(
    title="medrag",
    version=__version__,
    description="Agentic RAG over GLP-1 cardiovascular outcome literature.",
)

_agent: MedRagAgent | None = None


def get_agent() -> MedRagAgent:
    global _agent
    if _agent is None:
        _agent = MedRagAgent()
    return _agent


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=MAX_QUESTION_CHARS)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest, request: Request) -> Answer:
    # Keyed on the peer address. Behind a reverse proxy, run uvicorn with
    # --proxy-headers so this is the real client and not the proxy.
    client = request.client.host if request.client else "unknown"
    decision = get_limiter().acquire(client)
    if not decision.allowed:
        raise HTTPException(
            status_code=429,
            detail=decision.message,
            headers={"Retry-After": str(math.ceil(decision.retry_after_s))},
        )
    try:
        return get_agent().answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
