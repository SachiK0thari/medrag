"""FastAPI service exposing the agent over HTTP."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medrag import __version__
from medrag.agent.graph import MedRagAgent
from medrag.models import Answer

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
    question: str = Field(..., min_length=3, max_length=500)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    try:
        return get_agent().answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
