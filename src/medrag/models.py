"""Core data models for the medrag pipeline.

Every piece of data flowing through ingestion, retrieval, and generation is
typed via Pydantic. This is the contract between modules — keep it tight.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# --------------------------------------------------------------------------- #
# Source documents                                                             #
# --------------------------------------------------------------------------- #


class PaperSource(str, Enum):
    """Where a paper's full text came from."""

    PMC_XML = "pmc_xml"  # Open-access full text from PubMed Central
    PUBMED_ABSTRACT = "pubmed_abstract"  # Abstract only (no PMC OA full text)


class Paper(BaseModel):
    """A single scientific paper ingested from PubMed/PMC."""

    model_config = ConfigDict(frozen=True)

    pmid: str = Field(..., description="PubMed ID")
    pmcid: str | None = Field(None, description="PubMed Central ID, if open access")
    title: str
    abstract: str | None = None
    full_text: str | None = Field(None, description="Plain-text body if PMC OA available")
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    year: int | None = None
    doi: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    source: PaperSource
    url: HttpUrl | None = None

    @property
    def has_full_text(self) -> bool:
        return self.full_text is not None and len(self.full_text) > 500

    @property
    def display_citation(self) -> str:
        """Short citation string for inline use, e.g. 'Smith 2023, NEJM'."""
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        year = self.year or "n.d."
        journal = self.journal or ""
        return f"{first_author} {year}{', ' + journal if journal else ''}"


# --------------------------------------------------------------------------- #
# Chunks (the unit of retrieval)                                               #
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """A retrievable text chunk derived from a Paper."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Stable hash-based ID, used as Pinecone vector ID")
    pmid: str
    text: str
    section: str | None = Field(None, description="e.g. 'abstract', 'methods', 'results'")
    chunk_index: int = Field(..., ge=0)
    n_tokens: int = Field(..., ge=0)


class EmbeddedChunk(BaseModel):
    """A Chunk paired with its dense embedding vector."""

    chunk: Chunk
    embedding: list[float]


# --------------------------------------------------------------------------- #
# Retrieval                                                                    #
# --------------------------------------------------------------------------- #


class RetrievalMethod(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANKED = "hybrid_reranked"


class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval, with its score and provenance."""

    chunk: Chunk
    paper: Paper
    score: float
    rank: int = Field(..., ge=1)
    method: RetrievalMethod


class RetrievalResult(BaseModel):
    """Full result of a single retrieval call."""

    query: str
    method: RetrievalMethod
    chunks: list[RetrievedChunk]
    latency_ms: float


# --------------------------------------------------------------------------- #
# Agent + Generation                                                           #
# --------------------------------------------------------------------------- #


class SubQuery(BaseModel):
    """One decomposed sub-query produced by the planner step."""

    text: str
    rationale: str | None = None


class Citation(BaseModel):
    """An inline citation pointing back to a source chunk."""

    marker: str = Field(..., description="e.g. '[1]'")
    pmid: str
    chunk_id: str
    display_citation: str
    snippet: str = Field(..., description="The exact chunk text the claim is grounded in")


class Answer(BaseModel):
    """A cited answer produced by the agent."""

    question: str
    text: str
    citations: list[Citation]
    sub_queries: list[SubQuery] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    n_retrieval_calls: int = Field(..., ge=1)
    total_chunks_considered: int = Field(..., ge=0)


# --------------------------------------------------------------------------- #
# Tracing (lightweight observability — SQLite-backed)                          #
# --------------------------------------------------------------------------- #


class TraceEvent(BaseModel):
    """A single timed event in a query trace."""

    name: str
    started_at: datetime
    duration_ms: float
    metadata: dict[str, str | int | float] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    """End-to-end trace of one user query."""

    trace_id: str
    question: str
    started_at: datetime
    total_latency_ms: float
    events: list[TraceEvent]
    answer: Answer
    estimated_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
