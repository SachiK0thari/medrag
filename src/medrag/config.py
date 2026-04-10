"""Centralized configuration. All env vars and tunables live here.

Read once at process start; never read os.environ elsewhere in the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Application settings, populated from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Secrets ---------------------------------------------------------- #
    groq_api_key: SecretStr
    pinecone_api_key: SecretStr
    ncbi_api_key: SecretStr | None = None  # raises rate limit 3 -> 10 req/s
    ncbi_email: str = "you@example.com"  # NCBI asks for a contact email

    # ---- LLM -------------------------------------------------------------- #
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ---- Embeddings ------------------------------------------------------- #
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # ---- Reranker --------------------------------------------------------- #
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 10

    # ---- Pinecone --------------------------------------------------------- #
    pinecone_index_name: str = "medrag-glp1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- Chunking --------------------------------------------------------- #
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64

    # ---- Retrieval -------------------------------------------------------- #
    bm25_top_k: int = 30
    dense_top_k: int = 30
    rrf_k: int = 60  # constant in Reciprocal Rank Fusion

    # ---- Agent ------------------------------------------------------------ #
    max_subqueries: int = 4
    max_reflection_loops: int = 2
    min_confidence_threshold: float = 0.6

    # ---- Paths ------------------------------------------------------------ #
    data_dir: Path = DATA_DIR
    sqlite_path: Path = DATA_DIR / "medrag.db"

    # ---- Ingestion -------------------------------------------------------- #
    ingest_topic_query: str = (
        "GLP-1 receptor agonist AND (cardiovascular outcome OR MACE)"
    )
    ingest_max_papers: int = 500


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — call this everywhere instead of constructing Settings()."""
    return Settings()  # type: ignore[call-arg]
