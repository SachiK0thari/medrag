# Hugging Face Spaces-compatible Dockerfile for the Streamlit demo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for lxml + sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e .

# Pre-download embedding + reranker models so first request isn't a cold start
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Writable cache/data dirs for HF Spaces (repo dir is read-only)
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    DATA_DIR=/tmp/medrag \
    SQLITE_PATH=/tmp/medrag/medrag.db
RUN mkdir -p /tmp/hf_cache /tmp/medrag

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["streamlit", "run", "src/medrag/ui/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
