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
    SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache \
    DATA_DIR=/tmp/medrag \
    SQLITE_PATH=/tmp/medrag/medrag.db
RUN mkdir -p /tmp/hf_cache /tmp/medrag

# The corpus (papers + chunks) feeds BM25 and supplies the text behind every
# Pinecone hit; without it the app answers with no sources. It is not in git,
# so fetch it from the private dataset that scripts/upload_corpus.py writes,
# pinned to one commit: change CORPUS_REVISION to ship a new corpus.
# Needs the HF_TOKEN secret (read access to the dataset): a Space secret on
# Hugging Face, `docker build --secret id=HF_TOKEN,env=HF_TOKEN .` locally.
ARG CORPUS_REPO=SachiK/medrag-corpus
ARG CORPUS_REVISION=a8d7c42ff8e32d33716d7b0e426e025edabc7183
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true python -c "\
import sqlite3; \
from huggingface_hub import hf_hub_download; \
hf_hub_download('${CORPUS_REPO}', 'medrag.db', repo_type='dataset', \
    revision='${CORPUS_REVISION}', local_dir='/tmp/medrag', \
    token=open('/run/secrets/HF_TOKEN').read().strip()); \
n = sqlite3.connect('/tmp/medrag/medrag.db').execute('SELECT count(*) FROM chunks').fetchone()[0]; \
assert n, 'the downloaded corpus has no chunks'; \
print('corpus chunks:', n)" \
    && chmod -R a+rwX /tmp/medrag

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["streamlit", "run", "src/medrag/ui/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
