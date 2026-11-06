"""Upload the corpus database to the private Hugging Face dataset the Space builds from.

The Space image cannot take `data/medrag.db` from git (it is ignored, and it
holds paper full text that should not be republished), so the Dockerfile
downloads it at build time from a private dataset repo, pinned to one commit.

    python scripts/upload_corpus.py            # needs HF_TOKEN (write) in .env or the env

Prints the new dataset commit. Put it in the Dockerfile's `CORPUS_REVISION`
and redeploy; an unchanged revision keeps serving the old corpus.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

DEFAULT_REPO = "SachiK/medrag-corpus"
DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "medrag.db"


def _token() -> str:
    env = {k.upper(): v for k, v in dotenv_values(".env").items() if v}
    token = os.environ.get("HF_TOKEN") or env.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN (a token with write access) in the environment or .env.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        papers = conn.execute("SELECT count(*) FROM papers").fetchone()[0]
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if not chunks:
        raise SystemExit(f"{args.db} has no chunks; refusing to upload an empty corpus.")

    api = HfApi(token=_token())
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    commit = api.upload_file(
        path_or_fileobj=args.db,
        path_in_repo="medrag.db",
        repo_id=args.repo,
        repo_type="dataset",
        commit_message=f"corpus: {papers} papers, {chunks} chunks",
    )
    print(f"Uploaded {papers} papers, {chunks} chunks to {args.repo}")
    print(f"CORPUS_REVISION={commit.oid}")


if __name__ == "__main__":
    main()
