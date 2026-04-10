"""Run the eval harness across all retrieval configs.

Usage:
    python scripts/run_eval.py
"""

from __future__ import annotations

from medrag.eval.harness import run_all

if __name__ == "__main__":
    # TODO(sunday): instantiate the four+ RetrievalConfig implementations and pass in.
    # configs = [DenseOnlyConfig(), BM25OnlyConfig(), HybridRRFConfig(), HybridRerankedConfig(), AgenticConfig()]
    # run_all(configs)
    print("Implement RetrievalConfig wrappers Sunday morning, then call run_all().")
