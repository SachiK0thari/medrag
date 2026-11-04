"""Run the eval harness across all retrieval configs.

Usage:
    python scripts/run_eval.py
"""

from __future__ import annotations

from medrag.eval.configs import build_all_configs
from medrag.eval.harness import run_all

if __name__ == "__main__":
    run_all(build_all_configs())
