"""Run the eval harness across retrieval configs.

Usage:
    python scripts/run_eval.py                       # all configs -> data/eval_results.csv
    python scripts/run_eval.py --configs agentic,agentic_v2 --out data/eval_agentic.csv
    python scripts/run_eval.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medrag.eval.configs import build_configs, config_names
from medrag.eval.harness import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        help="Comma-separated config names (default: all). See --list.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="CSV path (default: data/eval_results.csv).",
    )
    parser.add_argument("--list", action="store_true", help="Print config names and exit.")
    args = parser.parse_args()

    if args.list:
        print("\n".join(config_names()))
        return

    names = [n.strip() for n in args.configs.split(",")] if args.configs else None
    run_all(build_configs(names), out_path=args.out)


if __name__ == "__main__":
    main()
