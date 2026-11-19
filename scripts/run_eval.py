"""Run the eval harness across retrieval configs.

Usage:
    python scripts/run_eval.py                       # all configs -> data/eval_results.csv
    python scripts/run_eval.py --configs agentic,agentic_v2 --out data/eval_agentic.csv
    python scripts/run_eval.py --configs agentic_v2 --keep-existing   # re-run one config in place
    python scripts/run_eval.py --list
    python scripts/run_eval.py --split test --configs hybrid_reranked,agentic_v2
                                                     # -> data/eval_results_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from medrag.config import get_settings
from medrag.eval.configs import build_configs, config_names
from medrag.eval.harness import run_all
from medrag.eval.splits import SPLITS, questions_for, split_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        help="Comma-separated config names (default: all). See --list.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="CSV path (default: data/eval_results.csv, or eval_results_test.csv for --split test).",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="dev",
        help="Question set: dev (the 33 tuning questions) or test (never used for tuning).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep rows/traces already in --out for configs that are not being re-run.",
    )
    parser.add_argument("--list", action="store_true", help="Print config names and exit.")
    args = parser.parse_args()

    if args.list:
        print("\n".join(config_names()))
        return

    names = [n.strip() for n in args.configs.split(",")] if args.configs else None
    out = args.out or split_files(args.split, get_settings().data_dir).results
    run_all(
        build_configs(names),
        out_path=out,
        keep_existing=args.keep_existing,
        questions=questions_for(args.split),
    )


if __name__ == "__main__":
    main()
