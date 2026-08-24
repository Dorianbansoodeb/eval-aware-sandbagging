#!/usr/bin/env python3
"""Run the full eval-aware sandbagging pipeline."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_items import main as build_items
from src.config import ROOT
from src.elicit import run_elicit
from src.eval_run import run_eval
from src.figures import main as make_figures
from src.interp import run_interp
from src.suffix_search import run_suffix_search


def _cleanup():
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage",
        default="all",
        choices=["data", "eval", "elicit", "suffix", "interp", "figures", "all"],
    )
    p.add_argument("--config", default=None)
    args = p.parse_args()
    ROOT.joinpath("results").mkdir(exist_ok=True)

    if args.stage in ("data", "all"):
        build_items()
    if args.stage in ("eval", "all"):
        print("=== eval ===")
        print(run_eval(args.config))
        _cleanup()
    if args.stage in ("elicit", "all"):
        print("=== elicit ===")
        print(run_elicit(args.config))
        _cleanup()
    if args.stage in ("suffix", "all"):
        print("=== suffix ===")
        print(run_suffix_search(args.config))
        _cleanup()
    if args.stage in ("interp", "all"):
        print("=== interp ===")
        print(run_interp(args.config))
        _cleanup()
    if args.stage in ("figures", "all"):
        make_figures()


if __name__ == "__main__":
    main()
