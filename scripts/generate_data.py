#!/usr/bin/env python3
"""Thin wrapper: generate the default synthetic dataset.

Usage: python scripts/generate_data.py [--n 600] [--seed 42] [--out data/synthetic.csv]
See scripts/cli.py generate for the same command with more options.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_platform import data as data_module  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic.csv")
    args = parser.parse_args()
    rows = data_module.generate_synthetic(n=args.n, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data_module.write_csv(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
