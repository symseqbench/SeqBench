# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Deprecated compatibility wrapper for ``seqbench inspect-batch``."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from seqbench.cli import inspect_batch


def parse_cli_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_arguments(argv)
    warnings.warn(
        "python -m seqbench.read_dataset is deprecated; use "
        "`seqbench inspect-batch CONFIG` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    inspect_batch(
        args.config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
