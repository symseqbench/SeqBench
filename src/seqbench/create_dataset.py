# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Deprecated compatibility wrapper for ``seqbench create``."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from seqbench.cli import create


def parse_cli_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument("--split", type=str, default=None, help="Optional split to build.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_arguments(argv)
    warnings.warn(
        "python -m seqbench.create_dataset is deprecated; use "
        "`seqbench create CONFIG` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    create(args.config, split=args.split)


if __name__ == "__main__":
    main()
