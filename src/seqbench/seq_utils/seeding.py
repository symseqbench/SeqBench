# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Per-trial seed derivation for symbolic generation and rendering.

Every random draw that belongs to one dataset item is derived from the tuple
``(seed, split, idx, stream, *sub)`` via :class:`numpy.random.SeedSequence`.
The result depends only on that tuple — not on access order, DataLoader
workers, epochs, or draws made for other items — so a single item can be
regenerated, and one random factor (e.g. gaps) can be changed without
shifting the draws of another (e.g. base-item choice).
"""

from __future__ import annotations

import zlib
from enum import IntEnum

import numpy as np
import torch


class Stream(IntEnum):
    """Independent random streams of one dataset item."""

    SYMBOL = 0  # symbolic trial drawn from the source
    BASE_ITEM = 1  # base-dataset item rendering each symbol occurrence
    GAP = 2  # gap occurrence, duration, filler and noise
    TRANSFORM = 3  # stochastic transforms (e.g. Poisson spike sampling)


def split_key(split: str | None) -> int:
    """Map a split name to a stable integer (0 when no split is given).

    Parameters
    ----------
    split : str or None
        Split name such as ``"train"`` or ``"test"``.

    Returns
    -------
    int
        CRC32 of the split name; stable across processes and Python sessions
        (unlike ``hash``).
    """
    if split is None:
        return 0
    return zlib.crc32(split.encode("utf-8"))


def trial_seed_sequence(
    seed: int, split: str | None, idx: int, stream: Stream, *sub: int
) -> np.random.SeedSequence:
    """Return the seed sequence of one random stream of one dataset item.

    Parameters
    ----------
    seed : int
        Master seed.
    split : str or None
        Split name; different splits yield independent trials.
    idx : int
        Item index within the split.
    stream : Stream
        Which random factor the sequence drives.
    *sub : int
        Optional finer keys, e.g. the element position within the trial.

    Returns
    -------
    numpy.random.SeedSequence
        Seed sequence keyed by ``(split, idx, stream, *sub)`` under ``seed``.
    """
    key = (split_key(split), int(idx), int(stream), *(int(s) for s in sub))
    return np.random.SeedSequence(entropy=int(seed), spawn_key=key)


def trial_rng(
    seed: int, split: str | None, idx: int, stream: Stream, *sub: int
) -> np.random.Generator:
    """Return a NumPy generator for one random stream of one dataset item.

    See :func:`trial_seed_sequence` for the parameters.
    """
    return np.random.default_rng(trial_seed_sequence(seed, split, idx, stream, *sub))


def torch_generator(seed_seq: np.random.SeedSequence) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded from a NumPy seed sequence.

    Parameters
    ----------
    seed_seq : numpy.random.SeedSequence
        Source of entropy.

    Returns
    -------
    torch.Generator
        CPU generator seeded with 63 bits derived from ``seed_seq``.
    """
    state = int(seed_seq.generate_state(1, np.uint64)[0]) >> 1
    generator = torch.Generator()
    generator.manual_seed(state)
    return generator
