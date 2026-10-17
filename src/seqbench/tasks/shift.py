# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Shift-based tasks: predict the class-id n steps back (memory) or n steps
forward (prediction) within a ``GeneratorSample.class_seq``.

Semantics mirror ``symseq.tasks.shift`` exactly so that ``source: symseq``
and ``source: seqbench`` with the same task type+params produce equivalent
training targets (modulo the symbols-vs-class_ids representation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from seqbench.tasks.base import Target, Task
from seqbench.tasks.registry import register

if TYPE_CHECKING:
    from seqbench.seq_utils.generator import GeneratorSample


@register("NStepMemory")
class NStepMemory(Task):
    """Per-token target: at position i, predict ``class_seq[i-n]``.

    Positions ``[0..n-1]`` are masked.
    """

    granularity = "per_token"
    label_space = "class_id"

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NStepMemory.n must be a positive int, got {n!r}")
        self.n = n

    def __call__(self, sample: GeneratorSample) -> Target:
        seq = list(sample.class_seq)
        L = len(seq)
        if self.n >= L:
            return Target(values=[None] * L, mask=[False] * L, granularity="per_token")
        values = [None] * self.n + seq[: L - self.n]
        mask = [False] * self.n + [True] * (L - self.n)
        return Target(values=values, mask=mask, granularity="per_token")


@register("NStepPrediction")
class NStepPrediction(Task):
    """Per-token target: at position i, predict ``class_seq[i+n]``.

    Last ``n`` positions are masked.
    """

    granularity = "per_token"
    label_space = "class_id"

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"NStepPrediction.n must be a positive int, got {n!r}")
        self.n = n

    def __call__(self, sample: GeneratorSample) -> Target:
        seq = list(sample.class_seq)
        L = len(seq)
        if self.n >= L:
            return Target(values=[None] * L, mask=[False] * L, granularity="per_token")
        values = seq[self.n:] + [None] * self.n
        mask = [True] * (L - self.n) + [False] * self.n
        return Target(values=values, mask=mask, granularity="per_token")
