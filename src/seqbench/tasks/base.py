# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Task ABC + Target — SeqBench's task layer, mirroring ``symseq.tasks.base``.

A SeqBench Task derives a Target from a ``GeneratorSample`` (the concatenated,
encoded sample produced by ``seqbench.seq_utils.generator.GeneratorSample``).
The symbolic, per-token logic operates on ``sample.class_seq`` (numeric class
ids); embedding-derived tasks may also reach into the encoded tensors.

The Target dataclass intentionally matches ``symseq.tasks.base.Target`` in
shape so that downstream code can treat both registries uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from seqbench.seq_utils.generator import GeneratorSample


@dataclass
class Target:
    values: list[Any] | Any
    mask: list[bool] | None
    granularity: Literal["per_token", "per_trial"]


class Task(ABC):
    """Anonymous task behavior; configured IDs own target identity."""

    # Output structure of the produced Target, known without running the task
    # (consumed by TaskTargetBuilder / SeqDataset to choose the output path).
    granularity: Literal["per_token", "per_trial"]
    # The label space the task's targets live in, used to compute num_classes:
    #   "class_id"        -> len(encoder) (alphabet + EOS)
    #   "unreduced_state" -> prob_generator.num_unreduced_states (grammar)
    #   "base_label"      -> len(base_dataset.class_dict)
    label_space: Literal["class_id", "unreduced_state", "base_label"] = "class_id"

    @abstractmethod
    def __call__(self, sample: GeneratorSample) -> Target: ...
