# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
TaskTargetBuilder — the single source of truth that turns a generated sample
into the model's training target, driven entirely by ``seqbench.task``.

It replaces the old ``SeqDataset.create_target_for_gensample`` special-case.
Two task sources are supported, resolving to the same class-id target space:

- ``source: seqbench`` — a registered :mod:`seqbench.tasks` Task runs on the
  EOS-terminated ``class_seq``/``state_seq``. Values are already class ids.
- ``source: symseq`` — the referenced :mod:`symseq.tasks` Task runs on the
  Trial's symbols (no EOS); the result is encoded to class ids via
  :class:`~seqbench.seq_utils.symbol_encoder.SymbolEncoder` and one masked slot
  is appended for the EOS position.

EOS-alignment invariant: a ``per_token`` target always has length
``len(class_seq)``, with the EOS position masked unless a seqbench task assigns
it (e.g. ``NStepPrediction`` predicts EOS at the last real token). Masked
positions become ``pad_index`` in the emitted numpy target.

The builder is invoked per-trial at draw time by ``SequenceGenerator`` (storing
the Target on ``GeneratorSample.targets[task_id]``); ``SeqDataset`` reads it
back via :meth:`to_target_seq` (which also recomputes seqbench-source targets
on the fly when a sample lacks them — e.g. legacy on-disk datasets).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from seqbench.config import TaskSource
from seqbench.seq_utils.generator import GeneratorSample
from seqbench.seq_utils.symbol_encoder import SymbolEncoder
from seqbench.tasks import registry as seqbench_registry
from seqbench.tasks.base import Target
from seqbench.tasks.classify import StateClassification


@dataclass
class ResolvedTarget:
    """A task target materialized into the class-id space.

    ``target_seq`` is a per-position int array (per_token, masked positions set
    to ``pad_index``) or a 0-d array holding the single label (per_trial).
    """

    target_seq: np.ndarray
    mask: np.ndarray | None
    granularity: Literal["per_token", "per_trial"]


class TaskTargetBuilder:
    def __init__(
        self,
        *,
        source: TaskSource,
        task_id: str,
        task: Any,
        encoder: SymbolEncoder,
        pad_index: int = -1,
        granularity: Literal["per_token", "per_trial"] = "per_token",
        label_space: str | None = None,
    ):
        self.source = source
        self.task_id = task_id
        self.task = task
        self.encoder = encoder
        self.pad_index = pad_index
        self._granularity = granularity
        # symseq-source tasks (and any task without the attribute) target the
        # encoded class-id space; seqbench tasks declare their own label_space.
        self.label_space = label_space or getattr(task, "label_space", "class_id")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_run_cfg(
        cls,
        run_cfg,
        *,
        encoder: SymbolEncoder,
        pad_index: int = -1,
        state_id_fn: Callable[[str], int] | None = None,
    ) -> TaskTargetBuilder:
        task_cfg = run_cfg.seqbench.task
        source = task_cfg.source

        if source == TaskSource.SEQBENCH:
            task_id = task_cfg.id
            task = seqbench_registry.build(task_cfg.type, **task_cfg.params)
            if isinstance(task, StateClassification):
                if state_id_fn is None:
                    raise ValueError(
                        "StateClassification requires a grammar source providing "
                        "a state_id_fn (RestrictedTargetProbGenerator.unred_state_to_id)."
                    )
                task.state_id_fn = state_id_fn
            granularity = task.granularity
        else:  # SYMSEQ
            if run_cfg.symseq is None:
                raise ValueError("seqbench.task.source='symseq' requires a symseq section")
            entry = next(
                (t for t in run_cfg.symseq.tasks if t.id == task_cfg.ref_id), None
            )
            if entry is None:
                raise ValueError(
                    f"seqbench.task.ref_id={task_cfg.ref_id!r} does not match any "
                    "symseq.tasks[*].id"
                )
            from symseq.tasks import registry as symseq_registry

            task_id = entry.id
            task = symseq_registry.build(entry.type, **entry.params)
            granularity = task.granularity

        if (
            getattr(task, "needs_base_dataset", False)
            and run_cfg.seqbench.composition.combine_sequences
        ):
            raise ValueError(
                "Classification(label_source='base') is incompatible with "
                "combine_sequences=true: base-dataset labels cannot be resolved "
                "per-trial after concatenation. Use label_source='first' or 'last'."
            )

        return cls(
            source=source,
            task_id=task_id,
            task=task,
            encoder=encoder,
            pad_index=pad_index,
            granularity=granularity,
        )

    @property
    def granularity(self) -> Literal["per_token", "per_trial"]:
        return self._granularity

    def num_classes(self, *, prob_generator=None, base_dataset=None) -> int:
        """Number of distinct classes in the task's target label space.

        Resolves the task's declared ``label_space`` against the runtime context:
        ``class_id`` -> encoder vocab size (alphabet + EOS, grammar-independent),
        ``unreduced_state`` -> grammar's unreduced-state count, ``base_label`` ->
        base dataset's class count.
        """
        if self.label_space == "class_id":
            return len(self.encoder)
        if self.label_space == "unreduced_state":
            if prob_generator is None or prob_generator.num_unreduced_states == 0:
                raise ValueError(
                    "StateClassification num_classes requires a grammar source "
                    "with transitions."
                )
            return prob_generator.num_unreduced_states
        if self.label_space == "base_label":
            if base_dataset is None:
                raise ValueError("base-label num_classes requires a base_dataset.")
            return len(base_dataset.class_dict)
        raise ValueError(f"unknown label_space {self.label_space!r}")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, trial, class_seq, state_seq) -> Target | None:
        """Resolve the configured task for one trial (used at draw time).

        Returns a class-id-space :class:`Target` aligned to the EOS-terminated
        ``class_seq`` (length ``len(class_seq)`` for per_token), or ``None``
        when the task is deferred (``task.needs_base_dataset`` is True) —
        signalling that resolution must happen later in
        :meth:`~seqbench.seq_dataset.SeqDataset.gensample_to_sample`.
        """
        if self.source == TaskSource.SEQBENCH:
            if getattr(self.task, "needs_base_dataset", False):
                return None
            return self._run_seqbench_task(
                GeneratorSample(class_seq, state_seq, len(class_seq))
            )
        target = trial.targets.get(self.task_id) if trial.targets else None
        if target is None:
            target = self.task(trial)
        return self._encode_symseq_target(target, target_len=len(class_seq))

    def to_target_seq(self, gensample: GeneratorSample) -> ResolvedTarget:
        """Materialize the target for a sample (used at sample-build time).

        Prefers the Target stored on the sample (resolved at draw time or
        deserialized from disk); for a seqbench-source task it recomputes from
        the sample when absent (e.g. legacy 3-field datasets).
        """
        target = None
        if gensample.targets and self.task_id in gensample.targets:
            target = gensample.targets[self.task_id]
        elif self.source == TaskSource.SEQBENCH:
            target = self._run_seqbench_task(gensample)
        else:
            raise RuntimeError(
                f"Target {self.task_id!r} is missing from the sample and cannot "
                "be recomputed for a symseq-source task. Regenerate the dataset "
                "(seqbench.storage.force_rebuild: true) or use an online mode."
            )
        return self._to_resolved(target, len(gensample.class_seq))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_seqbench_task(self, gensample: GeneratorSample) -> Target:
        if isinstance(self.task, StateClassification) and self.task.state_id_fn is None:
            raise ValueError("StateClassification.state_id_fn was not injected.")
        return self.task(gensample)

    def _encode_symseq_target(self, t, target_len: int) -> Target:
        if t.granularity == "per_trial":
            return Target(
                values=_scalarize(t.values), mask=None, granularity="per_trial"
            )

        values = list(t.values)
        mask = list(t.mask) if t.mask is not None else [True] * len(values)
        encoded: list = []
        for v, m in zip(values, mask, strict=True):
            if not m or v is None:
                encoded.append(None)
            elif isinstance(v, str):
                encoded.append(self.encoder.encode([v])[0])
            else:
                raise NotImplementedError(
                    f"symseq task produced a non-symbol target value {v!r}; "
                    "tuple/non-scalar targets (e.g. NGramChunk) are not supported "
                    "in the class-id training pipeline."
                )

        pad = target_len - len(encoded)
        if pad < 0:
            raise ValueError(
                f"symseq task target ({len(encoded)}) is longer than the "
                f"EOS-terminated class_seq ({target_len})."
            )
        encoded += [None] * pad  # EOS position(s)
        out_mask = [v is not None for v in encoded]
        return Target(values=encoded, mask=out_mask, granularity="per_token")

    def _to_resolved(self, target: Target, length: int) -> ResolvedTarget:
        if target.granularity == "per_trial":
            return ResolvedTarget(
                target_seq=np.asarray(target.values),
                mask=None,
                granularity="per_trial",
            )
        values = list(target.values)
        mask = (
            list(target.mask)
            if target.mask is not None
            else [v is not None for v in values]
        )
        out = np.full(len(values), self.pad_index, dtype=int)
        for i, (v, m) in enumerate(zip(values, mask, strict=True)):
            if m and v is not None:
                out[i] = int(v)
        return ResolvedTarget(
            target_seq=out,
            mask=np.asarray(mask, dtype=bool),
            granularity="per_token",
        )


def make_task_target_builder(run_cfg, source, *, pad_index: int = -1, prob_generator=None):
    """Build a :class:`TaskTargetBuilder` from a run config and a live source.

    Shared by :class:`~seqbench.seq_dataset.SeqDataset` and the standalone
    ``create_dataset`` script so both wire the task identically. ``prob_generator``
    (a ``RestrictedTargetProbGenerator`` with transitions loaded) supplies the
    ``state_id_fn`` required by ``StateClassification``.
    """
    state_id_fn = prob_generator.unred_state_to_id if prob_generator is not None else None
    return TaskTargetBuilder.from_run_cfg(
        run_cfg,
        encoder=SymbolEncoder(source.alphabet),
        pad_index=pad_index,
        state_id_fn=state_id_fn,
    )


def _scalarize(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
