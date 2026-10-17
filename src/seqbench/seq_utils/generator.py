# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Sequence generator utilities for creating synthetic sequence data.

This module is the boundary between SeqBench and any external trial source
(notably symseq generators). A ``source`` here is duck-typed against the
``TrialSource`` Protocol: it must expose ``alphabet`` and ``draw_trial()``.

Token → class-index conversion is handled by :class:`SymbolEncoder` rather
than the old hardcoded ``ord(s) - 64`` arithmetic, so any alphabet works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from seqbench.seq_utils.symbol_encoder import SymbolEncoder
from seqbench.tasks.base import Target

logger = logging.getLogger("generator")


def determine_decimal_digits(x):
    s = str(x)
    if not "." in s:
        return 0
    return len(s) - s.index(".") - 1


def rotate(l, n):
    return l[n:] + l[:n]


@dataclass
class GeneratorSample:
    class_seq: list[int]
    state_seq: list[str]
    length: int
    # Targets forwarded from the originating symseq Trial (intrinsic, e.g.
    # nback_match) and/or resolved by the configured task. Each per_token
    # Target is aligned to the EOS-terminated class_seq (the EOS slot is
    # masked). None when the source emits no targets. See TaskTargetBuilder.
    targets: dict[str, Target] | None = None
    # Free-form per-trial metadata from the originating Trial. Dropped when
    # sequences are combined (per-trial meta is ill-defined for a combination).
    meta: dict | None = None
    # Native base-dataset label, populated by SeqDataset for
    # Classification(label_source="base") tasks. None otherwise.
    label: int | None = None


class SequenceGenerator:

    def __init__(
        self,
        source,
        *,
        combine_sequences: bool,
        combined_seq_len: int,
        seed=None,
        compute_te=False,
        plot_transition_table=False,
        trial_params: dict | None = None,
        task_builder=None,
    ):
        self.task_builder = task_builder
        self.seed = seed
        self.compute_te = compute_te
        self.plot_transition_table = plot_transition_table
        self.num_illustration_seq = 4

        self.combine_sequences = combine_sequences
        self.combined_seq_len = combined_seq_len

        logger.info(f"Task RNG seed {self.seed}")

        self.source = source
        self._trial_params = trial_params or {}
        self.encoder = SymbolEncoder(source.alphabet)

        # Topological entropy — only available for grammar sources.
        if self.compute_te:
            if hasattr(self.source, "transition_table") and hasattr(
                self.source, "topological_entropy"
            ):
                transition_table = (
                    self.source.transition_table(correct=True, display=False) > 0
                ).astype(int)
                TE = self.source.topological_entropy(
                    transitions=transition_table, method="direct"
                )
                try:
                    import wandb

                    wandb.log({"TE": TE})
                except:
                    pass

                print(f"TE:\t{TE}")
                print("####")
            else:
                logger.info(
                    "compute_te requested but source has no transition_table; skipping."
                )

        # @BZ: can we remove this here and use symseq's plotting?
        # Transition-table plot — only available for grammar sources.
        if self.plot_transition_table:
            if hasattr(self.source, "transition_table") and hasattr(
                self.source, "states"
            ):
                import matplotlib.pyplot as plt
                from seqbench.seq_utils.markov_chain import MarkovChain

                P = self.source.transition_table(correct=False, display=True).T
                mc = MarkovChain(
                    P,
                    self.source.states,
                    node_fontsize=10,
                    node_radius=1.0,
                    fontsize=10,
                )

                for _ in range(self.num_illustration_seq):
                    x = self.generate_sequence()
                    seqs = x.state_seq
                    print(seqs)

                start_states = (
                    [str(item) for item in self.source.start_states]
                    if hasattr(self.source, "start_states")
                    else []
                )
                try:
                    mc.draw(
                        title=f"start states: {start_states}, TE: {round(TE, 2)}",
                        figsize=(5, 5),
                    )
                except:
                    mc.draw(title=f"start states: {start_states}", figsize=(5, 5))

                fname = "grammar"
                path = "."
                print(f"Save {path}/{fname}.pdf and {path}/{fname}.png")
                plt.savefig(f"{path}/{fname}.pdf")
                plt.savefig(f"{path}/{fname}.png", dpi=300)

                plt.close()
            else:
                logger.info(
                    "plot_transition_table requested but source has no transition_table; skipping."
                )

    def generate(self, idx, compute_length=True):

        # Per-sample seeding. symseq generators expose a writable .rng;
        # other sources may choose to ignore this.
        if self.seed is not None:
            self.source.rng = np.random.default_rng(self.seed * idx)
        else:
            self.source.rng = np.random.default_rng()

        if self.combine_sequences:
            gen_sample = self.generate_sequences()
        else:
            gen_sample = self.generate_sequence()

        if compute_length:
            gen_sample.length = self.__compute_sequence_length(gen_sample.class_seq)

        return gen_sample

    def __compute_sequence_length(self, seq):
        return len(seq)

    def generate_sequences(self):

        comb_sample = GeneratorSample(np.array([]), np.array([]), 0)

        seq_len = 0

        while seq_len < self.combined_seq_len:
            sample = self.generate_sequence()
            comb_sample = self.__concat_samples(comb_sample, sample)
            seq_len = comb_sample.class_seq.shape[0]

        n = self.combined_seq_len
        comb_sample.class_seq = comb_sample.class_seq[:n]
        comb_sample.state_seq = comb_sample.state_seq[:n]
        if comb_sample.targets:
            for t in comb_sample.targets.values():
                t.values = t.values[:n]
                t.mask = t.mask[:n]

        return comb_sample

    def __concat_samples(self, comb_sample, sample):
        comb_sample.class_seq = np.concatenate(
            (comb_sample.class_seq, sample.class_seq)
        )
        comb_sample.state_seq = np.concatenate(
            (comb_sample.state_seq, sample.state_seq)
        )

        comb_sample.class_seq = comb_sample.class_seq.astype(int)
        comb_sample.targets = self.__concat_targets(
            comb_sample.targets, sample.targets, len(sample.class_seq)
        )

        return comb_sample

    @staticmethod
    def __concat_targets(acc, new, new_seq_len: int):
        """Concatenate target values/masks across combined trials.

        ``per_token`` targets are concatenated directly.

        ``per_trial`` targets (scalar label per trial) are **spread** into a
        per_token target: the label is placed at the last position of the trial
        and all earlier positions are masked.  This preserves supervision for
        per_trial tasks (e.g. Classification) under combine_sequences without
        discarding any information.
        """
        if not new:
            return acc
        for name, t in new.items():
            if t.granularity == "per_trial":
                n = new_seq_len
                t = Target(
                    values=[None] * (n - 1) + [t.values],
                    mask=[False] * (n - 1) + [True],
                    granularity="per_token",
                )
            if acc is None:
                acc = {}
            if name not in acc:
                acc[name] = Target(values=[], mask=[], granularity="per_token")
            acc[name].values = list(acc[name].values) + list(t.values)
            acc[name].mask = list(acc[name].mask) + list(t.mask)
        return acc

    def generate_sequence(self):
        trial = self.source.draw_trial(**self._trial_params)
        symbols = list(trial.symbols)
        states = list(trial.states) if trial.states is not None else list(symbols)

        # Append EOS marker to both views; encoder maps '#' -> 0.
        state_seq = states + [SymbolEncoder.EOS_SYMBOL]
        state_seq = [sym.replace("(", "").replace(")", "") for sym in state_seq]

        class_indices = self.encoder.encode(symbols) + [SymbolEncoder.EOS_INDEX]

        assert len(class_indices) == len(state_seq)

        class_indices = np.array(class_indices)
        state_seq = np.array(state_seq)

        targets = (
            self.__forward_trial_targets(trial, target_len=len(class_indices))
            if trial is not None
            else None
        )

        # Resolve the configured training target (if a builder is wired in) and
        # store it alongside other configured targets, keyed by the task id.
        # resolve() returns None for deferred tasks (e.g. label_source="base")
        # that need base-dataset context only available in SeqDataset.
        if self.task_builder is not None and trial is not None:
            resolved = self.task_builder.resolve(trial, class_indices, state_seq)
            if resolved is not None:
                if targets is None:
                    targets = {}
                targets[self.task_builder.task_id] = resolved

        meta = dict(trial.meta) if trial is not None and trial.meta else None

        return GeneratorSample(class_indices, state_seq, 0, targets=targets, meta=meta)

    @staticmethod
    def __forward_trial_targets(trial, target_len):
        """Convert ``trial.targets`` into class_seq-aligned :class:`Target`s.

        Each ``per_token`` target produced by a symseq generator has length
        ``len(symbols) == target_len - 1`` (no EOS). We append one masked slot
        so every stored target lines up 1:1 with the EOS-terminated class_seq,
        which makes concatenation/truncation under ``combine_sequences`` trivial.
        ``per_trial`` targets are kept scalar.
        """
        if not trial.targets:
            return None

        out: dict[str, Target] = {}
        for name, t in trial.targets.items():
            if t.granularity == "per_token":
                values = list(t.values)
                mask = list(t.mask) if t.mask is not None else [True] * len(values)
                pad = target_len - len(values)
                if pad < 0:
                    raise ValueError(
                        f"per_token target {name!r} is longer ({len(values)}) than "
                        f"the EOS-terminated class_seq ({target_len})."
                    )
                values = values + [None] * pad  # EOS position(s)
                mask = mask + [False] * pad
                out[name] = Target(
                    values=values, mask=mask, granularity="per_token"
                )
            else:
                out[name] = Target(
                    values=t.values, mask=None, granularity="per_trial"
                )
        return out or None
