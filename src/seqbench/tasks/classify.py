# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Classification tasks: a single label per sample (per_trial) or one label
per token position (per_token).

Covers:
 - seqbench-standalone tasks where the base dataset carries its own labels
   (e.g. GSC keyword spotting) — ``label_source="base"``.
 - probes over the symbolic class_seq — ``label_source="first"|"last"|"current"``.
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from seqbench.tasks.base import Target, Task
from seqbench.tasks.registry import register

if TYPE_CHECKING:
    from seqbench.seq_utils.generator import GeneratorSample


LabelSource = Literal["base", "first", "last", "current"]
Granularity = Literal["per_trial", "per_token"]


@register("Classification")
class Classification(Task):
    """Per-trial or per-token classification.

    Parameters
    ----------
    label_source : {"base", "first", "last", "current"}
        How to determine the class label(s).

        - ``"base"``: native base-dataset label (``sample.label``), populated
          by :meth:`~seqbench.seq_dataset.SeqDataset.gensample_to_sample`.
          Use for seqbench-standalone tasks where the base dataset carries its
          own labels (e.g. GSC keyword spotting).
        - ``"first"``: ``class_seq[0]``.
        - ``"last"``:  ``class_seq[-1]``.
        - ``"current"``: ``class_seq[i]`` at each token position *i*
          (only valid with ``granularity="per_token"``).

    granularity : {"per_trial", "per_token"}
        Output granularity.

        - ``"per_trial"`` (default): one scalar label for the whole sequence.
        - ``"per_token"``: one label per token position. For ``"current"``
          the label at position *i* is ``class_seq[i]``; for ``"base"``,
          ``"first"``, and ``"last"`` the scalar label is broadcast across
          all non-EOS positions.  The EOS position is always masked.
    """

    # granularity and label_space are set dynamically in __init__
    granularity: str = "per_trial"
    label_space: str = "class_id"

    def __init__(
        self,
        label_source: LabelSource = "base",
        granularity: Granularity = "per_trial",
    ):
        if label_source not in ("base", "first", "last", "current"):
            raise ValueError(
                f"label_source must be one of 'base'|'first'|'last'|'current', "
                f"got {label_source!r}"
            )
        if granularity not in ("per_trial", "per_token"):
            raise ValueError(
                "granularity must be 'per_trial' or 'per_token', "
                f"got {granularity!r}"
            )
        if label_source == "current" and granularity == "per_trial":
            raise ValueError(
                "label_source='current' requires granularity='per_token' "
                "(a single 'current' label is undefined at trial level)."
            )
        self.label_source = label_source
        self.granularity = granularity  # overrides the class attribute
        # "base" labels live in the base dataset's class space; first/last/current
        # are symbolic class ids.
        self.label_space = "base_label" if label_source == "base" else "class_id"

    @property
    def needs_base_dataset(self) -> bool:
        """True when the target requires base_dataset context (cannot resolve at draw time)."""
        return self.label_source == "base"

    def __call__(self, sample: GeneratorSample) -> Target:
        seq = list(sample.class_seq)  # EOS-terminated: [..., EOS_INDEX]

        # --- "current" is only per_token ---
        if self.label_source == "current":
            mask = [True] * (len(seq) - 1) + [False]  # mask EOS slot
            return Target(values=seq, mask=mask, granularity="per_token")

        # --- scalar label for "base" / "first" / "last" ---
        if self.label_source == "base":
            label = getattr(sample, "label", None)
            if label is None:
                raise ValueError(
                    "Classification(label_source='base') requires sample.label "
                    "to be set by the input_mapping (base dataset label)."
                )
        elif not seq:
            label = None
        elif self.label_source == "first":
            label = seq[0]
        else:  # "last"
            label = seq[-1]

        if self.granularity == "per_trial":
            return Target(values=label, mask=None, granularity="per_trial")

        # per_token: broadcast scalar label across all non-EOS positions
        n = len(seq)
        mask = [True] * (n - 1) + [False]  # EOS position masked
        return Target(values=[label] * n, mask=mask, granularity="per_token")


@register("StateClassification")
class StateClassification(Task):
    """Per-token target: the unreduced-state id at each position.

    Each position's target is ``state_id_fn(state_seq[i])``, covering the full
    sequence including the EOS ``'#'`` slot (which maps to 0). Requires a
    grammar source; ``state_id_fn`` (typically
    ``RestrictedTargetProbGenerator.unred_state_to_id``) is injected by
    :class:`~seqbench.tasks.target_builder.TaskTargetBuilder`.
    """

    granularity = "per_token"
    label_space = "unreduced_state"

    def __init__(self):
        self.state_id_fn = None

    def __call__(self, sample: GeneratorSample) -> Target:
        if self.state_id_fn is None:
            raise ValueError(
                "StateClassification requires a state_id_fn (grammar source). "
                "It is injected by TaskTargetBuilder; ensure the run uses a "
                "grammar generator with transitions."
            )
        values = [self.state_id_fn(s) for s in sample.state_seq]
        return Target(
            values=values, mask=[True] * len(values), granularity="per_token"
        )
