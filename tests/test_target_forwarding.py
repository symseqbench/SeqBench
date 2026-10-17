# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Phase 1: ``SequenceGenerator`` forwards a Trial's configured targets onto the
``GeneratorSample``, aligned to the EOS-terminated class_seq, and concatenates/
truncates them under ``combine_sequences``. per_trial targets are rejected when
combining.
"""

import numpy as np
import pytest

try:
    import symseq  # noqa: F401
    from symseq.generators.nback import NBack
    from symseq.trial import Trial, Target as SymseqTarget
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench.seq_utils.generator import SequenceGenerator

pytestmark = pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")


def _nback_gen(combine, combined_seq_len=20):
    from symseq.tasks import ConfiguredTrialSource, build_tasks

    source = ConfiguredTrialSource(
        NBack(n=2, seq_length=8, alphabet_size=5, seed=1),
        build_tasks(
            [
                {"id": "nback_match", "type": "NBackMatch"},
                {"id": "nback_role", "type": "NBackRole"},
            ]
        ),
    )
    return SequenceGenerator(
        source,
        combine_sequences=combine,
        combined_seq_len=combined_seq_len,
        seed=7,
    )


class _PerTrialSource:
    """Minimal TrialSource emitting a per_trial target (for the combine guard)."""

    alphabet = ["a", "b"]

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def draw_trial(self, **kwargs):
        return Trial(
            symbols=["a", "b"],
            states=None,
            targets={
                "lbl": SymseqTarget(
                    values=1, mask=None, granularity="per_trial"
                )
            },
            meta={},
        )


def test_forwarded_targets_aligned_to_class_seq():
    gs = _nback_gen(combine=False).generate(idx=1, compute_length=True)

    assert gs.targets is not None
    assert set(gs.targets) == {"nback_match", "nback_role"}
    L = len(gs.class_seq)
    for name, t in gs.targets.items():
        assert t.granularity == "per_token"
        assert len(t.values) == L
        assert len(t.mask) == L
        # the appended EOS slot is masked out
        assert t.mask[-1] is False
        assert t.values[-1] is None


def test_forwarded_meta_present():
    gs = _nback_gen(combine=False).generate(idx=1, compute_length=True)
    assert gs.meta is not None
    assert gs.meta.get("paradigm") is not None


def test_combine_concatenates_and_truncates_targets():
    n = 20
    gs = _nback_gen(combine=True, combined_seq_len=n).generate(idx=1, compute_length=True)

    assert gs.class_seq.shape[0] == n
    assert gs.targets is not None
    for t in gs.targets.values():
        assert len(t.values) == n
        assert len(t.mask) == n
    # meta is dropped for combined samples
    assert gs.meta is None


def test_per_trial_target_spread_under_combine():
    # Under combine_sequences a per_trial label is spread into a per_token
    # target: the label sits at each trial's last position, earlier positions
    # masked. No supervision is discarded.
    gen = SequenceGenerator(
        _PerTrialSource(),
        combine_sequences=True,
        combined_seq_len=10,
        seed=1,
    )
    gs = gen.generate(idx=1, compute_length=True)
    assert gs.class_seq.shape[0] == 10
    assert gs.targets is not None and "lbl" in gs.targets
    t = gs.targets["lbl"]
    assert t.granularity == "per_token"
    assert len(t.values) == 10 and len(t.mask) == 10
    assert any(t.mask)  # at least one trial boundary carries the label
    for v, m in zip(t.values, t.mask):
        assert v == 1 if m else v is None


def test_per_trial_target_forwarded_without_combine():
    gen = SequenceGenerator(
        _PerTrialSource(),
        combine_sequences=False,
        combined_seq_len=10,
        seed=1,
    )
    gs = gen.generate(idx=1, compute_length=True)
    assert gs.targets is not None
    assert gs.targets["lbl"].granularity == "per_trial"
    assert gs.targets["lbl"].values == 1
