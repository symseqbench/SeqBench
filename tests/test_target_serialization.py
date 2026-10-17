# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Phase 2: GeneratorSample targets survive the file-mode serialization round-trip.

The on-disk format gains a 4th ``::targets`` (JSON) field; legacy 3-field lines
parse back with ``targets=None``.
"""

import io

import numpy as np
import pytest

from seqbench.dataset_generator import DatasetGenerator
from seqbench.seq_utils.generator import GeneratorSample
from seqbench.tasks.base import Target


def _roundtrip(gs):
    buf = io.StringIO()
    DatasetGenerator.write_gensample_to_file(buf, gs)
    line = buf.getvalue().rstrip("\n")
    return DatasetGenerator.create_gensample_from_str(line)


def test_roundtrip_no_targets():
    gs = GeneratorSample(np.array([1, 2, 0]), np.array(["a", "b", "#"], dtype=object), 3)
    out = _roundtrip(gs)
    assert out.class_seq == [1, 2, 0]
    assert out.state_seq == ["a", "b", "#"]
    assert out.targets is None


def test_roundtrip_per_token_targets_with_none_and_mask():
    gs = GeneratorSample(
        np.array([1, 2, 3, 0]),
        np.array(["a", "b", "c", "#"], dtype=object),
        4,
        targets={
            "next_token": Target(
                values=[2, 3, 0, None],
                mask=[True, True, True, False],
                granularity="per_token",
            )
        },
    )
    out = _roundtrip(gs)
    t = out.targets["next_token"]
    assert t.granularity == "per_token"
    assert t.values == [2, 3, 0, None]
    assert t.mask == [True, True, True, False]


def test_roundtrip_numpy_values():
    gs = GeneratorSample(
        np.array([1, 0]),
        np.array(["a", "#"], dtype=object),
        2,
        targets={
            "t": Target(
                values=list(np.array([5, 0])),
                mask=[True, True],
                granularity="per_token",
            )
        },
    )
    out = _roundtrip(gs)
    assert out.targets["t"].values == [5, 0]


def test_legacy_three_field_line_parses_with_no_targets():
    legacy = "[1, 2, 0]::['a', 'b', '#']::3"
    out = DatasetGenerator.create_gensample_from_str(legacy)
    assert out.class_seq == [1, 2, 0]
    assert out.targets is None


def test_parse_rejects_malformed_field_count():
    with pytest.raises(ValueError, match="3 or 4"):
        DatasetGenerator.create_gensample_from_str("[1, 2]::['a', 'b']")


def test_parse_rejects_mismatched_class_and_state_lengths():
    line = "[1, 2, 0]::['a', 'b']::3"
    with pytest.raises(ValueError, match="same length"):
        DatasetGenerator.create_gensample_from_str(line)


def test_parse_rejects_malformed_targets():
    line = "[1, 0]::['a', '#']::2::{bad json"
    with pytest.raises(ValueError, match="targets field"):
        DatasetGenerator.create_gensample_from_str(line)


def test_parse_rejects_legacy_target_vocabulary():
    legacy_targets = '{"t":{"values":[2,0],"mask":[true,true],"kind":"per_token"}}'
    line = f"[1, 0]::['a', '#']::2::{legacy_targets}"
    with pytest.raises(ValueError, match="targets field"):
        DatasetGenerator.create_gensample_from_str(line)


def test_write_rejects_mismatched_class_and_state_lengths():
    gs = GeneratorSample(np.array([1, 2, 0]), np.array(["a", "b"], dtype=object), 3)
    with pytest.raises(ValueError, match="same length"):
        DatasetGenerator.write_gensample_to_file(io.StringIO(), gs)


def test_roundtrip_configured_nback_targets():
    pytest.importorskip("symseq")
    from symseq.generators.nback import NBack
    from symseq.tasks import ConfiguredTrialSource, build_tasks
    from seqbench.seq_utils.generator import SequenceGenerator

    gen = SequenceGenerator(
        ConfiguredTrialSource(
            NBack(n=2, seq_length=8, alphabet_size=5, seed=1),
            build_tasks([{"id": "match", "type": "NBackMatch"}]),
        ),
        combine_sequences=False,
        combined_seq_len=20,
        seed=3,
    )
    gs = gen.generate(idx=1, compute_length=True)
    assert gs.targets is not None
    out = _roundtrip(gs)
    assert set(out.targets) == set(gs.targets)
    for name in gs.targets:
        assert out.targets[name].values == gs.targets[name].values
        assert out.targets[name].mask == gs.targets[name].mask
        assert out.targets[name].granularity == gs.targets[name].granularity
