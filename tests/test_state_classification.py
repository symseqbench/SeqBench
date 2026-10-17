# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Tests for the StateClassification task."""

import numpy as np
import pytest

from seqbench.seq_utils.generator import GeneratorSample
from seqbench.tasks import registry
from seqbench.tasks.classify import StateClassification


def _gs(state_seq):
    n = len(state_seq)
    return GeneratorSample(np.arange(n), np.array(state_seq, dtype=object), n)


def test_state_classification_maps_state_ids():
    id_map = {"a": 3, "b": 1, "c": 2, "#": 0}
    task = StateClassification()
    task.state_id_fn = lambda s: id_map[s]
    tgt = task(_gs(["a", "b", "c", "#"]))
    assert tgt.granularity == "per_token"
    assert tgt.values == [3, 1, 2, 0]
    assert tgt.mask == [True, True, True, True]


def test_state_classification_requires_state_id_fn():
    task = StateClassification()
    with pytest.raises(ValueError, match="state_id_fn"):
        task(_gs(["a", "#"]))


def test_registry_builds_state_classification():
    task = registry.build("StateClassification")
    assert isinstance(task, StateClassification)
    assert "StateClassification" in registry.registered_types()
