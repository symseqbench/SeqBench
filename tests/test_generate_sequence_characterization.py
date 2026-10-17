# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Characterization tests pinning the CURRENT behavior of
``SequenceGenerator.generate_sequence`` / ``generate_sequences`` BEFORE the
tasks-module migration:

- EOS is appended (class-id 0, symbol '#'); class_seq and state_seq stay equal
  length; parentheses are stripped from state symbols.
- combine_sequences truncates the combined sample to ``combined_seq_len``.

Phase 1 extends these to assert ``GeneratorSample.targets`` is forwarded.
"""

import os

import numpy as np
import pytest

try:
    from seqbench.sources import build_symseq_source
    import symseq  # noqa: F401
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import config as cfg_mod
from seqbench.seq_utils.generator import SequenceGenerator
from seqbench.seq_utils.symbol_encoder import SymbolEncoder

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "onehot_raw.yaml")

pytestmark = pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")


def _source():
    run_cfg = cfg_mod.load(CONFIG_PATH)
    return build_symseq_source(run_cfg), run_cfg


def test_generate_sequence_appends_eos():
    source, run_cfg = _source()
    gen = SequenceGenerator(
        source,
        combine_sequences=False,
        combined_seq_len=run_cfg.seqbench.composition.sample_length,
        seed=run_cfg.run.seed,
    )
    gs = gen.generate(idx=1, compute_length=True)

    assert gs.class_seq[-1] == SymbolEncoder.EOS_INDEX
    assert gs.state_seq[-1] == SymbolEncoder.EOS_SYMBOL
    assert len(gs.class_seq) == len(gs.state_seq)
    # parens stripped from every state symbol
    assert all("(" not in s and ")" not in s for s in gs.state_seq)


def test_generate_sequences_truncation():
    source, run_cfg = _source()
    combined = run_cfg.seqbench.composition.sample_length
    gen = SequenceGenerator(
        source,
        combine_sequences=True,
        combined_seq_len=combined,
        seed=run_cfg.run.seed,
    )
    gs = gen.generate(idx=1, compute_length=True)

    assert gs.class_seq.shape[0] == combined
    assert gs.state_seq.shape[0] == combined
