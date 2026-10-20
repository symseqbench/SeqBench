# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Symbolic draws depend only on (seed, split, idx)."""

import numpy as np
import pytest

try:
    import symseq  # noqa: F401
    from seqbench.sources import build_symseq_source
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import config as cfg_mod
from seqbench.seq_dataset import SeqDataset
from seqbench.seq_utils.seeding import Stream, split_key, trial_rng, trial_seed_sequence

pytestmark = pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")

_GRAMMAR = {
    "type": "ArtificialGrammar",
    "mode": "random",
    "params": {
        "label": "g",
        "ambiguities": 2,
        "ambiguity_depth": 2,
        "n_start_states": 1,
        "transition_density": 0.25,
        "assume_equiprobable": True,
    },
}


def _raw(seed=1, *, seqbench_seed=None, gap_profile=None, base="one_hot", transforms=()):
    seqbench = {
        "mode": "online",
        "splits": {"train": 16, "test": 16},
        "storage": {"path": "/tmp/sb_seeding_test"},
        "time_grid": {"dt": 0.1},
        "composition": {"combine_sequences": False, "sample_length": 20},
        "input_mapping": {"base": base, "base_params": {}, "transforms": list(transforms)},
        "task": {"source": "seqbench", "id": "current_symbol", "type": "Classification",
                 "params": {"label_source": "current", "granularity": "per_token"}},
    }
    if gap_profile is not None:
        seqbench["composition"]["gap_profile"] = gap_profile
    if seqbench_seed is not None:
        seqbench["seed"] = seqbench_seed
    return {
        "run": {"seed": seed},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "symseq": {
            "generator": _GRAMMAR,
            "trial_constraints": {"length": {"min": 3, "max": 20}},
        },
        "seqbench": seqbench,
    }


def _symbols(ds, idx):
    return list(ds.gs.generate(idx, compute_length=False).class_seq)


def _dataset(raw, split, **kwargs):
    return SeqDataset(cfg_mod.load(raw), split, **kwargs)


# ---------------------------------------------------------------- seed helpers


def test_seed_sequence_depends_on_every_key():
    base = trial_seed_sequence(1, "train", 3, Stream.SYMBOL).generate_state(4)
    variants = [
        trial_seed_sequence(2, "train", 3, Stream.SYMBOL),
        trial_seed_sequence(1, "test", 3, Stream.SYMBOL),
        trial_seed_sequence(1, "train", 4, Stream.SYMBOL),
        trial_seed_sequence(1, "train", 3, Stream.BASE_ITEM),
        trial_seed_sequence(1, "train", 3, Stream.SYMBOL, 0),
    ]
    for ss in variants:
        assert not np.array_equal(base, ss.generate_state(4))


def test_seed_sequence_has_no_product_collisions():
    # The former seed * idx scheme mapped (2, 3) and (3, 2) to the same stream
    # and made idx 0 identical for every seed.
    a = trial_rng(2, "train", 3, Stream.SYMBOL).integers(0, 2**32, 4)
    b = trial_rng(3, "train", 2, Stream.SYMBOL).integers(0, 2**32, 4)
    assert not np.array_equal(a, b)
    c = trial_rng(1, "train", 0, Stream.SYMBOL).integers(0, 2**32, 4)
    d = trial_rng(7, "train", 0, Stream.SYMBOL).integers(0, 2**32, 4)
    assert not np.array_equal(c, d)


def test_split_key_is_stable():
    assert split_key(None) == 0
    assert split_key("train") == split_key("train") != split_key("test")


# ---------------------------------------------------------- symbolic draws (S0)


def test_train_and_test_draw_different_trials():
    train = _dataset(_raw(), "train")
    test = _dataset(_raw(), "test")
    n_equal = sum(_symbols(train, i) == _symbols(test, i) for i in range(16))
    # Short grammar strings can coincide by chance, but not for every index.
    assert n_equal < 8


def test_index_zero_depends_on_master_seed():
    # run.seed fixes the (random) grammar; only the seqbench seed varies.
    draws = {
        tuple(_symbols(_dataset(_raw(seqbench_seed=s), "train"), 0)) for s in range(1, 9)
    }
    assert len(draws) > 1


def test_seeded_generator_requires_a_split():
    ds = _dataset(_raw(), "train")
    generator = ds.gs
    assert generator.split == "train"
    generator.split = None
    with pytest.raises(ValueError, match="needs a split"):
        generator.generate(0, compute_length=False)
    # An explicit split argument is enough.
    assert generator.generate(0, compute_length=False, split="test") is not None


def test_file_mode_train_and_test_files_differ(tmp_path):
    import yaml

    raw = _raw()
    raw["seqbench"]["mode"] = "file"
    raw["seqbench"]["storage"] = {"path": str(tmp_path)}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(raw))
    lines = {}
    for split in ("train", "test"):
        root = tmp_path / f"root_{split}"
        SeqDataset(
            cfg_mod.load(raw), split, dataset_root=str(root), config_file_path=str(cfg_path)
        )
        lines[split] = (root / split).read_text().splitlines()
    assert len(lines["train"]) == len(lines["test"]) == 16
    n_equal = sum(a == b for a, b in zip(lines["train"], lines["test"]))
    assert n_equal < 8


def test_symbols_reproducible_per_seed_split_idx():
    a = _dataset(_raw(), "train")
    b = _dataset(_raw(), "train")
    for idx in (5, 0, 11):
        assert _symbols(a, idx) == _symbols(b, idx)
