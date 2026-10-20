# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Per-trial seeding: symbolic draws depend only on (seed, split, idx)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

try:
    import symseq  # noqa: F401
    from seqbench.sources import build_symseq_source
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import config as cfg_mod
from seqbench.seq_dataset import SeqDataset
from seqbench.seq_utils.seeding import Stream, split_key, trial_rng, trial_seed_sequence
from seqbench.transforms.base import TimeGrid
from seqbench.transforms.compose import Compose
from seqbench.transforms.snn.poisson_encoding import PoissonEncoding

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


def _raw(seed=1, *, seqbench_seed=None, gap_profile=None, base="one_hot", transforms=(),
         render_seeding=None):
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
    if render_seeding is not None:
        seqbench["render_seeding"] = render_seeding
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


# ------------------------------------------------------ rendering (S4, opt-in)

_N_ITEMS = 6
_N_STEPS = 2
_GAPS_A = {"start": 0, "duration": {"dist": "uniform", "params": {"low": 0.1, "high": 0.5}}}
_GAPS_B = {"start": 0, "duration": {"dist": "lognormal", "params": {"mean": -1.0, "sigma": 0.5}}}
_POISSON = {"type": "PoissonEncoding",
            "params": {"temporal": True, "spike_dt": 0.1, "normalize": False}}


class _ValueCodedBase:
    """Base dataset with several items per class; each row reads ``[rate, code]``.

    ``code = item index + 1`` identifies the drawn base item; gaps are zero rows.
    """

    def __init__(self, n_classes):
        self.class_dict = {
            c: list(range(c * _N_ITEMS, (c + 1) * _N_ITEMS)) for c in range(n_classes)
        }

    def __len__(self):
        return len(self.class_dict) * _N_ITEMS

    def __getitem__(self, i):
        row = torch.tensor([[5.0, 2.0 + i]])
        return row.repeat(_N_STEPS, 1), i // _N_ITEMS


def _render_dataset(split="train", *, gaps=_GAPS_A, transforms=(), render_seeding="per_trial"):
    raw = _raw(base="value_coded", gap_profile=gaps, transforms=transforms,
               render_seeding=render_seeding)
    run_cfg = cfg_mod.load(raw)
    source = build_symseq_source(run_cfg)
    return SeqDataset(
        run_cfg, split,
        generator=source,
        base_dataset=_ValueCodedBase(len(source.alphabet) + 1),
        initial_time_grid=TimeGrid(0.1),
    )


def _data(item):
    return item[0].detach().cpu().numpy()


def _item_rows(ds, idx):
    """Boolean mask of stimulus (non-gap) rows of an item without transforms."""
    return _data(ds[idx])[:, 0] > 0


def _base_items(ds, idx):
    """Base-item code of every element of item ``idx`` (renders once)."""
    data = _data(ds[idx])
    return data[data[:, 0] > 0, 1][::_N_STEPS]


def test_render_seeding_validated():
    with pytest.raises(ValueError, match="render_seeding"):
        cfg_mod.load(_raw(render_seeding="random"))
    assert cfg_mod.load(_raw()).seqbench.render_seeding == "sequential"


def test_sequential_mode_depends_on_access_order():
    # Documents the default: one shared RNG consumed in access order.
    fwd = _render_dataset(render_seeding="sequential")
    a = [_base_items(fwd, i) for i in range(4)]
    rev = _render_dataset(render_seeding="sequential")
    b = [_base_items(rev, i) for i in reversed(range(4))][::-1]
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_per_trial_render_independent_of_access_order():
    ds = _render_dataset(transforms=[_POISSON])
    forward = [_data(ds[i]) for i in range(8)]
    for i in reversed(range(8)):
        np.testing.assert_array_equal(_data(ds[i]), forward[i])

    fresh = _render_dataset(transforms=[_POISSON])
    perm = [5, 2, 7, 0, 3, 6, 1, 4]
    loader = DataLoader(fresh, batch_size=None, sampler=perm, num_workers=2)
    for i, item in zip(perm, loader):
        np.testing.assert_array_equal(_data(item), forward[i])


def test_per_trial_base_items_and_spikes_unchanged_by_gap_changes():
    plain_a, plain_b = _render_dataset(gaps=_GAPS_A), _render_dataset(gaps=_GAPS_B)
    spikes_a = _render_dataset(gaps=_GAPS_A, transforms=[_POISSON])
    spikes_b = _render_dataset(gaps=_GAPS_B, transforms=[_POISSON])
    n_layout_changed = 0
    for idx in range(8):
        rows_a, rows_b = _item_rows(plain_a, idx), _item_rows(plain_b, idx)
        n_layout_changed += len(rows_a) != len(rows_b) or not np.array_equal(rows_a, rows_b)
        np.testing.assert_array_equal(_base_items(plain_a, idx), _base_items(plain_b, idx))
        np.testing.assert_array_equal(
            _data(spikes_a[idx])[rows_a], _data(spikes_b[idx])[rows_b]
        )
    assert n_layout_changed > 0  # the gap change actually moved the stimuli


def test_per_trial_train_and_test_render_differently():
    train, test = _render_dataset("train"), _render_dataset("test")
    assert any(
        not np.array_equal(_data(train[i]), _data(test[i])) for i in range(4)
    )


def test_compose_reseed_gives_each_step_its_own_stream():
    def poisson():
        return PoissonEncoding(temporal=False, duration=1.0, spike_dt=0.01)

    x = torch.full((8,), 20.0)
    ss = np.random.SeedSequence(3)
    pipe = Compose([poisson(), poisson()])
    pipe.reseed(ss)
    first, second = pipe.transforms[0](x), pipe.transforms[1](x)
    assert not torch.equal(first, second)

    again = Compose([poisson(), poisson()])
    again.reseed(np.random.SeedSequence(3))
    assert torch.equal(again.transforms[0](x), first)


def test_poisson_reseed_keeps_frozen_noise():
    enc = PoissonEncoding(temporal=False, duration=1.0, spike_dt=0.01,
                          frozen_noise=True, seed=7)
    x = torch.full((8,), 20.0)
    frozen = enc(x)
    enc.reseed(np.random.SeedSequence(123))
    assert torch.equal(enc(x), frozen)
