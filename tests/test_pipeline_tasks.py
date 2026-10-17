# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Phase 5: end-to-end pipeline tests — the configured task drives the target
through SeqDataset for both task sources, the per_trial path, and file mode."""

import os
import random

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

try:
    import symseq  # noqa: F401
    from seqbench.sources import build_symseq_source
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import config as cfg_mod
from seqbench.seq_dataset import SeqDataset, make_pad_sequence
from seqbench.utils import get_config_hash
from seqbench.dataset import create_base_dataset_from_config, initial_time_grid_from_config
from seqbench.transforms import compose_transforms_from_config

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


def _raw(task, *, symseq_tasks=None, generator=None, combine=False, mode="online",
         storage_path="/tmp/sb_pipeline_test"):
    cfg = {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "symseq": {
            "generator": generator or _GRAMMAR,
            "trial_constraints": {"length": {"min": 1, "max": 20}},
        },
        "seqbench": {
            "mode": mode,
            "prob_generator_type": "restricted",
            "splits": {"train": 8, "test": 4},
            "storage": {"path": storage_path},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": combine, "sample_length": 20},
            "input_mapping": {"base": "one_hot", "base_params": {}, "transforms": []},
            "task": task,
        },
    }
    if symseq_tasks is not None:
        cfg["symseq"]["tasks"] = symseq_tasks
    return cfg


def _build_dataset(raw, *, dataset_root=None, config_file_path=None, size=8):
    run_cfg = cfg_mod.load(raw)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    source = build_symseq_source(run_cfg)
    inp_map = run_cfg.seqbench.input_mapping
    kwargs = {}
    if inp_map.base == "one_hot":
        kwargs["alphabet_size"] = len(source.alphabet)
    base_dataset = create_base_dataset_from_config(
        inp_map, "train", final_dt=run_cfg.seqbench.time_grid.dt, **kwargs
    )
    initial_time_grid = initial_time_grid_from_config(
        inp_map, final_dt=run_cfg.seqbench.time_grid.dt
    )
    transforms = compose_transforms_from_config(inp_map)
    return SeqDataset(
        config=run_cfg,
        generator=source,
        base_dataset=base_dataset,
        is_train=True,
        dataset_size=size,
        config_file_path=config_file_path,
        pad_index=-1,
        dataset_root=dataset_root,
        transform=transforms,
        initial_time_grid=initial_time_grid,
    )


_SYMSEQ_NEXT = dict(
    task={"source": "symseq", "ref_id": "next_token"},
    symseq_tasks=[{"id": "next_token", "type": "NStepPrediction", "params": {"n": 1}}],
)


def test_symseq_next_token_is_prediction_not_classification():
    """#12 regression guard: the config (NStepPrediction) now drives a next-token
    prediction target — not the previously hardcoded state-id classification."""
    ds = _build_dataset(_raw(**_SYMSEQ_NEXT, combine=False))
    assert ds.is_per_trial is False
    assert ds.per_token_classify is False    # predict mode
    assert ds._wants_target_probs is True

    gs = ds.gs.generate(0, compute_length=False)
    sample = ds.gensample_to_sample(gs)
    cls = list(gs.class_seq)
    # next-token property: at non-masked positions target == next class id
    for i in range(len(cls) - 2):
        assert sample.target_seq[i] == cls[i + 1]
    # ... and it is NOT the state-id classification target
    state_ids = [ds.target_prob_generator.unred_state_to_id(s) for s in gs.state_seq]
    assert not np.array_equal(sample.target_seq, state_ids)

    item = ds[0]
    assert len(item) == 5  # predict item: data, target, class_seq, target_probs, gap_mask


def test_state_classification_reproduces_legacy_classify():
    ds = _build_dataset(_raw(task={"source": "seqbench", "id": "state_classification", "type": "StateClassification"}))
    assert ds.per_token_classify is True
    assert ds._wants_target_probs is False

    gs = ds.gs.generate(0, compute_length=False)
    sample = ds.gensample_to_sample(gs)
    expected = [ds.target_prob_generator.unred_state_to_id(s) for s in gs.state_seq]
    np.testing.assert_array_equal(sample.target_seq, expected)

    item = ds[0]
    assert len(item) == 4  # classify item: data, target, class_seq, gap_mask


def test_per_trial_classification_one_label_per_sample():
    ds = _build_dataset(
        _raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                   "params": {"label_source": "first"}}, combine=False)
    )
    assert ds.is_per_trial is True

    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=make_pad_sequence(ds, pad_index=-1), num_workers=0)
    batch = next(iter(loader))
    assert batch["labels"].dim() == 1                       # one label per sample
    assert batch["labels"].shape[0] == batch["data"].shape[0]


def test_combine_with_per_trial_task_labels_at_boundaries():
    """combine_sequences + per_trial task: label at last position of each trial, rest masked."""
    ds = _build_dataset(
        _raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                   "params": {"label_source": "first"}}, combine=True)
    )
    assert ds.is_per_trial is False
    assert ds.per_token_classify is True

    gs = ds.gs.generate(0, compute_length=False)
    sample = ds.gensample_to_sample(gs)

    # At least one position must carry a non-pad label (a trial boundary).
    # Masked positions are encoded as pad_index (-1) in target_seq.
    n_labelled = sum(v != -1 for v in sample.target_seq)
    n_masked = sum(v == -1 for v in sample.target_seq)
    assert n_labelled > 0, "Expected at least one labelled boundary in the combined target"
    assert n_masked > 0, "Expected masked positions between trial boundaries"


def test_combine_base_label_source_raises():
    """label_source='base' + combine_sequences must raise at construction time."""
    with pytest.raises(ValueError, match="label_source='base'"):
        _build_dataset(
            _raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                       "params": {"label_source": "base"}}, combine=True)
        )


def test_file_mode_roundtrip_matches_online(tmp_path):
    """A file-mode dataset (targets serialized) yields the same targets as online."""
    nback = {"type": "NBack", "params": {"n": 2, "alphabet_size": 4, "seq_length": 8}}

    # online reference (built on the fly)
    ds_online = _build_dataset(_raw(**_SYMSEQ_NEXT, generator=nback, combine=False))
    online_targets = [ds_online.gensample_to_sample(
        ds_online.gs.generate(i, compute_length=False)).target_seq for i in range(4)]

    # file mode: write config, generate to disk, then read back
    raw = _raw(**_SYMSEQ_NEXT, generator=nback, combine=False, mode="file",
               storage_path=str(tmp_path))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(raw))
    run_cfg = cfg_mod.load(raw)
    dataset_root = os.path.join(str(tmp_path), get_config_hash(run_cfg, dataset_size=8))
    ds_file = _build_dataset(raw, dataset_root=dataset_root,
                             config_file_path=str(cfg_path))

    # Only explicitly configured SymSeq targets are serialized.
    with open(os.path.join(dataset_root, "train")) as fh:
        serialized = fh.read()
        assert "next_token" in serialized
        assert "nback_match" not in serialized

    # reading must not raise (configured target was serialized) and match online
    for i in range(4):
        np.testing.assert_array_equal(
            ds_file.buffered_samples[i].target_seq, online_targets[i]
        )


class _LabeledBaseDataset:
    """Mock base dataset with list-style class_dict and native labels.

    Simulates a real dataset (e.g. SpeechCommands) where class_dict maps
    class_idx -> [sample_idx, ...] and __getitem__ returns (data, label).
    Labels are intentionally offset from class indices to catch mix-ups.
    """

    _LABEL_OFFSET = 100

    def __init__(self, alphabet_size):
        # class indices 1..alphabet_size (0 is EOS), each with one sample
        self.class_dict = {i: [i - 1] for i in range(1, alphabet_size + 1)}
        self._n = alphabet_size

    def __getitem__(self, idx):
        data = torch.zeros(1, self._n)
        data[0, idx % self._n] = 1.0
        label = torch.tensor(idx + self._LABEL_OFFSET)
        return data, label

    def __len__(self):
        return self._n


def test_label_source_base_reads_from_base_dataset():
    """label_source='base' must use base_dataset[rep_idx][1], not class_seq[0]."""
    alphabet_size = 4
    raw = {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": alphabet_size}, "eos": "#"},
        "symseq": {
            "generator": _GRAMMAR,
            "trial_constraints": {"length": {"min": 1, "max": 10}},
        },
        "seqbench": {
            "mode": "online",
            "prob_generator_type": "restricted",
            "splits": {"train": 8, "test": 4},
            "storage": {"path": "/tmp/sb_label_base_test"},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": False, "sample_length": 1},
            "input_mapping": {"base": "shd", "base_params": {}, "transforms": []},
            "task": {"source": "seqbench", "id": "classification", "type": "Classification",
                     "params": {"label_source": "base"}},
        },
    }
    run_cfg = cfg_mod.load(raw)
    source = build_symseq_source(run_cfg)
    base_dataset = _LabeledBaseDataset(alphabet_size)

    ds = SeqDataset(
        config=run_cfg,
        generator=source,
        base_dataset=base_dataset,
        is_train=True,
        dataset_size=8,
        config_file_path=None,
        pad_index=-1,
    )

    assert ds.is_per_trial is True

    for i in range(4):
        gs = ds.gs.generate(i, compute_length=False)
        # task is deferred — target must NOT be pre-cached at draw time
        assert ds.target_builder.task_id not in (gs.targets or {})

        sample = ds.gensample_to_sample(gs)
        class_idx = int(gs.class_seq[0])
        rep_idx = base_dataset.class_dict[class_idx][0]
        expected_label = rep_idx + _LabeledBaseDataset._LABEL_OFFSET
        assert int(sample.target_seq) == expected_label, (
            f"Expected label from base_dataset ({expected_label}), "
            f"got {int(sample.target_seq)} (class_idx={class_idx})"
        )
