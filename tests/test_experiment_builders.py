# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Tests for high-level SeqBench experiment setup helpers."""

import copy
import os

import pytest
import torch
import yaml

try:
    import symseq  # noqa: F401

    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import build_dataloader, build_dataset, config as cfg_mod

pytestmark = pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")


def _raw(storage_path):
    return {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "symseq": {
            "generator": {
                "type": "ArtificialGrammar",
                "mode": "random",
                "params": {
                    "label": "builder-test",
                    "ambiguities": 2,
                    "ambiguity_depth": 2,
                    "n_start_states": 1,
                    "transition_density": 0.25,
                    "assume_equiprobable": True,
                },
            },
            "trial_constraints": {"length": {"min": 1, "max": 8}},
            "tasks": [
                {
                    "id": "next_token",
                    "type": "NStepPrediction",
                    "params": {"n": 1},
                }
            ],
        },
        "seqbench": {
            "mode": "file",
            "prob_generator_type": "restricted",
            "splits": {"train": 6, "test": 3},
            "storage": {"path": str(storage_path), "force_rebuild": True},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": True, "sample_length": 8},
            "input_mapping": {
                "base": "one_hot",
                "base_params": {"duration": 0.2},
                "transforms": [],
            },
            "task": {"source": "symseq", "ref_id": "next_token"},
        },
    }


def _write_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(_raw(tmp_path / "cache"), f)
    return config_path


def _example_onehot_alpha_rate_spikes(tmp_path, name):
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "configs",
        "onehot_alpha_rate_spikes.yaml",
    )
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["seqbench"]["splits"] = {"train": 8, "test": 0}
    raw["seqbench"]["storage"]["path"] = str(tmp_path / name)
    raw["seqbench"]["storage"]["force_rebuild"] = True
    return raw


def _set_transform_param(raw, transform_name, **params):
    raw = copy.deepcopy(raw)
    for transform in raw["seqbench"]["input_mapping"]["transforms"]:
        if transform["type"] == transform_name:
            transform.setdefault("params", {}).update(params)
            return raw
    raise AssertionError(f"transform {transform_name!r} not found")


def _spike_count(raw):
    ds = build_dataset(raw, split="train", dataset_size=8)
    torch.manual_seed(0)
    return sum(float(ds[i][0].sum()) for i in range(len(ds)))


def test_build_dataset_from_config_path_wires_defaults(tmp_path):
    config_path = _write_config(tmp_path)

    ds = build_dataset(config_path, split="train")

    assert len(ds) == 6
    assert ds.split == "train"
    assert ds.dataset_root.startswith(str(tmp_path / "cache"))
    assert ds.initial_time_grid.dt == pytest.approx(0.1)
    assert ds.config.seqbench.task.ref_id == "next_token"
    assert ds.input_mapping.base == "one_hot"
    assert ds.task_config.ref_id == "next_token"
    assert ds.time_grid.dt == pytest.approx(0.1)
    assert ds.dt == pytest.approx(0.1)
    assert ds.returns_target_probs is True
    assert ds.output_kind == "prediction"
    sample = ds[0]
    assert len(sample) == 5


def test_build_dataset_accepts_raw_config_dict(tmp_path):
    ds = build_dataset(_raw(tmp_path / "cache"), split="train")

    assert len(ds) == 6
    assert ds._cache_manifest["source_config"]["run"]["seed"] == 1


def test_build_dataloader_wires_collate(tmp_path):
    config_path = _write_config(tmp_path)

    loader = build_dataloader(
        config_path,
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    assert batch["data"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    assert "target_probs" in batch


def test_split_aware_cache_roots_and_manifests(tmp_path):
    config_path = _write_config(tmp_path)

    train_ds = build_dataset(config_path, split="train")
    test_ds = build_dataset(config_path, split="test")

    assert train_ds.dataset_root != test_ds.dataset_root
    assert os.path.isfile(os.path.join(train_ds.dataset_root, "train"))
    assert os.path.isfile(os.path.join(test_ds.dataset_root, "test"))

    with open(os.path.join(train_ds.dataset_root, "manifest.yaml")) as f:
        train_manifest = yaml.safe_load(f)
    with open(os.path.join(test_ds.dataset_root, "manifest.yaml")) as f:
        test_manifest = yaml.safe_load(f)

    assert train_manifest["split"] == "train"
    assert train_manifest["split_size"] == 6
    assert test_manifest["split"] == "test"
    assert test_manifest["split_size"] == 3
    assert train_manifest["cache_key"] != test_manifest["cache_key"]
    assert train_manifest["source_config"]["seqbench"]["task"]["ref_id"] == "next_token"


def test_storage_cache_key_override_controls_cache_root(tmp_path):
    raw = _raw(tmp_path / "cache")
    raw["seqbench"]["storage"]["cache_key"] = "manual-key"

    ds = build_dataset(cfg_mod.load(raw), split="train")

    assert ds.dataset_root == os.path.join(str(tmp_path / "cache"), "manual-key")
    with open(os.path.join(ds.dataset_root, "manifest.yaml")) as f:
        manifest = yaml.safe_load(f)
    assert manifest["cache_key"] == "manual-key"


def test_seqdataset_can_infer_time_grid_from_loaded_config(tmp_path):
    run_cfg = cfg_mod.load(_raw(tmp_path / "cache"))

    ds = build_dataset(run_cfg, split="train")

    assert ds.initial_time_grid.dt == pytest.approx(0.1)


def test_onehot_alpha_rate_spikes_ratecoding_max_rate_controls_spike_count(tmp_path):
    raw = _example_onehot_alpha_rate_spikes(tmp_path, "rate")

    low = _set_transform_param(raw, "RateCoding", max_rate=20.0)
    high = _set_transform_param(raw, "RateCoding", max_rate=200.0)

    assert _spike_count(high) > _spike_count(low)


def test_onehot_alpha_rate_spikes_amplitude_normalized_by_ratecoding(tmp_path):
    raw = _example_onehot_alpha_rate_spikes(tmp_path, "amplitude")

    low = _set_transform_param(raw, "TemporalUnfold", amplitude=1.0)
    high = _set_transform_param(raw, "TemporalUnfold", amplitude=1000.0)

    # RateCoding(normalize=True) intentionally removes absolute upstream scale.
    assert _spike_count(high) == pytest.approx(_spike_count(low))
