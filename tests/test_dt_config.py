# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Tests for explicit final output time-grid configuration."""

import pytest

from seqbench import config as cfg_mod
from seqbench.config import InputMappingCfg
from seqbench.dataset import (
    create_base_dataset_from_config,
    has_time_creating_transform,
    initial_time_grid_from_config,
)
from seqbench.utils import to_plain_data


def _raw(**seqbench_overrides):
    """Minimal valid config; seqbench section is shallow-merged with overrides."""
    seqbench = {
        "mode": "online",
        "prob_generator_type": "restricted",
        "splits": {"train": 8, "test": 4},
        "storage": {"path": "/tmp/sb_dt_test"},
        "time_grid": {"dt": 0.1},
        "composition": {"combine_sequences": False, "sample_length": 20},
        "input_mapping": {"base": "one_hot", "base_params": {}, "transforms": []},
        "task": {"source": "seqbench", "id": "state_classification", "type": "StateClassification"},
    }
    seqbench.update(seqbench_overrides)
    return {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "seqbench": seqbench,
    }


def test_time_grid_is_required():
    seqbench = _raw()["seqbench"]
    seqbench.pop("time_grid")
    with pytest.raises(ValueError, match="missing required keys.*time_grid"):
        cfg_mod.load({
            "run": _raw()["run"],
            "symbol_space": _raw()["symbol_space"],
            "seqbench": seqbench,
        })


def test_time_grid_explicit():
    cfg = cfg_mod.load(_raw(time_grid={"dt": 0.05}))
    assert cfg.seqbench.time_grid.dt == 0.05
    assert cfg.seqbench.time_grid.validation == "error"


def test_old_dataset_schema_migrates_to_new_sections():
    raw = {
        "dataset": {
            "seed": 7,
            "alphabet": {"size": 4, "eos": "#"},
            "trial_length": {"min": 2, "max": 9, "distribution": "uniform"},
            "splits": {"train": 8, "test": 4},
        },
        "symseq": {
            "generator": {"type": "NBack", "params": {"n": 2, "seq_length": 8}},
            "trial_set": {"n_trials": 12, "gen_params": {"seq_length": 9}},
        },
        "seqbench": _raw()["seqbench"],
    }

    cfg = cfg_mod.load(raw)
    plain = to_plain_data(cfg)

    assert "dataset" not in plain
    assert plain["run"]["seed"] == 7
    assert plain["symbol_space"] == {"alphabet": {"size": 4, "symbols": None}, "eos": "#"}
    assert plain["symseq"]["trial_constraints"]["length"] == {"min": 2, "max": 9}
    assert plain["symseq"]["generator"]["trial_params"] == {"seq_length": 9}
    assert "gen_params" not in plain["symseq"].get("trial_set", {})
    assert plain["seqbench"]["splits"] == {"train": 8, "test": 4}


def test_time_grid_validation_mode():
    cfg = cfg_mod.load(_raw(time_grid={"dt": 0.05, "validation": "warn"}))
    assert cfg.seqbench.time_grid.validation == "warn"


@pytest.mark.parametrize("bad", [0, -0.1, "x"])
def test_time_grid_dt_must_be_positive(bad):
    with pytest.raises(ValueError, match="time_grid.dt must be a positive"):
        cfg_mod.load(_raw(time_grid={"dt": bad}))


def test_time_grid_validation_must_be_known():
    with pytest.raises(ValueError, match="time_grid.validation"):
        cfg_mod.load(_raw(time_grid={"dt": 0.1, "validation": "whatever"}))


def test_top_level_dt_is_rejected():
    with pytest.raises(ValueError, match="seqbench.dt is no longer supported"):
        cfg_mod.load(_raw(dt=0.1))


def test_gap_profile_dt_is_rejected():
    raw = _raw(
        composition={
            "combine_sequences": False,
            "sample_length": 20,
            "gap_profile": {
                "start": 0,
                "duration": {"dist": "uniform", "params": {"low": 5, "high": 5}},
                "add_nongramm_gap": False,
                "dt": 0.2,
            },
        }
    )
    with pytest.raises(ValueError, match="gap_profile.dt is no longer supported"):
        cfg_mod.load(raw)


def test_one_hot_duration_creates_initial_grid_without_time_creator():
    pytest.importorskip("torch")
    mapping = InputMappingCfg(
        base="one_hot",
        base_params={"duration": 0.3},
        transforms=[],
    )
    grid = initial_time_grid_from_config(mapping, final_dt=0.1)
    dataset = create_base_dataset_from_config(
        mapping, "train", final_dt=0.1, alphabet_size=4
    )
    assert grid.dt == pytest.approx(0.1)
    assert dataset.n_steps == 3


def test_one_hot_duration_rejected_with_time_creator():
    pytest.importorskip("torch")
    mapping = InputMappingCfg(
        base="one_hot",
        base_params={"duration": 0.3},
        transforms=[
            {
                "type": "TemporalUnfold",
                "params": {
                    "out_dt": 0.01,
                    "duration": 0.1,
                    "kernel_spec": {"shape": "box", "params": {"width": 0.1}},
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="one_hot.base_params.duration"):
        create_base_dataset_from_config(
            mapping, "train", final_dt=0.01, alphabet_size=4
        )


def test_one_hot_static_when_transform_creates_time():
    pytest.importorskip("torch")
    mapping = InputMappingCfg(
        base="one_hot",
        base_params={},
        transforms=[
            {
                "type": "TemporalUnfold",
                "params": {
                    "out_dt": 0.01,
                    "duration": 0.1,
                    "kernel_spec": {"shape": "box", "params": {"width": 0.1}},
                },
            }
        ],
    )
    grid = initial_time_grid_from_config(mapping, final_dt=0.01)
    dataset = create_base_dataset_from_config(
        mapping, "train", final_dt=0.01, alphabet_size=4
    )
    assert grid is None
    assert dataset.n_steps == 1


def test_poisson_time_creation_reads_nested_temporal_param():
    static = InputMappingCfg(
        base="one_hot",
        transforms=[{"type": "PoissonEncoding", "params": {"temporal": False}}],
    )
    temporal = InputMappingCfg(
        base="one_hot",
        transforms=[{"type": "PoissonEncoding", "params": {"temporal": True}}],
    )

    assert has_time_creating_transform(static) is True
    assert has_time_creating_transform(temporal) is False
