import copy

import pytest
from seqbench import config as cfg_mod
from seqbench.sources import build_symseq_source


def _raw():
    return {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "symseq": {
            "generator": {
                "type": "NBack",
                "params": {"n": 2, "alphabet_size": 4, "seq_length": 8},
            },
            "tasks": [
                {"id": "next_token", "type": "NStepPrediction", "params": {"n": 1}}
            ],
        },
        "seqbench": {
            "mode": "online",
            "splits": {"train": 4},
            "storage": {"path": "/tmp/seqbench-task-config"},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": False, "sample_length": 8},
            "input_mapping": {"base": "one_hot"},
            "task": {"source": "symseq", "ref_id": "next_token"},
        },
    }


def test_duplicate_symseq_task_id_rejected():
    raw = _raw()
    raw["symseq"]["tasks"].append(
        {"id": "next_token", "type": "NStepMemory", "params": {"n": 1}}
    )
    with pytest.raises(ValueError, match="must be unique"):
        cfg_mod.load(raw)


def test_unknown_ref_id_lists_available_ids():
    raw = _raw()
    raw["seqbench"]["task"]["ref_id"] = "missing"
    with pytest.raises(ValueError, match=r"available:.*next_token"):
        cfg_mod.load(raw)


@pytest.mark.parametrize(
    "key,value",
    [("id", "alias"), ("type", "NStepPrediction"), ("params", {})],
)
def test_symseq_reference_forbids_local_definition_fields(key, value):
    raw = _raw()
    raw["seqbench"]["task"][key] = value
    with pytest.raises(ValueError, match="forbids"):
        cfg_mod.load(raw)


def test_seqbench_native_task_requires_id():
    raw = _raw()
    raw["seqbench"]["task"] = {"source": "seqbench", "type": "Classification"}
    with pytest.raises(ValueError, match=r"missing required keys.*id"):
        cfg_mod.load(raw)


def test_seqbench_native_id_is_preserved():
    raw = _raw()
    raw["seqbench"]["task"] = {
        "source": "seqbench",
        "id": "base_classification",
        "type": "Classification",
    }
    cfg = cfg_mod.load(raw)
    assert cfg.seqbench.task.id == "base_classification"


def test_native_id_cannot_collide_with_symseq_task_id():
    raw = _raw()
    raw["seqbench"]["task"] = {
        "source": "seqbench",
        "id": "next_token",
        "type": "Classification",
    }
    with pytest.raises(ValueError, match="collides"):
        cfg_mod.load(raw)


def test_old_name_field_is_not_accepted():
    raw = copy.deepcopy(_raw())
    raw["symseq"]["tasks"][0]["name"] = raw["symseq"]["tasks"][0].pop("id")
    with pytest.raises(ValueError, match=r"unknown keys.*name"):
        cfg_mod.load(raw)


def test_intrinsic_task_generator_mismatch_fails_when_source_is_built():
    pytest.importorskip("symseq")
    raw = _raw()
    raw["symseq"]["tasks"] = [{"id": "valid", "type": "Grammaticality"}]
    raw["seqbench"]["task"] = {"source": "symseq", "ref_id": "valid"}

    config = cfg_mod.load(raw)
    with pytest.raises(ValueError, match=r"requires intrinsic target.*grammaticality"):
        build_symseq_source(config)
