# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Task-driven ``num_classes``: the target label-space size is derived from the
configured task (class-id vocab / unreduced-state count / base-label count),
not hardcoded to grammar state counts."""

from types import SimpleNamespace

import pytest

from seqbench.config import TaskSource
from seqbench.seq_utils.symbol_encoder import SymbolEncoder
from seqbench.tasks.classify import Classification, StateClassification
from seqbench.tasks.shift import NStepPrediction
from seqbench.tasks.target_builder import TaskTargetBuilder

try:
    import symseq  # noqa: F401
    from seqbench.sources import build_symseq_source
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False


def _enc():
    return SymbolEncoder(["A", "B", "C", "D"])  # len == 5 (alphabet + EOS)


def _builder(task):
    return TaskTargetBuilder(
        source=TaskSource.SEQBENCH,
        task_id="test_task",
        task=task,
        encoder=_enc(),
    )


# --------------------------------------------------------------------------
# Builder unit (label_space -> size), no dataset needed
# --------------------------------------------------------------------------

class TestBuilderNumClasses:
    def test_class_id_is_encoder_size(self):
        assert _builder(NStepPrediction(1)).num_classes() == 5
        assert _builder(Classification(label_source="first")).num_classes() == 5

    def test_unreduced_state_uses_grammar_count(self):
        b = _builder(StateClassification())
        pg = SimpleNamespace(num_unreduced_states=7)
        assert b.num_classes(prob_generator=pg) == 7

    def test_unreduced_state_requires_grammar(self):
        b = _builder(StateClassification())
        with pytest.raises(ValueError, match="grammar"):
            b.num_classes(prob_generator=None)
        with pytest.raises(ValueError, match="grammar"):
            b.num_classes(prob_generator=SimpleNamespace(num_unreduced_states=0))

    def test_base_label_uses_base_dataset_count(self):
        b = _builder(Classification(label_source="base"))
        base = SimpleNamespace(class_dict={0: [], 1: [], 2: []})
        assert b.num_classes(base_dataset=base) == 3

    def test_base_label_requires_base_dataset(self):
        b = _builder(Classification(label_source="base"))
        with pytest.raises(ValueError, match="base_dataset"):
            b.num_classes()


# --------------------------------------------------------------------------
# SeqDataset integration (online)
# --------------------------------------------------------------------------

_GRAMMAR = {
    "type": "ArtificialGrammar",
    "mode": "random",
    "params": {
        "label": "g", "ambiguities": 2, "ambiguity_depth": 2,
        "n_start_states": 1, "transition_density": 0.25, "assume_equiprobable": True,
    },
}


def _raw(task, *, symseq_tasks=None, generator=None):
    cfg = {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 4}, "eos": "#"},
        "symseq": {
            "generator": generator or _GRAMMAR,
            "trial_constraints": {"length": {"min": 1, "max": 20}},
        },
        "seqbench": {
            "mode": "online", "prob_generator_type": "restricted",
            "splits": {"train": 8, "test": 4},
            "storage": {"path": "/tmp/sb_numclasses_test"},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": False, "sample_length": 20},
            "input_mapping": {"base": "one_hot", "base_params": {}, "transforms": []},
            "task": task,
        },
    }
    if symseq_tasks is not None:
        cfg["symseq"]["tasks"] = symseq_tasks
    return cfg


@pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")
class TestSeqDatasetNumClasses:
    def _build(self, raw):
        import random
        import numpy as np
        import torch
        from seqbench import config as cfg_mod
        from seqbench.seq_dataset import SeqDataset
        from seqbench.dataset import create_base_dataset_from_config, initial_time_grid_from_config
        from seqbench.transforms import compose_transforms_from_config

        run_cfg = cfg_mod.load(raw)
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        source = build_symseq_source(run_cfg)
        inp_map = run_cfg.seqbench.input_mapping
        base_dataset = create_base_dataset_from_config(
            inp_map,
            "train",
            final_dt=run_cfg.seqbench.time_grid.dt,
            alphabet_size=len(source.alphabet),
        )
        initial_time_grid = initial_time_grid_from_config(
            inp_map, final_dt=run_cfg.seqbench.time_grid.dt
        )
        transforms = compose_transforms_from_config(inp_map)
        return SeqDataset(
            config=run_cfg, generator=source, base_dataset=base_dataset,
            is_train=True, dataset_size=8, config_file_path=None,
            pad_index=-1, dataset_root=None, transform=transforms,
            initial_time_grid=initial_time_grid)

    _SYMSEQ_NEXT = dict(
        task={"source": "symseq", "ref_id": "next_token"},
        symseq_tasks=[{"id": "next_token", "type": "NStepPrediction", "params": {"n": 1}}],
    )

    def test_prediction_over_grammar(self):
        ds = self._build(_raw(**self._SYMSEQ_NEXT))
        # class-id vocab == num_reduced_states for a grammar (behavior preserved)
        assert ds.num_classes == 5 == ds.target_prob_generator.num_reduced_states

    def test_state_classification_over_grammar(self):
        ds = self._build(_raw(task={"source": "seqbench", "id": "state_classification", "type": "StateClassification"}))
        assert ds.num_classes == ds.target_prob_generator.num_unreduced_states == 7

    def test_per_trial_classification_over_grammar(self):
        ds = self._build(_raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                                    "params": {"label_source": "first"}}))
        # class-id label space, NOT the unreduced-state count (was wrongly 7)
        assert ds.num_classes == 5

    def test_prediction_over_non_grammar_nback(self):
        # regression-fix guard: non-grammar source previously gave num_classes == 0
        nback = {"type": "NBack", "params": {"n": 2, "alphabet_size": 4, "seq_length": 8}}
        ds = self._build(_raw(**self._SYMSEQ_NEXT, generator=nback))
        assert ds.num_classes == 5  # alphabet_size + EOS
