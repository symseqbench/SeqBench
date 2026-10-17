# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Phase 3: the TaskTargetBuilder resolves targets for both task sources into a
single class-id space, honoring the EOS-alignment rule."""

import numpy as np
import pytest

from seqbench.config import TaskSource
from seqbench.seq_utils.generator import GeneratorSample
from seqbench.seq_utils.symbol_encoder import SymbolEncoder
from seqbench.tasks.classify import StateClassification
from seqbench.tasks.shift import NStepPrediction
from seqbench.tasks.target_builder import TaskTargetBuilder

HAS_SYMSEQ = True
try:
    import symseq  # noqa: F401
    from symseq.trial import Trial, Target as SymseqTarget
    from symseq.tasks import registry as symseq_registry
except ImportError:
    HAS_SYMSEQ = False


def _encoder():
    return SymbolEncoder(["A", "B", "C"])  # A=1 B=2 C=3 #=0


def _gs():
    return GeneratorSample(
        np.array([1, 2, 3, 0]), np.array(["A", "B", "C", "#"], dtype=object), 4
    )


# --------------------------------------------------------------------------
# source = seqbench
# --------------------------------------------------------------------------

class TestSeqbenchSource:
    def test_nstep_prediction_resolve(self):
        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="prediction",
            task=NStepPrediction(1),
            encoder=_encoder(),
        )
        gs = _gs()
        target = b.resolve(None, gs.class_seq, gs.state_seq)
        # seqbench task sees the EOS token: last real token predicts EOS (0)
        assert target.values == [2, 3, 0, None]
        res = b._to_resolved(target, 4)
        np.testing.assert_array_equal(res.target_seq, [2, 3, 0, -1])  # masked EOS -> pad

    def test_state_classification_matches_legacy(self):
        id_map = {"A": 1, "B": 2, "C": 3, "#": 0}
        task = StateClassification()
        task.state_id_fn = lambda s: id_map[s]
        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="states",
            task=task,
            encoder=_encoder(),
        )
        gs = _gs()
        res = b.to_target_seq(
            GeneratorSample(gs.class_seq, gs.state_seq, 4)
        )
        np.testing.assert_array_equal(res.target_seq, [1, 2, 3, 0])
        assert res.granularity == "per_token"

    def test_to_target_seq_recomputes_when_absent(self):
        # legacy on-disk sample (targets=None) -> seqbench task recomputed
        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="prediction",
            task=NStepPrediction(1),
            encoder=_encoder(),
        )
        res = b.to_target_seq(_gs())
        np.testing.assert_array_equal(res.target_seq, [2, 3, 0, -1])

    def test_to_target_seq_prefers_stored(self):
        from seqbench.tasks.base import Target

        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="prediction",
            task=NStepPrediction(1),
            encoder=_encoder(),
        )
        gs = _gs()
        gs.targets = {
            b.task_id: Target(
                values=[9, 9, 9, None],
                mask=[True, True, True, False],
                granularity="per_token",
            )
        }
        res = b.to_target_seq(gs)
        np.testing.assert_array_equal(res.target_seq, [9, 9, 9, -1])

    def test_classification_per_trial(self):
        from seqbench.tasks.classify import Classification

        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="classification",
            task=Classification(label_source="first"),
            encoder=_encoder(),
            granularity="per_trial",
        )
        res = b.to_target_seq(_gs())
        assert res.granularity == "per_trial"
        assert int(res.target_seq) == 1  # class_seq[0]

    def test_classification_per_token_current(self):
        """Per-token current-label targets align to class_seq with EOS masked."""
        from seqbench.tasks.classify import Classification

        task = Classification(label_source="current", granularity="per_token")
        assert task.granularity == "per_token"

        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="classification",
            task=task,
            encoder=_encoder(),
            pad_index=-1,
            granularity="per_token",
        )
        # _gs(): class_seq = [1, 2, 3, 0(EOS)]
        res = b.to_target_seq(_gs())
        assert res.granularity == "per_token"
        # EOS slot (index 3) is masked → replaced with pad_index=-1
        np.testing.assert_array_equal(res.target_seq, [1, 2, 3, -1])

    def test_classification_per_token_first(self):
        """Per-token first-label targets broadcast with EOS masked."""
        from seqbench.tasks.classify import Classification

        b = TaskTargetBuilder(
            source=TaskSource.SEQBENCH,
            task_id="classification",
            task=Classification(label_source="first", granularity="per_token"),
            encoder=_encoder(),
            pad_index=-1,
            granularity="per_token",
        )
        res = b.to_target_seq(_gs())
        assert res.granularity == "per_token"
        np.testing.assert_array_equal(res.target_seq, [1, 1, 1, -1])

    def test_classification_current_per_trial_raises(self):
        """label_source='current' with per-trial granularity must raise."""
        from seqbench.tasks.classify import Classification

        with pytest.raises(ValueError, match="current"):
            Classification(label_source="current", granularity="per_trial")


# --------------------------------------------------------------------------
# source = symseq
# --------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")
class TestSymseqSource:
    def _trial(self):
        return Trial(symbols=["A", "B", "C"], states=None, targets={}, meta={})

    def test_nstep_prediction_resolve_and_encode(self):
        task = symseq_registry.build("NStepPrediction", n=1)
        b = TaskTargetBuilder(
            source=TaskSource.SYMSEQ,
            task_id="next_token",
            task=task,
            encoder=_encoder(),
        )
        target = b.resolve(self._trial(), np.array([1, 2, 3, 0]), None)
        # symseq task runs on symbols (no EOS); last real token is masked,
        # EOS slot appended & masked -> differs from seqbench at position 2
        res = b._to_resolved(target, 4)
        np.testing.assert_array_equal(res.target_seq, [2, 3, -1, -1])

    def test_granularity_detected_per_token(self):
        task = symseq_registry.build("NStepPrediction", n=1)
        assert task.granularity == "per_token"

    def test_per_trial_intrinsic_scalarized(self):
        # a per_trial symseq Target (e.g. grammaticality) -> scalar int
        class _GramTask:
            granularity = "per_trial"

            def __call__(self, trial):
                return SymseqTarget(
                    values=True, mask=None, granularity="per_trial"
                )

        b = TaskTargetBuilder(
            source=TaskSource.SYMSEQ,
            task_id="grammaticality",
            task=_GramTask(),
            encoder=_encoder(),
            granularity="per_trial",
        )
        target = b.resolve(self._trial(), np.array([1, 2, 3, 0]), None)
        assert target.granularity == "per_trial"
        assert target.values == 1  # bool True -> int 1


# --------------------------------------------------------------------------
# from_run_cfg wiring (both sources)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")
def test_from_run_cfg_symseq_source():
    from seqbench import config as cfg_mod

    raw = {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 3}, "eos": "#"},
        "symseq": {
            "generator": {"type": "NBack", "params": {"n": 2, "alphabet_size": 3, "seq_length": 8}},
            "trial_constraints": {"length": {"min": 1, "max": 20}},
            "tasks": [{"id": "next_token", "type": "NStepPrediction", "params": {"n": 1}}],
        },
        "seqbench": {
            "mode": "online",
            "splits": {"train": 10, "test": 5},
            "storage": {"path": "/tmp/sb"},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": False, "sample_length": 20},
            "input_mapping": {"base": "one_hot"},
            "task": {"source": "symseq", "ref_id": "next_token"},
        },
    }
    run_cfg = cfg_mod.load(raw)
    b = TaskTargetBuilder.from_run_cfg(run_cfg, encoder=_encoder(), pad_index=-1)
    assert b.source == TaskSource.SYMSEQ
    assert b.granularity == "per_token"
    assert b.task_id == "next_token"


def test_from_run_cfg_seqbench_state_classification_requires_state_id_fn():
    from seqbench import config as cfg_mod

    raw = {
        "run": {"seed": 1},
        "symbol_space": {"alphabet": {"size": 3}, "eos": "#"},
        "seqbench": {
            "mode": "online",
            "splits": {"train": 10, "test": 5},
            "storage": {"path": "/tmp/sb"},
            "time_grid": {"dt": 0.1},
            "composition": {"combine_sequences": False, "sample_length": 20},
            "input_mapping": {"base": "one_hot"},
            "task": {"source": "seqbench", "id": "state_classification", "type": "StateClassification"},
        },
    }
    run_cfg = cfg_mod.load(raw)
    with pytest.raises(ValueError, match="state_id_fn"):
        TaskTargetBuilder.from_run_cfg(run_cfg, encoder=_encoder())
