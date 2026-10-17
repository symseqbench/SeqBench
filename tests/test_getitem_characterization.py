# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Characterization tests for SeqDataset.__getitem__ — freeze the exact output
of all three paths (per_trial Classification, per_token StateClassification,
predict NStepPrediction) so the per-token target-broadcast refactor is provably
behavior-preserving. These must pass BOTH before and after the refactor."""

import numpy as np
import pytest
import torch

from tests.test_pipeline_tasks import (  # reuse the existing harness
    HAS_SYMSEQ,
    _build_dataset,
    _raw,
    _SYMSEQ_NEXT,
)

pytestmark = pytest.mark.skipif(not HAS_SYMSEQ, reason="symseq not available")

# Deterministic gap: uniform(low=high=0.5) always returns 0.5, so the gap is
# RNG-independent. With dt=0.1 each element gets round(0.5/0.1)=5 gap timesteps
# on top of the one_hot stimulus footprint (1) -> 6 timesteps per element.
_GAP_DELAY_STEPS = 5
# one_hot stimulus footprint: round(duration/dt); duration defaults to dt -> 1.
_BASE_STEPS = 1


def _with_gap(raw, *, start=0):
    """Inject a deterministic gap profile into a raw config (in place)."""
    raw["seqbench"]["time_grid"] = {"dt": 0.1}
    raw["seqbench"]["composition"]["gap_profile"] = {
        "start": start,
        "duration": {"dist": "uniform", "params": {"low": 0.5, "high": 0.5}},
        "add_nongramm_gap": False,
    }
    return raw


def _expected_broadcast(target_seq, data_t=1, delay_dur=0):
    """Reference per-timestep target: the exact current formula."""
    return torch.cat(
        [torch.ones(data_t + delay_dur) * float(v) for v in target_seq]
    )


def test_getitem_predict_path_characterization():
    ds = _build_dataset(_raw(**_SYMSEQ_NEXT, combine=False))
    assert ds._wants_target_probs is True
    assert ds._gap_profile is None  # gap-free config: delay_dur == 0

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 5
    data, target, class_seq, target_probs, gap_mask = item

    # data: one_hot base, stimulus footprint == 1 -> one timestep per element
    assert data.shape[0] == len(sample.class_seq)

    # target: exact temporal broadcast of sample.target_seq (the refactor invariant)
    assert target.dtype == torch.float32
    expected_target = _expected_broadcast(sample.target_seq)
    assert torch.equal(target, expected_target)
    assert data.shape[0] == target.shape[0] == target_probs.shape[0]

    # class_seq: unreduced-state ids, one per element (debug channel)
    expected_cls = [
        ds.target_prob_generator.unred_state_to_id(s) for s in sample.state_seq
    ]
    assert list(class_seq) == expected_cls

    # target_probs: (T, C); gap-free -> repeat per element
    assert target_probs.shape[0] == data.shape[0]

    # gap_mask: gap-free predict path -> 1-D, data_t==1 -> all ones
    assert gap_mask.dim() == 1
    assert gap_mask.shape[0] == data.shape[0]
    assert torch.equal(gap_mask, torch.ones(data.shape[0]))


def test_getitem_state_classification_path_characterization():
    ds = _build_dataset(_raw(task={"source": "seqbench", "id": "state_classification", "type": "StateClassification"}))
    assert ds.per_token_classify is True
    assert ds._gap_profile is None

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 4
    data, target, class_seq, gap_mask = item

    assert target.dtype == torch.float32
    expected_target = _expected_broadcast(sample.target_seq)
    assert torch.equal(target, expected_target)
    assert data.shape[0] == target.shape[0]

    # class_seq is sample.class_seq (ndarray) on this path
    np.testing.assert_array_equal(np.asarray(class_seq), sample.class_seq)

    # gap_mask: per_token_classify -> 2-D padded (one row per element)
    assert gap_mask.dim() == 2
    assert gap_mask.shape[0] == len(sample.class_seq)


def test_getitem_per_trial_classification_path_characterization():
    ds = _build_dataset(
        _raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                   "params": {"label_source": "first"}}, combine=False)
    )
    assert ds.is_per_trial is True
    assert ds._gap_profile is None

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 4
    data, label, class_seq, gap_mask = item

    # per_trial: single scalar label, NOT broadcast
    assert isinstance(label, int)
    assert label == int(sample.target_seq)

    np.testing.assert_array_equal(np.asarray(class_seq), sample.class_seq)
    assert gap_mask.dim() == 2
    assert gap_mask.shape[0] == len(sample.class_seq)
    assert data.shape[0] >= len(sample.class_seq)  # >= because of any gap padding


# ---------------------------------------------------------------------------
# Gap scenarios: each element spans _BASE_STEPS + _GAP_DELAY_STEPS timesteps,
# so the target broadcast must repeat each per-element value over that wider
# span. These exercise the delay_dur > 0 branch of _expand_per_token_target.
# ---------------------------------------------------------------------------


def test_getitem_predict_path_with_gaps():
    ds = _build_dataset(_with_gap(_raw(**_SYMSEQ_NEXT, combine=False)))
    assert ds._wants_target_probs is True
    assert ds._gap_profile is not None
    assert ds.base_dataset.n_steps == _BASE_STEPS  # one_hot -> data_t == 1

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 5
    data, target, class_seq, target_probs, gap_mask = item

    dur = _BASE_STEPS + _GAP_DELAY_STEPS  # 6 timesteps per element
    assert data.shape[0] == len(sample.target_seq) * dur

    assert target.dtype == torch.float32
    expected_target = _expected_broadcast(
        sample.target_seq, data_t=_BASE_STEPS, delay_dur=_GAP_DELAY_STEPS
    )
    assert torch.equal(target, expected_target)
    assert data.shape[0] == target.shape[0] == target_probs.shape[0]

    # gap_mask stays 1-D and length-aligned to the timestep axis in predict mode
    assert gap_mask.dim() == 1
    assert gap_mask.shape[0] == data.shape[0]


def test_getitem_state_classification_path_with_gaps():
    ds = _build_dataset(
        _with_gap(_raw(task={"source": "seqbench", "id": "state_classification", "type": "StateClassification"}))
    )
    assert ds.per_token_classify is True
    assert ds._gap_profile is not None
    assert ds.base_dataset.n_steps == _BASE_STEPS

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 4
    data, target, class_seq, gap_mask = item

    dur = _BASE_STEPS + _GAP_DELAY_STEPS
    assert data.shape[0] == len(sample.target_seq) * dur

    assert target.dtype == torch.float32
    expected_target = _expected_broadcast(
        sample.target_seq, data_t=_BASE_STEPS, delay_dur=_GAP_DELAY_STEPS
    )
    assert torch.equal(target, expected_target)
    assert data.shape[0] == target.shape[0]

    np.testing.assert_array_equal(np.asarray(class_seq), sample.class_seq)
    assert gap_mask.dim() == 2
    assert gap_mask.shape[0] == len(sample.class_seq)


def test_getitem_per_trial_classification_path_with_gaps():
    ds = _build_dataset(
        _with_gap(
            _raw(task={"source": "seqbench", "id": "classification", "type": "Classification",
                       "params": {"label_source": "first"}}, combine=False)
        )
    )
    assert ds.is_per_trial is True
    assert ds._gap_profile is not None
    assert ds.base_dataset.n_steps == _BASE_STEPS

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    item = ds[0]
    assert len(item) == 4
    data, label, class_seq, gap_mask = item

    # per_trial: single scalar label, NOT broadcast — gaps only widen the input
    assert isinstance(label, int)
    assert label == int(sample.target_seq)

    dur = _BASE_STEPS + _GAP_DELAY_STEPS
    assert data.shape[0] == len(sample.class_seq) * dur
    np.testing.assert_array_equal(np.asarray(class_seq), sample.class_seq)
    assert gap_mask.dim() == 2
    assert gap_mask.shape[0] == len(sample.class_seq)


# ---------------------------------------------------------------------------
# Single real-time axis without time-creating transforms: stimulus and gap both
# convert through the resolved final grid.
# ---------------------------------------------------------------------------


def _raw_onehot(*, dt, stim_s, gap_s):
    """Predict-path one_hot config with explicit dt and durations (seconds)."""
    raw = _raw(**_SYMSEQ_NEXT, combine=False)
    raw["seqbench"]["time_grid"] = {"dt": dt}
    raw["seqbench"]["input_mapping"]["base_params"] = {"duration": stim_s}
    raw["seqbench"]["composition"]["gap_profile"] = {
        "start": 0,
        "duration": {"dist": "uniform", "params": {"low": gap_s, "high": gap_s}},
        "add_nongramm_gap": False,
    }
    return raw


def test_onehot_stimulus_footprint_from_duration():
    # duration=1.0s at dt=0.1 -> 10 stimulus steps; gap 5.0s -> 50 steps.
    ds = _build_dataset(_raw_onehot(dt=0.1, stim_s=1.0, gap_s=5.0))
    assert ds.base_dataset.n_steps == 10

    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    data = ds[0][0]
    assert data.shape[0] == len(sample.target_seq) * (10 + 50)


def test_onehot_duration_defaults_to_single_step():
    # No duration in base_params -> defaults to dt -> exactly one step.
    raw = _raw(**_SYMSEQ_NEXT, combine=False)
    raw["seqbench"]["time_grid"] = {"dt": 0.1}
    ds = _build_dataset(raw)
    assert ds.base_dataset.n_steps == 1


def test_grid_is_scale_invariant_in_dt():
    # Halving dt doubles both stimulus and gap step counts; the ratio is fixed.
    ds_coarse = _build_dataset(_raw_onehot(dt=0.1, stim_s=1.0, gap_s=5.0))
    ds_fine = _build_dataset(_raw_onehot(dt=0.05, stim_s=1.0, gap_s=5.0))

    assert ds_fine.base_dataset.n_steps == 2 * ds_coarse.base_dataset.n_steps

    sample_c = ds_coarse.gensample_to_sample(
        ds_coarse.gs.generate(0, compute_length=False)
    )
    sample_f = ds_fine.gensample_to_sample(
        ds_fine.gs.generate(0, compute_length=False)
    )
    # Same generator seed -> same symbolic sequence length.
    assert len(sample_c.target_seq) == len(sample_f.target_seq)
    assert ds_fine[0][0].shape[0] == 2 * ds_coarse[0][0].shape[0]


def test_gap_duration_not_rounded_to_one_decimal():
    ds = _build_dataset(_raw_onehot(dt=0.001, stim_s=0.001, gap_s=0.015))
    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    data = ds[0][0]
    assert data.shape[0] == len(sample.target_seq) * (1 + 15)


def test_nongrammatical_gap_filler_uses_transform_pipeline():
    raw = _raw(**_SYMSEQ_NEXT, combine=False)
    raw["seqbench"]["time_grid"] = {"dt": 0.001}
    raw["seqbench"]["input_mapping"]["base_params"] = {}
    raw["seqbench"]["input_mapping"]["transforms"] = [
        {
            "type": "TemporalUnfold",
            "params": {
                "kernel_spec": {"shape": "box", "params": {"width": 0.001}},
                "out_dt": 0.001,
                "duration": 0.010,
            },
        }
    ]
    raw["seqbench"]["composition"]["gap_profile"] = {
        "start": 0,
        "duration": {"dist": "uniform", "params": {"low": 0.015, "high": 0.015}},
        "add_nongramm_gap": True,
    }

    ds = _build_dataset(raw)
    sample = ds.gensample_to_sample(ds.gs.generate(0, compute_length=False))
    data = ds[0][0]
    assert data.shape[0] == len(sample.target_seq) * (10 + 15)
