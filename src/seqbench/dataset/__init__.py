# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Dataset Module for SeqBench

This module provides dataset classes and factory functions for creating base datasets
from configuration files. Supported datasets include:
- SHD (Spiking Heidelberg Digits)
- SSC (Spiking Speech Commands)
- GSC (Google Speech Commands)
- One-hot synthetic datasets
- Tonic datasets (DVSGesture, NMNIST, etc.)
"""

from seqbench.transforms.base import TimeGrid


_TIME_CREATING_TRANSFORMS = {"TemporalUnfold", "MFCC", "LogMel"}


def has_time_creating_transform(input_mapping):
    """Return True when config transforms create a new leading time axis."""
    for transform in input_mapping.transforms:
        behavior = transform.get("time_behavior")
        if isinstance(behavior, str):
            if behavior == "create":
                return True
        elif isinstance(behavior, dict) and behavior.get("kind") == "create":
            return True

        transform_type = transform["type"].split(".")[-1]
        params = transform.get("params") or {}
        if transform_type in _TIME_CREATING_TRANSFORMS:
            return True
        if transform_type == "PoissonEncoding" and not params.get("temporal", False):
            return True
    return False


def initial_time_grid_from_config(input_mapping, final_dt=None):
    """Return the base dataset's native time grid, if it has one."""
    bp = input_mapping.base_params
    if input_mapping.base in {"shd", "ssc"}:
        return TimeGrid(dt=bp["max_time"] / bp["nb_steps"])
    if input_mapping.base == "gsc" and not bp.get("return_raw", False):
        # SpeechCommands computes MFCCs internally with hop_length=160 and
        # sample_rate=16000 when return_raw=False.
        return TimeGrid(dt=160 / 16000)
    if input_mapping.base == "one_hot" and final_dt is not None:
        if not has_time_creating_transform(input_mapping):
            return TimeGrid(dt=final_dt)
    return None


def create_base_dataset_from_config(input_mapping, split, final_dt=None, **kwargs):
    """
    Factory function to create a base dataset from an
    :class:`~seqbench.config.InputMappingCfg`.

    Args:
        input_mapping: :class:`~seqbench.config.InputMappingCfg` instance.
        split: Dataset split ('train' or 'test').
        final_dt: Final grid resolution in seconds. Used to convert
            ``base_params.duration`` (seconds) into a stimulus step count for
            one_hot.
        **kwargs: Additional keyword arguments (e.g. ``alphabet_size`` for one_hot).
    """
    bp = input_mapping.base_params  # plain dict of base-specific parameters

    if input_mapping.base == "shd":
        from seqbench.dataset.shd_ssc import SpikingDataset

        return SpikingDataset(
            "shd",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if input_mapping.base == "ssc":
        from seqbench.dataset.shd_ssc import SpikingDataset

        return SpikingDataset(
            "ssc",
            bp["base_dataset_path"],
            split,
            nb_steps=bp["nb_steps"],
            max_time=bp["max_time"],
            num_bins=bp.get("num_bins", 1),
        )
    if input_mapping.base == "gsc":
        from seqbench.dataset.speech_commands import SpeechCommands

        split = "training" if split == "train" else "testing"
        return SpeechCommands(
            data_folder=bp["base_dataset_path"],
            split=split,
            return_raw=bp.get("return_raw", False),
        )
    if input_mapping.base == "one_hot":
        from seqbench.dataset.synthetic import OneHot

        if "alphabet_size" not in kwargs:
            raise ValueError("alphabet_size must be provided for one_hot dataset!")
        creates_time = has_time_creating_transform(input_mapping)
        if creates_time:
            if "duration" in bp:
                raise ValueError(
                    "one_hot.base_params.duration cannot be used with a "
                    "time-creating transform; put the duration on the transform"
                )
            n_steps = 1
        elif final_dt is not None:
            # Stimulus footprint = round(duration / final_dt) rows. ``duration``
            # defaults to ``final_dt`` so an unspecified stimulus is one row.
            duration = bp.get("duration", final_dt)
            n_steps = max(1, round(duration / final_dt))
        else:
            n_steps = 1
        return OneHot(kwargs["alphabet_size"] + 1, n_steps=n_steps)
    from seqbench.dataset.tonic_wrapper import TonicDatasetWrapper, TONIC_DATASET_REGISTRY

    if input_mapping.base.lower() in TONIC_DATASET_REGISTRY:
        canonical_name = TONIC_DATASET_REGISTRY[input_mapping.base.lower()]
        return TonicDatasetWrapper(
            dataset_name=canonical_name,
            root=bp["base_dataset_path"],
            split=split,
            **{k: v for k, v in bp.items() if k != "base_dataset_path"},
        )
    raise ValueError(f"Unknown input encoding: {input_mapping.base!r}")
