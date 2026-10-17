# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""High-level helpers for constructing SeqBench datasets and dataloaders."""

from __future__ import annotations

from os import PathLike
from typing import Any

import yaml
from torch.utils.data import DataLoader

from seqbench import config as cfg_mod
from seqbench.seq_dataset import SeqDataset, make_pad_sequence
from seqbench.utils import to_plain_data

ConfigLike = str | PathLike[str] | dict[str, Any] | cfg_mod.RunConfig


def build_dataset(
    config: ConfigLike,
    *,
    split: str = "train",
    dataset_size: int | None = None,
    pad_index: int = -1,
    **dataset_kwargs,
) -> SeqDataset:
    """Build a configured :class:`seqbench.seq_dataset.SeqDataset`.

    Args:
        config: YAML path, raw config dictionary, or loaded
            :class:`seqbench.config.RunConfig`.
        split: Dataset split to build.
        dataset_size: Optional sample-count override for ``split``.
        pad_index: Padding label used by the dataset and collate function.
        dataset_kwargs: Additional keyword arguments forwarded to
            :class:`seqbench.seq_dataset.SeqDataset`.

    Returns:
        Fully configured ``SeqDataset`` instance.
    """
    run_cfg, config_path, config_snapshot = _load_config(config)
    return SeqDataset(
        config=run_cfg,
        split=split,
        dataset_size=dataset_size,
        pad_index=pad_index,
        config_file_path=config_path,
        config_snapshot=config_snapshot,
        **dataset_kwargs,
    )


def build_dataloader(
    config: ConfigLike,
    *,
    split: str = "train",
    dataset_size: int | None = None,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pad_index: int = -1,
    **kwargs,
) -> DataLoader:
    """Build a PyTorch ``DataLoader`` for a configured SeqBench split.

    Args:
        config: YAML path, raw config dictionary, or loaded
            :class:`seqbench.config.RunConfig`.
        split: Dataset split to build.
        dataset_size: Optional sample-count override for ``split``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle batches. Defaults to ``True`` for
            ``split="train"`` and ``False`` otherwise.
        num_workers: Number of PyTorch dataloader workers.
        pad_index: Padding label used by the dataset and collate function.
        kwargs: Additional PyTorch ``DataLoader`` keyword arguments. Prefix a
            key with ``dataset__`` to forward it to ``build_dataset`` instead.

    Returns:
        PyTorch ``DataLoader`` instance.
    """
    dataset_kwargs = {}
    dataloader_kwargs = {}
    for key, value in kwargs.items():
        if key.startswith("dataset__"):
            dataset_kwargs[key[len("dataset__") :]] = value
        else:
            dataloader_kwargs[key] = value

    dataset = build_dataset(
        config,
        split=split,
        dataset_size=dataset_size,
        pad_index=pad_index,
        **dataset_kwargs,
    )
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_pad_sequence(dataset, pad_index=pad_index),
        num_workers=num_workers,
        **dataloader_kwargs,
    )


def _load_config(config):
    if isinstance(config, (str, PathLike)):
        config_path = str(config)
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"config file {config_path} is empty")
        run_cfg = cfg_mod.load(raw)
        return run_cfg, config_path, to_plain_data(run_cfg)
    if isinstance(config, dict):
        run_cfg = cfg_mod.load(config)
        return run_cfg, None, to_plain_data(run_cfg)
    return config, None, to_plain_data(config)
