# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Tests for the SeqBench dataset creation and DataLoader functionality.
"""

import pytest
import numpy as np
import random
import torch
import os

from torch.utils.data import DataLoader

try:
    from seqbench.sources import build_symseq_source
    import symseq  # noqa: F401  # verify availability
    HAS_SYMSEQ = True
except ImportError:
    HAS_SYMSEQ = False

from seqbench import config as cfg_mod
from seqbench.seq_dataset import SeqDataset, make_pad_sequence
from seqbench.utils import get_config_hash
from seqbench.dataset import create_base_dataset_from_config, initial_time_grid_from_config
from seqbench.transforms import compose_transforms_from_config


@pytest.fixture(scope="class")
def dataset_setup(tmp_path_factory):
    """Set up dataset for testing."""
    config_path = os.path.join(os.path.dirname(__file__), 'onehot_raw.yaml')

    if not os.path.exists(config_path):
        pytest.skip(f"Config file {config_path} not found")

    if not HAS_SYMSEQ:
        pytest.skip("symseq not available")

    run_cfg = cfg_mod.load(config_path)

    seed = run_cfg.run.seed
    dataset_size = min(int(run_cfg.seqbench.splits["train"]), 20)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    config_hash = get_config_hash(run_cfg, dataset_size=dataset_size)
    dataset_root = str(tmp_path_factory.mktemp("seqbench-data") / config_hash)

    dataset = SeqDataset(
        config=run_cfg,
        generator=source,
        base_dataset=base_dataset,
        is_train=True,
        dataset_size=dataset_size,
        config_file_path=config_path,
        pad_index=-1,
        dataset_root=dataset_root,
        transform=transforms,
        initial_time_grid=initial_time_grid,
    )

    return dataset, dataset_size, dataset.per_token_classify


class TestDataset:
    """Tests for dataset creation and basic functionality."""

    def test_dataset_creation(self, dataset_setup):
        """Test that dataset is created successfully."""
        dataset, dataset_size, do_classify = dataset_setup

        assert dataset is not None
        assert len(dataset) > 0
        assert len(dataset) == dataset_size

    def test_dataset_getitem(self, dataset_setup):
        """Test that __getitem__ returns a valid sample."""
        dataset, dataset_size, do_classify = dataset_setup

        sample = dataset[0]
        assert sample is not None
        # onehot_raw.yaml uses NStepPrediction over a grammar -> predict mode,
        # so __getitem__ returns 5 fields (extra target_probs).
        assert len(sample) == 5  # data, target, class_seq, target_probs, gap_mask

        data, target = sample[0], sample[1]
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape[0] == target.shape[0]


class TestDataLoader:
    """Tests for DataLoader integration."""

    def test_dataloader_iteration(self, dataset_setup):
        """Test that DataLoader can iterate over batches."""
        dataset, dataset_size, do_classify = dataset_setup

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=make_pad_sequence(dataset, pad_index=-1),
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert isinstance(batch, dict)
        assert 'data' in batch
        assert 'labels' in batch
        assert 'mask' in batch
        assert 'lens' in batch

    def test_dataloader_batch_shapes(self, dataset_setup):
        """Test that DataLoader batches have correct shapes."""
        dataset, dataset_size, do_classify = dataset_setup

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=make_pad_sequence(dataset, pad_index=-1),
            num_workers=0,
        )

        batch = next(iter(dataloader))

        assert batch['data'].dim() == 3  # (batch, time, features)
        assert batch['data'].shape[0] <= 4
        assert batch['labels'].shape[0] == batch['data'].shape[0]
        assert batch['labels'].shape[1] == batch['data'].shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
