# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Tests for the SeqBench command-line interface."""

from __future__ import annotations

from pathlib import Path

import torch
from typer.testing import CliRunner

from seqbench import cli


runner = CliRunner()
EXAMPLE_CONFIG = Path("examples/configs/onehot_raw.yaml")


def test_validate_example_config_succeeds():
    result = runner.invoke(cli.app, ["validate", str(EXAMPLE_CONFIG)])

    assert result.exit_code == 0
    assert "Valid config" in result.stdout
    assert "Input mapping: one_hot" in result.stdout
    assert "Task source: symseq" in result.stdout


def test_create_split_calls_build_dataset_once(monkeypatch):
    calls = []

    def fake_build_dataset(config, *, split):
        calls.append((config, split))

    monkeypatch.setattr(cli, "build_dataset", fake_build_dataset)

    result = runner.invoke(cli.app, ["create", str(EXAMPLE_CONFIG), "--split", "train"])

    assert result.exit_code == 0
    assert calls == [(EXAMPLE_CONFIG, "train")]


def test_create_defaults_to_all_configured_splits(monkeypatch):
    calls = []

    def fake_build_dataset(config, *, split):
        calls.append((config, split))

    monkeypatch.setattr(cli, "build_dataset", fake_build_dataset)

    result = runner.invoke(cli.app, ["create", str(EXAMPLE_CONFIG)])

    assert result.exit_code == 0
    assert calls == [(EXAMPLE_CONFIG, "train"), (EXAMPLE_CONFIG, "test")]


def test_inspect_batch_prints_actual_batch_keys(monkeypatch):
    class FakeDataset:
        split = "train"
        is_per_trial = False
        per_token_classify = False

        def __len__(self):
            return 4

        def returns_target_probs(self):
            return True

    class FakeLoader:
        dataset = FakeDataset()

        def __iter__(self):
            batch = {
                "data": torch.zeros(2, 3, 4),
                "labels": torch.ones(2, 3, dtype=torch.long),
                "lens": torch.tensor([3, 3]),
                "target_probs": torch.full((2, 3, 4), 0.25),
                "gap_mask": torch.ones(2, 3),
                "debug_class_seq": torch.arange(6).reshape(2, 3),
            }
            return iter([batch])

    def fake_build_dataloader(*args, **kwargs):
        return FakeLoader()

    monkeypatch.setattr(cli, "build_dataloader", fake_build_dataloader)

    result = runner.invoke(cli.app, ["inspect-batch", str(EXAMPLE_CONFIG)])

    assert result.exit_code == 0
    assert "Dataset length: 4" in result.stdout
    assert "- target_probs: shape=(2, 3, 4)" in result.stdout
    assert "- gap_mask: shape=(2, 3)" in result.stdout
    assert "- debug_class_seq: shape=(2, 3)" in result.stdout


def test_invalid_config_exits_cleanly(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("run: {}\n")

    result = runner.invoke(cli.app, ["validate", str(config_path)])

    assert result.exit_code != 0
    assert "missing required keys" in result.output


def test_create_dataset_wrapper_delegates(monkeypatch):
    from seqbench import create_dataset

    calls = []

    def fake_create(config, split=None):
        calls.append((config, split))

    monkeypatch.setattr(create_dataset, "create", fake_create)

    create_dataset.main(["--config", str(EXAMPLE_CONFIG), "--split", "train"])

    assert calls == [(EXAMPLE_CONFIG, "train")]


def test_read_dataset_wrapper_delegates(monkeypatch):
    from seqbench import read_dataset

    calls = []

    def fake_inspect_batch(config, split="train", batch_size=2, num_workers=0):
        calls.append((config, split, batch_size, num_workers))

    monkeypatch.setattr(read_dataset, "inspect_batch", fake_inspect_batch)

    read_dataset.main(
        [
            "--config",
            str(EXAMPLE_CONFIG),
            "--split",
            "test",
            "--batch-size",
            "3",
            "--num-workers",
            "1",
        ]
    )

    assert calls == [(EXAMPLE_CONFIG, "test", 3, 1)]
