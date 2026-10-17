# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""Command-line interface for SeqBench dataset workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import typer

from seqbench import build_dataloader, build_dataset, config as cfg_mod

app = typer.Typer(no_args_is_help=True)


def main() -> None:
    """Run the SeqBench command-line app."""
    app()


@app.command()
def validate(config: Path) -> None:
    """Validate a SeqBench config and print its execution summary."""
    run_cfg = _load_run_config(config)
    _require_seqbench(run_cfg)

    task = run_cfg.seqbench.task
    typer.echo(f"Valid config: {config}")
    typer.echo(f"Mode: {_enum_value(run_cfg.operating_mode)}")
    typer.echo(f"Splits: {', '.join(run_cfg.seqbench.splits)}")
    typer.echo(f"Storage path: {run_cfg.seqbench.storage.path}")
    typer.echo(f"Input mapping: {run_cfg.seqbench.input_mapping.base}")
    typer.echo(f"Task source: {_enum_value(task.source)}")
    task_id = task.ref_id if task.source == cfg_mod.TaskSource.SYMSEQ else task.id
    typer.echo(f"Task: {task_id}")
    if task.type is not None:
        typer.echo(f"Task type: {task.type}")


@app.command("list-tasks")
def list_tasks() -> None:
    """List registered SeqBench task types."""
    from seqbench.tasks.registry import registered_types

    for type_name in registered_types():
        typer.echo(type_name)


@app.command()
def create(
    config: Path,
    split: str | None = typer.Option(None, "--split", help="Build only this split."),
) -> None:
    """Materialize configured SeqBench datasets."""
    run_cfg = _load_run_config(config)
    _require_seqbench(run_cfg)

    splits = _selected_splits(run_cfg, split)
    for split_name in splits:
        typer.echo(f"Building split: {split_name}")
        build_dataset(config, split=split_name)
    typer.echo(f"Built {len(splits)} split(s).")


@app.command("inspect-batch")
def inspect_batch(
    config: Path,
    split: str = typer.Option("train", "--split"),
    batch_size: int = typer.Option(2, "--batch-size", min=1),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
) -> None:
    """Print the actual batch keys, shapes, and dtypes for one dataloader batch."""
    loader = build_dataloader(
        config,
        split=split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batch = next(iter(loader))

    dataset = loader.dataset
    typer.echo(f"Split: {getattr(dataset, 'split', split)}")
    try:
        typer.echo(f"Dataset length: {len(dataset)}")
    except TypeError:
        typer.echo("Dataset length: unknown")
    for flag in _dataset_flags(dataset):
        typer.echo(flag)

    typer.echo("Batch:")
    for key, value in batch.items():
        typer.echo(f"- {key}: {_value_summary(value)}")


@app.command("show-sample")
def show_sample(
    config: Path,
    split: str = typer.Option("train", "--split"),
    sample: int = typer.Option(0, "--sample", min=0),
    out_dir: Path = typer.Option(Path("img"), "--out-dir"),
) -> None:
    """Render one sample from a configured dataloader."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError as exc:
        raise typer.BadParameter(
            "show-sample requires matplotlib and numpy; install seqbench[viz]."
        ) from exc

    loader = build_dataloader(
        config,
        split=split,
        batch_size=max(sample + 1, 1),
        shuffle=True,
        num_workers=0,
    )
    dataset = loader.dataset
    batch = next(iter(loader))

    if "debug_class_seq" not in batch:
        raise typer.BadParameter(
            "show-sample requires batches with 'debug_class_seq'. "
            "Use inspect-batch to see this config's output keys."
        )

    data = batch["data"]
    labels = batch["labels"]
    lengths = batch["lens"]
    debug_class_seq = batch["debug_class_seq"]
    per_token_classify = getattr(dataset, "per_token_classify", False)
    target_probs = None if per_token_classify else batch.get("target_probs")
    input_mapping = getattr(dataset, "input_mapping", None)
    input_base = getattr(input_mapping, "base", "input")

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "classify" if per_token_classify else "prob"
    sample_path = out_dir / f"show_sample_{input_base}_{suffix}.png"
    stimulus_path = out_dir / f"stimulus_{input_base}.png"

    tensor = data[sample]
    if len(tensor.shape) >= 3:
        tensor = tensor.squeeze()

    sequence = debug_class_seq[sample].detach().cpu().numpy()
    sample_length = int(lengths[sample])

    tpg = getattr(dataset, "target_prob_generator", None)
    if tpg is not None and hasattr(tpg, "id_to_red_state"):
        sym_seq = [tpg.id_to_red_state(int(c)) for c in sequence[:sample_length]]
    else:
        sym_seq = sequence[:sample_length].tolist()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True)
    ax1.imshow(np.transpose(tensor.detach().cpu().numpy(), [1, 0]), aspect="auto", origin="lower", cmap="Greys")
    ax1.set_ylabel("Inp. ch.")
    ax1.set_title(f"Inputs={sym_seq}")

    if target_probs is not None:
        target_img = np.transpose(target_probs[sample].detach().cpu().numpy(), [1, 0])
    else:
        target_img = torch.nn.functional.one_hot(labels[sample, :sample_length].long())
        target_img = torch.swapaxes(target_img, 0, 1).detach().cpu().numpy()
    ax2.imshow(target_img, aspect="auto", origin="lower", cmap="Greys")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel("Target class")
    ax2.set_xlabel("Time steps")
    fig.tight_layout()
    fig.savefig(sample_path, dpi=600)
    plt.close(fig)

    fig = plt.figure(figsize=(5, 2.5))
    plt.imshow(np.transpose(tensor.detach().cpu().numpy(), [1, 0]), aspect="auto", origin="lower", cmap="Greys")
    plt.ylabel("Inp. ch.")
    plt.xlabel("Time steps")
    fig.tight_layout()
    fig.savefig(stimulus_path, dpi=600)
    plt.close(fig)

    typer.echo(f"Saved {sample_path}")
    typer.echo(f"Saved {stimulus_path}")


def _load_run_config(config: Path):
    try:
        return cfg_mod.load(config)
    except Exception as exc:
        raise typer.BadParameter(str(exc), param_hint="CONFIG") from exc


def _require_seqbench(run_cfg) -> None:
    if run_cfg.seqbench is None:
        raise typer.BadParameter("config must contain a 'seqbench' section", param_hint="CONFIG")


def _selected_splits(run_cfg, split: str | None) -> list[str]:
    available = list(run_cfg.seqbench.splits)
    if split is None:
        return available
    if split not in run_cfg.seqbench.splits:
        raise typer.BadParameter(
            f"unknown split {split!r}; available: {available}",
            param_hint="--split",
        )
    return [split]


def _dataset_flags(dataset: Any) -> list[str]:
    flags = []
    for attr, label in (
        ("is_per_trial", "Per-trial"),
        ("per_token_classify", "Per-token classify"),
        ("returns_target_probs", "Returns target probs"),
    ):
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            if callable(value):
                value = value()
            flags.append(f"{label}: {value}")
    return flags


def _value_summary(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    parts = []
    if shape is not None:
        parts.append(f"shape={tuple(shape)}")
    if dtype is not None:
        parts.append(f"dtype={dtype}")
    if torch.is_tensor(value) and value.numel() > 0 and value.dtype != torch.bool:
        parts.append(f"min={value.min().item():.4g}")
        parts.append(f"max={value.max().item():.4g}")
    if not parts:
        parts.append(type(value).__name__)
    return ", ".join(parts)


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


if __name__ == "__main__":
    main()
