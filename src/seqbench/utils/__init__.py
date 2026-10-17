# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utility functions for SeqBench.
"""

import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

import numpy as np


CACHE_KEY_VERSION = 3
GENERATED_DATASET_FORMAT_VERSION = 3


def parse_gap_duration(identifier):
    if identifier == "uniform":
        return np.random.uniform
    elif identifier == "lognormal":
        return np.random.lognormal
    else:
        raise ValueError("Unknown gap_isi identifier!")


def to_plain_data(value):
    """Convert dataclass config objects into JSON/YAML-safe builtin values."""
    if is_dataclass(value):
        return {
            field.name: to_plain_data(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def sequence_cache_key_data(run_cfg, split, split_size=None):
    """Return the normalized symbolic-generation inputs used for cache keys.

    The key intentionally excludes runtime embedding details such as
    ``seqbench.input_mapping`` and transforms. Those affect tensors produced by
    ``SeqDataset.__getitem__`` but not the generated symbolic sequence file.
    """
    if split_size is None:
        split_size = run_cfg.seqbench.split_size(split)

    symseq = run_cfg.symseq
    generator = {}
    symseq_tasks = []
    trial_params = {}
    effective_symseq_seed = run_cfg.run.seed
    trial_constraints = {}
    if symseq is not None:
        from seqbench.config import resolve_trial_params

        effective_symseq_seed = (
            symseq.seed if symseq.seed is not None else run_cfg.run.seed
        )
        generator = to_plain_data(symseq.generator)
        trial_params = to_plain_data(resolve_trial_params(symseq))
        symseq_tasks = to_plain_data(symseq.tasks)
        trial_constraints = to_plain_data(symseq.trial_constraints)
        _apply_inherited_generator_defaults(
            generator,
            symbol_space_cfg=run_cfg.symbol_space,
            seed=effective_symseq_seed,
        )

    seqbench_seed = None
    composition = {}
    task = {}
    prob_generator_type = None
    if run_cfg.seqbench is not None:
        seqbench_seed = (
            run_cfg.seqbench.seed
            if run_cfg.seqbench.seed is not None
            else run_cfg.run.seed
        )
        composition = to_plain_data(run_cfg.seqbench.composition)
        task = to_plain_data(run_cfg.seqbench.task)
        prob_generator_type = run_cfg.seqbench.prob_generator_type

    return {
        "cache_key_version": CACHE_KEY_VERSION,
        "generated_dataset_format_version": GENERATED_DATASET_FORMAT_VERSION,
        "split": split,
        "split_size": split_size,
        "run": to_plain_data(run_cfg.run),
        "symbol_space": to_plain_data(run_cfg.symbol_space),
        "symseq": {
            "seed": effective_symseq_seed,
            "generator": generator,
            "generator_trial_params": trial_params,
            "trial_constraints": trial_constraints,
            "tasks": symseq_tasks,
        },
        "seqbench": {
            "seed": seqbench_seed,
            "splits": to_plain_data(run_cfg.seqbench.splits if run_cfg.seqbench else {}),
            "composition": composition,
            "task": task,
            "prob_generator_type": prob_generator_type,
        },
    }


def build_sequence_cache_key(run_cfg, split, split_size=None):
    """Return a stable hash for generated symbolic sequence files."""
    if (
        run_cfg.seqbench is not None
        and run_cfg.seqbench.storage.cache_key is not None
    ):
        return str(run_cfg.seqbench.storage.cache_key)
    key_data = sequence_cache_key_data(run_cfg, split, split_size)
    encoded = json.dumps(key_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def dataset_cache_dir(run_cfg, split, split_size=None):
    """Return the cache directory for a generated SeqBench split."""
    if run_cfg.seqbench is None:
        raise ValueError("RunConfig must contain a 'seqbench' section")
    cache_key = build_sequence_cache_key(run_cfg, split, split_size)
    return os.path.join(run_cfg.seqbench.storage.path, cache_key)


def build_cache_manifest(
    run_cfg,
    split,
    split_size=None,
    *,
    source_config=None,
    source_config_path=None,
):
    """Return manifest metadata written next to generated sequence files."""
    if split_size is None:
        split_size = run_cfg.seqbench.split_size(split)
    cache_key = build_sequence_cache_key(run_cfg, split, split_size)
    manifest = {
        "cache_key_version": CACHE_KEY_VERSION,
        "cache_key": cache_key,
        "generated_dataset_format_version": GENERATED_DATASET_FORMAT_VERSION,
        "split": split,
        "split_size": split_size,
        "source_config": (
            source_config if source_config is not None else to_plain_data(run_cfg)
        ),
    }
    if source_config_path is not None:
        manifest["source_config_path"] = str(source_config_path)
    return manifest


def _apply_inherited_generator_defaults(generator, *, symbol_space_cfg, seed):
    """Mirror build_symseq_source inheritance so cache keys match real sources."""
    if not generator:
        return
    params = generator.setdefault("params", {})
    gen_type = generator.get("type")
    if gen_type == "ArtificialGrammar":
        mode = generator.get("mode")
        if mode in (None, "random") and symbol_space_cfg is not None:
            params.setdefault("alphabet_size", symbol_space_cfg.alphabet.size)
            if symbol_space_cfg.eos is not None:
                params.setdefault("eos", symbol_space_cfg.eos)
    elif gen_type == "NBack" and symbol_space_cfg is not None:
        if getattr(symbol_space_cfg.alphabet, "symbols", None) is not None:
            params.setdefault("alphabet", symbol_space_cfg.alphabet.resolved_symbols)
        else:
            params.setdefault("alphabet_size", symbol_space_cfg.alphabet.size)
    if seed is not None and "seed" not in params and "rng" not in params:
        params.setdefault("seed", seed)


def get_config_hash(run_cfg, dataset_size=None):
    """Deprecated: compute the legacy cache key for older callers."""
    return _get_config_hash_from_run_cfg(run_cfg, dataset_size)


def _get_config_hash_from_run_cfg(run_cfg, dataset_size=None):
    import hashlib
    from dataclasses import asdict
    from pprint import pformat

    key = {}
    if run_cfg.symseq is not None:
        gen = asdict(run_cfg.symseq.generator)
        # Normalise enum to its string value so the hash is stable across
        # Python sessions and independent of enum repr changes.
        if gen.get("mode") is not None:
            gen["mode"] = str(gen["mode"].value) if hasattr(gen["mode"], "value") else str(gen["mode"])
        key["generator"] = gen
        key["symseq_tasks"] = to_plain_data(run_cfg.symseq.tasks)
    else:
        key["generator"] = {}
        key["symseq_tasks"] = []

    length_cfg = (
        run_cfg.symseq.trial_constraints.length
        if run_cfg.symseq is not None
        and run_cfg.symseq.trial_constraints is not None
        else None
    )
    key["trial_constraints"] = (
        {"length": {"min": length_cfg.min, "max": length_cfg.max}}
        if length_cfg is not None
        else None
    )
    key["combined_seq_length"] = run_cfg.seqbench.composition.sample_length
    if dataset_size is not None:
        key["dataset_size"] = dataset_size
    key["seed"] = run_cfg.run.seed
    key["seqbench_task"] = to_plain_data(run_cfg.seqbench.task)

    return hashlib.md5(pformat(key).encode("utf-8")).hexdigest()
