# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Typed schema for the unified symseq + SeqBench YAML config.

Parses a YAML file (or pre-loaded dict) into a :class:`RunConfig` of nested
dataclasses, validating structural constraints by hand (no pydantic). The
schema is documented in detail in
``SeqBench/examples/configs/_schema_reference.yaml``.

This module is intentionally side-effect-free: it does NOT perform
``symbol_space → symseq`` inheritance, does NOT instantiate generators, and does
NOT touch disk. Those steps belong to the loader that consumes a parsed
:class:`RunConfig` (next step).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

# ----------------------------- enums ------------------------------------------


class OperatingMode(str, Enum):
    SYMSEQ_ONLY = "symseq_only"
    SYMSEQ_PLUS_SEQBENCH = "symseq_plus_seqbench"
    SEQBENCH_STANDALONE = "seqbench_standalone"


class AGMode(str, Enum):
    RANDOM = "random"
    PRESET = "preset"
    CUSTOM = "custom"


class TaskSource(str, Enum):
    SYMSEQ = "symseq"
    SEQBENCH = "seqbench"


# ----------------------------- run / symbol space -----------------------------


@dataclass
class RunMetaCfg:
    seed: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError(f"run.seed must be int, got {self.seed!r}")
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError(f"run.name must be a string or null, got {self.name!r}")


@dataclass
class SymbolAlphabetCfg:
    size: int
    symbols: list[str] | None = None

    def __post_init__(self) -> None:
        if self.symbols is not None:
            if not isinstance(self.symbols, list) or not all(
                isinstance(s, str) for s in self.symbols
            ):
                raise ValueError("symbol_space.alphabet.symbols must be a list of strings")
            if len(set(self.symbols)) != len(self.symbols):
                raise ValueError("symbol_space.alphabet.symbols must be unique")
            if self.size != len(self.symbols):
                raise ValueError(
                    "symbol_space.alphabet.size must match len(symbols), "
                    f"got size={self.size!r}, len(symbols)={len(self.symbols)}"
                )
        if not isinstance(self.size, int) or self.size < 1:
            raise ValueError(
                f"symbol_space.alphabet.size must be a positive int, got {self.size!r}"
            )

    @property
    def resolved_symbols(self) -> list[str]:
        if self.symbols is not None:
            return list(self.symbols)
        return [str(i) for i in range(self.size)]


@dataclass
class SymbolSpaceCfg:
    alphabet: SymbolAlphabetCfg
    eos: str | None = "#"

    def __post_init__(self) -> None:
        if self.eos is not None and not isinstance(self.eos, str):
            raise ValueError(f"symbol_space.eos must be a string or null, got {self.eos!r}")
        if self.eos is not None and self.eos in self.alphabet.resolved_symbols:
            raise ValueError(
                f"symbol_space.eos {self.eos!r} must not appear in alphabet symbols"
            )

    @property
    def symbols(self) -> list[str]:
        return self.alphabet.resolved_symbols


@dataclass
class LengthConstraintCfg:
    min: int
    max: int

    def __post_init__(self) -> None:
        if not (isinstance(self.min, int) and self.min >= 0):
            raise ValueError(
                f"trial_constraints.length.min must be a non-negative int, got {self.min!r}"
            )
        if not (isinstance(self.max, int) and self.max >= self.min):
            raise ValueError(
                "trial_constraints.length.max must be an int >= "
                f"min ({self.min}), got {self.max!r}"
            )


@dataclass
class TrialConstraintsCfg:
    length: LengthConstraintCfg | None = None


# ----------------------------- symseq -----------------------------------------


@dataclass
class GeneratorCfg:
    type: str
    mode: AGMode | None = None        # AG-only; ignored for other types
    params: dict[str, Any] = field(default_factory=dict)
    trial_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(f"generator.type must be a non-empty string, got {self.type!r}")
        if self.mode is not None and not isinstance(self.mode, AGMode):
            # Allow string -> enum coercion at parse time.
            self.mode = AGMode(self.mode)
        # AG-mode default is "random"; non-AG mode must be None.
        if self.type == "ArtificialGrammar" and self.mode is None:
            self.mode = AGMode.RANDOM


@dataclass
class SymseqTaskEntry:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(
                f"symseq.tasks[*].id must be a non-empty string, got {self.id!r}"
            )
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(
                f"symseq.tasks[*].type must be a non-empty string, got {self.type!r}"
            )
        if not isinstance(self.params, dict):
            raise ValueError(
                f"symseq.tasks[*].params must be a mapping, got {self.params!r}"
            )


@dataclass
class SymseqStorageCfg:
    path: str
    format: str = "pickle"

    def __post_init__(self) -> None:
        if self.format != "pickle":
            raise ValueError(
                f"symseq.storage.format only supports 'pickle' for now, got {self.format!r}"
            )


@dataclass
class SymseqTrialSetCfg:
    n_trials: int
    splits: dict[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.n_trials, int) or self.n_trials < 1:
            raise ValueError(
                f"symseq.trial_set.n_trials must be a positive int, got {self.n_trials!r}"
            )
        if self.splits:
            _validate_splits(self.splits, where="symseq.trial_set.splits")


@dataclass
class SymseqCfg:
    generator: GeneratorCfg
    seed: int | None = None
    trial_constraints: TrialConstraintsCfg | None = None
    trial_set: SymseqTrialSetCfg | None = None
    tasks: list[SymseqTaskEntry] = field(default_factory=list)
    storage: SymseqStorageCfg | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for t in self.tasks:
            if t.id in seen:
                raise ValueError(
                    f"symseq.tasks[*].id must be unique; duplicate {t.id!r}"
                )
            seen.add(t.id)


# ----------------------------- seqbench ---------------------------------------


@dataclass
class SeqbenchStorageCfg:
    path: str
    force_rebuild: bool = False
    cache_key: str | None = None


@dataclass
class TimeGridCfg:
    """Final output time-grid configuration for SeqBench samples.

    ``dt`` is the duration, in seconds, represented by one row of the final
    tensor after all transforms. ``validation`` controls how mismatches between
    declared and resolved transform grids are handled.
    """

    dt: float
    validation: Literal["error", "warn", "ignore"] = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError(f"seqbench.time_grid.dt must be a positive number, got {self.dt!r}")
        if self.validation not in ("error", "warn", "ignore"):
            raise ValueError(
                "seqbench.time_grid.validation must be 'error', 'warn', or 'ignore', "
                f"got {self.validation!r}"
            )


@dataclass
class GapProfileCfg:
    start: int = 0
    duration: dict[str, Any] = field(default_factory=dict)
    add_nongramm_gap: bool = False


@dataclass
class CompositionCfg:
    combine_sequences: bool
    sample_length: int
    gap_profile: GapProfileCfg | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_length, int) or self.sample_length < 1:
            raise ValueError(
                f"composition.sample_length must be a positive int, got {self.sample_length!r}"
            )


@dataclass
class InputMappingCfg:
    base: str
    base_params: dict[str, Any] = field(default_factory=dict)
    transforms: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.base, str) or not self.base:
            raise ValueError(f"input_mapping.base must be a non-empty string, got {self.base!r}")
        if not isinstance(self.transforms, list):
            raise ValueError(
                f"input_mapping.transforms must be a list, got {self.transforms!r}"
            )
        for i, t in enumerate(self.transforms):
            where = f"input_mapping.transforms[{i}]"
            if not isinstance(t, dict):
                raise ValueError(f"{where} must be a mapping, got {t!r}")
            unknown = set(t) - {"type", "params", "time_behavior"}
            if unknown:
                raise ValueError(
                    f"{where} has unknown keys {sorted(unknown)}; allowed keys are "
                    "['params', 'time_behavior', 'type']"
                )
            transform_type = t.get("type")
            if not isinstance(transform_type, str) or not transform_type:
                raise ValueError(
                    f"{where}.type must be a non-empty string, got {transform_type!r}"
                )
            params = t.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"{where}.params must be a mapping, got {params!r}")
            time_behavior = t.get("time_behavior")
            if time_behavior is not None and not isinstance(time_behavior, (str, dict)):
                raise ValueError(
                    f"{where}.time_behavior must be a string or mapping, "
                    f"got {time_behavior!r}"
                )


@dataclass
class SeqbenchTaskCfg:
    source: TaskSource
    id: str | None = None        # source=seqbench: configured target key
    ref_id: str | None = None    # source=symseq: referenced symseq.tasks id
    type: str | None = None      # source=seqbench: seqbench.tasks registry key
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, TaskSource):
            self.source = TaskSource(self.source)
        if self.source == TaskSource.SYMSEQ:
            if not self.ref_id:
                raise ValueError("seqbench.task with source='symseq' requires 'ref_id'")
            if self.id:
                raise ValueError(
                    "seqbench.task with source='symseq' must NOT set 'id'"
                )
            if self.type:
                raise ValueError(
                    "seqbench.task with source='symseq' must NOT set 'type' "
                    "(use 'ref_id' to point at a symseq.tasks entry)"
                )
            if self.params:
                raise ValueError(
                    "seqbench.task with source='symseq' must NOT set 'params'"
                )
        else:  # SEQBENCH
            if not self.id:
                raise ValueError("seqbench.task with source='seqbench' requires 'id'")
            if not self.type:
                raise ValueError("seqbench.task with source='seqbench' requires 'type'")
            if self.ref_id:
                raise ValueError(
                    "seqbench.task with source='seqbench' must NOT set 'ref_id'"
                )
        if self.id is not None and (not isinstance(self.id, str) or not self.id):
            raise ValueError(f"seqbench.task.id must be a non-empty string, got {self.id!r}")
        if self.ref_id is not None and (
            not isinstance(self.ref_id, str) or not self.ref_id
        ):
            raise ValueError(
                f"seqbench.task.ref_id must be a non-empty string, got {self.ref_id!r}"
            )
        if not isinstance(self.params, dict):
            raise ValueError(
                f"seqbench.task.params must be a mapping, got {self.params!r}"
            )


@dataclass
class SeqbenchCfg:
    mode: str
    splits: dict[str, int | float]
    storage: SeqbenchStorageCfg
    time_grid: TimeGridCfg
    composition: CompositionCfg
    input_mapping: InputMappingCfg
    task: SeqbenchTaskCfg
    seed: int | None = None
    prob_generator_type: str = "restricted"

    def __post_init__(self) -> None:
        if self.mode not in ("file", "offline", "online"):
            raise ValueError(
                f"seqbench.mode must be 'file' | 'offline' | 'online', got {self.mode!r}"
            )
        _validate_splits(self.splits, where="seqbench.splits")

    def split_size(self, name: str, total: int | None = None) -> int:
        """Return the absolute sample count for a named SeqBench split."""
        value = self.splits[name]
        if isinstance(value, float):
            if total is None:
                raise TypeError(
                    f"seqbench.splits[{name!r}] is a fraction ({value!r}); "
                    "provide a total sample count to resolve it"
                )
            return round(value * total)
        return value


# ----------------------------- root -------------------------------------------


@dataclass
class RunConfig:
    run: RunMetaCfg
    symbol_space: SymbolSpaceCfg | None = None
    symseq: SymseqCfg | None = None
    seqbench: SeqbenchCfg | None = None

    def __post_init__(self) -> None:
        if self.symseq is None and self.seqbench is None:
            raise ValueError("At least one of 'symseq' or 'seqbench' must be present")
        # Cross-section consistency: source='symseq' requires a symseq.tasks entry.
        if self.symseq is not None and self.seqbench is not None:
            if self.seqbench.task.source == TaskSource.SYMSEQ:
                ids = {t.id for t in self.symseq.tasks}
                if self.seqbench.task.ref_id not in ids:
                    raise ValueError(
                        f"seqbench.task.ref_id={self.seqbench.task.ref_id!r} does not match any "
                        f"symseq.tasks[*].id (available: {sorted(ids) or 'none'})"
                    )
            elif self.seqbench.task.id in {t.id for t in self.symseq.tasks}:
                raise ValueError(
                    f"seqbench.task.id={self.seqbench.task.id!r} collides with a "
                    "symseq.tasks[*].id"
                )
        if self.seqbench is not None and self.seqbench.task.source == TaskSource.SYMSEQ:
            if self.symseq is None:
                raise ValueError(
                    "seqbench.task.source='symseq' requires the top-level 'symseq' section"
                )

    @property
    def operating_mode(self) -> OperatingMode:
        if self.symseq is not None and self.seqbench is None:
            return OperatingMode.SYMSEQ_ONLY
        if self.symseq is not None and self.seqbench is not None:
            return OperatingMode.SYMSEQ_PLUS_SEQBENCH
        return OperatingMode.SEQBENCH_STANDALONE


# ----------------------------- parser -----------------------------------------


def load(source: str | Path | dict) -> RunConfig:
    """Load a YAML file (or pre-parsed dict) into a validated :class:`RunConfig`."""
    raw = _load_raw(source)
    raw = _migrate_raw(raw)
    return _parse_root(raw)


def _load_raw(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in (".yaml", ".yml"):
        raise ValueError(f"unsupported config extension {suffix!r}; use .yaml/.yml")
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"config file {path} is empty")
    return cfg


def _parse_root(raw: dict) -> RunConfig:
    _require_keys(raw, {"run"}, where="<root>")
    _require_keys(raw["run"], {"seed"}, where="run")
    run = RunMetaCfg(**raw["run"])
    symbol_space = (
        _parse_symbol_space(raw["symbol_space"]) if raw.get("symbol_space") else None
    )
    symseq = _parse_symseq(raw["symseq"]) if "symseq" in raw else None
    seqbench = _parse_seqbench(raw["seqbench"]) if "seqbench" in raw else None
    return RunConfig(
        run=run,
        symbol_space=symbol_space,
        symseq=symseq,
        seqbench=seqbench,
    )


def _parse_symbol_space(raw: dict) -> SymbolSpaceCfg:
    _require_keys(raw, {"alphabet"}, where="symbol_space")
    return SymbolSpaceCfg(
        alphabet=SymbolAlphabetCfg(**raw["alphabet"]),
        eos=raw.get("eos", "#"),
    )


def _parse_symseq(raw: dict) -> SymseqCfg:
    _require_keys(raw, {"generator"}, where="symseq")
    gen = GeneratorCfg(**raw["generator"])
    trial_constraints = None
    if raw.get("trial_constraints"):
        tc_raw = dict(raw["trial_constraints"])
        length = (
            LengthConstraintCfg(**tc_raw["length"])
            if tc_raw.get("length")
            else None
        )
        trial_constraints = TrialConstraintsCfg(length=length)
    tasks = [
        _parse_symseq_task(t, index=i)
        for i, t in enumerate(raw.get("tasks") or [])
    ]
    storage = SymseqStorageCfg(**raw["storage"]) if raw.get("storage") else None
    trial_set = (
        SymseqTrialSetCfg(
            n_trials=raw["trial_set"]["n_trials"],
            splits=dict(raw["trial_set"].get("splits") or {}),
        )
        if raw.get("trial_set")
        else None
    )
    return SymseqCfg(
        generator=gen,
        seed=raw.get("seed"),
        trial_constraints=trial_constraints,
        trial_set=trial_set,
        tasks=tasks,
        storage=storage,
    )


def _parse_seqbench(raw: dict) -> SeqbenchCfg:
    if "dt" in raw:
        raise ValueError("seqbench.dt is no longer supported; use seqbench.time_grid.dt")
    _require_keys(
        raw,
        {"mode", "splits", "storage", "time_grid", "composition", "input_mapping", "task"},
        where="seqbench",
    )
    storage = SeqbenchStorageCfg(**raw["storage"])
    time_grid = TimeGridCfg(**raw["time_grid"])
    comp_raw = dict(raw["composition"])
    gp_raw = dict(comp_raw.pop("gap_profile", None) or {}) or None
    if gp_raw is not None and "dt" in gp_raw:
        raise ValueError(
            "composition.gap_profile.dt is no longer supported; use seqbench.time_grid.dt"
        )

    composition = CompositionCfg(
        **comp_raw,
        gap_profile=GapProfileCfg(**gp_raw) if gp_raw else None,
    )
    input_mapping = InputMappingCfg(**raw["input_mapping"])
    task = _parse_seqbench_task(raw["task"])
    return SeqbenchCfg(
        mode=raw["mode"],
        splits=dict(raw["splits"]),
        storage=storage,
        time_grid=time_grid,
        composition=composition,
        input_mapping=input_mapping,
        task=task,
        seed=raw.get("seed"),
        prob_generator_type=raw.get("prob_generator_type", "restricted"),
    )


def _parse_symseq_task(raw: dict, *, index: int) -> SymseqTaskEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"symseq.tasks[{index}] must be a mapping, got {raw!r}")
    allowed = {"id", "type", "params"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"symseq.tasks[{index}] has unknown keys {sorted(unknown)}")
    _require_keys(raw, {"id", "type"}, where=f"symseq.tasks[{index}]")
    return SymseqTaskEntry(
        id=raw["id"], type=raw["type"], params=dict(raw.get("params") or {})
    )


def _parse_seqbench_task(raw: dict) -> SeqbenchTaskCfg:
    if not isinstance(raw, dict):
        raise ValueError(f"seqbench.task must be a mapping, got {raw!r}")
    allowed = {"source", "id", "ref_id", "type", "params"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"seqbench.task has unknown keys {sorted(unknown)}")
    _require_keys(raw, {"source"}, where="seqbench.task")
    source = TaskSource(raw["source"])
    if source == TaskSource.SYMSEQ:
        _require_keys(raw, {"ref_id"}, where="seqbench.task")
        forbidden = {"id", "type", "params"} & set(raw)
    else:
        _require_keys(raw, {"id", "type"}, where="seqbench.task")
        forbidden = {"ref_id"} & set(raw)
    if forbidden:
        raise ValueError(
            f"seqbench.task with source={source.value!r} forbids keys {sorted(forbidden)}"
        )
    return SeqbenchTaskCfg(
        source=source,
        id=raw.get("id"),
        ref_id=raw.get("ref_id"),
        type=raw.get("type"),
        params=dict(raw.get("params") or {}),
    )


def _require_keys(d: dict, keys: set[str], *, where: str) -> None:
    missing = keys - set(d)
    if missing:
        raise ValueError(f"{where}: missing required keys {sorted(missing)}")


def resolve_trial_params(symseq: SymseqCfg | None) -> dict[str, Any]:
    """Delegate SymSeq per-trial generation policy to ``symseq.config``."""
    if symseq is None:
        return {}
    try:
        from symseq.config import resolve_trial_params as _resolve
    except ImportError as exc:
        raise ImportError(
            "Resolving symseq trial parameters requires the `symseq` package."
        ) from exc
    return _resolve(symseq)


def _validate_splits(splits: dict[str, int | float], *, where: str) -> None:
    if not splits:
        raise ValueError(f"{where} must be non-empty")
    for name, value in splits.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{where}[{name!r}] must be int (count) or float in (0,1], "
                f"got {value!r}"
            )
        if isinstance(value, float) and not (0.0 < value <= 1.0):
            raise ValueError(f"{where}[{name!r}] float must be in (0,1], got {value!r}")
        if isinstance(value, int) and value < 0:
            raise ValueError(f"{where}[{name!r}] int must be >= 0, got {value!r}")


def _migrate_raw(raw: dict) -> dict:
    """Normalize older config shapes to the current schema."""
    raw = _deep_copy_config(raw)

    dataset = raw.pop("dataset", None)
    if dataset is not None:
        raw.setdefault("run", {})
        raw["run"].setdefault("seed", dataset.get("seed"))

        alphabet = dict(dataset.get("alphabet") or {})
        if alphabet:
            eos = alphabet.pop("eos", "#")
            raw.setdefault("symbol_space", {})
            raw["symbol_space"].setdefault("alphabet", alphabet)
            raw["symbol_space"].setdefault("eos", eos)

        if "trial_length" in dataset and raw.get("symseq") is not None:
            length = dict(dataset["trial_length"])
            length.pop("distribution", None)
            raw["symseq"].setdefault("trial_constraints", {})
            raw["symseq"]["trial_constraints"].setdefault("length", length)

        if "splits" in dataset and raw.get("seqbench") is not None:
            raw["seqbench"].setdefault("splits", dict(dataset["splits"]))

    if raw.get("symseq") is not None:
        symseq = raw["symseq"]
        generator = symseq.setdefault("generator", {})
        trial_set = symseq.get("trial_set")
        if trial_set and "gen_params" in trial_set:
            generator.setdefault("trial_params", dict(trial_set.pop("gen_params") or {}))

    return raw


def _deep_copy_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_config(v) for v in value]
    return value
