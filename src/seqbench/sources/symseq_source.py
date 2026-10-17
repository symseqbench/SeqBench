# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
symseq trial source.

This is the only place in SeqBench that imports ``symseq``. Everywhere else,
the source is duck-typed against the structural ``TrialSource`` Protocol
(``alphabet``, ``draw_trial``, ``draw_batch``). Keeping the import here means
``import seqbench`` does not require symseq to be installed; only callers that
actually construct a symseq source pay the dependency.
"""

from __future__ import annotations

from typing import Any


def build_symseq_source(config: Any) -> Any:
    """Build a live symseq generator from a config block.

    Parameters
    ----------
    config
        Either a :class:`~seqbench.config.RunConfig`, a
        :class:`~seqbench.config.GeneratorCfg` dataclass instance, or a plain
        dict shaped like ``{"type": <registry_name>, "params": {...}}``.

    Returns
    -------
    A ``symseq.core.sequencer.SymbolicSequencer`` subclass instance. The
    object structurally satisfies ``symseq.TrialSource`` (exposes
    ``alphabet``, ``draw_trial``, ``draw_batch``).
    """
    try:
        from symseq.generators.registry import build
    except ImportError as exc:
        raise ImportError(
            "Building a symseq trial source requires the `symseq` package. "
            "Install it, or supply a different TrialSource."
        ) from exc

    generator_cfg, symbol_space_cfg, seed = _normalize_config(config)

    # Normalise: accept a GeneratorCfg dataclass or a plain dict.
    if isinstance(generator_cfg, dict):
        gen_type = generator_cfg.get("type")
        params = dict(generator_cfg.get("params") or {})
        # Legacy symseq standalone schema: top-level "preset" key in the dict.
        ag_preset = generator_cfg.get("preset")
        ag_mode = _enum_value(generator_cfg.get("mode"))
    else:
        # GeneratorCfg dataclass: fields accessed as attributes.
        gen_type = generator_cfg.type
        params = dict(generator_cfg.params)
        ag_preset = None
        ag_mode = _enum_value(generator_cfg.mode)  # AGMode enum or None

    if not gen_type:
        raise ValueError(
            "symseq generator config requires a 'type' key (registry name)."
        )

    # Special case: ArtificialGrammar preset constructor.
    if gen_type == "ArtificialGrammar":
        from symseq.generators.ag import ArtificialGrammar

        if ag_mode == "preset":
            # Dataclass path: preset name lives in params["preset"].
            preset_name = params.pop("preset", None)
            if not preset_name:
                raise ValueError(
                    "ArtificialGrammar with mode='preset' requires a 'preset' key "
                    "inside generator.params (e.g. params: {preset: Elman})."
                )
            source = ArtificialGrammar.from_preset(
                preset_name=preset_name,
                seed=params.get("seed", seed if seed is not None else 42),
            )
            _validate_symbol_space(source, symbol_space_cfg)
            return _with_configured_tasks(source, config)
        if ag_preset:
            # Legacy dict path: top-level "preset" key.
            source = ArtificialGrammar.from_preset(
                preset_name=ag_preset,
                seed=params.get("seed", seed if seed is not None else 42),
            )
            _validate_symbol_space(source, symbol_space_cfg)
            return _with_configured_tasks(source, config)
        if ag_mode in (None, "random"):
            _inherit_symbol_space_defaults(params, gen_type, ag_mode, symbol_space_cfg)
            _inherit_seed(params, seed)
            source = ArtificialGrammar.from_constraints(**params)
            _validate_symbol_space(source, symbol_space_cfg)
            return _with_configured_tasks(source, config)

    _inherit_symbol_space_defaults(params, gen_type, ag_mode, symbol_space_cfg)
    _inherit_seed(params, seed)

    source = build(gen_type, **params)
    _validate_symbol_space(source, symbol_space_cfg)
    return _with_configured_tasks(source, config)


def _with_configured_tasks(source: Any, config: Any) -> Any:
    if not hasattr(config, "symseq") or config.symseq is None:
        return source
    from symseq.tasks import ConfiguredTrialSource, build_tasks

    tasks = build_tasks(config.symseq.tasks)
    return ConfiguredTrialSource(source, tasks)


def _normalize_config(config: Any) -> tuple[Any, Any | None, int | None]:
    """Return ``(generator_cfg, symbol_space_cfg, effective_seed)``.

    ``RunConfig`` is preferred because it gives this boundary enough context to
    apply documented symbol_space -> symseq inheritance. Bare generator configs still
    work for older call sites and tests.
    """
    if hasattr(config, "symseq") and hasattr(config, "run"):
        if config.symseq is None:
            raise ValueError("RunConfig must contain a 'symseq' section")
        seed = config.symseq.seed if config.symseq.seed is not None else config.run.seed
        return config.symseq.generator, config.symbol_space, seed
    return config, None, None


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _inherit_symbol_space_defaults(
    params: dict[str, Any],
    gen_type: str,
    ag_mode: Any,
    symbol_space_cfg: Any | None,
) -> None:
    if symbol_space_cfg is None:
        return
    if gen_type == "ArtificialGrammar" and ag_mode in (None, "random"):
        params.setdefault("alphabet_size", symbol_space_cfg.alphabet.size)
        if symbol_space_cfg.eos is not None:
            params.setdefault("eos", symbol_space_cfg.eos)
    elif gen_type == "NBack":
        if getattr(symbol_space_cfg.alphabet, "symbols", None) is not None:
            params.setdefault("alphabet", symbol_space_cfg.alphabet.resolved_symbols)
        else:
            params.setdefault("alphabet_size", symbol_space_cfg.alphabet.size)


def _inherit_seed(params: dict[str, Any], seed: int | None) -> None:
    if seed is not None:
        params.setdefault("seed", seed)


def _validate_symbol_space(source: Any, symbol_space_cfg: Any | None) -> None:
    if symbol_space_cfg is None:
        return
    actual = list(source.alphabet)
    if getattr(symbol_space_cfg.alphabet, "symbols", None) is None:
        if len(actual) != symbol_space_cfg.alphabet.size:
            raise ValueError(
                "symseq generator alphabet size does not match symbol_space.alphabet.size: "
                f"expected {symbol_space_cfg.alphabet.size!r}, got {len(actual)!r}"
            )
        return
    expected = symbol_space_cfg.alphabet.resolved_symbols
    if actual != expected:
        raise ValueError(
            "symseq generator alphabet does not match symbol_space.alphabet: "
            f"expected {expected!r}, got {actual!r}"
        )
