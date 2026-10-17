# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Task registry — mirrors ``symseq.tasks.registry``.

Each task class registers itself via ``@register("Type")``; ``build(type_name, **params)``
instantiates a registered task.
"""

from __future__ import annotations

from seqbench.tasks.base import Task

_REGISTRY: dict[str, type[Task]] = {}


def register(type_name: str):
    """Class decorator that registers a Task under ``type_name``."""

    def _decorator(cls: type[Task]) -> type[Task]:
        if type_name in _REGISTRY:
            raise ValueError(
                f"Task {type_name!r} is already registered to "
                f"{_REGISTRY[type_name].__name__}."
            )
        _REGISTRY[type_name] = cls
        return cls

    return _decorator


def build(type_name: str, **params) -> Task:
    """Instantiate a registered Task by type."""
    if type_name not in _REGISTRY:
        raise KeyError(
            f"Unknown task type {type_name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[type_name](**params)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)
