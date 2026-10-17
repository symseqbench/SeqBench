# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Transform Module for SeqBench

This module provides data transformation utilities and composition functionality.
"""

import pkgutil
import importlib
import inspect
import ast
from pathlib import Path

from seqbench.transforms.compose import Compose
from seqbench.transforms.functional import FunctionalTransform
from seqbench.transforms.base import DeclaredTimeBehavior, TransformTimeSpec

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = []


def _auto_import_classes(package_name: str, package_path: Path):
    """
    Recursively import all classes from the given package and subpackages.
    """

    # Ensure package_path is a Path object
    package_path = Path(package_path)

    for finder, module_name, ispkg in pkgutil.walk_packages([str(package_path)], prefix=package_name + "."):
        # Skip private modules
        if any(part.startswith("_") for part in module_name.split(".")):
            continue

        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue  # or log

        # Import all classes defined in this module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name:
                globals()[name] = obj
                __all__.append(name)


# Run for current package
_package_name = __name__
_package_path = Path(__file__).parent

_auto_import_classes(_package_name, _package_path)


def _normalize_params(params):
    """Normalize YAML params: 'None' -> None, '(a,b)' -> tuple, 'torch.X' -> torch.X"""
    normalized = {}
    for k, v in params.items():
        if isinstance(v, str):
            if v.lower() == 'none':
                v = None
            elif v.startswith('torch.') and TORCH_AVAILABLE:
                v = getattr(torch, v.split('.', 1)[1], v)
            elif v.startswith('(') and v.endswith(')'):
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
        normalized[k] = v
    return normalized


def _parse_time_behavior(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError("time_behavior must be a string or dict")

    kind = raw.get("kind")
    if kind not in {"preserve", "create", "resample", "require", "unknown"}:
        raise ValueError(
            "time_behavior.kind must be one of "
            "'preserve', 'create', 'resample', 'require', or 'unknown'"
        )
    return TransformTimeSpec(
        kind,
        out_dt=raw.get("out_dt"),
        expected_in_dt=raw.get("expected_in_dt"),
    )


def compose_transforms_from_config(input_mapping):
    """
    Create a Compose object from the transforms defined in an
    :class:`~seqbench.config.InputMappingCfg`.

    Supports dotted paths (e.g., "tonic.transforms.ToFrame") and plain
    callables (wrapped in :class:`FunctionalTransform`).

    Returns ``None`` if no transforms are configured.
    """
    transforms_spec = input_mapping.transforms
    if not transforms_spec:
        return None

    transforms = []
    for spec in transforms_spec:
        transform_type = spec["type"]
        params = _normalize_params(dict(spec.get("params") or {}))
        time_spec = _parse_time_behavior(spec.get("time_behavior"))

        try:
            # Handle dotted paths (e.g., "tonic.transforms.ToFrame")
            if "." in transform_type:
                parts = transform_type.split(".")
                module = importlib.import_module(".".join(parts[:-1]))
                obj = getattr(module, parts[-1])
            else:
                obj = globals()[transform_type]

            # Wrap functions, instantiate classes
            if inspect.isclass(obj):
                t = obj(**params)
            elif callable(obj):
                t = FunctionalTransform(obj, **params)
            else:
                raise ValueError(f"{transform_type} is not callable")

        except Exception as e:
            raise RuntimeError(
                f"Could not create transform {transform_type} with params {params}"
            ) from e

        if time_spec is not None:
            t = DeclaredTimeBehavior(t, time_spec)
        transforms.append(t)

    return Compose(transforms)
