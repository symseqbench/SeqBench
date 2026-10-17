# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Transform composition utilities for chaining multiple transforms together.
"""

import warnings
from dataclasses import dataclass

from seqbench.transforms.base import (
    UNKNOWN_TIME_GRID,
    TimeGrid,
    TransformTimeSpec,
    UnknownTimeGrid,
)


@dataclass(frozen=True)
class ResolvedTransformStep:
    transform: object
    spec: TransformTimeSpec
    input_grid: TimeGrid | UnknownTimeGrid | None
    output_grid: TimeGrid | UnknownTimeGrid | None

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        self.initial_time_grid = None
        self.current_time_grid = None
        self.output_time_grid = None
        self.time_specs = []
        self._resolved_steps = []
        self._time_validation = "error"

    def __call__(self, x):
        if not self._resolved_steps:
            for t in self.transforms:
                x = t(x)
            return x

        for step in self._resolved_steps:
            before = x
            x = step.transform(x)
            self._validate_tensor_time_contract(step, before, x)
        return x

    def resolve_time_grid(self, initial_grid, expected_final_grid, validation="error"):
        """Resolve the transform chain's final time grid.

        Args:
            initial_grid: ``TimeGrid`` or ``None`` for static/event-like input.
            expected_final_grid: required final ``TimeGrid``.
            validation: ``"error"``, ``"warn"``, or ``"ignore"``.

        Returns:
            The resolved output ``TimeGrid``.
        """
        if validation not in {"error", "warn", "ignore"}:
            raise ValueError(f"Unknown time-grid validation mode {validation!r}")

        current = initial_grid
        steps = []
        for transform in self.transforms:
            spec = self._get_time_spec(transform, current)
            next_grid = self._apply_spec(transform, spec, current, validation)
            steps.append(ResolvedTransformStep(transform, spec, current, next_grid))
            current = next_grid

        if current is None:
            current = expected_final_grid
        elif current is UNKNOWN_TIME_GRID:
            self._handle_issue(
                "transform pipeline output time grid is unknown; "
                "using seqbench.time_grid.dt as the final grid",
                validation,
            )
            current = expected_final_grid

        self._handle_dt_mismatch(
            "transform pipeline",
            current.dt,
            expected_final_grid.dt,
            validation,
        )

        self.initial_time_grid = initial_grid
        self.current_time_grid = current
        self.output_time_grid = current
        self.time_specs = [(step.transform, step.spec) for step in steps]
        self._resolved_steps = steps
        self._time_validation = validation
        self._bind_time_grids(steps, validation)
        return current

    def _get_time_spec(self, transform, current):
        if hasattr(transform, "time_spec"):
            input_grid = current if isinstance(current, TimeGrid) else None
            spec = transform.time_spec(input_grid)
            if not isinstance(spec, TransformTimeSpec):
                raise TypeError(
                    f"{transform.__class__.__name__}.time_spec() must return "
                    "TransformTimeSpec"
                )
            return spec
        return TransformTimeSpec("unknown")

    def _apply_spec(self, transform, spec, current, validation):
        name = transform.__class__.__name__

        if spec.kind == "unknown":
            self._handle_unknown(name, validation)
            return UNKNOWN_TIME_GRID

        if spec.expected_in_dt is not None:
            if not isinstance(current, TimeGrid):
                self._handle_missing_grid(name, validation)
            else:
                self._handle_dt_mismatch(
                    name,
                    current.dt,
                    spec.expected_in_dt,
                    validation,
                )

        if spec.kind == "preserve":
            return current

        if spec.kind == "require":
            if not isinstance(current, TimeGrid):
                self._handle_missing_grid(name, validation)
            return current

        if spec.kind == "create":
            if isinstance(current, TimeGrid):
                self._handle_existing_grid(name, validation)
            elif current is UNKNOWN_TIME_GRID:
                self._handle_issue(
                    f"{name} creates a time grid but the input time grid is unknown",
                    validation,
                )
            return TimeGrid(dt=spec.out_dt)

        if spec.kind == "resample":
            if not isinstance(current, TimeGrid):
                self._handle_missing_grid(name, validation)
            return TimeGrid(dt=spec.out_dt)

        raise ValueError(f"Unhandled transform time spec kind {spec.kind!r}")

    def _bind_time_grids(self, steps, validation):
        for step in steps:
            transform = step.transform
            if not hasattr(transform, "bind_time_grid"):
                continue
            if not isinstance(step.input_grid, (TimeGrid, type(None))) or not isinstance(
                step.output_grid, TimeGrid
            ):
                self._handle_issue(
                    f"{transform.__class__.__name__} cannot bind an unknown time grid",
                    validation,
                )
                continue
            transform.bind_time_grid(step.input_grid, step.output_grid)

    def _validate_tensor_time_contract(self, step, before, after):
        validation = self._time_validation
        in_steps = self._time_len(before)
        out_steps = self._time_len(after)
        name = step.transform.__class__.__name__

        if isinstance(step.output_grid, TimeGrid) and out_steps is None:
            self._handle_issue(
                f"{name} declares a time grid but returned data without a leading time axis",
                validation,
            )
            return

        expected = self._expected_output_steps(step.transform, in_steps, before)
        if expected is None and step.spec.kind in {"preserve", "require"}:
            if isinstance(step.input_grid, TimeGrid) and isinstance(step.output_grid, TimeGrid):
                expected = in_steps

        if expected is None or out_steps is None:
            return

        if out_steps != expected:
            self._handle_issue(
                f"{name} output time length mismatch: got {out_steps}, expected {expected}",
                validation,
            )

    def _expected_output_steps(self, transform, in_steps, before):
        if in_steps is None or not hasattr(transform, "expected_time_steps"):
            return None
        return transform.expected_time_steps(in_steps, getattr(before, "shape", None))

    def _time_len(self, x):
        shape = getattr(x, "shape", None)
        if shape is None or len(shape) == 0:
            return None
        return int(shape[0])

    def _handle_unknown(self, name, validation):
        self._handle_issue(
            f"{name} does not declare time-grid behavior",
            validation,
        )

    def _handle_missing_grid(self, name, validation):
        self._handle_issue(
            f"{name} requires an input time grid, but none is available",
            validation,
        )

    def _handle_existing_grid(self, name, validation):
        self._handle_issue(
            f"{name} creates a time grid but the input already has one",
            validation,
        )

    def _handle_dt_mismatch(self, name, actual, expected, validation):
        if abs(actual - expected) <= 1e-9:
            return
        self._handle_issue(
            f"{name} time-grid mismatch: got dt={actual:g}s, expected dt={expected:g}s",
            validation,
        )

    def _handle_issue(self, message, validation):
        if validation == "error":
            raise ValueError(message)
        if validation == "warn":
            warnings.warn(message, stacklevel=3)


class ApplyToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, sample):
        sample = dict(sample)
        sample[self.key] = self.transform(sample[self.key])
        return sample
