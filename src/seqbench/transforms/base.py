# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Base transform protocol and time-grid metadata for transform pipelines.
"""

from dataclasses import dataclass
from typing import Protocol, Any, Literal


class Transform(Protocol):
    def __call__(self, x: Any) -> Any: ...


@dataclass(frozen=True)
class TimeGrid:
    """Resolved time-axis metadata for a tensor in a transform pipeline.

    ``dt`` is always canonical seconds per row. SeqBench currently supports
    only time on axis 0; other units must be converted to seconds at config or
    dataset-loading boundaries.
    """

    dt: float
    axis: int = 0

    def __post_init__(self):
        if not isinstance(self.dt, (int, float)) or self.dt <= 0:
            raise ValueError(f"TimeGrid.dt must be a positive number, got {self.dt!r}")
        if self.axis != 0:
            raise ValueError("SeqBench currently supports time on axis 0 only")


@dataclass(frozen=True)
class UnknownTimeGrid:
    """Marker for pipelines whose time axis can no longer be trusted.

    ``None`` means no temporal grid has been created yet. ``UnknownTimeGrid``
    means a transform may have changed the temporal semantics without declaring
    how, so downstream code must not assume the previous ``TimeGrid`` still
    applies.
    """


UNKNOWN_TIME_GRID = UnknownTimeGrid()


@dataclass(frozen=True)
class TransformTimeSpec:
    """A transform's declared effect on the pipeline time grid.

    ``kind`` describes whether a transform preserves, creates, resamples, or
    requires a time grid. ``out_dt`` and ``expected_in_dt`` are canonical
    seconds per row.
    """

    kind: Literal["preserve", "create", "resample", "require", "unknown"]
    out_dt: float | None = None
    expected_in_dt: float | None = None

    def __post_init__(self):
        if self.kind not in {"preserve", "create", "resample", "require", "unknown"}:
            raise ValueError(f"Unknown transform time spec kind: {self.kind!r}")
        if self.kind in {"create", "resample"} and self.out_dt is None:
            raise ValueError(f"{self.kind!r} transform time spec requires out_dt")
        if self.out_dt is not None and self.out_dt <= 0:
            raise ValueError(f"out_dt must be positive, got {self.out_dt!r}")
        if self.expected_in_dt is not None and self.expected_in_dt <= 0:
            raise ValueError(f"expected_in_dt must be positive, got {self.expected_in_dt!r}")


class DeclaredTimeBehavior:
    """Attach a config-declared time-grid contract to an arbitrary transform.

    This is intended for third-party or functional transforms that cannot
    implement SeqBench's ``time_spec`` protocol directly. It delegates all data
    transformation to the wrapped object and only supplies the declared
    ``TransformTimeSpec`` to ``Compose``.
    """

    def __init__(self, transform: Transform, spec: TransformTimeSpec):
        self.transform = transform
        self.spec = spec

    def __call__(self, x: Any) -> Any:
        return self.transform(x)

    def time_spec(self, input_grid):
        return self.spec

    def __getattr__(self, name):
        return getattr(self.transform, name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.transform!r}, spec={self.spec!r})"
