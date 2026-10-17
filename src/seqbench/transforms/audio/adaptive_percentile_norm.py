# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Adaptive percentile normalization transform for audio data.
"""

import torch

from seqbench.transforms.base import TransformTimeSpec


class AdaptivePercentileNorm:
    """
    Normalize a tensor to [0, 1] using adaptive clipping based on
    lower/upper percentiles computed over the tensor values.
    """

    def __init__(self, floor_percentile=5.0, ceil_percentile=95.0):
        if not 0.0 <= floor_percentile < ceil_percentile <= 100.0:
            raise ValueError(
                "percentiles must satisfy 0 <= floor_percentile < "
                "ceil_percentile <= 100"
            )
        self.floor_percentile = floor_percentile
        self.ceil_percentile = ceil_percentile

    def time_spec(self, input_grid):
        return TransformTimeSpec("preserve")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Compute percentiles
        lo = torch.quantile(x, self.floor_percentile / 100.0)
        hi = torch.quantile(x, self.ceil_percentile / 100.0)

        # Degenerate case
        if hi <= lo:
            return torch.zeros_like(x)

        # Clip and normalize
        clipped = torch.clamp(x, min=lo.item(), max=hi.item())
        return (clipped - lo) / (hi - lo)
