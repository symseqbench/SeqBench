# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Axis resizing transform for expanding existing tensor dimensions.
"""

import torch

from seqbench.transforms.base import TransformTimeSpec


class ExpandDim:
    """
    Resize an existing dimension `axis` of a tensor from N_in to N_out by
    repeated index selection. This transform does not insert a new axis.

    Works for any tensor shape. This is useful for expanding the number of
    input neurons in a spiking model, for example.
    """

    def __init__(self, axis: int, target_size: int):
        """"""
        self.axis = axis
        self.target_size = target_size

    def time_spec(self, input_grid):
        if input_grid is not None and self.axis == 0:
            raise ValueError("ExpandDim(axis=0) cannot preserve an existing time grid")
        return TransformTimeSpec("preserve")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: input tensor
        returns: tensor with axis `axis` expanded to `target_size`
        """
        axis = self.axis
        target_size = self.target_size

        # Normalize negative axis
        if axis < 0:
            axis = x.ndim + axis
        if axis < 0 or axis >= x.ndim:
            raise ValueError(f"axis={self.axis} out of bounds for tensor with {x.ndim} dims")

        N_in = x.shape[axis]
        N_out = target_size

        if N_out < N_in:
            raise ValueError(f"target_size {N_out} smaller than input size {N_in} along axis={axis}")

        # Create indices with proper binning
        device = x.device
        idx = ((torch.arange(N_out, device=device) + 0.5) * N_in / N_out).floor().long()
        idx = torch.clamp(idx, 0, N_in - 1)  # ensure no overflow

        # Select elements along the axis
        expanded = torch.index_select(x, axis, idx)

        return expanded
