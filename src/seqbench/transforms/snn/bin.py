# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Binning transform for spiking neural network data along specified axes.
"""

import torch

from seqbench.transforms.base import TransformTimeSpec


class BinAlongAxis:
    """
    Bins a tensor along a specified axis. For example, this can be used to bin spikes into
    coarse-grained representations.
    """

    def __init__(self, bin_size, bin_axis=-1, reduction="sum"):
        """
        Parameters
        ----------
        bin_size : int
            Size of each bin along the specified axis.
        bin_axis : int
            Axis along which to apply binning.
        reduction : str
            "sum", "mean", "binary"
        """
        if reduction not in {"sum", "mean", "binary"}:
            raise ValueError("reduction must be one of 'sum', 'mean', or 'binary'")
        self.bin_size = bin_size
        self.bin_axis = bin_axis
        self.reduction = reduction

    def time_spec(self, input_grid):
        if self.bin_axis != 0:
            return TransformTimeSpec("preserve")
        if input_grid is None:
            return TransformTimeSpec("require")
        return TransformTimeSpec("resample", out_dt=input_grid.dt * self.bin_size)

    def expected_time_steps(self, input_steps, input_shape=None):
        if self.bin_axis != 0:
            return None
        return input_steps // self.bin_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize axis for negative indexing
        axis = self.bin_axis % x.ndim
        B = self.bin_size
        N = x.shape[axis]

        if N % B != 0:
            raise ValueError(f"axis length {N} is not divisible by bin_size {B}")

        # Move axis to last dimension for easy reshaping
        x_perm = x.moveaxis(axis, -1)

        # Collapse last dimension into bins
        # new shape: (..., N/B, B)
        new_shape = x_perm.shape[:-1] + (N // B, B)
        x_binned = x_perm.reshape(new_shape)

        # Reduce within-bin dimension
        if self.reduction == "sum":
            out = x_binned.sum(dim=-1)
        elif self.reduction == "binary":
            out = x_binned.sum(dim=-1) > 0
        else:
            out = x_binned.mean(dim=-1)

        # Move axis back to original position
        out = out.moveaxis(-1, axis)
        return out


# # TODO we need a better name for this
# class BinNeurons:
#     """
#     Sum spikes across groups of neurons/units, resulting in a coarse-grained/binned representation.
#     """

#     def __init__(self, bin_size, bin_axis=1):
#         """
#         Parameters
#         ----------
#         bin_size : int
#             Number of neurons/units per bin.
#         """
#         self.bin_size = bin_size
#         self.bin_axis = bin_axis

#     def __call__(self, x):
#         assert len(x.shape) == 2
#         T = x.shape[0]  # number of time steps
#         J = x.shape[1]  # number of neurons / units

#         # binning
#         with torch.no_grad():
#             x = x.contiguous().view(T, J // self.bin_size, self.bin_size).sum(-1)

#         return x


# old function, check if we need
# def bin_spiking_data(x, nb_steps, neuron_bin_size):
#     """
#     TODO for now, this matches the binning process in the SpikingDataset

#     x : torch.Tensor [T, J]
#         Dense spike matrix (time × neurons), binary or counts.
#     time_bin_size : int
#         Number of fine time steps to group into one temporal bin.
#     neuron_bin_size : int
#         Number of neurons to group into one pooled unit.

#     Returns
#     -------
#     x_binned : torch.Tensor [T_binned, J_binned]
#         Temporally binned and neuron-grouped spike counts.
#     """
#     T, N = x.shape
#     # time_bins = np.linspace(0, max_time, num=nb_steps)

#     if T % (nb_steps - 1) != 0:
#         raise ValueError(f"Number of time steps {T} is not divisible by nb_steps-1={nb_steps-1}.")

#     # number of fine time steps per coarse bin
#     time_bin_size = T // (nb_steps - 1)

#     # --- Temporal binning ---
#     x = x.view(nb_steps - 1, time_bin_size, N).sum(1)
#     # shape: [T_binned = nb_steps-1, N]

#     # --- Neuron grouping ---
#     J_trim = (N // neuron_bin_size) * neuron_bin_size
#     x = x[:, :J_trim]  # trim so it's divisible
#     x = x.view(x.shape[0], J_trim // neuron_bin_size, neuron_bin_size).sum(-1)
#     # shape: [T_binned, N_binned]

#     return x
