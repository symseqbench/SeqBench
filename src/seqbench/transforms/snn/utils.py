# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utility functions for spiking neural network transforms.
"""

import torch
import numpy as np


# TODO this ignores multiple spikes in one bin, only binary. We need to fix this?
def spike_times_to_tensor(spike_times, units, time_bins, num_units=700, dense=True):
    """
    Convert spike times to a sparse or dense tensor with shape (`num_units`, `time_bins`).

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times.
    units : np.ndarray
        Units corresponding to the spike times.
    time_bins : np.ndarray
        Time bins.
    num_units : int
        Number of total units (neurons, channels) used for contructing the tensor.
    dense : bool
        Whether to return a dense tensor or a sparse tensor.

    Returns
    -------
    x : torch.Tensor
        Sparse or dense tensor.
    """
    times = np.digitize(spike_times, time_bins)
    # Size the time axis from the grid, not the data: np.digitize returns indices
    # in [0, len(time_bins)], so len(time_bins) + 1 rows hold every possible index
    # and give a consistent shape across samples (cf. the fixed `num_units`).
    length = len(time_bins) + 1

    x_idx = torch.LongTensor(np.array([times, units]))  # 2xN tensor of times and units
    x_val = torch.FloatTensor(np.ones(len(times)))
    x_size = torch.Size([length, num_units])

    # Indices are in-bounds by construction (see `length` above), so skip the
    # invariant checks; the context manager also silences torch's warning about
    # the implicit default.
    with torch.sparse.check_sparse_tensor_invariants(False):
        x = torch.sparse_coo_tensor(x_idx, x_val, x_size)

    if dense:
        x = x.to_dense()

    return x
