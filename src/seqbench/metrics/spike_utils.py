# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Utilities for processing and analyzing spike data, including PCA and visualization.
"""

# PCA on (time x neurons) spike matrices
# - Converts per-neuron spike-time lists into a (T x N) binned count matrix
# - Optional smoothing -> rates
# - Z-score per neuron
# - PCA with samples=time bins, features=neurons
# - Simple visualizations

import numpy as np
try:
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA
except ImportError as exc:
    raise ImportError(
        "SeqBench spike metrics require scipy and scikit-learn. "
        "Install SeqBench with the metrics extra: pip install 'seqbench[metrics]'."
    ) from exc

# ---------- Utilities ----------


def bin_spike_times(spike_times_per_neuron, t_start, t_end, dt):
    """
    spike_times_per_neuron: list of length N; each item is 1D array of spike times (s)
    t_start, t_end: analysis window (s)
    dt: bin width (s)

    Returns
      X: (T x N) spike counts per bin
      t: (T,) bin centers (s)
    """
    spike_times_per_neuron = [np.asarray(st) for st in spike_times_per_neuron]
    edges = np.arange(t_start, t_end + dt, dt)
    T = len(edges) - 1
    N = len(spike_times_per_neuron)
    X = np.zeros((T, N), dtype=float)
    for j, times in enumerate(spike_times_per_neuron):
        times = times[(times >= t_start) & (times < t_end)]
        counts, _ = np.histogram(times, bins=edges)
        X[:, j] = counts
    t = (edges[:-1] + edges[1:]) / 2.0
    return X, t


def smooth_rates(X, dt, sigma_ms=None):
    """
    Smooth counts to get rates (spikes/s). Gaussian smoothing along time.
    sigma_ms: std of Gaussian kernel in ms. If None or <=0, returns counts/dt.
    """
    if sigma_ms is None or sigma_ms <= 0:
        return X / dt
    sigma_bins = (sigma_ms / 1000.0) / dt
    R = gaussian_filter1d(X, sigma=sigma_bins, axis=0, mode="nearest")
    return R / dt


def zscore_timewise(X):
    """Z-score each neuron (column) across time."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def run_pca_time_by_neuron(X, n_components=6):
    """PCA where samples = time bins, features = neurons."""
    pca = PCA(n_components=n_components, svd_solver="full")
    scores = pca.fit_transform(X)  # (T x n_components)
    return pca, scores


# ---------- Example with synthetic data (replace with your own) ----------

rng = np.random.default_rng(7)

# Analysis window & binning
t_start, t_end = 0.0, 0.25  # 250 ms per trial
dt = 0.005  # 5 ms bins

N = 20  # neurons
stim_bumps = [0.07, 0.12, 0.18]  # 3 stimuli, different transient times


def simulate_spike_times(N, t_start, t_end, bump_t, bump_width=0.03, base_rate=5.0, bump_rate=35.0, dt=0.001, rng=None):
    """Simple inhomogeneous Poisson with a Gaussian bump."""
    if rng is None:
        rng = np.random.default_rng()
    times = np.arange(t_start, t_end, dt)
    lam = base_rate * np.ones_like(times)  # Hz
    lam += bump_rate * np.exp(-0.5 * ((times - bump_t) / bump_width) ** 2)
    spike_times_per_neuron = []
    for _ in range(N):
        p = lam * dt
        spikes = rng.random(size=times.shape) < p
        spike_times_per_neuron.append(times[spikes])
    return spike_times_per_neuron


# # Build (time x neurons) by concatenating trials in time
# X_all, t_all = [], []
# for s, bump_t in enumerate(stim_bumps):
#     spikes = simulate_spike_times(N, t_start, t_end, bump_t=bump_t, rng=rng)
#     X, t = bin_spike_times(spikes, t_start, t_end, dt)
#     X_all.append(X)
#     t_all.append(t + (t_end - t_start) * s)  # shift to make trials contiguous

# X_all = np.vstack(X_all)          # (T_total x N)
# t_all = np.concatenate(t_all)

# # Optional smoothing and normalization
# R_all = smooth_rates(X_all, dt, sigma_ms=10)  # try 5–20 ms
# Z_all = zscore_timewise(R_all)
