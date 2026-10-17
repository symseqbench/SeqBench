# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Poisson encoding transform for converting continuous signals to spike trains.
TODO BZ: add support for variable durations / token and test it!
"""

import numpy as np
import torch
import logging

from seqbench.transforms.base import TransformTimeSpec

logger = logging.getLogger(__name__)


class PoissonEncoding:
    """
    Generate a Poisson-distributed spike train from continuous input.

    Temporal input uses the current pipeline ``TimeGrid`` as the input frame
    duration and emits spikes on ``spike_dt`` seconds per row. Static input has
    no input grid; ``duration`` gives the physical spike-train duration in
    seconds.

    By default, input values are interpreted directly as firing rates in Hz.
    Set ``normalize=True`` only when the input should be min-max normalized to
    ``[0, max_rate]`` before sampling.

    Spike probability is computed correctly via:
        p = 1 - exp(-rate * dt)

    """

    def __init__(
        self,
        temporal: bool = False,
        duration: float = None,
        spike_dt: float = 0.001,
        max_rate: float = 100.0,
        normalize: bool = False,
        frozen_noise: bool = False,
        seed: int = None,
    ):
        """
        Parameters
        ----------
        temporal : bool
            Whether input has a temporal dimension (time-varying).
        duration : float
            Static-mode spike-train duration in seconds. Required when
            ``temporal`` is False and ignored when ``temporal`` is True.
        spike_dt : float
            Output spike-grid resolution in seconds per row. Default
            1 ms = 0.001.
        max_rate : float
            Maximum firing rate in Hz when ``normalize=True``.
        normalize : bool
            If True, normalize input to [0, max_rate]. If False, input is
            treated as rates in Hz.
        seed : int
            Random seed for reproducibility.
        frozen_noise : bool
            If True, always generate the same static spike pattern (frozen noise) by re-seeding the generator.
        """

        self.temporal = temporal
        self.duration = duration
        self.spike_dt = spike_dt
        self.max_rate = max_rate
        self.normalize = normalize
        self.frozen_noise = frozen_noise
        self.trng = torch.Generator()
        self.input_dt = None

        if seed is not None:
            self.seed = seed
            self.trng.manual_seed(seed)
        else:
            self.seed = np.random.randint(0, 10000)

        if not temporal and duration is None:
            raise ValueError("duration must be specified for static PoissonEncoding")

        if spike_dt <= 0:
            raise ValueError("spike_dt must be positive")

    def time_spec(self, input_grid):
        if self.temporal:
            if input_grid is None:
                return TransformTimeSpec("require")
            return TransformTimeSpec(
                "resample",
                out_dt=self.spike_dt,
                expected_in_dt=input_grid.dt,
            )
        return TransformTimeSpec("create", out_dt=self.spike_dt)

    def bind_time_grid(self, input_grid, output_grid):
        if not self.temporal:
            return
        if input_grid is None:
            raise ValueError("Temporal PoissonEncoding requires an input time grid")
        ratio = input_grid.dt / self.spike_dt
        bins_per_frame = int(round(ratio))
        if bins_per_frame <= 0:
            raise ValueError(f"input dt ({input_grid.dt}) must be >= spike_dt ({self.spike_dt})")
        if abs(ratio - bins_per_frame) > 1e-9:
            raise ValueError(
                "Temporal PoissonEncoding requires input_dt / spike_dt to be an "
                f"integer ratio, got {input_grid.dt:g} / {self.spike_dt:g}"
            )
        self.input_dt = input_grid.dt

    def expected_time_steps(self, input_steps, input_shape=None):
        if self.temporal:
            if self.input_dt is None:
                return None
            return input_steps * int(round(self.input_dt / self.spike_dt))
        return int(round(self.duration / self.spike_dt))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor
            Input data:
                - Temporal: [T_in, ...] first dim = time
                - Static: any shape, last dim = units

        Returns
        -------
        spikes : torch.Tensor
            Binary spike train of shape [T_out, ...] (temporal) or same as input (static)
        """
        if self.temporal:
            return self._generate_temporal_spikes(x)
        else:
            return self._generate_static_spikes(x)

    # ---------------- Temporal ----------------
    def _generate_temporal_spikes(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to max_rate if needed
        rates = self._normalize(x) if self.normalize else x

        # Expand each frame in time
        if self.input_dt is None:
            raise RuntimeError(
                "Temporal PoissonEncoding requires Compose.resolve_time_grid() "
                "before it can be called"
            )
        bins_per_frame = int(round(self.input_dt / self.spike_dt))
        if bins_per_frame <= 0:
            raise ValueError(f"input dt ({self.input_dt}) must be >= spike_dt ({self.spike_dt})")

        # # Expand time dimension: [T_in, ...] -> [T_in * bins_per_frame, ...]
        rates_exp = rates.repeat_interleave(bins_per_frame, dim=0)

        # Poisson spike probability per bin
        p_spike = 1.0 - torch.exp(-rates_exp * self.spike_dt)

        if p_spike.max().item() > 1.0:
            logger.warning(
                f"Maximum spike probability {p_spike.max().item():.3f} > 1.0. " f"Check max_rate or spike_dt."
            )

        # Generate spikes
        spikes = (torch.rand_like(p_spike) < p_spike).float()
        return spikes

    # ---------------- Static ----------------
    def _generate_static_spikes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate static Poisson (frozen noise) spike pattern.
        """
        if self.frozen_noise:
            # Re-seed the generator for reproducibility
            self.trng.manual_seed(self.seed)

        # Treat last dimension as units, expand over time according to max_rate & spike_dt
        n_bins = int(round(self.duration / self.spike_dt))
        if n_bins <= 0:
            raise ValueError(f"Duration {self.duration} must be >= spike_dt ({self.spike_dt})")

        x_expanded = x.unsqueeze(0).repeat(n_bins, *([1] * (x.ndim)))  # [T_out, ...]
        rates_hz = self._normalize(x_expanded) if self.normalize else x_expanded
        p_spike = 1.0 - torch.exp(-rates_hz * self.spike_dt)

        rand_vals = torch.rand(p_spike.shape, device=p_spike.device, dtype=p_spike.dtype, generator=self.trng)
        spikes = (rand_vals < p_spike).float()
        return spikes

    # ---------------- Normalization ----------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [0, max_rate]
        x_min, x_max = x.min(), x.max()
        if x_max <= x_min:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min) * self.max_rate


# def frozen_noise(self, n_processes=10, duration=50.0, rates=10.0, dt=0.1, jitter=None):
#     """
#     Generate a spatiotemporal spike pattern for each token as frozen Poisson noise.

#     Each token is represented by a spatiotemporal embedding composed of multiple independent Poisson processes.

#     Parameters
#     ----------
#     n_processes : int
#         Number of independent Poisson processes (channels) per spike pattern.

#     pattern_duration : float or list of float
#         Duration in seconds for each spike pattern. Can be a fixed scalar (applied to all tokens),
#         or a list specifying different durations per token.

#     rate : float or array-like of shape (n_tokens,)
#         Mean firing rate(s) in Hz for the Poisson processes. Can be a scalar (same for all tokens)
#         or an array specifying one rate per token.

#     resolution : float
#         Temporal resolution in seconds for the generated spike patterns.

#     jitter : NOT YET IMPLEMENTED! None or tuple (float, bool), optional
#         Temporal jitter to apply to the spike times. If provided, should be a tuple
#         (jitter_value, compensate), where:
#             - jitter_value : float
#                 Maximum jitter in seconds applied to spike times.
#             - compensate : bool
#                 Whether to adjust the pattern duration to maintain mean firing rate after jitter.

#     rng : torch.Generator or None, optional
#         Torch random number generator for reproducibility.

#     Returns
#     -------
#     patterns : torch.Tensor of shape (n_processes, n_steps), dtype=torch.int64
#         Binary spike patterns where 1 indicates a spike.
#     """
#     return homogeneous_spike_pattern(n_processes, rates=rates, duration=duration, dt=dt, rng=self.rng)


# # TODO this can be merged with generate_static_spikes
# def homogeneous_spike_pattern(
#     n_processes: int,
#     duration: Union[float, Dict[str, Any]],
#     rates: Union[float, torch.Tensor],
#     dt: float = 1.0,
#     rng: torch.Generator = None,
# ):
#     """
#     Generate a spatiotemporal spike template as a collection of independent Poisson processes
#     parameterized by `rates`. If `rates` is a tensor, the function converts a 1-D vector into spike trains.

#     Parameters
#     ----------
#     n_processes : int
#         Number of independent processes (neurons).
#     duration : float | dict
#         Duration of the spike train in seconds. Can be a scalar or a dictionary with keys
#         ['dist', 'params'] describing a sampler.
#     rates : float or 1D torch.Tensor
#         Firing rate(s) in Hz. Scalar (applied to all processes) or a tensor of shape [n_processes].
#     dt : float, optional
#         Time resolution in seconds.
#     rng : torch.Generator or None, optional
#         Random number generator for reproducibility. If None, a new generator is created and
#         seeded from torch.seed() (non-deterministic).

#     Returns
#     -------
#     spikes : torch.Tensor, shape (n_processes, n_steps), dtype=torch.int64
#         Binary tensor where 1 indicates a spike.
#     """
#     # Device/dtype: follow `rates` if it's a tensor, else default to CPU/float32
#     device, dtype = get_device_and_dtype(rates)

#     if rng is None:
#         rng = torch.Generator(device=device)
#         rng.manual_seed(torch.seed())

#     # Optional stochastic duration from dict {'dist': callable, 'params': {...}}
#     if isinstance(duration, dict):
#         rounding_precision = -(Decimal(str(dt)).as_tuple().exponent)  # number of decimal digits in dt
#         sampled = duration["dist"](**duration["params"])
#         duration = round(float(sampled), rounding_precision)  # Round to multiples of dt precision

#     num_steps = int(round(float(duration) / float(dt)))

#     # rates -> 1D tensor of length n_processes
#     rates_t = torch.as_tensor(rates, device=device, dtype=dtype).flatten()
#     if rates_t.numel() == 1:
#         rates_t = torch.full((n_processes,), rates_t.item(), device=device, dtype=dtype)
#     elif rates_t.shape[0] != n_processes:
#         raise ValueError("`rates` must be scalar or length n_processes.")

#     # Probability per bin (broadcasted across time)
#     p = rates_t[:, None] * (dt / 1000.0)  # shape [n_processes, 1]
#     p = p.expand(n_processes, num_steps).clone()  # repeat along time like np.tile
#     assert p.shape == (n_processes, num_steps)
#     p = torch.clamp(p, 0.0, 1.0)

#     # Vectorized Bernoulli draws (0/1) (float)
#     spike_pattern = torch.bernoulli(p, generator=rng)

#     return spike_pattern


# # TODO BZ:
# def generate_static_spikes(data, duration, spike_resolution, max_rate, normalize, time_unit):
#     """Generate spikes for static data, preserving spatial dimensions."""
#     original_shape = data.shape

#     # Convert to rates
#     rates = _norm_rates(data, max_rate) if normalize else data.clone()

#     # fig, ax = plt.subplots(figsize=(8, 4))
#     # # for n in range(len(rates[0, 0])):
#     # plt.show()

#     # Calculate number of time bins
#     T_out = int(duration / spike_resolution)
#     if T_out == 0:
#         raise ValueError(f"Duration ({duration}) is smaller than spike_resolution ({spike_resolution})")

#     # Create output tensor with time dimension first, preserving spatial structure
#     output_shape = (T_out,) + original_shape
#     rates_expanded = rates.unsqueeze(0).expand(output_shape)

#     # Convert ms to seconds or use resolution directly as time unit (steps mode)
#     dt = spike_resolution / 1000.0 if time_unit == "ms" else spike_resolution

#     # Generate spikes using Poisson process
#     spike_prob = rates_expanded * dt

#     # Warning for high spike probabilities
#     max_prob = spike_prob.max().item()
#     if max_prob > 1.0:
#         print(
#             f"Warning: Maximum spike probability {max_prob:.3f} > 1.0. Consider reducing max_rate or spike_resolution."
#         )

#     spike_train = (torch.rand_like(spike_prob) < spike_prob).float()

#     return spike_train
