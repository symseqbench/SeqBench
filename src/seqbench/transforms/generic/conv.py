# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Temporal unfolding and filtering transforms for sequence data processing.
"""

from decimal import Decimal
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import logging

from seqbench.transforms.generic.utils import get_device_and_dtype
from seqbench.transforms.base import TransformTimeSpec

logger = logging.getLogger(__name__)


def _validate_kernel_spec(kernel_spec, transform_name: str) -> dict:
    if not isinstance(kernel_spec, dict):
        raise TypeError(f"{transform_name} kernel_spec must be a dict with 'shape' and 'params'")

    missing = {"shape", "params"} - set(kernel_spec)
    if missing:
        raise ValueError(f"{transform_name} kernel_spec missing required key(s): {sorted(missing)}")

    if not isinstance(kernel_spec["params"], dict):
        raise TypeError(f"{transform_name} kernel_spec['params'] must be a dict")

    return kernel_spec


def make_temporal_kernel(
    shape: str,
    width: float = 3.0,
    height: float = 1.0,
    dt: float = 1.0,
    normalize: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    """
    Create a smoothing kernel for convolution operations using PyTorch tensors.

    Parameters
    ----------
    shape : str
        Kernel type {'box', 'exp', 'alpha', 'double_exp', 'gauss', 'tri', 'sin'}
    width : float, optional
        Kernel width in seconds, by default 3.0.
    height : float, optional
        Peak amplitude of the kernel, by default 1.0
    dt : float, optional
        Kernel sampling interval in seconds per point, by default 1.0.
    normalize : bool, optional
        Whether to normalize the kernel, by default False
    device : torch.device, optional
        PyTorch device to place tensor on, by default None (uses CPU)
    dtype : torch.dtype, optional
        PyTorch data type, by default torch.float32
    **kwargs : dict
        Additional parameters specific to each kernel type:
        - 'exp': tau (time constant)
        - 'double_exp': tau_1, tau_2 (time constants)
        - 'alpha': tau (time constant)
        - 'gauss': mu (mean), sigma (standard deviation)
        - 'sin': frequency, phase_shift, mean_amplitude

    Returns
    -------
    torch.Tensor
        1D kernel tensor
    """
    if "out_dt" in kwargs:
        out_dt = kwargs.pop("out_dt")
        if dt != 1.0 and not np.isclose(dt, out_dt):
            raise ValueError("Specify either dt or out_dt for make_temporal_kernel, not both with different values")
        dt = out_dt

    if device is None:
        device = torch.device("cpu")

    # Create time axis
    n_points = int((width / dt) + 1)
    x = torch.arange(0.0, n_points, dtype=dtype, device=device) * dt

    if shape == "box":
        k = torch.ones_like(x) * height

    elif shape == "exp":
        if "tau" not in kwargs:
            raise ValueError("for exponential kernel, please specify tau")
        tau = kwargs["tau"]
        k = torch.exp(-x / tau) * height

    elif shape == "double_exp":
        if "tau_1" not in kwargs:
            raise ValueError("for double exponential kernel, please specify tau_1")
        if "tau_2" not in kwargs:
            raise ValueError("for double exponential kernel, please specify tau_2")
        tau_1 = kwargs["tau_1"]
        tau_2 = kwargs["tau_2"]
        tmp_k = -torch.exp(-x / tau_1) + torch.exp(-x / tau_2)
        k = tmp_k * (height / torch.max(tmp_k))

    elif shape == "alpha":
        if "tau" not in kwargs:
            raise ValueError("for alpha kernel, please specify tau")
        tau = kwargs["tau"]
        tmp_k = (x / tau) * torch.exp(-x / tau)
        k = tmp_k * (height / torch.max(tmp_k))

    elif shape == "gauss":
        if "mu" not in kwargs:
            raise ValueError("for Gaussian kernel, please specify mu")
        if "sigma" not in kwargs:
            raise ValueError("for Gaussian kernel, please specify sigma")
        sigma = kwargs["sigma"]
        mu = kwargs["mu"]
        tmp_k = (1.0 / (sigma * torch.sqrt(2.0 * torch.tensor(torch.pi, device=device)))) * torch.exp(
            -((x - mu) ** 2.0 / (2.0 * (sigma**2.0)))
        )
        k = tmp_k * (height / torch.max(tmp_k))

    elif shape == "tri":
        halfwidth = int(width / 2)
        trileft = torch.arange(1, halfwidth + 2, dtype=dtype, device=device)
        triright = torch.arange(halfwidth, 0, -1, dtype=dtype, device=device)
        k = torch.cat([trileft, triright])
        k = k * height / torch.max(k)  # Normalize to height

    elif shape == "sin":
        if "frequency" not in kwargs:
            raise ValueError("for sin kernel, please specify frequency")
        if "phase_shift" not in kwargs:
            raise ValueError("for sin kernel, please specify phase_shift")
        if "mean_amplitude" not in kwargs:
            raise ValueError("for sin kernel, please specify mean_amplitude")
        frequency = kwargs["frequency"]
        phase_shift = kwargs["phase_shift"]
        mean_amplitude = kwargs["mean_amplitude"]
        k = torch.sin(2 * torch.pi * x / width * frequency + phase_shift) * height
        k = k + mean_amplitude

    else:
        logger.info(
            f"Kernel '{shape}' not implemented, please choose from "
            "{'box', 'exp', 'alpha', 'double_exp', 'gauss', 'tri', 'sin'}"
        )
        k = torch.zeros(1, dtype=dtype, device=device)

    if normalize:
        k = k / torch.sum(k)

    return k


class TemporalUnfold:
    """Expand a static tensor into a temporal kernel response.

    The input is treated as non-temporal feature data. The transform creates a
    new leading time axis sampled at ``out_dt`` seconds per row and places one
    impulse response per active input value. ``duration`` and kernel time
    constants are also expressed in seconds.
    """

    def __init__(
        self,
        kernel_spec,
        out_dt,
        duration=None,
        num_steps=None,
        amplitude: float | dict = 1.0,
        impulse_pos="start",
        **torch_kwargs,
    ):
        """
        Unfolds a static tensor into a temporal kernel response by adding a
        leading time axis.


        Parameters
        ----------
        kernel_spec : dict
            Tuple of (kernel_type, kernel_params) where kernel_type is a string and
            kernel_params is a dict of parameters for the kernel
        out_dt : float
            Output grid resolution in seconds per row.
        amplitude : float or torch.Tensor, optional
            Amplitude scaling factor. Can be scalar, tensor broadcastable to `x`, or a distribution. Default is 1.0.
        duration : float, optional
            Physical duration of unfolded signal in seconds. If ``num_steps`` is
            omitted, the output length is ``floor(duration / out_dt)``.
        num_steps : int, optional
            Dimensionless output row count. If both ``duration`` and
            ``num_steps`` are supplied, they must satisfy
            ``num_steps == floor(duration / out_dt)``.
        impulse_pos : str or int, optional
            Where to center the kernel response:
            - 'center': middle of time axis
            - 'start': beginning of time axis
            - 'end': end of time axis
            - int: specific time index
            - float: specific time value (will find nearest index)
            By default 'start'.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (T, *x.shape).
            Each active element in ``x`` becomes a temporal kernel response.
        """

        kernel_spec = _validate_kernel_spec(kernel_spec, "TemporalUnfold")

        # case where amplitudes and/or durations are drawn from distribution
        if isinstance(amplitude, dict):
            amplitude = amplitude["dist"](**amplitude["params"])
        elif isinstance(amplitude, list):
            raise ValueError("List amplitudes are not supported; use a scalar, tensor, or distribution dict")
        else:
            amplitude = amplitude

        if isinstance(duration, dict):
            rounding_precision = -(Decimal(str(out_dt)).as_tuple().exponent)  # nr of decimal digits
            duration = np.round(duration["dist"](**duration["params"]), rounding_precision)

        self.kernel_spec = kernel_spec
        self.out_dt = out_dt
        self.duration = duration
        self.amp = amplitude
        self.impulse_pos = impulse_pos

        bad = set(torch_kwargs) - {"device", "dtype"}
        if bad:
            raise ValueError(f"Unknown parameter(s) for TemporalUnfold: {sorted(bad)}")

        kwargs = kernel_spec["params"] | torch_kwargs
        self.kernel = make_temporal_kernel(
            kernel_spec["shape"], height=1.0, dt=out_dt, normalize=False, **kwargs
        )

        if duration is not None:
            ratio = float(duration) / float(out_dt)
            duration_steps = int(round(ratio))
            if not np.isclose(ratio, duration_steps, rtol=0.0, atol=1e-9):
                raise ValueError(
                    "TemporalUnfold duration must align with out_dt: "
                    f"duration / out_dt must be an integer, got {duration!r} / {out_dt!r}"
                )
            if num_steps is not None and num_steps != duration_steps:
                raise ValueError("When both are specified, out_dt, num_steps and duration must be consistent!")
            num_steps = duration_steps
        elif num_steps is None:
            raise ValueError("TemporalUnfold requires either duration or num_steps")

        self.time_axis = torch.arange(num_steps) * out_dt

    def time_spec(self, input_grid):
        return TransformTimeSpec("create", out_dt=self.out_dt)

    def expected_time_steps(self, input_steps, input_shape=None):
        return self.time_axis.numel()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape. Active elements will generate temporal responses.

        Returns
        -------
        torch.Tensor
            Output tensor with shape (T, *x.shape).
            Each active element in ``x`` becomes a temporal kernel response.
        """
        # ----- 1. Determine spike index -----
        device, dtype = get_device_and_dtype(x)
        T = self.time_axis.numel()

        if isinstance(self.impulse_pos, int):
            idx = self.impulse_pos
        else:
            if self.impulse_pos == "start":
                idx = 0
            elif self.impulse_pos == "center":
                idx = T // 2
            elif self.impulse_pos == "end":
                idx = T - 1
            else:
                raise ValueError(f"Invalid impulse_pos: {self.impulse_pos}")

        # ----- 2. Compute amplitudes -----
        if self.amp is not None:
            amps = x * self.amp
        elif not is_binary(x):
            amps = x
        else:
            amps = (x != 0).float()

        # Create spike trains with output shape: (*x_shape, T)
        shape = x.shape
        spikes = torch.zeros(*shape, T, device=x.device, dtype=torch.float32)
        spikes[..., idx] = amps  # place all impulses at chosen index

        # Prepare convolution kernel
        kernel = self.kernel.to(x.device).float()
        kernel_len = kernel.numel()

        # F.conv1d does cross-correlation, so flip kernel for true convolution
        kernel_flipped = kernel.flip(0).view(1, 1, -1)

        # For causal response: kernel[0] at spike_pos, kernel[1] at spike_pos+1, etc.
        # We need left-padding only (pad before the signal, not after)
        pad_left = kernel_len - 1

        # Batched convolution
        spikes_flat = spikes.reshape(-1, 1, T)  # (N, 1, T)

        # Manual padding - only on the left for causal response
        spikes_padded = F.pad(spikes_flat, (pad_left, 0), mode="constant", value=0)

        # Convolve with no additional padding
        conv_flat = F.conv1d(spikes_padded, kernel_flipped, padding=0)

        # The output should now be exactly length T
        # Verify and trim if needed
        if conv_flat.shape[2] > T:
            conv_flat = conv_flat[:, :, :T]

        conv = conv_flat.reshape(*shape, T)  # reshape back
        conv = conv.permute(-1, *range(conv.ndim - 1))  # Move time axis to front

        return conv
        # sns.heatmap(x.T, cmap="viridis")
        # plt.show()


class TemporalFilter:
    """Apply a causal temporal kernel to an existing leading time axis.

    The transform requires a resolved input ``TimeGrid`` from ``Compose`` and
    preserves that grid. Kernels are sampled using the input grid's ``dt``.
    """

    def __init__(self, kernel_spec, **torch_kwargs):
        """
        Apply a causal temporal filter to an existing leading time axis.

        The input time grid is supplied by ``Compose.resolve_time_grid``. The
        filter preserves the input grid resolution.
        """
        kernel_spec = _validate_kernel_spec(kernel_spec, "TemporalFilter")

        bad = set(torch_kwargs) - {"device", "dtype"}
        if bad:
            raise ValueError(f"Unknown parameter(s) for TemporalFilter: {sorted(bad)}")

        self.kernel_spec = kernel_spec
        self.torch_kwargs = torch_kwargs
        self.dt = None
        self.kernel = None

    def time_spec(self, input_grid):
        if input_grid is None:
            return TransformTimeSpec("require")
        return TransformTimeSpec("preserve", expected_in_dt=input_grid.dt)

    def bind_time_grid(self, input_grid, output_grid):
        if input_grid is None:
            raise ValueError("TemporalFilter requires an input time grid")
        self.dt = input_grid.dt
        kwargs = self.kernel_spec["params"] | self.torch_kwargs
        self.kernel = make_temporal_kernel(
            self.kernel_spec["shape"],
            height=1.0,
            dt=input_grid.dt,
            normalize=False,
            **kwargs,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            raise RuntimeError(
                "TemporalFilter requires Compose.resolve_time_grid() before it can be called"
            )
        if x.ndim < 1:
            raise ValueError("TemporalFilter expects a leading time axis")

        T = x.shape[0]
        feature_shape = x.shape[1:]
        kernel = self.kernel.to(x.device).float()
        kernel_flipped = kernel.flip(0).view(1, 1, -1)
        pad_left = kernel.numel() - 1

        x_flat = x.reshape(T, -1).transpose(0, 1).unsqueeze(1)
        x_padded = F.pad(x_flat, (pad_left, 0), mode="constant", value=0)
        filtered = F.conv1d(x_padded, kernel_flipped, padding=0)
        if filtered.shape[2] > T:
            filtered = filtered[:, :, :T]
        return filtered.squeeze(1).transpose(0, 1).reshape(T, *feature_shape)
        # sns.lineplot(kernel)
        # plt.show()


def is_binary(tensor: torch.Tensor, threshold: float = 1e-6) -> bool:
    """
    Check if tensor contains only binary values (0 and 1).

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to check
    threshold : float, optional
        Tolerance for floating point comparison, by default 1e-6

    Returns
    -------
    bool
        True if tensor contains only binary values
    """
    unique_vals = torch.unique(tensor)
    return len(unique_vals) <= 2 and torch.allclose(unique_vals, torch.round(unique_vals), atol=threshold)
