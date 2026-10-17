import torch
import pytest
import torch.nn.functional as F

from seqbench.transforms.base import TimeGrid
from seqbench.transforms.generic.conv import TemporalFilter, TemporalUnfold
from seqbench.transforms.generic.conv import make_temporal_kernel


class TestMakeTemporalKernel:
    """Tests for the make_temporal_kernel function."""

    def test_box_kernel(self):
        """Test box kernel generation."""
        kernel = make_temporal_kernel(shape="box", width=5.0, height=2.0, dt=1.0)
        assert kernel.shape[0] == 6  # 0 to 5 inclusive
        assert torch.allclose(kernel, torch.ones(6) * 2.0)

    def test_exp_kernel(self):
        """Test exponential kernel generation."""
        kernel = make_temporal_kernel(shape="exp", width=10.0, height=1.0, dt=1.0, tau=2.0)
        assert kernel.shape[0] == 11
        assert kernel[0] == 1.0  # At t=0, exp(-0/tau) = 1
        assert kernel[-1] < kernel[0]  # Decaying

    def test_double_exp_kernel(self):
        """Test double exponential kernel generation."""
        kernel = make_temporal_kernel(shape="double_exp", width=10.0, height=1.0, dt=1.0, tau_1=1.0, tau_2=3.0)
        assert kernel.shape[0] == 11
        assert torch.max(kernel) == 1.0  # Normalized to height

    def test_alpha_kernel(self):
        """Test alpha function kernel generation."""
        kernel = make_temporal_kernel(shape="alpha", width=10.0, height=1.0, dt=1.0, tau=2.0)
        assert kernel.shape[0] == 11
        assert kernel[0] == 0.0  # Alpha function starts at 0
        assert torch.max(kernel) == 1.0

    def test_gauss_kernel(self):
        """Test Gaussian kernel generation."""
        kernel = make_temporal_kernel(shape="gauss", width=10.0, height=1.0, dt=1.0, mu=5.0, sigma=1.0)
        assert kernel.shape[0] == 11
        assert torch.max(kernel) == 1.0
        # Peak should be near mu
        peak_idx = torch.argmax(kernel)
        assert abs(peak_idx - 5) <= 1

    def test_tri_kernel(self):
        """Test triangular kernel generation."""
        kernel = make_temporal_kernel(shape="tri", width=10.0, height=1.0, dt=1.0)
        assert torch.max(kernel) == 1.0
        # Should be symmetric (approximately)
        assert kernel.shape[0] > 0

    def test_sin_kernel(self):
        """Test sinusoidal kernel generation."""
        kernel = make_temporal_kernel(
            shape="sin", width=10.0, height=2.0, out_dt=0.1, frequency=1.0, phase_shift=0.0, mean_amplitude=1.0
        )
        assert kernel.shape[0] == 101
        # Check that values oscillate around mean
        assert torch.min(kernel) < 1.0
        assert torch.max(kernel) > 1.0

    def test_normalization(self):
        """Test kernel normalization."""
        kernel = make_temporal_kernel(shape="box", width=5.0, height=2.0, dt=1.0, normalize=True)
        assert torch.allclose(torch.sum(kernel), torch.tensor(1.0))

    def test_device_and_dtype(self):
        """Test kernel creation on different devices and dtypes."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        dtype = torch.float64
        kernel = make_temporal_kernel(shape="box", width=5.0, height=1.0, dt=1.0, device=device, dtype=dtype)
        assert kernel.device.type == device.type
        assert kernel.dtype == dtype

    def test_missing_parameters(self):
        """Test that missing required parameters raise explicit errors."""
        with pytest.raises(ValueError):
            make_temporal_kernel(shape="exp", width=5.0)

        with pytest.raises(ValueError):
            make_temporal_kernel(shape="double_exp", width=5.0, tau_1=1.0)

        with pytest.raises(ValueError):
            make_temporal_kernel(shape="alpha", width=5.0)

        with pytest.raises(ValueError):
            make_temporal_kernel(shape="gauss", width=5.0, mu=0.0)

        with pytest.raises(ValueError):
            make_temporal_kernel(shape="sin", width=5.0, frequency=1.0, phase_shift=0.0)


class TestTemporalUnfold:
    """Tests for the TemporalUnfold class."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)
        assert tc.out_dt == 0.1
        assert tc.duration == 1.0
        assert tc.time_axis.numel() == 10

    def test_duration_must_align_with_out_dt(self):
        """Test that duration must map to an integer number of output steps."""
        kernel_spec = {"shape": "box", "params": {}}
        with pytest.raises(ValueError, match="duration must align with out_dt"):
            TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.05)

    def test_kernel_spec_must_be_dict(self):
        """Test that invalid kernel specs raise explicit type errors."""
        with pytest.raises(TypeError, match="kernel_spec must be a dict"):
            TemporalUnfold(kernel_spec="box", out_dt=0.1, duration=1.0)

    def test_kernel_spec_requires_shape_and_params(self):
        """Test that incomplete kernel specs raise explicit value errors."""
        with pytest.raises(ValueError, match="missing required key"):
            TemporalUnfold(kernel_spec={"shape": "box"}, out_dt=0.1, duration=1.0)

    def test_kernel_spec_params_must_be_dict(self):
        """Test that kernel params must be a dict."""
        with pytest.raises(TypeError, match=r"params.*must be a dict"):
            TemporalUnfold(kernel_spec={"shape": "box", "params": []}, out_dt=0.1, duration=1.0)

    def test_list_amplitudes_are_rejected(self):
        """Test that unsupported list amplitudes fail explicitly."""
        kernel_spec = {"shape": "box", "params": {}}
        with pytest.raises(ValueError, match="List amplitudes are not supported"):
            TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, amplitude=[1.0])

    def test_initialization_with_num_steps(self):
        """Test initialization with explicit num_steps."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, num_steps=20, duration=2.0)
        assert tc.time_axis.numel() == 20

    def test_initialization_with_num_steps_only(self):
        """Test initialization when only explicit num_steps is provided."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, num_steps=20)
        assert tc.time_axis.numel() == 20

    def test_inconsistent_parameters(self):
        """Test that inconsistent parameters raise error."""
        kernel_spec = {"shape": "box", "params": {}}
        with pytest.raises(ValueError):
            TemporalUnfold(
                kernel_spec=kernel_spec, out_dt=0.1, num_steps=20, duration=1.0  # Inconsistent: should be 2.0
            )

    def test_amplitude_scaling(self):
        """Test amplitude scaling."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, amplitude=2.0)

        x = torch.ones(5)
        result = tc(x)

        # Result should be scaled by amplitude
        assert result.shape == (10, 5)
        assert torch.max(result) > 1.0

    def test_spike_position_center(self):
        """Test spike positioned at center."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, impulse_pos="center")

        x = torch.tensor([1.0])
        result = tc(x)

        assert result.shape == (10, 1)
        # Peak should be near center
        peak_idx = torch.argmax(result[:, 0])
        assert abs(peak_idx - 5) <= 1

    def test_spike_position_start(self):
        """Test spike positioned at start."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, impulse_pos="start")

        x = torch.tensor([1.0])
        result = tc(x)

        # Response should start early
        assert result[0, 0] > 0

    def test_spike_position_end(self):
        """Test spike positioned at end."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, impulse_pos="end")

        x = torch.tensor([1.0])
        result = tc(x)

        # Response should peak near end
        assert result[-1, 0] > 0

    def test_spike_position_integer(self):
        """Test spike positioned at specific index."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0, impulse_pos=3)

        x = torch.tensor([1.0])
        result = tc(x)

        # Peak should be near index 3
        peak_idx = torch.argmax(result[:, 0])
        assert abs(peak_idx - 3) <= 1

    def test_multidimensional_input(self):
        """Test with multi-dimensional input."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)

        x = torch.ones(3, 4, 5)
        result = tc(x)

        # Time dimension should be first
        assert result.shape == (10, 3, 4, 5)

    def test_exponential_kernel_convolution(self):
        """Test convolution with exponential kernel."""
        kernel_spec = {"shape": "exp", "params": {"tau": 0.2}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.05, duration=1.0, impulse_pos="start")

        x = torch.tensor([1.0])
        result = tc(x)

        # Should decay exponentially
        assert result.shape == (20, 1)
        assert result[0, 0] > result[5, 0] > result[10, 0]

    def test_gaussian_kernel_convolution(self):
        """Test convolution with Gaussian kernel."""
        kernel_spec = {"shape": "gauss", "params": {"mu": 0.0, "sigma": 0.1}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.05, duration=1.0, impulse_pos="center")

        x = torch.tensor([1.0])
        result = tc(x)

        # Should have Gaussian shape centered
        assert result.shape == (20, 1)
        peak_idx = torch.argmax(result[:, 0])
        assert abs(peak_idx - 10) <= 2

    def test_multiple_inputs(self):
        """Test with multiple non-zero inputs."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)

        x = torch.tensor([1.0, 0.0, 0.5, 0.0, 2.0])
        result = tc(x)

        # Each non-zero element should produce a response
        assert result.shape == (10, 5)
        assert torch.sum(result[:, 0]) > 0  # First element
        assert torch.sum(result[:, 1]) == 0  # Second element (zero)
        assert torch.sum(result[:, 2]) > 0  # Third element
        assert torch.sum(result[:, 4]) > torch.sum(result[:, 0])  # Scaled by 2

    def test_3d_tensor_shape(self):
        """Test output shape for 3D tensor input."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)

        # Test with shape (2, 3, 4) - batch, height, width
        x = torch.ones(2, 3, 4)
        result = tc(x)

        # Expected: (T, 2, 3, 4) - time first, then original dimensions
        assert result.shape == (10, 2, 3, 4), f"Expected (10, 2, 3, 4), got {result.shape}"

    def test_zero_input(self):
        """Test with all-zero input."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)

        x = torch.zeros(5)
        result = tc(x)

        assert result.shape == (10, 5)
        assert torch.allclose(result, torch.zeros_like(result))

    def test_device_consistency(self):
        """Test that output is on same device as input."""
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.1, duration=1.0)

        if torch.cuda.is_available():
            x = torch.ones(5, device="cuda")
            result = tc(x)
            assert result.device.type == "cuda"
        else:
            x = torch.ones(5, device="cpu")
            result = tc(x)
            assert result.device.type == "cpu"

    def test_convolution_centering(self):
        """Test that convolution properly centers the kernel at spike position."""
        # Use a simple box kernel for easy verification
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=1.0, duration=20.0, impulse_pos=10)  # Spike at index 10

        x = torch.tensor([1.0])
        result = tc(x)

        print(f"\nKernel length: {tc.kernel.numel()}")
        print(f"Time axis length: {tc.time_axis.numel()}")
        print(f"Result shape: {result.shape}")
        print(f"Result values around spike position:")
        for i in range(max(0, 10 - 3), min(result.shape[0], 10 + 4)):
            print(f"  t={i}: {result[i, 0].item():.4f}")

        # The response should be centered around the spike position
        # For a box kernel of width 3, we expect non-zero values around index 10
        assert result[10, 0] > 0, "Expected non-zero response at spike position"

    def test_exponential_decay_direction(self):
        """Test that exponential kernel decays in the correct direction (forward in time)."""
        kernel_spec = {"shape": "exp", "params": {"tau": 2.0}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=1.0, duration=20.0, impulse_pos=5)  # Spike at index 5

        x = torch.tensor([1.0])
        result = tc(x)

        print(f"\nExponential kernel convolution result:")
        for i in range(result.shape[0]):
            print(f"  t={i}: {result[i, 0].item():.4f}")

        # Find the peak
        peak_idx = torch.argmax(result[:, 0]).item()
        print(f"Peak at index: {peak_idx}")

        # After the peak, values should decay
        # This tests if the convolution is properly oriented
        if peak_idx < result.shape[0] - 3:
            assert result[peak_idx, 0] > result[peak_idx + 2, 0], "Expected exponential decay after peak"

    def test_impulse_response_location(self):
        """Test that impulse response appears at correct time for different spike positions."""
        kernel_spec = {"shape": "gauss", "params": {"mu": 0.0, "sigma": 1.0}}

        for spike_pos in [0, 5, 10]:
            tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=0.5, duration=10.0, impulse_pos=spike_pos)

            x = torch.tensor([1.0])
            result = tc(x)

            peak_idx = torch.argmax(result[:, 0]).item()
            print(f"\nSpike pos {spike_pos}: Peak at index {peak_idx}")

            # Peak should be near the spike position (within kernel width)
            assert abs(peak_idx - spike_pos) <= 3, f"Peak at {peak_idx} too far from spike position {spike_pos}"

    def test_kernel_application_manual_verification(self):
        """Manually verify convolution with a simple box kernel."""
        # Create a box kernel of width 4 (5 points: 0,1,2,3,4)
        kernel_spec = {"shape": "box", "params": {}}
        tc = TemporalUnfold(kernel_spec=kernel_spec, out_dt=1.0, duration=10.0, impulse_pos=5, amplitude=1.0)

        # Override kernel with simple one for testing
        tc.kernel = torch.tensor([1.0, 1.0, 1.0])  # 3-point box

        x = torch.tensor([1.0])
        result = tc(x)

        print(f"\nManual verification test:")
        print(f"Kernel: {tc.kernel}")
        print(f"Spike at index: 5")
        print(f"Result:")
        for i in range(result.shape[0]):
            print(f"  t={i}: {result[i, 0].item():.4f}")

        # With a 3-point kernel [1,1,1] and spike at position 5,
        # using "same" padding, we expect non-zero values around index 5
        non_zero_count = (result[:, 0] > 0.01).sum().item()
        print(f"Non-zero time points: {non_zero_count}")

        # Should have approximately kernel_length non-zero points
        assert non_zero_count >= 2, "Expected at least 2 non-zero points from convolution"

    def test_conv1d_behavior(self):
        """Test to understand F.conv1d behavior - correlation vs convolution."""
        print("\n=== Testing F.conv1d behavior ===")

        # Create a simple signal with a spike at position 3
        signal = torch.zeros(1, 1, 10)
        signal[0, 0, 3] = 1.0

        # Create an asymmetric kernel [1, 2, 3] to see direction
        kernel = torch.tensor([1.0, 2.0, 3.0]).view(1, 1, -1)

        # Test 1: Without flipping
        result_no_flip = F.conv1d(signal, kernel, padding="same")
        print("\nKernel: [1, 2, 3]")
        print("Signal: spike at index 3")
        print("Result WITHOUT flipping kernel:")
        print(result_no_flip[0, 0])

        # Test 2: With flipping
        kernel_flipped = kernel.flip(-1)  # [3, 2, 1]
        result_with_flip = F.conv1d(signal, kernel_flipped, padding="same")
        print("\nKernel flipped: [3, 2, 1]")
        print("Result WITH flipping kernel:")
        print(result_with_flip[0, 0])

        # Test 3: What we want for causal response
        print("\n=== What we WANT for causal (exponential-like) kernel ===")
        print("Kernel [3, 2, 1] represents decay: [peak, decay1, decay2]")
        print("Spike at t=3, response should be:")
        print("  t=3: 3.0 (peak)")
        print("  t=4: 2.0 (first decay)")
        print("  t=5: 1.0 (second decay)")

    def test_exponential_kernel_manually(self):
        """Manually verify exponential kernel behavior."""
        print("\n=== Manual exponential kernel test ===")

        # Create exponential kernel
        tau = 2.0
        dt = 1.0
        width = 10.0
        kernel = make_temporal_kernel(shape="exp", width=width, height=1.0, dt=dt, tau=tau)

        print(f"Exponential kernel (tau={tau}):")
        print(kernel)
        print(f"Kernel should decay: [1.0, 0.606, 0.368, 0.223, ...]")

        # Create spike at position 5
        signal = torch.zeros(1, 1, 20)
        signal[0, 0, 5] = 1.0

        # Test without flipping
        kernel_conv = kernel.view(1, 1, -1)
        result_no_flip = F.conv1d(signal, kernel_conv, padding="same")

        print("\nResult WITHOUT flipping (cross-correlation):")
        for i in range(20):
            if result_no_flip[0, 0, i] > 0.01:
                print(f"  t={i}: {result_no_flip[0, 0, i].item():.4f}")

        # Test with flipping
        kernel_flipped = kernel_conv.flip(-1)
        result_with_flip = F.conv1d(signal, kernel_flipped, padding="same")

        print("\nResult WITH flipping (true convolution):")
        for i in range(20):
            if result_with_flip[0, 0, i] > 0.01:
                print(f"  t={i}: {result_with_flip[0, 0, i].item():.4f}")

        print("\nFor CAUSAL response (decay forward in time after spike at t=5):")
        print("We want: t=5: 1.0, t=6: 0.606, t=7: 0.368, t=8: 0.223")
        print("This means we should NOT flip the kernel!")

    def test_correct_causal_convolution(self):
        """Test the correct way to apply causal convolution."""
        print("\n=== Testing correct causal convolution approach ===")

        # Exponential kernel for causal filtering
        tau = 2.0
        dt = 1.0
        width = 10.0
        kernel = make_temporal_kernel(shape="exp", width=width, height=1.0, dt=dt, tau=tau)

        # Spike at position 5
        T = 20
        spike_pos = 5
        signal = torch.zeros(1, 1, T)
        signal[0, 0, spike_pos] = 1.0

        # For causal response, we want kernel[0] at spike_pos, kernel[1] at spike_pos+1, etc.
        # F.conv1d does: output[t] = sum_k kernel[k] * input[t - k + offset]
        # Without flip and with 'same' padding, this is cross-correlation

        kernel_conv = kernel.view(1, 1, -1)

        # Try different approaches
        print("Approach 1: No flip, 'same' padding")
        result1 = F.conv1d(signal, kernel_conv, padding="same")
        print(f"Peak at: {torch.argmax(result1[0,0]).item()}, value: {torch.max(result1[0,0]).item():.4f}")

        print("\nApproach 2: Flip kernel, 'same' padding")
        result2 = F.conv1d(signal, kernel_conv.flip(-1), padding="same")
        print(f"Peak at: {torch.argmax(result2[0,0]).item()}, value: {torch.max(result2[0,0]).item():.4f}")

        print(f"\nSpike was at position: {spike_pos}")
        print("For causal (forward) decay, peak should be at spike position or just after")

    ################################################################################

    def test_fixed_convolution(self):
        """Test that the fixed convolution produces correct causal response."""
        print("\n=== Testing FIXED convolution ===")

        # Create exponential kernel
        tau = 2.0
        dt = 1.0
        width = 10.0
        from seqbench.transforms.generic.conv import make_temporal_kernel

        kernel = make_temporal_kernel(shape="exp", width=width, height=1.0, dt=dt, tau=tau)

        print(f"Exponential kernel: {kernel[:5]} ...")

        # Create spike at position 5
        T = 20
        spike_pos = 5
        signal = torch.zeros(1, 1, T)
        signal[0, 0, spike_pos] = 1.0

        # Apply fixed approach
        kernel_len = kernel.numel()
        kernel_flipped = kernel.flip(0).view(1, 1, -1)
        pad_left = kernel_len - 1

        signal_padded = F.pad(signal, (pad_left, 0), mode="constant", value=0)
        result = F.conv1d(signal_padded, kernel_flipped, padding=0)

        # Trim to original length
        result = result[:, :, :T]

        print(f"\nResult with fixed approach (spike at t={spike_pos}):")
        for i in range(T):
            if result[0, 0, i] > 0.01:
                print(f"  t={i}: {result[0, 0, i].item():.4f}")

        print(f"\nExpected: t=5: 1.0, t=6: 0.606, t=7: 0.368, t=8: 0.223")
        print(f"Peak at: t={torch.argmax(result[0, 0]).item()}")

        # Verify
        assert torch.argmax(result[0, 0]).item() == spike_pos, "Peak should be at spike position"
        assert abs(result[0, 0, spike_pos].item() - 1.0) < 0.01, "Peak value should be ~1.0"
        assert result[0, 0, spike_pos + 1].item() < result[0, 0, spike_pos].item(), "Should decay"


class TestTemporalFilter:
    def test_kernel_spec_must_be_dict(self):
        with pytest.raises(TypeError, match="kernel_spec must be a dict"):
            TemporalFilter(kernel_spec="box")

    def test_kernel_spec_requires_shape_and_params(self):
        with pytest.raises(ValueError, match="missing required key"):
            TemporalFilter(kernel_spec={"shape": "box"})

    def test_requires_resolved_time_grid_before_call(self):
        filt = TemporalFilter(kernel_spec={"shape": "box", "params": {"width": 0.2}})
        with pytest.raises(RuntimeError, match="resolve_time_grid"):
            filt(torch.ones(10, 2))

    def test_time_spec_requires_input_grid(self):
        filt = TemporalFilter(kernel_spec={"shape": "box", "params": {"width": 0.2}})
        assert filt.time_spec(None).kind == "require"

    def test_preserves_time_grid_and_shape(self):
        filt = TemporalFilter(kernel_spec={"shape": "box", "params": {"width": 0.2}})
        spec = filt.time_spec(TimeGrid(0.1))
        assert spec.kind == "preserve"
        assert filt.kernel is None
        filt.bind_time_grid(TimeGrid(0.1), TimeGrid(0.1))
        x = torch.zeros(10, 2)
        x[2, 0] = 1.0
        out = filt(x)
        assert out.shape == x.shape
        assert out[2:, 0].sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
