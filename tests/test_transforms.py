# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Tests for the ``seqbench.transforms`` package.

Covers the package machinery (auto-import, config-driven composition, param
normalization), the composition/functional wrappers, and the individual
transform classes under ``generic/``, ``vision/``, ``snn/`` and ``audio/``.

The temporal-convolution kernel/transform (``generic/conv.py``) has dedicated
coverage in ``test_conv.py`` and is only touched here for the cross-cutting
helpers (``get_device_and_dtype``, ``is_binary``).
"""

import numpy as np
import pytest
import torch

import seqbench.transforms as T
from seqbench.transforms import (
    compose_transforms_from_config,
    _normalize_params,
)
from seqbench.transforms.base import Transform, TimeGrid, TransformTimeSpec
from seqbench.transforms.compose import Compose, ApplyToKey
from seqbench.transforms.functional import FunctionalTransform
from seqbench.transforms.generic.expand_dim import ExpandDim
from seqbench.transforms.generic.utils import get_device_and_dtype
from seqbench.transforms.generic.conv import is_binary
from seqbench.transforms.vision.center_crop import CenterCrop
from seqbench.transforms.vision.flatten import Flatten
from seqbench.transforms.snn.bin import BinAlongAxis
from seqbench.transforms.snn.rate_coding import RateCoding
from seqbench.transforms.snn.poisson_encoding import PoissonEncoding
from seqbench.transforms.snn.utils import spike_times_to_tensor
from seqbench.transforms.audio.power_law import PowerLaw
from seqbench.transforms.audio.adaptive_percentile_norm import AdaptivePercentileNorm

from seqbench.config import InputMappingCfg


# ---------------------------------------------------------------------------
# base.Transform protocol
# ---------------------------------------------------------------------------
class TestTransformProtocol:
    def test_plain_callable_is_a_transform(self):
        # Transform is a structural Protocol: anything with __call__(x) matches.
        def fn(x):
            return x

        t: Transform = fn  # static structural check; also exercise at runtime
        assert t(3) == 3

    def test_transform_objects_satisfy_protocol(self):
        assert callable(PowerLaw())
        assert callable(Compose([]))


# ---------------------------------------------------------------------------
# compose.Compose / compose.ApplyToKey
# ---------------------------------------------------------------------------
class TestCompose:
    def test_empty_compose_is_identity(self):
        c = Compose([])
        obj = object()
        assert c(obj) is obj

    def test_applies_in_order(self):
        c = Compose([lambda x: x + 1, lambda x: x * 2])
        # (3 + 1) * 2 = 8, order matters
        assert c(3) == 8

    def test_order_is_not_commutative(self):
        forward = Compose([lambda x: x + 1, lambda x: x * 2])
        reverse = Compose([lambda x: x * 2, lambda x: x + 1])
        assert forward(3) != reverse(3)

    def test_single_transform(self):
        c = Compose([lambda x: x.flatten(start_dim=0)])
        out = c(torch.zeros(2, 3))
        assert out.shape == (6,)

    def test_transforms_attribute_preserved(self):
        ts = [lambda x: x]
        assert Compose(ts).transforms is ts

    def test_resolve_preserve_uses_expected_grid_for_static_input(self):
        c = Compose([ExpandDim(axis=0, target_size=4)])
        out = c.resolve_time_grid(None, TimeGrid(0.1))
        assert out.dt == pytest.approx(0.1)

    def test_resolve_create_sets_current_grid(self):
        class Creator:
            def time_spec(self, input_grid):
                return TransformTimeSpec("create", out_dt=0.01)

            def __call__(self, x):
                return x

        c = Compose([Creator()])
        out = c.resolve_time_grid(None, TimeGrid(0.01))
        assert out.dt == pytest.approx(0.01)

    def test_resolve_resample_updates_grid(self):
        c = Compose([BinAlongAxis(bin_size=2, bin_axis=0)])
        out = c.resolve_time_grid(TimeGrid(0.01), TimeGrid(0.02))
        assert out.dt == pytest.approx(0.02)

    def test_expand_dim_time_axis_rejected_for_temporal_input(self):
        c = Compose([ExpandDim(axis=0, target_size=4)])
        with pytest.raises(ValueError, match="ExpandDim\\(axis=0\\)"):
            c.resolve_time_grid(TimeGrid(0.1), TimeGrid(0.1))

    def test_require_without_input_grid_raises(self):
        c = Compose([PoissonEncoding(temporal=True, spike_dt=0.001)])
        with pytest.raises(ValueError, match="requires an input time grid"):
            c.resolve_time_grid(None, TimeGrid(0.001))

    def test_unknown_transform_raises_in_error_mode(self):
        c = Compose([lambda x: x])
        with pytest.raises(ValueError, match="does not declare time-grid behavior"):
            c.resolve_time_grid(None, TimeGrid(0.1), validation="error")

    def test_unknown_transform_warns_and_uses_expected_grid_in_warn_mode(self):
        c = Compose([lambda x: x])
        with pytest.warns(UserWarning, match="unknown"):
            out = c.resolve_time_grid(None, TimeGrid(0.1), validation="warn")
        assert out.dt == pytest.approx(0.1)

    def test_final_mismatch_raises(self):
        c = Compose([BinAlongAxis(bin_size=2, bin_axis=0)])
        with pytest.raises(ValueError, match="time-grid mismatch"):
            c.resolve_time_grid(TimeGrid(0.01), TimeGrid(0.01))

    def test_runtime_preserve_length_mismatch_raises(self):
        class LyingPreserve:
            def time_spec(self, input_grid):
                return TransformTimeSpec("preserve")

            def __call__(self, x):
                return x[:1]

        c = Compose([LyingPreserve()])
        c.resolve_time_grid(TimeGrid(0.1), TimeGrid(0.1))
        with pytest.raises(ValueError, match="output time length mismatch"):
            c(torch.zeros(3, 2))

    def test_runtime_declared_expected_length_mismatch_raises(self):
        class BadCreator:
            def time_spec(self, input_grid):
                return TransformTimeSpec("create", out_dt=0.1)

            def expected_time_steps(self, input_steps, input_shape=None):
                return 4

            def __call__(self, x):
                return torch.zeros(3, *x.shape)

        c = Compose([BadCreator()])
        c.resolve_time_grid(None, TimeGrid(0.1))
        with pytest.raises(ValueError, match="got 3, expected 4"):
            c(torch.zeros(2))

    def test_config_time_behavior_wraps_foreign_transform(self):
        comp = compose_transforms_from_config(
            InputMappingCfg(
                base="one_hot",
                transforms=[
                    {"type": "torch.clone", "time_behavior": "preserve"},
                ],
            )
        )
        out = comp.resolve_time_grid(TimeGrid(0.1), TimeGrid(0.1))
        assert out.dt == pytest.approx(0.1)
        x = torch.zeros(2, 3)
        assert comp(x).shape == x.shape

    def test_config_time_behavior_create_declares_out_dt(self):
        comp = compose_transforms_from_config(
            InputMappingCfg(
                base="one_hot",
                transforms=[
                    {
                        "type": "torch.clone",
                        "time_behavior": {"kind": "create", "out_dt": 0.05},
                    },
                ],
            )
        )
        out = comp.resolve_time_grid(None, TimeGrid(0.05))
        assert out.dt == pytest.approx(0.05)


class TestApplyToKey:
    def test_transforms_only_target_key(self):
        t = ApplyToKey("data", lambda v: v * 10)
        out = t({"data": 2, "label": 7})
        assert out == {"data": 20, "label": 7}

    def test_does_not_mutate_input(self):
        sample = {"data": 2, "label": 7}
        ApplyToKey("data", lambda v: v + 1)(sample)
        assert sample == {"data": 2, "label": 7}

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            ApplyToKey("missing", lambda v: v)({"data": 1})

    def test_returns_a_new_mapping(self):
        sample = {"data": 1}
        out = ApplyToKey("data", lambda v: v)(sample)
        assert out is not sample


# ---------------------------------------------------------------------------
# functional.FunctionalTransform
# ---------------------------------------------------------------------------
class TestFunctionalTransform:
    def test_wraps_and_calls(self):
        t = FunctionalTransform(torch.add, other=5)
        assert t(torch.tensor(2)).item() == 7

    def test_binds_kwargs(self):
        def f(x, *, scale):
            return x * scale

        assert FunctionalTransform(f, scale=3)(4) == 12

    def test_no_kwargs(self):
        t = FunctionalTransform(abs)
        assert t(-9) == 9

    def test_non_callable_raises(self):
        with pytest.raises(ValueError, match="must be callable"):
            FunctionalTransform(42)

    def test_repr_includes_func_name_and_kwargs(self):
        t = FunctionalTransform(torch.clamp, min=0.0, max=1.0)
        r = repr(t)
        assert "FunctionalTransform" in r
        assert "clamp" in r
        assert "min=0.0" in r and "max=1.0" in r

    def test_repr_no_kwargs(self):
        r = repr(FunctionalTransform(abs))
        assert "abs(" in r


# ---------------------------------------------------------------------------
# package machinery: auto-import + _normalize_params + config composition
# ---------------------------------------------------------------------------
class TestAutoImport:
    @pytest.mark.parametrize(
        "name",
        [
            "Compose",
            "FunctionalTransform",
            "ExpandDim",
            "CenterCrop",
            "Flatten",
            "BinAlongAxis",
            "RateCoding",
            "PoissonEncoding",
            "PowerLaw",
            "AdaptivePercentileNorm",
            "TemporalUnfold",
            "TemporalFilter",
        ],
    )
    def test_transform_exported_at_package_level(self, name):
        assert name in T.__all__
        assert hasattr(T, name)

    def test_private_modules_not_imported(self):
        # Names starting with "_" (e.g. utils helpers) are skipped by walk.
        assert "get_device_and_dtype" not in T.__all__


class TestNormalizeParams:
    def test_string_none_becomes_none(self):
        assert _normalize_params({"a": "None", "b": "none"}) == {"a": None, "b": None}

    def test_torch_dotted_resolves_to_attr(self):
        out = _normalize_params({"dtype": "torch.float64"})
        assert out["dtype"] is torch.float64

    def test_unknown_torch_attr_falls_back_to_string(self):
        out = _normalize_params({"dtype": "torch.not_a_real_attr"})
        assert out["dtype"] == "torch.not_a_real_attr"

    def test_tuple_literal_parsed(self):
        out = _normalize_params({"size": "(2, 3)"})
        assert out["size"] == (2, 3)

    def test_malformed_tuple_left_as_string(self):
        out = _normalize_params({"size": "(2, oops)"})
        assert out["size"] == "(2, oops)"

    def test_non_string_values_untouched(self):
        params = {"a": 3, "b": 1.5, "c": True, "d": [1, 2]}
        assert _normalize_params(params) == params


class TestTransformConfig:
    def test_type_with_optional_params_and_time_behavior(self):
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[
                {"type": "torch.clone"},
                {
                    "type": "ExpandDim",
                    "params": {"axis": 0, "target_size": 4},
                    "time_behavior": "preserve",
                },
            ],
        )
        assert cfg.transforms[0] == {"type": "torch.clone"}

    def test_transforms_must_be_a_list(self):
        with pytest.raises(ValueError, match="transforms must be a list"):
            InputMappingCfg(base="one_hot", transforms={"type": "torch.clone"})

    def test_transform_entry_must_be_a_mapping(self):
        with pytest.raises(ValueError, match=r"transforms\[0\] must be a mapping"):
            InputMappingCfg(base="one_hot", transforms=["torch.clone"])

    @pytest.mark.parametrize("transform_type", [None, "", 3])
    def test_type_is_required_and_non_empty(self, transform_type):
        entry = {} if transform_type is None else {"type": transform_type}
        with pytest.raises(ValueError, match=r"transforms\[0\]\.type"):
            InputMappingCfg(base="one_hot", transforms=[entry])

    def test_params_must_be_a_mapping(self):
        with pytest.raises(ValueError, match=r"transforms\[0\]\.params"):
            InputMappingCfg(
                base="one_hot",
                transforms=[{"type": "ExpandDim", "params": ["axis", 0]}],
            )

    def test_legacy_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown keys.*name"):
            InputMappingCfg(base="one_hot", transforms=[{"name": "ExpandDim"}])

    def test_flattened_constructor_arguments_are_rejected(self):
        with pytest.raises(ValueError, match="unknown keys.*axis"):
            InputMappingCfg(
                base="one_hot",
                transforms=[{"type": "ExpandDim", "axis": 0}],
            )

    def test_unknown_entry_keys_are_rejected(self):
        with pytest.raises(ValueError, match="unknown keys.*label"):
            InputMappingCfg(
                base="one_hot",
                transforms=[{"type": "torch.clone", "label": "clone"}],
            )

    def test_time_behavior_must_be_a_string_or_mapping(self):
        with pytest.raises(ValueError, match=r"transforms\[0\]\.time_behavior"):
            InputMappingCfg(
                base="one_hot",
                transforms=[{"type": "torch.clone", "time_behavior": ["preserve"]}],
            )


class TestComposeFromConfig:
    def test_no_transforms_returns_none(self):
        cfg = InputMappingCfg(base="one_hot", transforms=[])
        assert compose_transforms_from_config(cfg) is None

    def test_builds_class_transform_with_params(self):
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[
                {
                    "type": "ExpandDim",
                    "params": {"axis": 0, "target_size": 6},
                }
            ],
        )
        comp = compose_transforms_from_config(cfg)
        assert isinstance(comp, Compose)
        assert len(comp.transforms) == 1
        out = comp(torch.arange(3).float())
        assert out.shape == (6,)

    def test_dotted_path_resolves_callable_and_wraps(self):
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[{"type": "torch.flatten", "params": {"start_dim": 0}}],
        )
        comp = compose_transforms_from_config(cfg)
        # A plain function gets wrapped in FunctionalTransform.
        assert isinstance(comp.transforms[0], FunctionalTransform)
        assert comp(torch.zeros(2, 2)).shape == (4,)

    def test_chains_multiple_transforms_in_order(self):
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[
                {"type": "PowerLaw", "params": {"gamma": 2.0}},
                {"type": "Flatten", "params": {"start_dim": 0}},
            ],
        )
        comp = compose_transforms_from_config(cfg)
        assert len(comp.transforms) == 2
        out = comp(torch.tensor([[2.0, 3.0]]))
        assert torch.allclose(out, torch.tensor([4.0, 9.0]))

    def test_params_are_normalized(self):
        # "(40, 40)" / "None" strings should be normalized before instantiation.
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[
                {
                    "type": "CenterCrop",
                    "params": {
                        "sensor_size": "(4, 4, 1)",
                        "output_size": "(2, 2)",
                    },
                }
            ],
        )
        comp = compose_transforms_from_config(cfg)
        out = comp(torch.arange(16).reshape(4, 4))
        assert out.shape == (2, 2)

    def test_unknown_type_raises_runtime_error(self):
        cfg = InputMappingCfg(base="one_hot", transforms=[{"type": "DoesNotExist"}])
        with pytest.raises(RuntimeError, match="Could not create transform DoesNotExist"):
            compose_transforms_from_config(cfg)

    def test_bad_params_wrapped_in_runtime_error(self):
        cfg = InputMappingCfg(
            base="one_hot",
            transforms=[{"type": "ExpandDim", "params": {"nonexistent_arg": 1}}],
        )
        with pytest.raises(RuntimeError, match="Could not create transform ExpandDim"):
            compose_transforms_from_config(cfg)


# ---------------------------------------------------------------------------
# generic/utils.get_device_and_dtype
# ---------------------------------------------------------------------------
class TestGetDeviceAndDtype:
    def test_float_tensor_keeps_dtype(self):
        device, dtype = get_device_and_dtype(torch.zeros(3, dtype=torch.float64))
        assert dtype == torch.float64
        assert device.type == "cpu"

    def test_integer_tensor_promoted_to_float32(self):
        _, dtype = get_device_and_dtype(torch.zeros(3, dtype=torch.long))
        assert dtype == torch.float32

    def test_non_tensor_defaults_to_float32(self):
        device, dtype = get_device_and_dtype([1, 2, 3])
        assert dtype == torch.float32
        assert device.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# generic/conv.is_binary
# ---------------------------------------------------------------------------
class TestIsBinary:
    def test_binary_tensor(self):
        assert is_binary(torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_all_zeros_is_binary(self):
        assert is_binary(torch.zeros(5))

    def test_non_binary_tensor(self):
        assert not is_binary(torch.tensor([0.0, 0.5, 1.0]))

    def test_more_than_two_integer_values_not_binary(self):
        assert not is_binary(torch.tensor([0.0, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# generic/expand_dim.ExpandDim
# ---------------------------------------------------------------------------
class TestExpandDim:
    def test_expands_along_axis(self):
        x = torch.tensor([[1.0, 2.0]])  # shape (1, 2)
        out = ExpandDim(axis=1, target_size=4)(x)
        assert out.shape == (1, 4)

    def test_equal_size_is_identity_like(self):
        x = torch.arange(3).float()
        out = ExpandDim(axis=0, target_size=3)(x)
        assert torch.equal(out, x)

    def test_repeats_values_via_binning(self):
        x = torch.tensor([10.0, 20.0])
        out = ExpandDim(axis=0, target_size=4)(x)
        # idx = floor((arange(4)+0.5)*2/4) = [0,0,1,1]
        assert torch.equal(out, torch.tensor([10.0, 10.0, 20.0, 20.0]))

    def test_negative_axis_normalized(self):
        x = torch.ones(2, 3)
        out = ExpandDim(axis=-1, target_size=6)(x)
        assert out.shape == (2, 6)

    def test_target_smaller_than_input_raises(self):
        with pytest.raises(ValueError, match="smaller than input size"):
            ExpandDim(axis=0, target_size=2)(torch.zeros(4))

    @pytest.mark.parametrize("axis", [2, -3])
    def test_axis_out_of_bounds_raises(self, axis):
        with pytest.raises(ValueError, match="out of bounds"):
            ExpandDim(axis=axis, target_size=4)(torch.zeros(2))

    def test_preserves_other_dims(self):
        x = torch.randn(5, 2, 7)
        out = ExpandDim(axis=1, target_size=8)(x)
        assert out.shape == (5, 8, 7)


# ---------------------------------------------------------------------------
# vision/center_crop.CenterCrop
# ---------------------------------------------------------------------------
class TestCenterCrop:
    def test_crops_centered_region(self):
        img = torch.arange(16).reshape(4, 4)
        out = CenterCrop(sensor_size=(4, 4, 1), output_size=(2, 2))(img)
        # top=left=1 -> rows/cols 1:3
        assert torch.equal(out, img[1:3, 1:3])

    def test_int_output_size_becomes_square(self):
        img = torch.arange(16).reshape(4, 4)
        out = CenterCrop(sensor_size=(4, 4, 1), output_size=2)(img)
        assert out.shape == (2, 2)

    def test_works_with_leading_dims(self):
        img = torch.randn(3, 6, 6)  # (..., H, W)
        out = CenterCrop(sensor_size=(6, 6, 1), output_size=(4, 4))(img)
        assert out.shape == (3, 4, 4)

    def test_full_size_crop_is_identity(self):
        img = torch.arange(9).reshape(3, 3)
        out = CenterCrop(sensor_size=(3, 3, 1), output_size=(3, 3))(img)
        assert torch.equal(out, img)


# ---------------------------------------------------------------------------
# vision/flatten.Flatten
# ---------------------------------------------------------------------------
class TestFlatten:
    def test_flatten_from_zero(self):
        out = Flatten(start_dim=0)(torch.zeros(2, 3, 4))
        assert out.shape == (24,)

    def test_flatten_preserves_leading_dims(self):
        out = Flatten(start_dim=1)(torch.zeros(2, 3, 4))
        assert out.shape == (2, 12)

    def test_values_preserved(self):
        x = torch.arange(6).reshape(2, 3)
        assert torch.equal(Flatten(start_dim=0)(x), torch.arange(6))


# ---------------------------------------------------------------------------
# snn/bin.BinAlongAxis
# ---------------------------------------------------------------------------
class TestBinAlongAxis:
    def test_sum_reduction(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = BinAlongAxis(bin_size=2, bin_axis=0, reduction="sum")(x)
        assert torch.equal(out, torch.tensor([3.0, 7.0]))

    def test_mean_reduction(self):
        x = torch.tensor([1.0, 3.0, 5.0, 7.0])
        out = BinAlongAxis(bin_size=2, bin_axis=0, reduction="mean")(x)
        assert torch.equal(out, torch.tensor([2.0, 6.0]))

    def test_binary_reduction(self):
        x = torch.tensor([0.0, 0.0, 1.0, 0.0])
        out = BinAlongAxis(bin_size=2, bin_axis=0, reduction="binary")(x)
        assert out.dtype == torch.bool
        assert torch.equal(out, torch.tensor([False, True]))

    def test_negative_axis(self):
        x = torch.ones(3, 4)
        out = BinAlongAxis(bin_size=2, bin_axis=-1, reduction="sum")(x)
        assert out.shape == (3, 2)
        assert torch.all(out == 2.0)

    def test_bins_only_target_axis(self):
        x = torch.ones(6, 5)
        out = BinAlongAxis(bin_size=3, bin_axis=0, reduction="sum")(x)
        assert out.shape == (2, 5)

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="not divisible by bin_size"):
            BinAlongAxis(bin_size=3, bin_axis=0)(torch.ones(4))

    def test_unknown_reduction_raises(self):
        with pytest.raises(ValueError, match="reduction must be one of"):
            BinAlongAxis(bin_size=2, bin_axis=0, reduction="whatever")


# ---------------------------------------------------------------------------
# snn/rate_coding.RateCoding
# ---------------------------------------------------------------------------
class TestRateCoding:
    def test_scales_by_max_rate(self):
        out = RateCoding(max_rate=100.0, normalize=False, clamp=False)(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(out, torch.tensor([0.0, 50.0, 100.0]))

    def test_clamp_bounds_output(self):
        out = RateCoding(max_rate=10.0, normalize=False, clamp=True)(torch.tensor([-1.0, 0.5, 5.0]))
        assert out.min() >= 0.0
        assert out.max() <= 10.0
        assert torch.equal(out, torch.tensor([0.0, 5.0, 10.0]))

    def test_no_clamp_allows_negatives(self):
        out = RateCoding(max_rate=10.0, normalize=False, clamp=False)(torch.tensor([-1.0]))
        assert out.item() == pytest.approx(-10.0)

    def test_integer_input_is_cast_to_float(self):
        out = RateCoding(max_rate=2.0, normalize=False, clamp=False)(torch.tensor([1, 2, 3]))
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0]))

    def test_normalize_degenerate_returns_zeros(self):
        # All-equal input hits the early `x_max <= x_min` zeros return.
        out = RateCoding(max_rate=100.0, normalize=True)(torch.full((4,), 3.0))
        assert torch.equal(out, torch.zeros(4))

    def test_normalize_rescales_to_full_range(self):
        # Input range [0, 4] is normalized to [0, 1] then scaled to [0, max_rate].
        out = RateCoding(max_rate=10.0, normalize=True, clamp=True)(torch.tensor([0.0, 2.0, 4.0]))
        assert torch.allclose(out, torch.tensor([0.0, 5.0, 10.0]))

    def test_normalize_handles_offset_range(self):
        # Min/max normalization should map the smallest value to 0 and largest
        # to max_rate even when the input does not start at 0.
        out = RateCoding(max_rate=100.0, normalize=True, clamp=True)(torch.tensor([10.0, 20.0]))
        assert torch.allclose(out, torch.tensor([0.0, 100.0]))

    def test_repr_is_a_string(self):
        assert isinstance(repr(RateCoding()), str)


# ---------------------------------------------------------------------------
# snn/poisson_encoding.PoissonEncoding
# ---------------------------------------------------------------------------
class TestPoissonEncoding:
    def test_default_does_not_normalize(self):
        enc = PoissonEncoding(temporal=True, spike_dt=0.001)
        assert enc.normalize is False

    def test_static_requires_duration(self):
        with pytest.raises(ValueError, match="duration must be specified"):
            PoissonEncoding(temporal=False)

    def test_spike_dt_must_be_positive(self):
        with pytest.raises(ValueError, match="spike_dt must be positive"):
            PoissonEncoding(temporal=False, duration=0.05, spike_dt=0.0)

    def test_static_output_is_binary(self):
        enc = PoissonEncoding(temporal=False, duration=0.05, spike_dt=0.001, seed=0)
        out = enc(torch.rand(8))
        assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})

    def test_static_expands_time_dim(self):
        enc = PoissonEncoding(temporal=False, duration=0.05, spike_dt=0.001, seed=0)
        out = enc(torch.rand(8))
        # n_bins = round(0.05 / 0.001) = 50, prepended as new leading dim.
        assert out.shape == (50, 8)

    def test_frozen_noise_is_reproducible(self):
        x = torch.rand(16)
        enc = PoissonEncoding(temporal=False, duration=0.05, frozen_noise=True, seed=123)
        assert torch.equal(enc(x), enc(x))

    def test_temporal_output_shape_and_binary(self):
        enc = PoissonEncoding(temporal=True, spike_dt=0.001, seed=0, normalize=True)
        spec = enc.time_spec(TimeGrid(0.01))
        assert spec.out_dt == pytest.approx(0.001)
        assert enc.input_dt is None
        enc.bind_time_grid(TimeGrid(0.01), TimeGrid(0.001))
        x = torch.rand(5, 4)  # [T_in, units]
        out = enc(x)
        # bins_per_frame = round(0.01 / 0.001) = 10 -> T_out = 5 * 10
        assert out.shape == (50, 4)
        assert set(torch.unique(out).tolist()).issubset({0.0, 1.0})

    def test_high_rate_gives_more_spikes_than_low(self):
        torch.manual_seed(0)
        enc = PoissonEncoding(temporal=True, spike_dt=0.001, seed=0)
        enc.bind_time_grid(TimeGrid(0.1), TimeGrid(0.001))
        low = enc(torch.full((4, 50), 1.0)).sum()
        high = enc(torch.full((4, 50), 200.0)).sum()
        assert high > low

    def test_temporal_requires_integer_dt_ratio(self):
        enc = PoissonEncoding(temporal=True, spike_dt=0.003, seed=0)
        with pytest.raises(ValueError, match="integer ratio"):
            enc.bind_time_grid(TimeGrid(0.01), TimeGrid(0.003))

    def test_normalize_constant_input_yields_no_spikes(self):
        enc = PoissonEncoding(temporal=False, duration=0.05, normalize=True, seed=0)
        out = enc(torch.full((6,), 5.0))
        assert torch.count_nonzero(out) == 0

    def test_explicit_normalize_uses_max_rate(self):
        x = torch.tensor([0.0, 2.0, 4.0])
        enc = PoissonEncoding(temporal=False, duration=0.001, normalize=True, max_rate=50.0)
        assert torch.allclose(enc._normalize(x), torch.tensor([0.0, 25.0, 50.0]))


# ---------------------------------------------------------------------------
# snn/utils.spike_times_to_tensor
# ---------------------------------------------------------------------------
class TestSpikeTimesToTensor:
    # The time axis is sized from the grid (len(time_bins) + 1), not the data,
    # so every spike index is in bounds and the shape is consistent across
    # samples regardless of when the last spike falls.
    def test_dense_shape_is_grid_based(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        spike_times = np.array([0.15, 0.55])  # digitize -> [2, 6]
        units = np.array([0, 2])
        x = spike_times_to_tensor(spike_times, units, time_bins, num_units=4, dense=True)
        assert x.shape == (12, 4)

    def test_shape_independent_of_last_spike_time(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        early = spike_times_to_tensor(np.array([0.15]), np.array([0]), time_bins, num_units=4)
        late = spike_times_to_tensor(np.array([0.95]), np.array([0]), time_bins, num_units=4)
        assert early.shape == late.shape == (12, 4)

    def test_spike_at_or_after_last_edge_is_placed(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        # digitize -> len(time_bins) == 11, the maximum possible index.
        x = spike_times_to_tensor(np.array([1.5]), np.array([2]), time_bins, num_units=4, dense=True)
        assert x[11, 2].item() == pytest.approx(1.0)
        assert x.sum().item() == pytest.approx(1.0)

    def test_all_spikes_placed(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        spike_times = np.array([0.15, 0.55])  # digitize -> [2, 6]
        units = np.array([0, 2])
        x = spike_times_to_tensor(spike_times, units, time_bins, num_units=4, dense=True)
        assert x[2, 0].item() == pytest.approx(1.0)
        assert x[6, 2].item() == pytest.approx(1.0)
        assert x.sum().item() == pytest.approx(2.0)

    def test_returns_sparse_when_requested(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        spike_times = np.array([0.15, 0.55])
        units = np.array([0, 2])
        x = spike_times_to_tensor(spike_times, units, time_bins, num_units=4, dense=False)
        assert x.is_sparse

    def test_units_index_correct_column(self):
        time_bins = np.linspace(0.0, 1.0, 11)
        x = spike_times_to_tensor(
            np.array([0.25]), np.array([3]), time_bins, num_units=5, dense=True
        )
        # Spike (digitize -> 3) sits on unit column 3; all other columns empty.
        assert x[:, 3].sum().item() == pytest.approx(1.0)
        assert x[:, [0, 1, 2, 4]].sum().item() == 0.0


# ---------------------------------------------------------------------------
# audio/power_law.PowerLaw
# ---------------------------------------------------------------------------
class TestPowerLaw:
    def test_squares_input(self):
        out = PowerLaw(gamma=2.0)(torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(out, torch.tensor([0.0, 0.25, 1.0]))

    def test_default_gamma(self):
        out = PowerLaw()(torch.tensor([1.0]))
        assert out.item() == pytest.approx(1.0)

    def test_gamma_one_is_identity(self):
        x = torch.tensor([0.2, 0.7, 0.9])
        assert torch.allclose(PowerLaw(gamma=1.0)(x), x)

    def test_monotonic_on_unit_interval(self):
        x = torch.linspace(0, 1, 20)
        out = PowerLaw(gamma=5.0)(x)
        assert torch.all(out[1:] >= out[:-1])


# ---------------------------------------------------------------------------
# audio/adaptive_percentile_norm.AdaptivePercentileNorm
# ---------------------------------------------------------------------------
class TestAdaptivePercentileNorm:
    @pytest.mark.parametrize(
        "floor_percentile, ceil_percentile",
        [(-1.0, 95.0), (5.0, 101.0), (50.0, 50.0), (75.0, 25.0)],
    )
    def test_invalid_percentiles_raise(self, floor_percentile, ceil_percentile):
        with pytest.raises(ValueError, match="percentiles must satisfy"):
            AdaptivePercentileNorm(
                floor_percentile=floor_percentile,
                ceil_percentile=ceil_percentile,
            )

    def test_output_in_unit_range(self):
        x = torch.linspace(0, 100, 101)
        out = AdaptivePercentileNorm(floor_percentile=5, ceil_percentile=95)(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_clips_below_and_above_percentiles(self):
        x = torch.linspace(0, 100, 101)
        out = AdaptivePercentileNorm(floor_percentile=10, ceil_percentile=90)(x)
        # Values at/under the floor percentile clamp to 0; at/over ceil clamp to 1.
        assert out[0].item() == pytest.approx(0.0)
        assert out[-1].item() == pytest.approx(1.0)

    def test_degenerate_constant_returns_zeros(self):
        out = AdaptivePercentileNorm()(torch.full((10,), 7.0))
        assert torch.equal(out, torch.zeros(10))

    def test_preserves_shape(self):
        x = torch.randn(3, 8)
        out = AdaptivePercentileNorm()(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# audio/mel.LogMel and audio/mfcc.MFCC (require torchaudio)
# ---------------------------------------------------------------------------
torchaudio = pytest.importorskip("torchaudio")
from seqbench.transforms.audio.mel import LogMel  # noqa: E402
from seqbench.transforms.audio.mfcc import MFCC  # noqa: E402


class TestLogMel:
    def test_output_time_by_mels(self):
        wav = torch.randn(1, 16000)
        out = LogMel(sample_rate=16000, n_mels=40)(wav)
        assert out.shape[1] == 40

    def test_runs_on_plain_waveform(self):
        out = LogMel(n_mels=23)(torch.randn(16000))
        assert out.shape[1] == 23
        assert torch.isfinite(out).all()


class TestMFCC:
    def test_output_time_by_coeffs(self):
        wav = torch.randn(1, 16000)
        out = MFCC(n_mfcc=13, n_mels=40)(wav)
        # __call__ squeezes then permute(1, 0) -> (time, n_mfcc)
        assert out.shape[1] == 13

    def test_finite_output(self):
        out = MFCC(n_mfcc=10)(torch.randn(1, 8000))
        assert torch.isfinite(out).all()
