#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Tests for the Conv operation, focused on grouped / depthwise convolution.

The MIOpen weight descriptor must use the per-group input-channel count
(input_c / group), not input_c. For group == 1 the two are equal, but for
grouped (1 < group < C) and depthwise (group == C) convolutions, using input_c
describes the wrong filter shape and produces silently incorrect results. These
tests exercise all three regimes against the ORT CPU reference.
"""

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


def _make_conv_model(
    in_c: int,
    out_c: int,
    group: int,
    k: int = 3,
    spatial: int = 8,
    seed: int = 7,
):
    """Build an fp16 2D Conv model with weight+bias initializers.

    input:   [1, in_c, spatial, spatial] f16
    weights: [out_c, in_c // group, k, k] f16  (initializer)
    bias:    [out_c] f16                        (initializer)
    output:  [1, out_c, spatial, spatial] f16   (stride 1, pad k//2 keeps size)
    """
    assert in_c % group == 0 and out_c % group == 0
    pad = k // 2

    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT16, [1, in_c, spatial, spatial]
    )
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, out_c, spatial, spatial]
    )

    rng = np.random.default_rng(seed)
    w = rng.uniform(-0.1, 0.1, [out_c, in_c // group, k, k]).astype(np.float16)
    b = rng.uniform(-0.1, 0.1, [out_c]).astype(np.float16)
    w_init = numpy_helper.from_array(w, name="weight")
    b_init = numpy_helper.from_array(b, name="bias")

    node = helper.make_node(
        "Conv",
        ["input", "weight", "bias"],
        ["output"],
        group=group,
        kernel_shape=[k, k],
        strides=[1, 1],
        pads=[pad, pad, pad, pad],
        dilations=[1, 1],
    )
    return make_model_from_nodes([node], [inp], [out], initializers=[w_init, b_init])


class TestConv:
    @pytest.mark.parametrize(
        "in_c,out_c,group",
        [
            (4, 8, 1),  # standard conv (group=1: input_c/group == input_c)
            (8, 16, 2),  # grouped conv (intermediate group)
            (8, 16, 4),  # grouped conv (intermediate group)
            (8, 8, 8),  # depthwise conv (group == channels)
            (16, 16, 16),  # depthwise conv
        ],
    )
    def test_conv_grouped(self, model_runner, in_c, out_c, group):
        """Conv across group regimes; weights are [out_c, in_c/group, k, k].

        Regression guard: using input_c instead of input_c/group for the MIOpen
        weight descriptor yields uncorrelated output (cosine ~0) for group > 1.
        """
        model = _make_conv_model(in_c, out_c, group)

        rng = np.random.default_rng(123)
        x = rng.uniform(-1.0, 1.0, [1, in_c, 8, 8]).astype(np.float16)

        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)

    def test_conv3d_temporal(self, model_runner):
        """Rank-5 NCDHW conv: depth kernel 3, spatial kernel 1x1, pad 1 on D.

        Output depth stays 6:
        (6 + 1 + 1 - (3 - 1) - 1) / 1 + 1 = 6.
        """
        depth, height, width = 6, 4, 4
        channels = 4
        inp = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT16, [1, channels, depth, height, width]
        )
        out = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT16, [1, channels, depth, height, width]
        )
        rng = np.random.default_rng(7)
        w = rng.uniform(-0.1, 0.1, [channels, channels, 3, 1, 1]).astype(np.float16)
        b = rng.uniform(-0.1, 0.1, [channels]).astype(np.float16)
        node = helper.make_node(
            "Conv",
            ["input", "weight", "bias"],
            ["output"],
            group=1,
            kernel_shape=[3, 1, 1],
            strides=[1, 1, 1],
            pads=[1, 0, 0, 1, 0, 0],
            dilations=[1, 1, 1],
        )
        model = make_model_from_nodes(
            [node],
            [inp],
            [out],
            initializers=[
                numpy_helper.from_array(w, name="weight"),
                numpy_helper.from_array(b, name="bias"),
            ],
        )
        x = rng.uniform(-1.0, 1.0, [1, channels, depth, height, width]).astype(
            np.float16
        )
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=1e-2, rtol=1e-2, cos_threshold=0.999)


def _make_conv_model_nd(
    dtype, n, in_c, h, w, out_c, k, stride=1, pad=None, group=1, bias=True, seed=11
):
    """2D Conv model in `dtype` (TensorProto FLOAT or FLOAT16), pad = k // 2 by default."""
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16
    pad = k // 2 if pad is None else pad
    oh = (h + 2 * pad - k) // stride + 1
    ow = (w + 2 * pad - k) // stride + 1
    rng = np.random.default_rng(seed)
    wt = rng.uniform(-0.1, 0.1, [out_c, in_c // group, k, k]).astype(np_dtype)
    inits = [numpy_helper.from_array(wt, name="weight")]
    names = ["input", "weight"]
    if bias:
        inits.append(
            numpy_helper.from_array(
                rng.uniform(-0.1, 0.1, [out_c]).astype(np_dtype), name="bias"
            )
        )
        names.append("bias")
    node = helper.make_node(
        "Conv",
        names,
        ["output"],
        group=group,
        kernel_shape=[k, k],
        strides=[stride, stride],
        pads=[pad] * 4,
        dilations=[1, 1],
    )
    inp = helper.make_tensor_value_info("input", dtype, [n, in_c, h, w])
    out = helper.make_tensor_value_info("output", dtype, [n, out_c, oh, ow])
    return make_model_from_nodes([node], [inp], [out], initializers=inits), np_dtype


class TestConvFewOutputChannels:
    """Output-channel counts below the smallest scalar tile's 32 rows. These used
    to fall back to the one-thread-per-output direct kernel; they now run on a
    tile with dead rows, which must be zero-filled on load and skipped on store.
    The shapes are large enough (>= 1M multiply-accumulates) to leave the direct
    kernel behind."""

    @pytest.mark.parametrize("dtype", [TensorProto.FLOAT, TensorProto.FLOAT16])
    @pytest.mark.parametrize(
        "n,in_c,hw,out_c,k,group",
        [
            (1, 64, 40, 3, 3, 1),  # single-digit channel count
            (1, 256, 24, 16, 1, 1),  # 1x1
            (1, 64, 32, 24, 3, 1),
            (2, 48, 28, 30, 3, 1),  # batch > 1, one row short of a tile
            (1, 64, 40, 24, 3, 2),  # grouped: 12 channels per group
        ],
    )
    def test_conv_few_output_channels(
        self, model_runner, dtype, n, in_c, hw, out_c, k, group
    ):
        model, np_dtype = _make_conv_model_nd(
            dtype, n, in_c, hw, hw, out_c, k, group=group
        )
        rng = np.random.default_rng(9)
        x = rng.uniform(-1.0, 1.0, [n, in_c, hw, hw]).astype(np_dtype)
        actual, expected = model_runner.run_sample(model, [x])
        tol = 1e-3 if dtype == TensorProto.FLOAT else 2e-2
        compare_outputs(actual, expected, atol=tol, rtol=tol, cos_threshold=0.9999)


class TestConv1x1Gemm:
    """fp16 1x1 convolutions with at least 128 input channels per group run as a
    GEMM with their own kernel. The shapes cover each global-load width it uses
    (output plane a multiple of 8, of 4, or neither), a reduction that is not a
    multiple of its 64-tap step, ragged output channels, groups, batch and a
    missing bias."""

    @pytest.mark.parametrize(
        "n,in_c,h,w,out_c,group,bias",
        [
            (1, 256, 16, 16, 64, 1, True),  # plane 256: 16-byte loads
            (2, 192, 12, 12, 70, 1, True),  # K = 192 = 3 steps, ragged M, batch
            (1, 200, 8, 16, 130, 1, True),  # K = 200, not a multiple of 64
            (1, 256, 10, 14, 128, 1, True),  # plane 140: 8-byte loads
            (1, 512, 7, 7, 2048, 1, True),  # plane 49: 2-byte loads
            (1, 256, 9, 9, 40, 1, True),  # plane 81, ragged M
            (2, 256, 16, 8, 192, 2, True),  # grouped
            (1, 128, 24, 24, 192, 1, False),  # K at the minimum, no bias
        ],
    )
    def test_conv_1x1(self, model_runner, n, in_c, h, w, out_c, group, bias):
        model, np_dtype = _make_conv_model_nd(
            TensorProto.FLOAT16, n, in_c, h, w, out_c, 1, pad=0, group=group, bias=bias
        )
        rng = np.random.default_rng(3)
        x = rng.uniform(-1.0, 1.0, [n, in_c, h, w]).astype(np_dtype)
        actual, expected = model_runner.run_sample(model, [x])
        compare_outputs(actual, expected, atol=2e-2, rtol=2e-2, cos_threshold=0.9999)
