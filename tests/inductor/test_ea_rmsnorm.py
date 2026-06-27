# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RMSNorm with FP16→FP32 upcasting via ElementArrangement propagation.

RMSNorm pattern (issue #2508):
  hidden_states.to(fp32) → pow → mean(-1) → rsqrt → mul → weight * result.to(fp16)

Expected EA flow:
  1. to(float32): STANDARD → DL16_TO_FP32
  2. pow(2): DL16_TO_FP32 → DL16_TO_FP32
  3. mean(-1): DL16_TO_FP32 → STANDARD  (reduction on stick)
  4. rsqrt: STANDARD → STANDARD
  5. mul(broadcast): DL16_TO_FP32 * STANDARD → DL16_TO_FP32
  6. to(float16): DL16_TO_FP32 → STANDARD  (restoration)
  7. mul: STANDARD * STANDARD → STANDARD
"""

import pytest

import torch

from utils_inductor import cached_randn, compare_with_cpu


def _rmsnorm(x, weight, eps=1e-6):
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normed.to(x.dtype)


def test_rmsnorm_full_pattern():
    """Test complete RMSNorm pattern from issue #2508."""
    x = cached_randn((4, 4096))
    weight = cached_randn((4096,), differentiation="weight")
    compare_with_cpu(_rmsnorm, x, weight, atol=1e-2, rtol=1e-2, run_eager=False)


def test_rmsnorm_fp32_to_fp16_restoration():
    """Test that FP32→FP16 downcast of a DL16_TO_FP32 tensor restores STANDARD EA."""

    def fn(x):
        return (x.to(torch.float32) * 2.0).to(torch.float16)

    x = cached_randn((4, 4096))
    compare_with_cpu(fn, x, atol=1e-3, rtol=1e-3, run_eager=False)


def test_rmsnorm_variance_computation():
    """Test pow + mean reduction in the RMSNorm variance step."""

    def fn(x):
        x_fp32 = x.to(torch.float32)
        return x_fp32.pow(2).mean(-1, keepdim=True)

    x = cached_randn((4, 4096))
    compare_with_cpu(fn, x, atol=1e-3, rtol=1e-3, run_eager=False)


def test_rmsnorm_normalization_step():
    """Test broadcast mul between DL16_TO_FP32 and STANDARD tensors."""

    def fn(x):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + 1e-6)
        return (x_fp32 * rsqrt).to(torch.float16)

    x = cached_randn((4, 4096))
    compare_with_cpu(fn, x, atol=1e-2, rtol=1e-2, run_eager=False)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4096),
        (1, 12, 4096),
        (1, 64, 4096),
        (4, 1, 4096),
        (8, 12, 4096),
        (16, 64, 4096),
    ],
)
def test_rmsnorm_with_different_shapes(shape):
    """Test RMSNorm full pattern with various tensor shapes."""
    x = cached_randn(shape, differentiation=str(shape))
    weight = cached_randn((shape[-1],), differentiation=f"weight_{shape}")
    compare_with_cpu(_rmsnorm, x, weight, atol=1e-2, rtol=1e-2, run_eager=False)


def test_rmsnorm_without_weight():
    """Test RMSNorm without final weight multiplication."""

    def rmsnorm_no_weight(x, eps=1e-6):
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(variance + eps)).to(x.dtype)

    x = cached_randn((4, 4096))
    compare_with_cpu(rmsnorm_no_weight, x, atol=1e-2, rtol=1e-2, run_eager=False)
