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

"""Tests for get_spyre_tensor_layout() EA correctness (issue #2788)."""

import pytest
import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 128),  # Original repro case
        (1, 64),  # Small batch
        (8, 256),  # Larger dimensions
        (16, 32),  # Different aspect ratio
        (2, 512),  # Wide tensor
    ],
)
def test_fp16_to_fp32_produces_dl16_to_fp32(device, shape):
    """Test that FP16→FP32 conversion produces DL16_TO_FP32 EA."""

    @torch.compile
    def fn(x):
        return x.to(torch.float32)

    x = torch.randn(*shape, device=device, dtype=torch.float16)
    result = fn(x)

    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, (
        f"Shape {shape}: Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    )
