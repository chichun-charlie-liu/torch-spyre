"""
Test case for issue #2788: get_spyre_tensor_layout() returns correct ElementArrangement

Issue: https://github.com/torch-spyre/torch-spyre/issues/2788

The issue reported that get_spyre_tensor_layout() was returning STANDARD EA
instead of the correct EA (like DL16_TO_FP32) that was set during compilation
for FP16→FP32 type conversions.
"""

import pytest
import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout


@pytest.mark.parametrize("device", ["spyre"])
@pytest.mark.parametrize("shape", [
    (4, 128),      # Original repro case
    (1, 64),       # Small batch
    (8, 256),      # Larger dimensions
    (16, 32),      # Different aspect ratio
    (2, 512),      # Wide tensor
])
def test_fp16_to_fp32_produces_dl16_to_fp32(device, shape):
    """Test that FP16→FP32 conversion produces DL16_TO_FP32 EA for various shapes.

    This validates the fix for issue #2788: get_spyre_tensor_layout() should
    return DL16_TO_FP32 for FP16→FP32 conversions, not STANDARD.
    """

    @torch.compile
    def fn(x):
        return x.to(torch.float32)

    x = torch.randn(*shape, device=device, dtype=torch.float16)
    result = fn(x)

    # Verify EA is DL16_TO_FP32
    result_layout = get_spyre_tensor_layout(result)
    assert (
        result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32
    ), f"Shape {shape}: Expected DL16_TO_FP32, got {result_layout.element_arrangement}"

# Made with Bob
