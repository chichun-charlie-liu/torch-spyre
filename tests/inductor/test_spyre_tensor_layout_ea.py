"""
Test cases for issue #2788: get_spyre_tensor_layout() returns correct ElementArrangement

Issue: https://github.com/torch-spyre/torch-spyre/issues/2788

The issue reported that get_spyre_tensor_layout() was returning STANDARD EA
instead of the correct EA (like DL16_TO_FP32) that was set during compilation
for FP16→FP32 type conversions.

This test suite validates the core fix: get_spyre_tensor_layout() now correctly
returns DL16_TO_FP32 for FP16→FP32 conversions after kernel execution.
"""

import pytest
import torch
from torch_spyre._C import ElementArrangement, get_spyre_tensor_layout


@pytest.mark.parametrize("device", ["spyre"])
def test_fp16_to_fp32_produces_dl16_to_fp32(device):
    """Test that FP16→FP32 conversion produces DL16_TO_FP32 EA.
    
    This is the core issue from #2788: get_spyre_tensor_layout() should
    return DL16_TO_FP32 for FP16→FP32 conversions, not STANDARD.
    """
    
    @torch.compile
    def fn(x):
        return x.to(torch.float32)
    
    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    result = fn(x)
    
    # Verify EA is DL16_TO_FP32
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, \
        f"Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    
    print("✓ FP16→FP32 produces DL16_TO_FP32")


@pytest.mark.parametrize("device", ["spyre"])
def test_different_tensor_sizes(device):
    """Test EA tracking works with different tensor sizes."""
    
    @torch.compile
    def fn(x):
        return x.to(torch.float32)
    
    # Test various sizes
    sizes = [(4, 64), (8, 128), (2, 256), (16, 32)]
    
    for size in sizes:
        x = torch.randn(*size, device=device, dtype=torch.float16)
        result = fn(x)
        
        result_layout = get_spyre_tensor_layout(result)
        assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, \
            f"Size {size}: Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    
    print(f"✓ EA tracking works for {len(sizes)} different tensor sizes")


@pytest.mark.parametrize("device", ["spyre"])
def test_batch_dimension_variations(device):
    """Test EA tracking with different batch dimensions."""
    
    @torch.compile
    def fn(x):
        return x.to(torch.float32)
    
    # Test various batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 128, device=device, dtype=torch.float16)
        result = fn(x)
        
        result_layout = get_spyre_tensor_layout(result)
        assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, \
            f"Batch {batch_size}: Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    
    print(f"✓ EA tracking works for {len(batch_sizes)} different batch sizes")


@pytest.mark.parametrize("device", ["spyre"])
def test_original_repro_case(device):
    """Test the original repro case from issue #2788."""
    
    @torch.compile
    def fn(x):
        return x.to(torch.float32)
    
    x = torch.randn(4, 128, device=device, dtype=torch.float16)
    result = fn(x)
    
    # This is the exact check from the original issue
    result_layout = get_spyre_tensor_layout(result)
    assert result_layout.element_arrangement == ElementArrangement.DL16_TO_FP32, \
        f"Original repro failed: Expected DL16_TO_FP32, got {result_layout.element_arrangement}"
    
    print("✓ Original repro case from issue #2788 passes")


if __name__ == "__main__":
    # Run tests manually for quick validation
    import sys
    
    device = "spyre"
    
    print("Running issue #2788 test suite...")
    print("=" * 60)
    
    try:
        test_fp16_to_fp32_produces_dl16_to_fp32(device)
        test_different_tensor_sizes(device)
        test_batch_dimension_variations(device)
        test_original_repro_case(device)
        
        print("=" * 60)
        print("✅ All tests passed!")
        print("\nIssue #2788 is fixed:")
        print("- get_spyre_tensor_layout() now correctly returns DL16_TO_FP32")
        print("- EA is properly written back to tensor after kernel execution")
        print("- Fix works across different tensor sizes and batch dimensions")
        sys.exit(0)
    except AssertionError as e:
        print("=" * 60)
        print(f"❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print("=" * 60)
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
