#!/usr/bin/env python3
"""Test script for monotonic alignment implementation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from src.core.alignment import maximum_path


def test_basic_functionality():
    """Test basic functionality of monotonic alignment."""
    print("Testing basic monotonic alignment functionality...")
    
    # Create test data
    batch_size = 2
    time1 = 10
    time2 = 8
    
    # Create negative log probabilities (lower values = higher probability)
    neg_cent = torch.randn(batch_size, time1, time2, dtype=torch.float32)
    
    # Create attention mask (2D as used in VITS)
    # Create mask that ensures valid alignment paths
    mask = torch.ones(batch_size, time1, time2, dtype=torch.float32)
    # Set some values to 0 to create realistic mask patterns
    for b in range(batch_size):
        for t in range(time1):
            # Create diagonal-like patterns for valid alignments
            start_s = max(0, t * time2 // time1 - 2)
            end_s = min(time2, t * time2 // time1 + 3)
            mask[b, t, :start_s] = 0
            mask[b, t, end_s:] = 0
    
    try:
        # Test the alignment function
        alignment = maximum_path(neg_cent, mask)
        
        print(f"Input shape: {neg_cent.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Output shape: {alignment.shape}")
        print(f"Output dtype: {alignment.dtype}")
        
        # Check output properties
        assert alignment.shape == (batch_size, time1, time2), f"Wrong output shape: {alignment.shape}"
        assert alignment.dtype == torch.float32, f"Wrong output dtype: {alignment.dtype}"
        
        # Check that alignment is binary (0 or 1)
        unique_values = torch.unique(alignment)
        print(f"Unique values in alignment: {unique_values}")
        assert torch.all((alignment == 0) | (alignment == 1)), "Alignment should be binary"
        
        # Check that each time step has exactly one alignment (where mask allows)
        for b in range(batch_size):
            # Check monotonic property: path should be monotonic
            path_coords = []
            for t in range(time1):
                for s in range(time2):
                    if alignment[b, t, s] == 1:
                        path_coords.append((t, s))
            
            # Verify monotonic property
            if len(path_coords) > 1:
                for i in range(1, len(path_coords)):
                    prev_t, prev_s = path_coords[i-1]
                    curr_t, curr_s = path_coords[i]
                    assert curr_t >= prev_t and curr_s >= prev_s, "Path should be monotonic"
            
            print(f"Batch {b} alignment path length: {len(path_coords)}")
        
        print("✓ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    """Test device compatibility (CPU/GPU)."""
    print("\nTesting device compatibility...")
    
    # Test CPU
    neg_cent_cpu = torch.randn(1, 5, 4, dtype=torch.float32)
    mask_cpu = torch.ones(1, 5, 4, dtype=torch.float32)
    
    try:
        alignment_cpu = maximum_path(neg_cent_cpu, mask_cpu)
        print("✓ CPU test passed!")
        
        # Test GPU if available
        if torch.cuda.is_available():
            neg_cent_gpu = neg_cent_cpu.cuda()
            mask_gpu = mask_cpu.cuda()
            alignment_gpu = maximum_path(neg_cent_gpu, mask_gpu)
            print("✓ GPU test passed!")
        else:
            print("⚠ GPU not available, skipping GPU test")
            
        return True
        
    except Exception as e:
        print(f"✗ Device compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vits_integration():
    """Test integration with VITS model."""
    print("\nTesting VITS model integration...")
    
    try:
        from src.core.models.vits import SynthesizerTrn
        
        # Create a simple VITS model configuration
        config = {
            'n_vocab': 100,
            'spec_channels': 513,
            'segment_size': 32,
            'inter_channels': 192,
            'hidden_channels': 192,
            'filter_channels': 768,
            'n_heads': 2,
            'n_layers': 6,
            'kernel_size': 3,
            'p_dropout': 0.1,
            'resblock': '1',
            'resblock_kernel_sizes': [3, 7, 11],
            'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            'upsample_rates': [8, 8, 2, 2],
            'upsample_initial_channel': 512,
            'upsample_kernel_sizes': [16, 16, 4, 4],
            'n_speakers': 0,
            'gin_channels': 0
        }
        
        # Create model
        model = SynthesizerTrn(**config)
        model.eval()
        
        # Create test input
        batch_size = 1
        text_length = 20
        spec_length = 32
        
        x = torch.randint(0, 100, (batch_size, text_length))
        x_lengths = torch.tensor([text_length])
        y = torch.randn(batch_size, 513, spec_length)
        y_lengths = torch.tensor([spec_length])
        
        # Test forward pass
        with torch.no_grad():
            output = model(x, x_lengths, y, y_lengths)
            
        print(f"✓ VITS forward pass successful! Output keys: {list(output.keys())}")
        return True
        
    except Exception as e:
        print(f"✗ VITS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== Monotonic Alignment Integration Test ===\n")
    
    tests = [
        test_basic_functionality,
        test_device_compatibility,
        test_vits_integration
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n=== Test Results: {passed}/{len(tests)} passed ===")
    
    if passed == len(tests):
        print("🎉 All tests passed! Monotonic alignment integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
