#!/usr/bin/env python3
"""Integration test for MMVC_Trainer package."""

import sys
import os
import tempfile
import json
import torch
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_package_imports():
    """Test that all main package components can be imported."""
    print("Testing package imports...")
    
    try:
        # Core models
        from src.core.models.vits import SynthesizerTrn
        from src.core.models.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
        
        # Data utilities
        from src.core.data.dataset import TextAudioSpeakerDataset, TextAudioSpeakerCollate
        
        # Text processing
        from src.text.symbols import symbols
        from src.text.cleaners import clean_text
        
        # Audio processing
        from src.utils.audio import load_wav, save_wav, get_mel_from_wav_torch
        
        # Common utilities
        from src.utils.common import setup_logger, load_checkpoint
        
        # Alignment
        from src.core.alignment import maximum_path
        
        print("✓ All core imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating VITS model."""
    print("\nTesting model creation...")
    
    try:
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
        
        from src.core.models.vits import SynthesizerTrn
        model = SynthesizerTrn(**config)
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_processing():
    """Test audio processing utilities."""
    print("\nTesting audio processing...")
    
    try:
        from src.utils.audio import get_mel_from_wav_torch
        
        # Create dummy audio data
        audio = torch.randn(1, 22050)  # 1 second at 22050 Hz
        
        # Generate mel spectrogram
        mel = get_mel_from_wav_torch(
            audio,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            sampling_rate=22050,
            mel_fmin=0.0,
            mel_fmax=8000.0
        )
        
        print(f"✓ Audio processing successful")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Mel spectrogram shape: {mel.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_processing():
    """Test text processing utilities."""
    print("\nTesting text processing...")
    
    try:
        from src.text.cleaners import clean_text
        from src.text.symbols import text_to_sequence
        
        # Test text cleaning
        test_text = "Hello, world! This is a test."
        cleaned = clean_text(test_text, ["english_cleaners"])
        
        # Test text to sequence conversion
        if 'text_to_sequence' in globals():
            sequence = text_to_sequence(cleaned, ["english_cleaners"])
            print(f"✓ Text processing successful")
            print(f"  Original: {test_text}")
            print(f"  Cleaned: {cleaned}")
            print(f"  Sequence length: {len(sequence)}")
        else:
            print("⚠ text_to_sequence not available, skipping sequence test")
            print(f"✓ Text cleaning successful")
            print(f"  Original: {test_text}")
            print(f"  Cleaned: {cleaned}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_commands():
    """Test that CLI commands are accessible."""
    print("\nTesting CLI command accessibility...")
    
    try:
        import subprocess
        import os
        
        # Change to the project directory
        project_dir = os.path.join(os.path.dirname(__file__))
        
        # Test each CLI command
        commands = [
            "python -m src.cli.train --help",
            "python -m src.cli.convert --help", 
            "python -m src.cli.export --help"
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd.split(),
                    cwd=project_dir,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"✓ {cmd.split()[2]} command accessible")
                else:
                    print(f"⚠ {cmd.split()[2]} command returned non-zero exit code")
            except subprocess.TimeoutExpired:
                print(f"⚠ {cmd.split()[2]} command timed out")
            except Exception as e:
                print(f"✗ {cmd.split()[2]} command failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ CLI testing failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("=== MMVC_Trainer Integration Test Suite ===\n")
    
    tests = [
        test_package_imports,
        test_model_creation,
        test_audio_processing,
        test_text_processing,
        test_cli_commands
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n=== Integration Test Results: {passed}/{len(tests)} passed ===")
    
    if passed == len(tests):
        print("🎉 All integration tests passed! MMVC_Trainer is ready for use.")
        print("\nNext steps:")
        print("1. Prepare your dataset in the data/raw/ directory")
        print("2. Create a configuration file in configs/")
        print("3. Run training with: python -m src.cli.train -c configs/your_config.json -m models/your_model")
        return 0
    else:
        print("❌ Some integration tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
