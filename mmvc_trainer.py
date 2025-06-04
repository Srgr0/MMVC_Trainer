#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='MMVC Trainer Setup and Utilities')
    parser.add_argument('--setup', action='store_true',
                        help='Setup monotonic_align Cython extension')
    parser.add_argument('--create-dataset', type=str, metavar='DATASET_DIR',
                        help='Create dataset from textful directory')
    parser.add_argument('--train', action='store_true',
                        help='Start training (single speaker)')
    parser.add_argument('--train-ms', action='store_true',
                        help='Start multi-speaker training')
    parser.add_argument('--config', type=str, default='configs/baseconfig.json',
                        help='Configuration file path')
    parser.add_argument('--model-dir', type=str, default='logs',
                        help='Model output directory')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_monotonic_align()
    elif args.create_dataset:
        create_dataset(args.create_dataset)
    elif args.train:
        start_training(args.config, args.model_dir, multi_speaker=False)
    elif args.train_ms:
        start_training(args.config, args.model_dir, multi_speaker=True)
    else:
        print_usage()


def print_usage():
    """Print usage information"""
    print("MMVC Trainer - AI Voice Conversion Training System")
    print("=" * 50)
    print()
    print("Available commands:")
    print("  --setup                 Setup Cython extensions")
    print("  --create-dataset DIR    Create dataset from textful directory")
    print("  --train                 Start single-speaker training")
    print("  --train-ms              Start multi-speaker training")
    print()
    print("Example usage:")
    print("  python mmvc_trainer.py --setup")
    print("  python mmvc_trainer.py --create-dataset dataset/")
    print("  python mmvc_trainer.py --train --config configs/baseconfig.json")
    print("  python mmvc_trainer.py --train-ms --config configs/multi_speaker.json")


def setup_monotonic_align():
    """Setup monotonic_align Cython extension"""
    print("Setting up monotonic_align Cython extension...")
    
    os.chdir('monotonic_align')
    
    # Build extension
    import subprocess
    result = subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ monotonic_align setup completed successfully!")
        print("Output:", result.stdout)
    else:
        print("✗ monotonic_align setup failed!")
        print("Error:", result.stderr)
        return False
    
    os.chdir('..')
    
    # Test import
    try:
        import monotonic_align
        print("✓ monotonic_align import test successful!")
        return True
    except ImportError as e:
        print(f"✗ monotonic_align import test failed: {e}")
        return False


def create_dataset(dataset_dir):
    """Create dataset using create_dataset_jtalk.py"""
    print(f"Creating dataset from: {dataset_dir}")
    
    import subprocess
    cmd = [
        sys.executable, 'create_dataset_jtalk.py',
        '--dataset_dir', dataset_dir,
        '--output_dir', 'filelists',
        '--config_template', 'configs/dataset_config.json'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Dataset creation completed successfully!")
        print(result.stdout)
    else:
        print("✗ Dataset creation failed!")
        print("Error:", result.stderr)


def start_training(config_path, model_dir, multi_speaker=False):
    """Start training process"""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        return
    
    # Check if dataset files exist
    if multi_speaker:
        required_files = ['filelists/train.txt', 'filelists/val.txt']
        script_name = 'train_ms.py'
        print("Starting multi-speaker training...")
    else:
        required_files = ['filelists/train.txt', 'filelists/val.txt']
        script_name = 'train.py'
        print("Starting single-speaker training...")
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            print("Please create dataset first using --create-dataset")
            return
    
    # Create model directory
    Path(model_dir).mkdir(exist_ok=True)
    
    # Copy config to model directory
    import shutil
    config_copy = os.path.join(model_dir, 'config.json')
    shutil.copy(config_path, config_copy)
    
    print(f"Using configuration: {config_path}")
    print(f"Model output directory: {model_dir}")
    print(f"Training script: {script_name}")
    
    # Start training
    import subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU by default
    
    cmd = [sys.executable, script_name]
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")


def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'torch',
        'librosa',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
        'pyopenjtalk',
        'Cython'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_system():
    """Check system requirements"""
    print("MMVC Trainer System Check")
    print("=" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available (GPUs: {torch.cuda.device_count()})")
        else:
            print("⚠ CUDA not available (CPU training will be slow)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Check key directories
    required_dirs = ['configs', 'dataset', 'filelists', 'monotonic_align', 'text']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory: {dir_name}")
        else:
            print(f"✗ Missing directory: {dir_name}")
            return False
    
    return True


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments provided, show system check and usage
        if check_system():
            print("\n" + "=" * 50)
            print_usage()
    else:
        main()
