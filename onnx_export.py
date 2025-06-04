#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import torch
import onnx
from onnxsim import simplify
import numpy as np

import utils
from models import SynthesizerTrn
from text.symbols import symbols


def export_onnx(checkpoint_path, config_path, output_path, opset_version=16):
    """
    Export PyTorch model to ONNX format
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Create model
    hps = utils.HParams(**config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=getattr(hps.data, 'n_speakers', 0),
        **hps.model
    )
    
    # Load checkpoint
    _ = utils.load_checkpoint(checkpoint_path, net_g, None)
    net_g.eval()
    
    # Remove weight norm for inference
    net_g.dec.remove_weight_norm()
    
    # Prepare dummy inputs for tracing
    sequence = torch.randint(low=0, high=len(symbols), size=(1, 50), dtype=torch.long)
    sequence_length = torch.LongTensor([sequence.size(1)])
    
    if getattr(hps.data, 'n_speakers', 0) > 0:
        speaker_id = torch.LongTensor([0])
        dummy_input = (sequence, sequence_length, speaker_id)
        input_names = ['input', 'input_lengths', 'sid']
    else:
        dummy_input = (sequence, sequence_length)
        input_names = ['input', 'input_lengths']
    
    output_names = ['output']
    
    # Set dynamic axes
    dynamic_axes = {
        'input': {1: 'sequence_length'},
        'input_lengths': {0: 'batch_size'},
        'output': {2: 'audio_length'}
    }
    
    if getattr(hps.data, 'n_speakers', 0) > 0:
        dynamic_axes['sid'] = {0: 'batch_size'}
    
    print("Exporting model to ONNX...")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            net_g.infer,
            dummy_input,
            output_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    print(f"ONNX model exported to: {output_path}")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful!")
        
        # Simplify ONNX model
        print("Simplifying ONNX model...")
        simplified_model, check = simplify(onnx_model)
        if check:
            simplified_path = output_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)
            print(f"Simplified ONNX model saved to: {simplified_path}")
        else:
            print("Warning: ONNX model simplification failed")
            
    except Exception as e:
        print(f"ONNX model verification failed: {e}")
    
    return output_path


def test_onnx_inference(onnx_path, config_path):
    """
    Test ONNX model inference
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        return
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    hps = utils.HParams(**config)
    
    # Create ONNX Runtime session
    providers = ['CPUExecutionProvider']
    if torch.cuda.is_available():
        providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Prepare test input
    sequence = np.random.randint(low=0, high=len(symbols), size=(1, 30), dtype=np.int64)
    sequence_length = np.array([sequence.shape[1]], dtype=np.int64)
    
    input_feed = {
        'input': sequence,
        'input_lengths': sequence_length
    }
    
    if getattr(hps.data, 'n_speakers', 0) > 0:
        speaker_id = np.array([0], dtype=np.int64)
        input_feed['sid'] = speaker_id
    
    print("Running ONNX inference test...")
    
    # Run inference
    outputs = session.run(None, input_feed)
    audio_output = outputs[0]
    
    print(f"ONNX inference successful!")
    print(f"Input shape: {sequence.shape}")
    print(f"Output shape: {audio_output.shape}")
    print(f"Output audio length: {audio_output.shape[-1] / hps.data.sampling_rate:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch VITS model to ONNX')
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-o', '--output', type=str, default='model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version')
    parser.add_argument('--test', action='store_true',
                        help='Test ONNX model after export')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Export model
    try:
        onnx_path = export_onnx(args.model, args.config, args.output, args.opset)
        
        if args.test:
            test_onnx_inference(onnx_path, args.config)
            
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
