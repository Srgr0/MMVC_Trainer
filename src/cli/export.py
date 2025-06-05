#!/usr/bin/env python3
# filepath: /Users/srypt/Documents/GitHub/MMVC_Trainer-my/src/cli/export.py
"""Export MMVC model to ONNX format."""

import argparse
import json
import os
import torch
import torch.onnx
from pathlib import Path
from typing import Optional

from ..core.models.vits import SynthesizerTrn
from ..utils.common import setup_logger, load_checkpoint


def setup_args() -> argparse.ArgumentParser:
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export MMVC model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint", "-ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Export arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--export-generator-only",
        action="store_true",
        help="Export only the generator (decoder) part"
    )
    parser.add_argument(
        "--export-encoder-only",
        action="store_true",
        help="Export only the text encoder part"
    )
    
    # ONNX export parameters
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--dynamic-axes",
        action="store_true",
        help="Use dynamic axes for variable length inputs"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize the exported model"
    )
    
    # Input shape parameters
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=200,
        help="Maximum text length for export"
    )
    parser.add_argument(
        "--max-audio-length",
        type=int,
        default=8192,
        help="Maximum audio length for export"
    )
    
    # Processing arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def load_model(config_path: str, checkpoint_path: str, device: torch.device) -> SynthesizerTrn:
    """Load model from checkpoint."""
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Create model
    model = SynthesizerTrn(
        n_vocab=config["data"]["n_vocab"],
        spec_channels=config["data"]["filter_length"] // 2 + 1,
        segment_size=config["train"]["segment_size"] // config["data"]["hop_length"],
        **config["model"]
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    return model


class TextEncoderWrapper(torch.nn.Module):
    """Wrapper for text encoder to make it ONNX-exportable."""
    
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.enc_p = model.enc_p
        self.dp = model.dp
        self.flow = model.flow
        self.emb_g = model.emb_g if hasattr(model, 'emb_g') else None
        self.n_speakers = model.n_speakers
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, 
                sid: Optional[torch.Tensor] = None, 
                noise_scale: float = 0.667, length_scale: float = 1.0):
        # Text encoding
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        # Speaker embedding
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        
        # Duration prediction
        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        
        # Generate path
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        
        # Sample from prior
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        
        return z_p, y_lengths, w_ceil


class GeneratorWrapper(torch.nn.Module):
    """Wrapper for generator to make it ONNX-exportable."""
    
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.dec = model.dec
        self.flow = model.flow
        self.emb_g = model.emb_g if hasattr(model, 'emb_g') else None
        self.n_speakers = model.n_speakers
    
    def forward(self, z: torch.Tensor, sid: Optional[torch.Tensor] = None):
        # Speaker embedding
        if self.n_speakers > 0 and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
        
        # Flow (posterior to prior)
        z = self.flow(z, g=g, reverse=True)
        
        # Generate audio
        audio = self.dec(z, g=g)
        
        return audio


def export_full_model(model: SynthesizerTrn, output_path: str, 
                     max_text_length: int, max_audio_length: int,
                     dynamic_axes: bool, opset_version: int,
                     device: torch.device):
    """Export full model for inference."""
    # Create dummy inputs
    x = torch.randint(0, model.n_vocab, (1, max_text_length), device=device)
    x_lengths = torch.LongTensor([max_text_length]).to(device)
    sid = torch.LongTensor([0]).to(device) if model.n_speakers > 0 else None
    
    # Input names
    input_names = ['text', 'text_lengths']
    output_names = ['audio', 'attention', 'audio_lengths']
    
    if model.n_speakers > 0:
        input_names.append('speaker_id')
    
    # Dynamic axes
    dynamic_axes_dict = {}
    if dynamic_axes:
        dynamic_axes_dict = {
            'text': {1: 'text_length'},
            'audio': {2: 'audio_length'},
            'attention': {2: 'text_length', 3: 'audio_length'}
        }
    
    # Export
    torch.onnx.export(
        model,
        (x, x_lengths, sid) if sid is not None else (x, x_lengths),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )


def export_encoder_only(model: SynthesizerTrn, output_path: str,
                       max_text_length: int, dynamic_axes: bool,
                       opset_version: int, device: torch.device):
    """Export text encoder only."""
    encoder_wrapper = TextEncoderWrapper(model)
    
    # Create dummy inputs
    x = torch.randint(0, model.n_vocab, (1, max_text_length), device=device)
    x_lengths = torch.LongTensor([max_text_length]).to(device)
    sid = torch.LongTensor([0]).to(device) if model.n_speakers > 0 else None
    
    # Input names
    input_names = ['text', 'text_lengths']
    output_names = ['latent', 'audio_lengths', 'durations']
    
    if model.n_speakers > 0:
        input_names.append('speaker_id')
    
    # Dynamic axes
    dynamic_axes_dict = {}
    if dynamic_axes:
        dynamic_axes_dict = {
            'text': {1: 'text_length'},
            'latent': {2: 'latent_length'},
            'durations': {2: 'text_length'}
        }
    
    # Export
    torch.onnx.export(
        encoder_wrapper,
        (x, x_lengths, sid) if sid is not None else (x, x_lengths),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )


def export_generator_only(model: SynthesizerTrn, output_path: str,
                         max_audio_length: int, dynamic_axes: bool,
                         opset_version: int, device: torch.device):
    """Export generator only."""
    generator_wrapper = GeneratorWrapper(model)
    
    # Create dummy inputs
    z = torch.randn(1, model.inter_channels, max_audio_length, device=device)
    sid = torch.LongTensor([0]).to(device) if model.n_speakers > 0 else None
    
    # Input names
    input_names = ['latent']
    output_names = ['audio']
    
    if model.n_speakers > 0:
        input_names.append('speaker_id')
    
    # Dynamic axes
    dynamic_axes_dict = {}
    if dynamic_axes:
        dynamic_axes_dict = {
            'latent': {2: 'latent_length'},
            'audio': {2: 'audio_length'}
        }
    
    # Export
    torch.onnx.export(
        generator_wrapper,
        (z, sid) if sid is not None else (z,),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )


def main():
    """Main function."""
    parser = setup_args()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mmvc_export", "DEBUG" if args.verbose else "INFO")
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, device)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    print(f"Exporting model to {args.output}")
    
    try:
        if args.export_encoder_only:
            print("Exporting text encoder only...")
            export_encoder_only(
                model, str(output_path), args.max_text_length,
                args.dynamic_axes, args.opset_version, device
            )
        elif args.export_generator_only:
            print("Exporting generator only...")
            export_generator_only(
                model, str(output_path), args.max_audio_length,
                args.dynamic_axes, args.opset_version, device
            )
        else:
            print("Exporting full model...")
            export_full_model(
                model, str(output_path), args.max_text_length, args.max_audio_length,
                args.dynamic_axes, args.opset_version, device
            )
        
        print("Export completed successfully!")
        
        # Optionally optimize the model
        if args.optimize:
            try:
                import onnx
                from onnxoptimizer import optimize
                
                print("Optimizing ONNX model...")
                onnx_model = onnx.load(str(output_path))
                optimized_model = optimize(onnx_model)
                onnx.save(optimized_model, str(output_path))
                print("Optimization completed!")
                
            except ImportError:
                print("Warning: onnxoptimizer not available, skipping optimization")
    
    except Exception as e:
        print(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
