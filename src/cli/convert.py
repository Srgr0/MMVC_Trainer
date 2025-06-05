#!/usr/bin/env python3
# filepath: /Users/srypt/Documents/GitHub/MMVC_Trainer-my/src/cli/convert.py
"""Voice conversion script for MMVC."""

import argparse
import json
import os
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from ..core.models.vits import SynthesizerTrn
from ..utils.audio import load_wav, save_wav, get_mel_from_wav_torch
from ..utils.common import setup_logger, load_checkpoint
from ..text.cleaners import clean_text


def setup_args() -> argparse.ArgumentParser:
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice conversion using MMVC",
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
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input audio file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output audio file path"
    )
    parser.add_argument(
        "--target-speaker", "-ts",
        type=int,
        default=0,
        help="Target speaker ID"
    )
    
    # Generation parameters
    parser.add_argument(
        "--noise-scale", "-ns",
        type=float,
        default=0.667,
        help="Noise scale for generation"
    )
    parser.add_argument(
        "--noise-scale-w", "-nsw",
        type=float,
        default=0.8,
        help="Noise scale for duration predictor"
    )
    parser.add_argument(
        "--length-scale", "-ls",
        type=float,
        default=1.0,
        help="Length scale for generation"
    )
    
    # Processing arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference"
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


def extract_content_features(audio_path: str, model: SynthesizerTrn, 
                           config: dict, device: torch.device) -> torch.Tensor:
    """Extract content features from audio."""
    # Load audio
    audio = load_wav(audio_path, config["data"]["sampling_rate"])
    
    # Convert to mel spectrogram
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    mel = get_mel_from_wav_torch(
        audio_tensor,
        config["data"]["filter_length"],
        config["data"]["hop_length"],
        config["data"]["win_length"],
        config["data"]["n_mel_channels"],
        config["data"]["sampling_rate"],
        config["data"]["mel_fmin"],
        config["data"]["mel_fmax"]
    )
    
    # Convert to tensor
    mel = mel.unsqueeze(0).to(device)
    mel_lengths = torch.LongTensor([mel.size(2)]).to(device)
    
    # Extract content features using posterior encoder
    with torch.no_grad():
        z, _, _, _ = model.enc_q(mel, mel_lengths)
    
    return z


def convert_voice(source_audio: str, target_speaker: int, model: SynthesizerTrn,
                 config: dict, device: torch.device, 
                 noise_scale: float = 0.667, noise_scale_w: float = 0.8,
                 length_scale: float = 1.0) -> torch.Tensor:
    """Convert voice from source to target speaker."""
    # Extract content features
    content_features = extract_content_features(source_audio, model, config, device)
    
    # Set target speaker
    if model.n_speakers > 0:
        sid = torch.LongTensor([target_speaker]).to(device)
        g = model.emb_g(sid).unsqueeze(-1)
    else:
        g = None
    
    # Generate audio
    with torch.no_grad():
        # Create dummy text input (not used in voice conversion)
        x = torch.LongTensor([[0]]).to(device)  # dummy text
        x_lengths = torch.LongTensor([1]).to(device)
        
        # Use content features as latent representation
        z = content_features
        y_mask = torch.ones(1, 1, z.size(2), dtype=torch.float32, device=device)
        
        # Generate audio
        audio = model.dec(z * y_mask, g=g)
    
    return audio.squeeze()


def main():
    """Main function."""
    parser = setup_args()
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("mmvc_convert", "DEBUG" if args.verbose else "INFO")
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, device)
    
    # Perform voice conversion
    print(f"Converting voice from {args.input}")
    print(f"Target speaker: {args.target_speaker}")
    
    converted_audio = convert_voice(
        args.input,
        args.target_speaker,
        model,
        config,
        device,
        args.noise_scale,
        args.noise_scale_w,
        args.length_scale
    )
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving converted audio to {args.output}")
    save_wav(converted_audio.cpu().numpy(), args.output, config["data"]["sampling_rate"])
    
    print("Voice conversion completed!")


if __name__ == "__main__":
    main()
