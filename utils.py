"""
Optimized Utilities for MMVC_Trainer
Features: Efficient checkpoint handling, streamlined logging, memory-optimized operations
"""
import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import wave
import csv
from mel_processing import spec_to_mel_torch_data
import warnings
from typing import Optional, Dict, Any, List, Tuple

# Global flags and logger
MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

class OptimizedCheckpoint:
    """Optimized checkpoint handling with error recovery"""
    
    @staticmethod
    def load(checkpoint_path: str, model: torch.nn.Module, 
             optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple:
        """Load checkpoint with robust error handling"""
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
            iteration = checkpoint_dict.get('iteration', 0)
            learning_rate = checkpoint_dict.get('learning_rate', 0.0001)
            
            # Load optimizer state
            if optimizer and 'optimizer' in checkpoint_dict:
                try:
                    optimizer.load_state_dict(checkpoint_dict['optimizer'])
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load model state with compatibility check
            saved_state = checkpoint_dict['model']
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            # Smart state dict loading
            new_state = {}
            for key, value in model_state.items():
                if key in saved_state:
                    try:
                        new_state[key] = saved_state[key]
                    except Exception as e:
                        logger.warning(f"Failed to load {key}: {e}")
                        new_state[key] = value
                else:
                    logger.info(f"Parameter {key} not found in checkpoint")
                    new_state[key] = value
            
            # Apply loaded state
            if hasattr(model, 'module'):
                model.module.load_state_dict(new_state)
            else:
                model.load_state_dict(new_state)
            
            logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
            return model, optimizer, learning_rate, iteration
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    @staticmethod
    def save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
             learning_rate: float, iteration: int, checkpoint_path: str) -> None:
        """Save checkpoint with atomic write"""
        logger.info(f"Saving checkpoint at iteration {iteration} to {checkpoint_path}")
        
        # Prepare state dict
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint_data = {
            'model': state_dict,
            'iteration': iteration,
            'optimizer': optimizer.state_dict(),
            'learning_rate': learning_rate
        }
        
        # Atomic save (write to temp file first)
        temp_path = checkpoint_path + '.tmp'
        try:
            torch.save(checkpoint_data, temp_path)
            os.rename(temp_path, checkpoint_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Failed to save checkpoint: {e}")
            raise

# Backward compatibility aliases
load_checkpoint = OptimizedCheckpoint.load
save_checkpoint = OptimizedCheckpoint.save


def save_vc_sample(hps, loader, collate, generator, name):
    """Optimized voice conversion sample saving"""
    if not (hasattr(hps.others, "input_filename") and 
            os.path.isfile(hps.others.input_filename)):
        return
    
    input_filename = hps.others.input_filename
    source_id = hps.others.source_id
    target_ids = hps.others.target_id if isinstance(hps.others.target_id, list) else [hps.others.target_id]

    # Load and process data
    dataset = loader("", hps.data, no_use_textfile=True, disable_tqdm=True)
    data = dataset.get_audio_text_speaker_pair([input_filename, source_id, "a"])
    data = collate()([data])
    
    with torch.no_grad():
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda(0) for x in data]
        
        # Use mel if specified
        if hps.model.use_mel_train:
            spec = spec_to_mel_torch_data(spec, hps.data)
        
        # Generate samples for each target
        for target_id in target_ids:
            sid_tgt = torch.LongTensor([target_id]).cuda(0)
            audio = generator.module.voice_conversion(spec, spec_lengths, 
                                                    sid_src=sid_src, sid_tgt=sid_tgt)[0][0,0]
            audio = (audio.data.cpu().float().numpy() * hps.data.max_wav_value).astype(np.int16)
            
            # Save audio file
            output_path = os.path.join(hps.model_dir, f"vc_{target_id}_{name}.wav")
            with wave.open(output_path, 'wb') as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(hps.data.sampling_rate)
                fh.writeframes(audio.tobytes())

def save_best_log(best_log_path: str, global_step: int, loss_mel_value: float, date: str):
    """Save best training log entry"""
    with open(best_log_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([global_step, loss_mel_value, date])

def summarize(writer, global_step: int, scalars: Dict = None, histograms: Dict = None, 
              images: Dict = None, audios: Dict = None, audio_sampling_rate: int = 22050):
    """Optimized tensorboard logging"""
    for data_dict, add_func in [
        (scalars, writer.add_scalar),
        (histograms, writer.add_histogram),
        (images, lambda k, v, s: writer.add_image(k, v, s, dataformats='HWC')),
        (audios, lambda k, v, s: writer.add_audio(k, v, s, audio_sampling_rate))
    ]:
        if data_dict:
            for k, v in data_dict.items():
                add_func(k, v, global_step)

def latest_checkpoint_path(dir_path: str, regex: str = "G_*.pth") -> str:
    """Find latest checkpoint with improved sorting"""
    pattern = os.path.join(dir_path, regex)
    f_list = glob.glob(pattern)
    if not f_list:
        raise FileNotFoundError(f"No checkpoints found matching {pattern}")
    
    # Sort by numeric value in filename
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    latest = f_list[-1]
    logger.info(f"Latest checkpoint: {latest}")
    return latest


def _init_matplotlib():
    """Initialize matplotlib with optimized settings"""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)

def plot_spectrogram_to_numpy(spectrogram):
    """Optimized spectrogram plotting"""
    _init_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def plot_alignment_to_numpy(alignment, info=None):
    """Optimized alignment plotting"""
    _init_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    
    xlabel = 'Decoder timestep'
    if info:
        xlabel += '\n\n' + info
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Encoder timestep')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

def load_wav_to_torch(full_path: str) -> Tuple[torch.FloatTensor, int]:
    """Load WAV file with warning suppression"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename: str, split: str = "|") -> List[List[str]]:
    """Load file paths and text from file"""
    with open(filename, encoding='utf-8') as f:
        return [line.strip().split(split) for line in f]


class HParams:
    """Optimized hyperparameters container with dynamic attributes"""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            setattr(self, key, value)
    
    def keys(self):
        return self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def values(self):
        return self.__dict__.values()
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def __repr__(self):
        return f"HParams({self.__dict__})"
    
    def update(self, other_dict):
        """Update parameters from dictionary"""
        for key, value in other_dict.items():
            if isinstance(value, dict):
                value = HParams(**value)
            setattr(self, key, value)

def get_hparams(init=True):
    """Optimized hyperparameter loading"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                       help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='Model name')
    parser.add_argument('-fg', '--fine_tuning_g', type=str, default=None,
                       help='Fine tuning generator model path')
    parser.add_argument('-fd', '--fine_tuning_d', type=str, default=None,
                       help='Fine tuning discriminator model path')
    
    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Handle best loss tracking
    best_log_path = os.path.join(model_dir, "best.log")
    if not os.path.exists(best_log_path):
        with open(best_log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss g/mel", "date"])
    
    # Read best loss
    try:
        with open(best_log_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            best_loss_mel = float(rows[-1][1]) if len(rows) > 1 else 9999.0
    except (IndexError, ValueError):
        best_loss_mel = 9999.0
    
    # Load config
    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    
    if init:
        # Copy config to model directory
        with open(config_path, "r") as f:
            config_data = f.read()
        with open(config_save_path, "w") as f:
            f.write(config_data)
    else:
        # Load from model directory
        with open(config_save_path, "r") as f:
            config_data = f.read()
    
    config = json.loads(config_data)
    
    # Add fine-tuning flags
    config['fine_flag'] = bool(args.fine_tuning_g and args.fine_tuning_d)
    if config['fine_flag']:
        config['fine_model_g'] = args.fine_tuning_g
        config['fine_model_d'] = args.fine_tuning_d
    
    # Create HParams object
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.best_log_path = best_log_path
    hparams.best_loss_mel = best_loss_mel
    
    return hparams

def get_hparams_from_dir(model_dir: str) -> HParams:
    """Load hyperparameters from model directory"""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams

def get_hparams_from_file(config_path: str) -> HParams:
    """Load hyperparameters from config file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return HParams(**config)

def check_git_hash(model_dir: str):
    """Check and save git hash for reproducibility"""
    source_dir = os.path.dirname(os.path.realpath(__file__))
    git_dir = os.path.join(source_dir, ".git")
    
    if not os.path.exists(git_dir):
        logger.warning(f"{source_dir} is not a git repository")
        return
    
    try:
        cur_hash = subprocess.getoutput("git rev-parse HEAD")
        hash_file = os.path.join(model_dir, "githash")
        
        if os.path.exists(hash_file):
            with open(hash_file) as f:
                saved_hash = f.read().strip()
            if saved_hash != cur_hash:
                logger.warning(f"Git hash mismatch: {saved_hash[:8]} (saved) != {cur_hash[:8]} (current)")
        else:
            with open(hash_file, "w") as f:
                f.write(cur_hash)
    except Exception as e:
        logger.warning(f"Failed to check git hash: {e}")

def get_logger(model_dir: str, filename: str = "train.log"):
    """Create optimized logger"""
    global logger
    logger_name = os.path.basename(model_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
        os.makedirs(model_dir, exist_ok=True)
        
        handler = logging.FileHandler(os.path.join(model_dir, filename))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
