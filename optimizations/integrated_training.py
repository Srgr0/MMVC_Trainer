"""
Integrated Optimized Training Script for MMVC_Trainer
All optimizations integrated into a single, efficient training pipeline.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import utils
import commons
from models import SynthesizerTrn, MultiPeriodDiscriminator
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollateOptimized
from losses import LossOptimizer, AdaptiveLossWeighting
from optimizations.performance_monitor import OptimizedPerformanceMonitor
from optimizations.memory_optimizer import MemoryOptimizer
from optimizations.training_optimizer import TrainingOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import monotonic_align with fallback
try:
    from monotonic_align_fallback import safe_import_monotonic_align
    MONOTONIC_ALIGN_SUCCESS, monotonic_align_module, monotonic_align_error = safe_import_monotonic_align()
    if not MONOTONIC_ALIGN_SUCCESS:
        logger.warning(f"monotonic_align fallback activated: {monotonic_align_error}")
    else:
        logger.info("monotonic_align standard module loaded successfully")
except ImportError:
    logger.error("monotonic_align_fallback module not found")
    MONOTONIC_ALIGN_SUCCESS = False
    monotonic_align_module = None
    monotonic_align_error = "Fallback module not available"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedMMVCTrainer:
    """Fully optimized MMVC trainer with all enhancements integrated."""
    
    def __init__(self, config_path: str, model_dir: str, log_dir: str):
        """
        Initialize the optimized trainer.
        
        Args:
            config_path: Path to configuration file
            model_dir: Directory to save models
            log_dir: Directory to save logs
        """
        self.config = self._load_config(config_path)
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize optimizers and monitors
        self.memory_optimizer = MemoryOptimizer()
        self.performance_monitor = OptimizedPerformanceMonitor()
        self.training_optimizer = TrainingOptimizer(self.device)
        
        # Initialize loss components
        self.loss_optimizer = LossOptimizer(
            device=self.device,
            use_amp=self.config.train.get('use_amp', True),
            gradient_clip_val=self.config.train.get('gradient_clip_val', 1.0)
        )
        
        self.adaptive_loss_weighting = AdaptiveLossWeighting(
            initial_weights={
                'discriminator': 1.0,
                'generator': 1.0,
                'feature_matching': 2.0,
                'kl_divergence': 1.0,
                'mel_spectrogram': 45.0
            }
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Initialize models and data loaders
        self._setup_models()
        self._setup_data_loaders()
        self._setup_optimizers()
        
    def _load_config(self, config_path: str) -> utils.HParams:
        """Load and validate configuration."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Add optimized default values
        config_dict.setdefault('train', {}).update({
            'use_amp': True,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'checkpoint_interval': 1000,
            'validation_interval': 1000,
            'log_interval': 100
        })
        
        return utils.HParams(**config_dict)
    
    def _setup_models(self):
        """Initialize and configure models."""
        logger.info("Initializing models...")
        
        # Generator (Synthesizer)
        self.net_g = SynthesizerTrn(
            n_vocab=len(getattr(utils, 'symbols', [])) or self.config.data.get('n_vocab', 256),
            spec_channels=self.config.data.filter_length // 2 + 1,
            segment_size=self.config.train.segment_size // self.config.data.hop_length,
            **self.config.model
        ).to(self.device)
        
        # Discriminator
        self.net_d = MultiPeriodDiscriminator(
            use_spectral_norm=self.config.model.get('use_spectral_norm', False)
        ).to(self.device)
        
        # Apply memory optimizations
        self.memory_optimizer.optimize_model(self.net_g)
        self.memory_optimizer.optimize_model(self.net_d)
        
        logger.info(f"Generator parameters: {sum(p.numel() for p in self.net_g.parameters()):,}")
        logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.net_d.parameters()):,}")
    
    def _setup_data_loaders(self):
        """Setup optimized data loaders."""
        logger.info("Setting up data loaders...")
        
        # Training dataset
        train_dataset = TextAudioSpeakerLoader(
            self.config.data.training_files,
            self.config.data
        )
        
        # Validation dataset
        val_dataset = TextAudioSpeakerLoader(
            self.config.data.validation_files,
            self.config.data
        )
        
        # Optimized collate function
        collate_fn = TextAudioSpeakerCollateOptimized()
        
        # Data loaders with optimization
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.train.get('num_workers', 4),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def _setup_optimizers(self):
        """Setup optimized optimizers and schedulers."""
        logger.info("Setting up optimizers...")
        
        # Generator optimizer
        self.optim_g = optim.AdamW(
            self.net_g.parameters(),
            lr=self.config.train.learning_rate,
            betas=self.config.train.get('betas', [0.8, 0.99]),
            eps=self.config.train.get('eps', 1e-9),
            weight_decay=self.config.train.get('weight_decay', 0.01)
        )
        
        # Discriminator optimizer
        self.optim_d = optim.AdamW(
            self.net_d.parameters(),
            lr=self.config.train.learning_rate,
            betas=self.config.train.get('betas', [0.8, 0.99]),
            eps=self.config.train.get('eps', 1e-9),
            weight_decay=self.config.train.get('weight_decay', 0.01)
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.ExponentialLR(
            self.optim_g,
            gamma=self.config.train.get('lr_decay', 0.999875)
        )
        
        self.scheduler_d = optim.lr_scheduler.ExponentialLR(
            self.optim_d,
            gamma=self.config.train.get('lr_decay', 0.999875)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one optimized training step."""
        step_start_time = time.time()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=True)
        
        # Extract batch components
        x, x_lengths = batch['text'], batch['text_lengths']
        spec, spec_lengths = batch['spec'], batch['spec_lengths']
        y, y_lengths = batch['wav'], batch['wav_lengths']
        speakers = batch.get('speaker')
        
        # Forward pass timing
        forward_start = time.time()
        
        with autocast(enabled=self.loss_optimizer.use_amp):
            # Generator forward pass
            y_hat, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), vc_loss = self.net_g(
                x, x_lengths, spec, spec_lengths, speakers
            )
            
            # Mel-spectrogram computation
            mel = commons.mel_spectrogram_torch(
                y.squeeze(1),
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.hop_length,
                self.config.data.win_length,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax
            )
            
            y_mel = commons.slice_segments(
                mel, ids_slice, self.config.train.segment_size // self.config.data.hop_length
            )
            
            y_hat_mel = commons.mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.hop_length,
                self.config.data.win_length,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax
            )
            
            # Slice real audio for discriminator
            y = commons.slice_segments(
                y, ids_slice * self.config.data.hop_length, self.config.train.segment_size
            )
        
        forward_time = time.time() - forward_start
        
        # Discriminator training
        self.optim_d.zero_grad()
        
        with autocast(enabled=self.loss_optimizer.use_amp):
            # Discriminator forward pass
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat.detach())
        
        # Discriminator loss
        d_loss, d_metrics = self.loss_optimizer.compute_discriminator_loss(y_d_hat_r, y_d_hat_g)
        
        # Discriminator backward pass
        backward_start = time.time()
        d_grad_metrics = self.loss_optimizer.backward_and_step(d_loss, self.optim_d)
        d_backward_time = time.time() - backward_start
        
        # Generator training
        self.optim_g.zero_grad()
        
        with autocast(enabled=self.loss_optimizer.use_amp):
            # Generator discriminator outputs
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        
        # Generator losses
        g_loss, g_metrics = self.loss_optimizer.compute_generator_loss(y_d_hat_g, fmap_r, fmap_g)
        mel_loss, mel_metrics = self.loss_optimizer.compute_mel_loss(y_mel, y_hat_mel)
        kl_loss, kl_metrics = self.loss_optimizer.compute_kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        
        # Total generator loss
        total_g_loss = g_loss + mel_loss + kl_loss
        if vc_loss is not None:
            total_g_loss += vc_loss * 0.1  # Voice conversion loss weight
        
        # Generator backward pass
        g_grad_metrics = self.loss_optimizer.backward_and_step(total_g_loss, self.optim_g)
        g_backward_time = time.time() - backward_start - d_backward_time
        
        # Update learning rates
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        # Compile loss metrics
        loss_components = {
            'discriminator': d_loss.item(),
            'generator': g_loss.item(),
            'feature_matching': g_metrics.get('feature_matching_loss', 0),
            'kl_divergence': kl_loss.item(),
            'mel_spectrogram': mel_loss.item(),
            'total_loss': total_g_loss.item() + d_loss.item()
        }
        
        # Update adaptive loss weighting
        self.adaptive_loss_weighting.update_weights(loss_components)
        self.loss_optimizer.update_loss_weights(self.adaptive_loss_weighting.get_weights())
        
        # Performance metrics
        total_step_time = time.time() - step_start_time
        
        step_metrics = {
            'step_time': total_step_time,
            'forward_pass_time': forward_time,
            'backward_pass_time': d_backward_time + g_backward_time,
            'data_loading_time': 0,  # Would need separate timing
            'memory_usage': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None,
            'learning_rate': self.scheduler_g.get_last_lr()[0],
            'gradient_norm': max(d_grad_metrics.get('gradient_norm', 0), g_grad_metrics.get('gradient_norm', 0)),
            'loss_components': loss_components,
            'total_loss': loss_components['total_loss']
        }
        
        return step_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.net_g.train()
        self.net_d.train()
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Training step
            step_metrics = self.train_step(batch)
            epoch_metrics.append(step_metrics)
            
            # Log performance
            self.performance_monitor.log_detailed_step(step_metrics)
            
            # Periodic logging
            if self.global_step % self.config.train.log_interval == 0:
                avg_loss = sum(m['total_loss'] for m in epoch_metrics[-self.config.train.log_interval:]) / min(len(epoch_metrics), self.config.train.log_interval)
                logger.info(f"Epoch {self.epoch}, Step {self.global_step}: Loss = {avg_loss:.4f}, LR = {step_metrics['learning_rate']:.2e}")
            
            # Validation
            if self.global_step % self.config.train.validation_interval == 0:
                val_metrics = self.validate()
                logger.info(f"Validation Loss: {val_metrics['total_loss']:.4f}")
                
                # Save best model
                if val_metrics['total_loss'] < self.best_loss:
                    self.best_loss = val_metrics['total_loss']
                    self.save_checkpoint('best_model.pth')
            
            # Save checkpoint
            if self.global_step % self.config.train.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_{self.global_step}.pth')
            
            self.global_step += 1
        
        # Epoch summary
        epoch_summary = {
            'avg_loss': sum(m['total_loss'] for m in epoch_metrics) / len(epoch_metrics),
            'avg_step_time': sum(m['step_time'] for m in epoch_metrics) / len(epoch_metrics),
            'total_steps': len(epoch_metrics)
        }
        
        return epoch_summary
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.net_g.eval()
        self.net_d.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device, non_blocking=True)
                
                x, x_lengths = batch['text'], batch['text_lengths']
                spec, spec_lengths = batch['spec'], batch['spec_lengths']
                y, y_lengths = batch['wav'], batch['wav_lengths']
                speakers = batch.get('speaker')
                
                # Forward pass
                y_hat, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), vc_loss = self.net_g(
                    x, x_lengths, spec, spec_lengths, speakers
                )
                
                # Compute validation loss (simplified)
                mel = commons.mel_spectrogram_torch(
                    y.squeeze(1),
                    self.config.data.filter_length,
                    self.config.data.n_mel_channels,
                    self.config.data.sampling_rate,
                    self.config.data.hop_length,
                    self.config.data.win_length,
                    self.config.data.mel_fmin,
                    self.config.data.mel_fmax
                )
                
                y_mel = commons.slice_segments(
                    mel, ids_slice, self.config.train.segment_size // self.config.data.hop_length
                )
                
                y_hat_mel = commons.mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    self.config.data.filter_length,
                    self.config.data.n_mel_channels,
                    self.config.data.sampling_rate,
                    self.config.data.hop_length,
                    self.config.data.win_length,
                    self.config.data.mel_fmin,
                    self.config.data.mel_fmax
                )
                
                mel_loss = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
                val_losses.append(mel_loss.item())
        
        return {'total_loss': sum(val_losses) / len(val_losses)}
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'generator': self.net_g.state_dict(),
            'discriminator': self.net_d.state_dict(),
            'optimizer_g': self.optim_g.state_dict(),
            'optimizer_d': self.optim_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, self.model_dir / filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load training checkpoint."""
        checkpoint = torch.load(self.model_dir / filename, map_location=self.device)
        
        self.net_g.load_state_dict(checkpoint['generator'])
        self.net_d.load_state_dict(checkpoint['discriminator'])
        self.optim_g.load_state_dict(checkpoint['optimizer_g'])
        self.optim_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        logger.info(f"Checkpoint loaded: {filename}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info(f"Resuming training from epoch {self.epoch}, step {self.global_step}")
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        try:
            for epoch in range(self.epoch, num_epochs):
                self.epoch = epoch
                
                epoch_metrics = self.train_epoch()
                logger.info(f"Epoch {epoch} completed: {epoch_metrics}")
                
                # Generate performance report
                if epoch % 10 == 0:
                    report = self.performance_monitor.generate_optimization_report()
                    report_path = self.log_dir / f'performance_report_epoch_{epoch}.json'
                    self.performance_monitor.save_performance_log(str(report_path))
                    
                    # Log optimization suggestions
                    suggestions = report.get('optimization_suggestions', [])
                    if suggestions:
                        logger.info("Optimization suggestions:")
                        for suggestion in suggestions:
                            logger.info(f"  - {suggestion}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint('interrupted_checkpoint.pth')
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self.save_checkpoint('error_checkpoint.pth')
            raise
        
        finally:
            # Final performance report
            final_report = self.performance_monitor.generate_optimization_report()
            final_report_path = self.log_dir / 'final_performance_report.json'
            self.performance_monitor.save_performance_log(str(final_report_path))
            logger.info(f"Final performance report saved: {final_report_path}")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Optimized MMVC Training')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--model_dir', default='./models', help='Directory to save models')
    parser.add_argument('--log_dir', default='./logs', help='Directory to save logs')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--resume_from', help='Checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = OptimizedMMVCTrainer(
        config_path=args.config,
        model_dir=args.model_dir,
        log_dir=args.log_dir
    )
    
    # Start training
    trainer.train(
        num_epochs=args.num_epochs,
        resume_from=args.resume_from
    )


if __name__ == '__main__':
    main()
