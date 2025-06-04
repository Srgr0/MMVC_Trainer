"""
Loss Optimization Module for MMVC Training
Provides efficient loss computation strategies and automatic mixed precision support.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import time
from contextlib import contextmanager

class LossOptimizer:
    """Advanced loss computation optimizer with automatic mixed precision and gradient scaling."""
    
    def __init__(self, device: torch.device, use_amp: bool = True, 
                 gradient_clip_val: float = 1.0, loss_weights: Optional[Dict[str, float]] = None):
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
        
        # Default loss weights
        self.loss_weights = loss_weights or {
            'discriminator': 1.0,
            'generator': 1.0,
            'feature_matching': 2.0,
            'kl_divergence': 1.0,
            'mel_spectrogram': 45.0,
            'spectral_convergence': 0.1,
            'log_stft_magnitude': 0.1
        }
        
        # Performance tracking
        self.loss_history = {}
        self.computation_times = {}
        
    @contextmanager
    def autocast_context(self):
        """Context manager for automatic mixed precision."""
        if self.use_amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def compute_discriminator_loss(self, real_outputs: List[torch.Tensor], 
                                 fake_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Optimized discriminator loss computation with detailed metrics.
        
        Args:
            real_outputs: Discriminator outputs for real samples
            fake_outputs: Discriminator outputs for fake samples
            
        Returns:
            Total loss and detailed loss metrics
        """
        start_time = time.time()
        
        with self.autocast_context():
            total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            real_losses = []
            fake_losses = []
            
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                # Compute losses for each discriminator
                real_loss = torch.mean((1.0 - real_out) ** 2)
                fake_loss = torch.mean(fake_out ** 2)
                
                total_loss += (real_loss + fake_loss)
                real_losses.append(real_loss.item())
                fake_losses.append(fake_loss.item())
        
        # Apply loss weight
        weighted_loss = total_loss * self.loss_weights['discriminator']
        
        # Track metrics
        metrics = {
            'total_loss': weighted_loss.item(),
            'real_losses': real_losses,
            'fake_losses': fake_losses,
            'avg_real_loss': sum(real_losses) / len(real_losses),
            'avg_fake_loss': sum(fake_losses) / len(fake_losses)
        }
        
        self.computation_times['discriminator'] = time.time() - start_time
        return weighted_loss, metrics
    
    def compute_generator_loss(self, fake_outputs: List[torch.Tensor], 
                             feature_maps_real: List[List[torch.Tensor]], 
                             feature_maps_fake: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Optimized generator loss computation including adversarial and feature matching losses.
        
        Args:
            fake_outputs: Discriminator outputs for generated samples
            feature_maps_real: Real feature maps from discriminator
            feature_maps_fake: Fake feature maps from discriminator
            
        Returns:
            Total loss and detailed loss metrics
        """
        start_time = time.time()
        
        with self.autocast_context():
            # Adversarial loss
            adv_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for fake_out in fake_outputs:
                adv_loss += torch.mean((1.0 - fake_out) ** 2)
            
            # Feature matching loss
            fm_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for fmap_r, fmap_f in zip(feature_maps_real, feature_maps_fake):
                for real_feat, fake_feat in zip(fmap_r, fmap_f):
                    fm_loss += F.l1_loss(fake_feat, real_feat.detach())
        
        # Apply loss weights
        weighted_adv_loss = adv_loss * self.loss_weights['generator']
        weighted_fm_loss = fm_loss * self.loss_weights['feature_matching']
        total_loss = weighted_adv_loss + weighted_fm_loss
        
        # Track metrics
        metrics = {
            'total_loss': total_loss.item(),
            'adversarial_loss': weighted_adv_loss.item(),
            'feature_matching_loss': weighted_fm_loss.item(),
            'adv_ratio': weighted_adv_loss.item() / (total_loss.item() + 1e-7),
            'fm_ratio': weighted_fm_loss.item() / (total_loss.item() + 1e-7)
        }
        
        self.computation_times['generator'] = time.time() - start_time
        return total_loss, metrics
    
    def compute_kl_loss(self, z_p: torch.Tensor, logs_q: torch.Tensor, 
                       m_p: torch.Tensor, logs_p: torch.Tensor, 
                       z_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Optimized KL divergence loss computation.
        
        Args:
            z_p: Posterior samples
            logs_q: Posterior log-variance
            m_p: Prior mean
            logs_p: Prior log-variance
            z_mask: Sequence mask
            
        Returns:
            KL loss and metrics
        """
        start_time = time.time()
        
        with self.autocast_context():
            # Ensure float32 precision
            z_p = z_p.float()
            logs_q = logs_q.float()
            m_p = m_p.float()
            logs_p = logs_p.float()
            z_mask = z_mask.float()
            
            # Compute KL divergence
            kl = logs_p - logs_q - 0.5
            kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
            
            # Apply mask and normalize
            kl_masked = torch.sum(kl * z_mask)
            mask_sum = torch.clamp(torch.sum(z_mask), min=1e-7)
            kl_loss = kl_masked / mask_sum
        
        # Apply loss weight
        weighted_loss = kl_loss * self.loss_weights['kl_divergence']
        
        # Track metrics
        metrics = {
            'kl_loss': weighted_loss.item(),
            'raw_kl_loss': kl_loss.item(),
            'kl_per_timestep': kl_loss.item() / mask_sum.item(),
            'effective_sequence_length': mask_sum.item()
        }
        
        self.computation_times['kl_divergence'] = time.time() - start_time
        return weighted_loss, metrics
    
    def compute_mel_loss(self, y_mel: torch.Tensor, y_hat_mel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Optimized mel-spectrogram loss computation.
        
        Args:
            y_mel: Target mel-spectrogram
            y_hat_mel: Generated mel-spectrogram
            
        Returns:
            Mel loss and metrics
        """
        start_time = time.time()
        
        with self.autocast_context():
            mel_loss = F.l1_loss(y_mel, y_hat_mel)
        
        # Apply loss weight
        weighted_loss = mel_loss * self.loss_weights['mel_spectrogram']
        
        # Track metrics
        metrics = {
            'mel_loss': weighted_loss.item(),
            'raw_mel_loss': mel_loss.item(),
            'mel_l1_distance': mel_loss.item()
        }
        
        self.computation_times['mel_spectrogram'] = time.time() - start_time
        return weighted_loss, metrics
    
    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, 
                         retain_graph: bool = False) -> Dict[str, float]:
        """
        Optimized backward pass with gradient scaling and clipping.
        
        Args:
            loss: Loss tensor to backpropagate
            optimizer: Optimizer to step
            retain_graph: Whether to retain computation graph
            
        Returns:
            Gradient metrics
        """
        start_time = time.time()
        
        if self.scaler is not None:
            # Scaled backward pass for mixed precision
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    self.gradient_clip_val
                )
            else:
                grad_norm = 0.0
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward(retain_graph=retain_graph)
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']],
                    self.gradient_clip_val
                )
            else:
                grad_norm = 0.0
            
            # Optimizer step
            optimizer.step()
        
        # Track metrics
        backward_time = time.time() - start_time
        metrics = {
            'backward_time': backward_time,
            'gradient_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'gradient_clipped': grad_norm > self.gradient_clip_val if self.gradient_clip_val > 0 else False
        }
        
        return metrics
    
    def update_loss_weights(self, new_weights: Dict[str, float]):
        """Update loss weights dynamically during training."""
        self.loss_weights.update(new_weights)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for loss computations."""
        return {
            'total_computation_time': sum(self.computation_times.values()),
            **self.computation_times
        }
    
    def reset_stats(self):
        """Reset performance tracking statistics."""
        self.loss_history.clear()
        self.computation_times.clear()


class AdaptiveLossWeighting:
    """Adaptive loss weighting strategy based on loss magnitudes and training progress."""
    
    def __init__(self, initial_weights: Dict[str, float], adaptation_rate: float = 0.1):
        self.weights = initial_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.loss_history = {name: [] for name in initial_weights.keys()}
        self.update_count = 0
    
    def update_weights(self, current_losses: Dict[str, float], target_ratios: Optional[Dict[str, float]] = None):
        """
        Update loss weights based on current loss magnitudes.
        
        Args:
            current_losses: Current loss values
            target_ratios: Target ratios for each loss component
        """
        self.update_count += 1
        
        # Store loss history
        for name, loss_val in current_losses.items():
            if name in self.loss_history:
                self.loss_history[name].append(loss_val)
                # Keep only recent history
                if len(self.loss_history[name]) > 100:
                    self.loss_history[name] = self.loss_history[name][-100:]
        
        # Adaptive weighting based on loss magnitudes
        if self.update_count > 10:  # Start adapting after some iterations
            total_loss = sum(current_losses.values())
            
            for name in self.weights.keys():
                if name in current_losses and total_loss > 0:
                    # Calculate recent average
                    recent_avg = sum(self.loss_history[name][-10:]) / min(10, len(self.loss_history[name]))
                    
                    # Adaptive adjustment
                    current_ratio = current_losses[name] / total_loss
                    target_ratio = target_ratios.get(name, 1.0 / len(self.weights)) if target_ratios else 1.0 / len(self.weights)
                    
                    # Adjust weight to move towards target ratio
                    if current_ratio > target_ratio:
                        self.weights[name] *= (1.0 - self.adaptation_rate)
                    else:
                        self.weights[name] *= (1.0 + self.adaptation_rate)
                    
                    # Clamp weights to reasonable range
                    self.weights[name] = max(0.01, min(10.0, self.weights[name]))
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return self.weights.copy()
