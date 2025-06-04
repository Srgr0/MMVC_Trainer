"""
Optimized Loss Functions for MMVC Training
Enhanced with efficient tensor operations and JIT compilation support.
"""

import torch 
from torch.nn import functional as F
from typing import List, Tuple, Optional

import commons


@torch.jit.script
def _feature_loss_inner(rl: torch.Tensor, gl: torch.Tensor) -> torch.Tensor:
    """JIT-compiled inner loop for feature loss calculation."""
    return torch.mean(torch.abs(rl - gl))


def feature_loss(fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """
    Optimized feature matching loss with efficient tensor operations.
    
    Args:
        fmap_r: Real feature maps from discriminator
        fmap_g: Generated feature maps from discriminator
        
    Returns:
        Feature matching loss
    """
    loss = torch.tensor(0.0, dtype=torch.float32, device=fmap_r[0][0].device)
    
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # Ensure consistent dtypes and detach real features
            rl = rl.float().detach()
            gl = gl.float()
            loss += _feature_loss_inner(rl, gl)
    
    return loss * 2.0


@torch.jit.script
def _discriminator_loss_step(dr: torch.Tensor, dg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT-compiled discriminator loss computation for single step."""
    r_loss = torch.mean((1.0 - dr) ** 2)
    g_loss = torch.mean(dg ** 2)
    return r_loss, g_loss


def discriminator_loss(disc_real_outputs: List[torch.Tensor], 
                      disc_generated_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    Optimized discriminator loss with efficient batch processing.
    
    Args:
        disc_real_outputs: Real discriminator outputs
        disc_generated_outputs: Generated discriminator outputs
        
    Returns:
        Total loss, real losses, generated losses
    """
    device = disc_real_outputs[0].device
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        
        r_loss, g_loss = _discriminator_loss_step(dr, dg)
        total_loss += (r_loss + g_loss)
        
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return total_loss, r_losses, g_losses


@torch.jit.script
def _generator_loss_step(dg: torch.Tensor) -> torch.Tensor:
    """JIT-compiled generator loss computation for single step."""
    return torch.mean((1.0 - dg) ** 2)


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Optimized generator adversarial loss.
    
    Args:
        disc_outputs: Discriminator outputs for generated samples
        
    Returns:
        Total loss, individual losses
    """
    device = disc_outputs[0].device
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    gen_losses = []
    
    for dg in disc_outputs:
        dg = dg.float()
        loss_step = _generator_loss_step(dg)
        gen_losses.append(loss_step)
        total_loss += loss_step
    
    return total_loss, gen_losses


@torch.jit.script
def _kl_divergence_core(z_p: torch.Tensor, logs_q: torch.Tensor, 
                       m_p: torch.Tensor, logs_p: torch.Tensor) -> torch.Tensor:
    """JIT-compiled core KL divergence computation."""
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return kl


def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, 
           logs_p: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
    """
    Optimized KL divergence loss with efficient tensor operations.
    
    Args:
        z_p: Posterior samples [b, h, t_t]
        logs_q: Posterior log-variance [b, h, t_t]
        m_p: Prior mean [b, h, t_t]
        logs_p: Prior log-variance [b, h, t_t]
        z_mask: Sequence mask [b, 1, t_t]
        
    Returns:
        KL divergence loss
    """
    # Ensure all tensors are float32 for consistency
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    
    # Compute KL divergence efficiently
    kl = _kl_divergence_core(z_p, logs_q, m_p, logs_p)
    
    # Apply mask and normalize
    kl_masked = torch.sum(kl * z_mask)
    mask_sum = torch.sum(z_mask)
    
    # Avoid division by zero
    return kl_masked / torch.clamp(mask_sum, min=1e-7)


# Additional optimized loss functions for advanced training

def spectral_convergence_loss(y_mag: torch.Tensor, y_hat_mag: torch.Tensor) -> torch.Tensor:
    """
    Spectral convergence loss for improved audio quality.
    
    Args:
        y_mag: Target magnitude spectrogram
        y_hat_mag: Generated magnitude spectrogram
        
    Returns:
        Spectral convergence loss
    """
    return torch.norm(y_mag - y_hat_mag, p="fro") / torch.clamp(torch.norm(y_mag, p="fro"), min=1e-7)


def log_stft_magnitude_loss(y_mag: torch.Tensor, y_hat_mag: torch.Tensor) -> torch.Tensor:
    """
    Log STFT magnitude loss for improved audio quality.
    
    Args:
        y_mag: Target magnitude spectrogram
        y_hat_mag: Generated magnitude spectrogram
        
    Returns:
        Log STFT magnitude loss
    """
    return F.l1_loss(torch.log(torch.clamp(y_mag, min=1e-7)), 
                     torch.log(torch.clamp(y_hat_mag, min=1e-7)))


class OptimizedLossContainer:
    """Container for all optimized loss functions with caching."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self._cache = {}
    
    def compute_all_losses(self, real_outputs, generated_outputs, fmap_r, fmap_g,
                          z_p, logs_q, m_p, logs_p, z_mask) -> dict:
        """Compute all losses in one efficient pass."""
        losses = {}
        
        # Discriminator loss
        losses['d_loss'], losses['r_losses'], losses['g_losses'] = discriminator_loss(
            real_outputs, generated_outputs
        )
        
        # Generator loss
        losses['g_loss'], losses['gen_losses'] = generator_loss(generated_outputs)
        
        # Feature matching loss
        losses['fm_loss'] = feature_loss(fmap_r, fmap_g)
        
        # KL divergence loss
        losses['kl_loss'] = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
        
        return losses
