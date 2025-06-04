"""
Optimized Common Utilities for MMVC_Trainer
Features: 
- JIT-compiled tensor operations for GPU acceleration
- Memory-efficient segment slicing
- Optimized timing signal generation
- Enhanced gradient clipping with proper norm computation
- Comprehensive type hints and documentation
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Union

def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize convolutional layer weights"""
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for convolution"""
    return (kernel_size * dilation - dilation) // 2

def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert padding shape for PyTorch"""
    return [item for sublist in reversed(pad_shape) for item in sublist]

def intersperse(lst: List, item) -> List:
    """Insert item between each element in list efficiently"""
    if not lst:
        return [item]
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def kl_divergence(m_p: torch.Tensor, logs_p: torch.Tensor, 
                  m_q: torch.Tensor, logs_q: torch.Tensor) -> torch.Tensor:
    """Optimized KL divergence KL(P||Q)"""
    exp_logs_p_2 = torch.exp(2.0 * logs_p)
    exp_logs_q_neg2 = torch.exp(-2.0 * logs_q)
    
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (exp_logs_p_2 + (m_p - m_q)**2) * exp_logs_q_neg2
    return kl

def rand_gumbel(shape: torch.Size, device: torch.device = None, 
                dtype: torch.dtype = None) -> torch.Tensor:
    """Sample from Gumbel distribution with overflow protection"""
    uniform = torch.rand(shape, device=device, dtype=dtype)
    uniform = uniform * 0.99998 + 0.00001  # Prevent log(0)
    return -torch.log(-torch.log(uniform))

def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Gumbel distribution with same shape as x"""
    return rand_gumbel(x.size(), x.device, x.dtype)

@torch.jit.script
def slice_segments_optimized(x: torch.Tensor, ids_str: torch.Tensor, 
                           segment_size: int) -> torch.Tensor:
    """Optimized segment slicing using advanced indexing"""
    b, d, t = x.size()
    
    # Create index tensor for all batches
    batch_indices = torch.arange(b, device=x.device).unsqueeze(1).unsqueeze(1)
    channel_indices = torch.arange(d, device=x.device).unsqueeze(0).unsqueeze(2)
    time_indices = ids_str.unsqueeze(1).unsqueeze(2) + torch.arange(segment_size, device=x.device)
    
    # Ensure indices are within bounds
    time_indices = torch.clamp(time_indices, 0, t - 1)
    
    return x[batch_indices, channel_indices, time_indices]

def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, 
                  segment_size: int = 4) -> torch.Tensor:
    """Slice segments from tensor (fallback for compatibility)"""
    if x.is_cuda and segment_size <= 64:  # Use optimized version for small segments
        return slice_segments_optimized(x, ids_str, segment_size)
    
    # Original implementation for large segments or CPU
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i].item()
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret

def rand_slice_segments(x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, 
                       segment_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random segment slicing with length constraints"""
    b, d, t = x.size()
    
    if x_lengths is None:
        x_lengths = torch.full((b,), t, device=x.device, dtype=torch.long)
    
    # Ensure x_lengths is not None and compute valid start positions
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids_str = (torch.rand(b, device=x.device) * ids_str_max.float()).long()
    
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


@torch.jit.script
def get_timing_signal_1d(length: int, channels: int, min_timescale: float = 1.0, 
                        max_timescale: float = 1.0e4, device: Optional[torch.device] = None) -> torch.Tensor:
    """Optimized sinusoidal timing signal generation"""
    position = torch.arange(length, dtype=torch.float, device=device)
    num_timescales = channels // 2
    
    if num_timescales <= 1:
        inv_timescales = torch.tensor([min_timescale], dtype=torch.float, device=device)
    else:
        log_timescale_increment = math.log(max_timescale / min_timescale) / (num_timescales - 1)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float, device=device) * -log_timescale_increment)
    
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    
    if channels % 2:
        signal = F.pad(signal, [0, 0, 0, 1])
    
    return signal.view(1, channels, length)

def add_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, 
                        max_timescale: float = 1.0e4) -> torch.Tensor:
    """Add timing signal to tensor"""
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale, x.device)
    return x + signal.to(dtype=x.dtype)

def cat_timing_signal_1d(x: torch.Tensor, min_timescale: float = 1.0, 
                        max_timescale: float = 1.0e4, axis: int = 1) -> torch.Tensor:
    """Concatenate timing signal to tensor"""
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale, x.device)
    return torch.cat([x, signal.to(dtype=x.dtype)], axis)


@torch.jit.script
def subsequent_mask(length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Create subsequent mask for sequence attention"""
    mask = torch.tril(torch.ones(length, length, device=device)).unsqueeze(0).unsqueeze(0)
    return mask

@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a: torch.Tensor, input_b: torch.Tensor, 
                                   n_channels: int) -> torch.Tensor:
    """Fused activation function for gated convolutions"""
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels, :])
    s_act = torch.sigmoid(in_act[:, n_channels:, :])
    return t_act * s_act

def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """Shift tensor by 1 position along last dimension"""
    return F.pad(x, [1, 0])[:, :, :-1]

def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Create sequence mask based on lengths"""
    if max_length is None:
        max_length = length.max().item()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@torch.jit.script
def generate_path_optimized(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Optimized path generation for alignment"""
    device = duration.device
    b, _, t_y, t_x = mask.shape
    
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    
    # Efficient difference computation
    shifted_path = F.pad(path, [0, 0, 1, 0])[:, :-1]
    path = path - shifted_path
    
    return path.unsqueeze(1).transpose(2, 3) * mask

def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate path for alignment (compatibility wrapper)"""
    return generate_path_optimized(duration, mask)

def clip_grad_value_(parameters, clip_value: Optional[float] = None, 
                    norm_type: float = 2.0) -> float:
    """Optimized gradient clipping with norm computation"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    
    norm_type = float(norm_type)
    total_norm = 0.0
    
    if clip_value is not None:
        clip_value = float(clip_value)
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            p.grad.data.clamp_(-clip_value, clip_value)
    else:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)
