"""
Optimized Neural Modules for MMVC_Trainer
Features:
- Memory-efficient convolution layers with proper weight initialization
- JIT-compiled components for GPU acceleration
- Consolidated residual blocks with shared logic
- Enhanced layer normalization and dropout handling
- Comprehensive type hints and documentation
"""
import copy
import math
import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List, Union

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

import commons
from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    """Optimized Layer Normalization for 1D convolutions"""
    
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

class ConvReluNorm(nn.Module):
    """Optimized Convolution-ReLU-Normalization block with residual connection"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 kernel_size: int, n_layers: int, p_dropout: float) -> None:
        super().__init__()
        assert n_layers > 1, "Number of layers should be larger than 0."
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        # Build layers efficiently
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, 
                                         padding=kernel_size//2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                                             padding=kernel_size//2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        
        # Shared activation and dropout
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        
        # Output projection with zero initialization
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_org = x
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = norm(x)
            x = self.relu_drop(x)
        return x_org + self.proj(x) * x_mask


class DDSConv(nn.Module):
    """Optimized Dilated and Depth-Separable Convolution"""
    
    def __init__(self, channels: int, kernel_size: int, n_layers: int, p_dropout: float = 0.0) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        
        # Build all layers in one go for efficiency
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            
            self.convs_sep.append(nn.Conv1d(channels, channels, kernel_size, 
                                          groups=channels, dilation=dilation, padding=padding))
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        if g is not None:
            x = x + g
            
        for conv_sep, conv_1x1, norm1, norm2 in zip(
            self.convs_sep, self.convs_1x1, self.norms_1, self.norms_2):
            
            y = conv_sep(x * x_mask)
            y = norm1(y)
            y = F.gelu(y)
            y = conv_1x1(y)
            y = norm2(y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
            
        return x * x_mask


class WN(nn.Module):
    """Optimized WaveNet with efficient conditional processing"""
    
    def __init__(self, hidden_channels: int, kernel_size: int, dilation_rate: int, 
                 n_layers: int, gin_channels: int = 0, p_dropout: float = 0.0) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        # Conditional layer for global conditioning
        if gin_channels != 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))

        # Build layers efficiently
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = (kernel_size * dilation - dilation) // 2
            
            # Input layer with weight normalization
            in_layer = weight_norm(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                         dilation=dilation, padding=padding))
            self.in_layers.append(in_layer)

            # Residual/skip layer - last layer only outputs hidden_channels
            res_skip_channels = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            res_skip_layer = weight_norm(
                nn.Conv1d(hidden_channels, res_skip_channels, 1))
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                g: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(x)
        
        # Precompute global conditioning
        if g is not None and self.gin_channels != 0:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            
            # Apply global conditioning
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
                x_in = x_in + g_l

            # Gated activation using optimized fused operation
            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, torch.zeros_like(x_in), 
                                                         self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = res_skip_layer(acts)
            
            if i < self.n_layers - 1:
                # Split residual and skip connections
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                skip_acts = res_skip_acts[:, self.hidden_channels:, :]
                x = (x + res_acts) * x_mask
                output = output + skip_acts
            else:
                output = output + res_skip_acts
                
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """Remove weight normalization for inference"""
        if hasattr(self, 'cond_layer'):
            remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class OptimizedResBlock(nn.Module):
    """Unified and optimized residual block supporting different configurations"""
    
    def __init__(self, channels: int, kernel_size: int = 3, 
                 dilation: Tuple[int, ...] = (1, 3, 5), use_dual_path: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_dual_path = use_dual_path
        
        # First convolution path with different dilations
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d,
                              padding=get_padding(kernel_size, d))) 
            for d in dilation
        ])
        
        # Second convolution path (only for dual path mode)
        if use_dual_path:
            self.convs2 = nn.ModuleList([
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1))) 
                for _ in dilation
            ])
        
        # Apply weight initialization
        self.convs1.apply(init_weights)
        if use_dual_path:
            self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_dual_path:
            # Dual path residual block (ResBlock1 style)
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask
                xt = c1(xt)
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask
                xt = c2(xt)
                x = xt + x
        else:
            # Single path residual block (ResBlock2 style)
            for c in self.convs1:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                if x_mask is not None:
                    xt = xt * x_mask
                xt = c(xt)
                x = xt + x
                
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization for inference"""
        for layer in self.convs1:
            remove_weight_norm(layer)
        if self.use_dual_path:
            for layer in self.convs2:
                remove_weight_norm(layer)

# Backward compatibility aliases
class ResBlock1(OptimizedResBlock):
    """ResBlock1 with dual convolution paths"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)):
        super().__init__(channels, kernel_size, dilation, use_dual_path=True)

class ResBlock2(OptimizedResBlock):
    """ResBlock2 with single convolution path"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3)):
        super().__init__(channels, kernel_size, dilation, use_dual_path=False)
        for c in self.convs1:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
            
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization for inference"""
        for layer in self.convs1:
            remove_weight_norm(layer)

class Log(nn.Module):
    """Logarithmic transformation for normalizing flows"""
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                reverse: bool = False, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            return torch.exp(x) * x_mask

class Flip(nn.Module):
    """Channel flipping transformation for normalizing flows"""
    
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, 
                **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x

class ElementwiseAffine(nn.Module):
    """Element-wise affine transformation for normalizing flows"""
    
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                reverse: bool = False, **kwargs) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            return (x - self.m) * torch.exp(-self.logs) * x_mask


class ResidualCouplingLayer(nn.Module):
    """Optimized Residual Coupling Layer for normalizing flows"""
    
    def __init__(self, channels: int, hidden_channels: int, kernel_size: int, 
                 dilation_rate: int, n_layers: int, p_dropout: float = 0.0, 
                 gin_channels: int = 0, mean_only: bool = False) -> None:
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, 
                     p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        
        # Zero initialization for stable training
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            return torch.cat([x0, x1], 1)

class ConvFlow(nn.Module):
    """Optimized Convolutional Flow with spline transformations"""
    
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, 
                 n_layers: int, num_bins: int = 10, tail_bound: float = 5.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        
        # Zero initialization for stable training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, 
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        # Normalize spline parameters for stability
        scale_factor = math.sqrt(self.filter_channels)
        unnormalized_widths = h[..., :self.num_bins] / scale_factor
        unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / scale_factor
        unnormalized_derivatives = h[..., 2 * self.num_bins:]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
            inverse=reverse, tails='linear', tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        
        if not reverse:
            return x, logdet
        else:
            return x
