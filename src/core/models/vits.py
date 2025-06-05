"""VITS model implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .commons import (
    sequence_mask, generate_path, rand_slice_segments,
    ResidualCouplingBlock, ConvReluNorm, LayerNorm, WaveNet
)
from .attentions import Encoder


class TextEncoder(nn.Module):
    """テキストエンコーダー."""
    
    def __init__(self,
                 n_vocab: int,
                 out_channels: int,
                 hidden_channels: int,
                 filter_channels: int,
                 n_heads: int,
                 n_layers: int,
                 kernel_size: int,
                 p_dropout: float):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlocks(nn.Module):
    """複数の残差結合ブロック."""
    
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels
                )
            )
    
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    """事後分布エンコーダー."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 gin_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(nn.Module):
    """HiFi-GANスタイルのジェネレーター."""
    
    def __init__(self,
                 initial_channel: int,
                 resblock: str,
                 resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list,
                 upsample_rates: list,
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: list,
                 gin_channels: int = 0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    k, u, padding=(k-u)//2
                )
            )
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    
    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class ResBlock1(nn.Module):
    """残差ブロック1."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                      padding=get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                      padding=get_padding(kernel_size, dilation[2]))
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1)),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1))
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    """残差ブロック2."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                      padding=get_padding(kernel_size, dilation[0])),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1]))
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x


class DurationPredictor(nn.Module):
    """継続時間予測器."""
    
    def __init__(self,
                 in_channels: int,
                 filter_channels: int,
                 kernel_size: int,
                 p_dropout: float,
                 gin_channels: int = 0):
        super().__init__()
        
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class SynthesizerTrn(nn.Module):
    """VITS synthesizer."""
    
    def __init__(self,
                 n_vocab: int,
                 spec_channels: int,
                 segment_size: int,
                 inter_channels: int,
                 hidden_channels: int,
                 filter_channels: int,
                 n_heads: int,
                 n_layers: int,
                 kernel_size: int,
                 p_dropout: float,
                 resblock: str,
                 resblock_kernel_sizes: list,
                 resblock_dilation_sizes: list,
                 upsample_rates: list,
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: list,
                 n_speakers: int = 0,
                 gin_channels: int = 0,
                 use_sdp: bool = True,
                 **kwargs):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout
        )
        
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels
        )
        
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels
        )
        
        self.flow = ResidualCouplingBlocks(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels
        )
        
        self.dp = DurationPredictor(
            hidden_channels,
            256,
            3,
            0.5,
            gin_channels
        )
        
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
    
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, y: torch.Tensor, y_lengths: torch.Tensor,
                sid: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        
        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
        
        w = attn.sum(2)
        
        if self.use_sdp:
            l_length = self.dp(x, x_mask, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask)  # for averaging
        
        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        
        return {
            'o': o,
            'l_length': l_length,
            'attn': attn,
            'ids_slice': ids_slice,
            'x_mask': x_mask,
            'y_mask': y_mask,
            'm_p': m_p,
            'logs_p': logs_p,
            'm_q': m_q,
            'logs_q': logs_q,
            'z': z,
            'z_p': z_p,
        }
    
    def infer(self, x: torch.Tensor, x_lengths: torch.Tensor,
              sid: Optional[torch.Tensor] = None, noise_scale: float = 1.0,
              length_scale: float = 1.0, noise_scale_w: float = 1.0,
              max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
        else:
            logw = self.dp(x, x_mask, g=g)
            w = torch.exp(logw) * x_mask * length_scale
        
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)
        
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        
        return o, attn, y_mask


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """パディングサイズを計算する."""
    return int((kernel_size * dilation - dilation) / 2)


# Import monotonic alignment module
from ..alignment import maximum_path

class MonotonicAlign:
    """Monotonic alignment search wrapper."""
    
    @staticmethod
    def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Maximum path search using optimized implementation."""
        return maximum_path(neg_cent, mask)


# Global instance for compatibility
monotonic_align = MonotonicAlign()
