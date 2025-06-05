"""Common modules for VITS model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """レイヤー正規化."""
    
    def __init__(self, channels: int, eps: float = 1e-5):
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
    """Conv1d + ReLU + LayerNorm のブロック."""
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 n_layers: int,
                 p_dropout: float):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # 最初の層
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        
        # 中間層
        for _ in range(n_layers - 2):
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        
        # 最後の層
        if n_layers > 1:
            self.conv_layers.append(
                nn.Conv1d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            self.norm_layers.append(LayerNorm(out_channels))
        else:
            # n_layers == 1の場合
            self.conv_layers[0] = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.norm_layers[0] = LayerNorm(out_channels)
        
        self.dropout = nn.Dropout(p_dropout)
    
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x = conv(x)
            
            if x_mask is not None:
                x = x * x_mask
            
            # 最後の層以外はReLUを適用
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
            
            x = norm(x)
        
        if x_mask is not None:
            x = x * x_mask
        
        return x


class WaveNet(nn.Module):
    """WaveNetスタイルの残差ブロック."""
    
    def __init__(self,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 gin_channels: int = 0,
                 p_dropout: float = 0):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p_dropout)
        
        if gin_channels != 0:
            cond_layer = nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')
        
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            
            in_layer = nn.Conv1d(
                hidden_channels, 
                2 * hidden_channels, 
                kernel_size,
                dilation=dilation, 
                padding=padding
            )
            in_layer = nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)
            
            # 残差とスキップ接続用
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            
            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)
    
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None, 
                g: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])
        
        if g is not None:
            g = self.cond_layer(g)
        
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            
            acts = x_in + g_l
            acts = F.glu(acts, dim=1)
            
            acts = self.dropout(acts)
            
            res_skip_acts = self.res_skip_layers[i](acts)
            
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask if x_mask is not None else x + res_acts
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        
        return output * x_mask if x_mask is not None else output


class ResidualCouplingBlock(nn.Module):
    """残差結合ブロック（Normalizing Flow用）."""
    
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 n_flows: int = 4,
                 gin_channels: int = 0):
        super().__init__()
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True
                )
            )
            self.flows.append(Flip())
    
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> torch.Tensor:
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    """アフィン結合層."""
    
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 p_dropout: float = 0,
                 gin_channels: int = 0,
                 mean_only: bool = False):
        super().__init__()
        
        assert channels % 2 == 0, "channels should be divisible by 2"
        
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
            p_dropout=p_dropout
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
    
    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                g: Optional[torch.Tensor] = None, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h)
        
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask if x_mask is not None else m + x1 * torch.exp(logs)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask if x_mask is not None else (x1 - m) * torch.exp(-logs)
            x = torch.cat([x0, x1], 1)
            return x, None


class Flip(nn.Module):
    """チャンネルを反転させる変換."""
    
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs) -> Tuple[torch.Tensor, None]:
        x = torch.flip(x, [1])
        return x, None


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """シーケンス用のマスクを生成する."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """継続時間からパスを生成する."""
    device = duration.device
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def convert_pad_shape(pad_shape):
    """パディング形状を変換する."""
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    """リストの要素間にアイテムを挿入する."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """テンソルからセグメントを切り出す."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, segment_size: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """ランダムにセグメントを切り出す."""
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str
