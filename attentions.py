"""
Optimized Attention Mechanisms for MMVC_Trainer
Features: Streamlined MultiHeadAttention, Memory-efficient operations
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
from modules import LayerNorm
   

class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
      
      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x


class MultiHeadAttention(nn.Module):
  """Optimized Multi-Head Attention with relative positional encoding support"""
  
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., 
               window_size=None, heads_share=True, block_length=None, 
               proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.k_channels = channels // n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.attn = None

    # Projection layers
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    # Relative position embeddings
    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels**-0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    # Initialize weights
    for conv in [self.conv_q, self.conv_k, self.conv_v]:
      nn.init.xavier_uniform_(conv.weight)
    
    if proximal_init:
      self.conv_k.weight.data.copy_(self.conv_q.weight.data)
      self.conv_k.bias.data.copy_(self.conv_q.bias.data)

  def forward(self, x, c, attn_mask=None):
    q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
    return self.conv_o(x)

  def attention(self, query, key, value, mask=None):
    b, d, t_s, t_t = (*key.size(), query.size(2))
    
    # Reshape and transpose: [b, d, t] -> [b, n_h, t, d_k]
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    # Compute attention scores
    scale = math.sqrt(self.k_channels)
    scores = torch.matmul(query / scale, key.transpose(-2, -1))

    # Add relative position information
    if self.window_size is not None and t_s == t_t:
      scores = self._add_relative_attention(scores, query / scale, t_s)
    
    # Add proximal bias
    if self.proximal_bias and t_s == t_t:
      scores = scores + self._get_proximal_bias(t_s, scores.device, scores.dtype)

    # Apply masks
    scores = self._apply_masks(scores, mask, t_s, t_t)
    
    # Softmax and dropout
    p_attn = F.softmax(scores, dim=-1)
    p_attn = self.drop(p_attn)
    
    # Compute output
    output = torch.matmul(p_attn, value)
    
    # Add relative values
    if self.window_size is not None and t_s == t_t:
      output = self._add_relative_values(output, p_attn, t_s)
    
    # Reshape back: [b, n_h, t_t, d_k] -> [b, d, t_t]
    output = output.transpose(2, 3).contiguous().view(b, d, t_t)
    return output, p_attn

  def _add_relative_attention(self, scores, scaled_query, t_s):
    """Add relative position attention scores"""
    rel_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
    rel_logits = torch.matmul(scaled_query, rel_embeddings.unsqueeze(0).transpose(-2, -1))
    rel_scores = self._rel_to_abs_position(rel_logits)
    return scores + rel_scores

  def _add_relative_values(self, output, p_attn, t_s):
    """Add relative position values"""
    rel_weights = self._abs_to_rel_position(p_attn)
    rel_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
    rel_output = torch.matmul(rel_weights, rel_embeddings.unsqueeze(0))
    return output + rel_output

  def _apply_masks(self, scores, mask, t_s, t_t):
    """Apply attention and block masks"""
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
    
    if self.block_length is not None and t_s == t_t:
      block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
      scores = scores.masked_fill(block_mask == 0, -1e4)
    
    return scores

  def _get_relative_embeddings(self, relative_embeddings, length):
    """Get relative embeddings for given length"""
    max_pos = 2 * self.window_size + 1
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start = max((self.window_size + 1) - length, 0)
    slice_end = slice_start + 2 * length - 1
    
    if pad_length > 0:
      padded = F.pad(relative_embeddings, [0, 0, pad_length, pad_length])
    else:
      padded = relative_embeddings
    
    return padded[:, slice_start:slice_end]

  def _rel_to_abs_position(self, x):
    """Convert relative position to absolute position"""
    batch, heads, length, _ = x.size()
    x = F.pad(x, [0, 1])
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, [0, length - 1])
    x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])
    return x_final[:, :, :length, length - 1:]

  def _abs_to_rel_position(self, x):
    """Convert absolute position to relative position"""
    batch, heads, length, _ = x.size()
    x = F.pad(x, [0, length - 1])
    x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
    x_flat = F.pad(x_flat, [length, 0])
    return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

  def _get_proximal_bias(self, length, device, dtype):
    """Compute proximal bias for self-attention"""
    r = torch.arange(length, dtype=dtype, device=device)
    diff = r.unsqueeze(0) - r.unsqueeze(1)
    return -torch.log1p(torch.abs(diff)).unsqueeze(0).unsqueeze(0)


class FFN(nn.Module):
  """Optimized Feed-Forward Network with efficient padding"""
  
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, 
               p_dropout=0., activation=None, causal=False):
    super().__init__()
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    # First convolution with padding
    x = self._apply_padding_and_conv(x * x_mask, self.conv_1)
    
    # Activation
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)  # Approximate GELU
    else:
      x = F.relu(x)
    
    x = self.drop(x)
    
    # Second convolution with padding
    x = self._apply_padding_and_conv(x * x_mask, self.conv_2)
    
    return x * x_mask

  def _apply_padding_and_conv(self, x, conv):
    """Apply appropriate padding and convolution"""
    if self.kernel_size == 1:
      return conv(x)
    
    if self.causal:
      pad_l, pad_r = self.kernel_size - 1, 0
    else:
      pad_l = (self.kernel_size - 1) // 2
      pad_r = self.kernel_size // 2
    
    x = F.pad(x, [pad_l, pad_r])
    return conv(x)
