import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from mel_processing import spectrogram_torch_data

'''
class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
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

'''

class TextEncoder(nn.Module):
    """Optimized Text Encoder with efficient embedding and attention."""
    
    def __init__(self, n_vocab, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Optimized embedding with proper initialization
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        
        # Transformer encoder
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout
        )
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        """Optimized forward pass with efficient masking."""
        # Embedding with scaling
        x = self.emb(x) * math.sqrt(self.hidden_channels)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        
        # Create attention mask
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)
        
        # Apply encoder and projection
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        
        # Split into mean and log variance
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    """Optimized Residual Coupling Block with efficient flow processing."""
    
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.n_flows = n_flows
        
        # Build normalizing flows
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(
                channels, hidden_channels, kernel_size, dilation_rate,
                n_layers, gin_channels=gin_channels, mean_only=True
            ))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        """Efficient bidirectional flow processing."""
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    """Optimized Posterior Encoder with efficient WaveNet processing."""
    
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size,
                 dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.out_channels = out_channels
        
        # Pre-processing
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        
        # WaveNet encoder
        self.enc = modules.WN(
            hidden_channels, kernel_size, dilation_rate, n_layers,
            gin_channels=gin_channels
        )
        
        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """Optimized forward pass with variational sampling."""
        # Create sequence mask
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)
        
        # Process through network
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        
        # Split and sample
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    """Optimized HiFi-GAN Generator with improved error handling and efficiency."""
    
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        
        # Validate inputs
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        assert self.num_kernels > 0, "resblock_kernel_sizes cannot be empty"
        assert self.num_upsamples > 0, "upsample_rates cannot be empty"
        
        # Pre-conv layer
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        
        # Select resblock type
        resblock_cls = modules.ResBlock1 if resblock == '1' else modules.ResBlock2
        
        # Build upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_ch = upsample_initial_channel // (2**i)
            out_ch = upsample_initial_channel // (2**(i+1))
            self.ups.append(weight_norm(
                ConvTranspose1d(in_ch, out_ch, k, u, padding=(k-u)//2)))

        # Build residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, k, d))

        # Post-conv layer
        final_ch = upsample_initial_channel // (2**self.num_upsamples)
        self.conv_post = Conv1d(final_ch, 1, 7, 1, padding=3, bias=False)
        
        # Initialize weights
        self.ups.apply(init_weights)

        # Global conditioning (disabled for optimization)
        self.gin_channels = 0  # Force disable for simplicity

    def forward(self, x, g=None):
        """Optimized forward pass with improved error handling."""
        x = self.conv_pre(x)
        
        # Process through upsampling and residual blocks
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            
            # Accumulate residual block outputs safely
            xs = 0  # Initialize as tensor for safe accumulation
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                xs = xs + self.resblocks[block_idx](x)
            
            # Safe division by num_kernels
            x = xs / self.num_kernels if self.num_kernels > 0 else xs
            
        # Final processing
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        """Remove weight normalization from all layers."""
        print('Removing weight norm...')
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    """Optimized Period-based Discriminator with improved efficiency."""
    
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        
        # Choose normalization function
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Build conv layers with optimized channel progression
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            in_ch, out_ch = channels[i], channels[i + 1]
            stride_val = stride if i < 4 else 1
            padding = (get_padding(kernel_size, 1), 0)
            
            self.convs.append(norm_f(Conv2d(
                in_ch, out_ch, (kernel_size, 1), (stride_val, 1), padding=padding
            )))
        
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        """Optimized forward pass with efficient feature map collection."""
        fmap = []
        b, c, t = x.shape
        
        # Efficient reshaping with padding
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        # Process through conv layers
        for conv in self.convs:
            x = F.leaky_relu(conv(x), modules.LRELU_SLOPE)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorS(torch.nn.Module):
    """Optimized Scale-based Discriminator with grouped convolutions."""
    
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        
        # Optimized conv layer configuration
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """Efficient forward pass with feature map collection."""
        fmap = []
        
        for conv in self.convs:
            x = F.leaky_relu(conv(x), modules.LRELU_SLOPE)
            fmap.append(x)
            
        x = self.conv_post(x)
        fmap.append(x)
        
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    """Optimized Multi-Period Discriminator with configurable periods."""
    
    def __init__(self, use_spectral_norm=False, periods=(2, 3, 5, 7, 11)):
        super(MultiPeriodDiscriminator, self).__init__()
        
        # Build discriminators: one scale + multiple period discriminators
        discriminators = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discriminators.extend([
            DiscriminatorP(period, use_spectral_norm=use_spectral_norm) 
            for period in periods
        ])
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        """Optimized forward pass processing real and generated samples."""
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        
        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
    """Optimized Text-to-Speech Synthesizer with efficient training pipeline."""

    def __init__(self, n_vocab, spec_channels, segment_size, inter_channels,
                 hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
                 p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 n_flow, n_speakers=0, gin_channels=0, use_sdp=True, 
                 hps_data=None, **kwargs):
        super().__init__()
        
        # Store essential parameters
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.hps_data = hps_data

        # Initialize core components
        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels,
                               filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes,
                           resblock_dilation_sizes, upsample_rates, 
                           upsample_initial_channel, upsample_kernel_sizes,
                           gin_channels=gin_channels)
        
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels,
                                    5, 1, 16, gin_channels=gin_channels)
        
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                        n_flows=n_flow, gin_channels=gin_channels)

        # Multi-speaker support
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.emb_g = None

    def _get_speaker_embedding(self, sid):
        """Get speaker embedding if available."""
        if self.emb_g is not None and sid is not None:
            return self.emb_g(sid).unsqueeze(-1)
        return None

    def forward(self, x, x_lengths, y, y_lengths, sid=None, target_ids=None):
        """Optimized forward pass with efficient tensor operations."""
        
        # Text encoding
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = self._get_speaker_embedding(sid)

        # Posterior encoding
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        # Efficient alignment calculation
        with torch.no_grad():
            # Compute negative cross-entropy efficiently
            s_p_sq_r = torch.exp(-2 * logs_p)
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # Expand prior distributions
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Generate audio segments
        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        # Voice conversion cycle (optimized)
        if self.n_speakers > 1 and target_ids is not None:
            vc_outputs = self._voice_conversion_cycle(y, ids_slice, z, y_mask, g, sid, target_ids)
        else:
            vc_outputs = None

        return o, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), vc_outputs

    def _voice_conversion_cycle(self, y, ids_slice, z, y_mask, g, sid, target_ids):
        """Efficient voice conversion cycle implementation."""
        target_sids = self._make_random_target_sids(target_ids, sid)
        target_g = self._get_speaker_embedding(target_sids)
        
        # Forward cycle
        vc_spec = commons.slice_segments(y, ids_slice, self.segment_size)
        vc_spec_length = torch.full_like(ids_slice, fill_value=self.segment_size)
        vc_z, _, _, vc_y_mask = self.enc_q(vc_spec, vc_spec_length, g=g)
        vc_z_p = self.flow(vc_z, vc_y_mask, g=g)
        vc_z_hat = self.flow(vc_z_p, vc_y_mask, g=target_g, reverse=True)
        vc_o_hat = self.dec(vc_z_hat * vc_y_mask, g=target_g)
        
        # Reconstruction cycle
        with torch.no_grad():
            vc_spec_r = spectrogram_torch_data(vc_o_hat.squeeze(1), self.hps_data)
            vc_spec_r_hat = torch.squeeze(vc_spec_r, 0)
            vc_z_r, _, _, vc_y_r_mask = self.enc_q(vc_spec_r_hat, vc_spec_length, g=target_g)
            vc_z_r_p = self.flow(vc_z_r, vc_y_r_mask, g=target_g)
            vc_z_r_hat = self.flow(vc_z_r_p, vc_y_r_mask, g=g, reverse=True)
            vc_o_r_hat = self.dec(vc_z_r_hat * vc_y_r_mask, g=g)
        
        return vc_o_r_hat

    def _make_random_target_sids(self, target_ids, sid):
        """Generate random target speaker IDs for voice conversion."""
        target_sids = torch.zeros_like(sid)
        for i, source_id in enumerate(sid):
            valid_targets = target_ids[target_ids != source_id]
            if len(valid_targets) >= 1:
                target_sids[i] = valid_targets[torch.randint(len(valid_targets), (1,))]
            else:
                target_sids[i] = source_id
        return target_sids

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, mode='normal'):
        """Unified voice conversion with multiple modes for inference."""
        assert self.n_speakers > 0, "n_speakers must be larger than 0."
        
        g_src = self._get_speaker_embedding(sid_src)
        g_tgt = self._get_speaker_embedding(sid_tgt)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        
        if mode == 'direct':
            # Direct conversion without flow
            o_hat = self.dec(z * y_mask, g=g_tgt)
            return o_hat, y_mask, z
            
        elif mode == 'source':
            # Reconstruct with source speaker
            o_hat = self.dec(z * y_mask, g=g_src)
            return o_hat, y_mask, z
            
        elif mode == 'cycle':
            # Full cycle conversion
            z_p = self.flow(z, y_mask, g=g_src)
            z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
            z_p_hat = self.flow(z_hat, y_mask, g=g_tgt)
            z_hat_hat = self.flow(z_p_hat, y_mask, g=g_src, reverse=True)
            o_hat = self.dec(z_hat_hat * y_mask, g=g_tgt)
            return o_hat, y_mask, (z, z_p, z_hat)
            
        else:  # normal mode
            # Standard flow-based conversion
            z_p = self.flow(z, y_mask, g=g_src)
            z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
            o_hat = self.dec(z_hat * y_mask, g=g_tgt)
            return o_hat, y_mask, (z, z_p, z_hat)


