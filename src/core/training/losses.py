"""Loss functions for MMVC training."""

import torch
import torch.nn.functional as F
from typing import List, Tuple


def feature_loss(fmap_r: List[List[torch.Tensor]], 
                fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
    """特徴マッチング損失を計算する."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += F.l1_loss(rl, gl)
    return loss * 2


def discriminator_loss(disc_real_outputs: List[torch.Tensor],
                      disc_generated_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """識別器の損失を計算する."""
    loss = 0
    r_losses = []
    g_losses = []
    
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        
        # 本物の音声に対する損失
        r_loss = torch.mean((1 - dr) ** 2)
        
        # 生成された音声に対する損失
        g_loss = torch.mean(dg ** 2)
        
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    
    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """生成器の敵対的損失を計算する."""
    loss = 0
    gen_losses = []
    
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    
    return loss, gen_losses


def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, 
           m_p: torch.Tensor, logs_p: torch.Tensor, 
           z_mask: torch.Tensor) -> torch.Tensor:
    """KLダイバージェンス損失を計算する."""
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()
    
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def mel_spectrogram_loss(y_mel: torch.Tensor, 
                        y_hat_mel: torch.Tensor) -> torch.Tensor:
    """メルスペクトログラム復元損失を計算する."""
    return F.l1_loss(y_mel, y_hat_mel)


def voice_conversion_loss(y_mel: torch.Tensor,
                         vc_o_hat_mel: torch.Tensor,
                         dispose_ratio: float = 0.25) -> torch.Tensor:
    """音声変換損失を計算する（中央部分のみ使用）."""
    # 両端を除去して中央部分のみを使用
    dispose_length = int(y_mel.size(2) * dispose_ratio)
    
    if dispose_length > 0:
        disposed_y_mel = y_mel[:, :, dispose_length:-dispose_length]
        disposed_vc_o_hat_mel = vc_o_hat_mel[:, :, dispose_length:-dispose_length]
    else:
        disposed_y_mel = y_mel
        disposed_vc_o_hat_mel = vc_o_hat_mel
    
    return F.l1_loss(disposed_y_mel, disposed_vc_o_hat_mel)


class MultiScaleSTFTLoss(torch.nn.Module):
    """マルチスケールSTFT損失."""
    
    def __init__(self, 
                 fft_sizes: List[int] = [1024, 2048, 512],
                 hop_sizes: List[int] = [120, 240, 50],
                 win_lengths: List[int] = [600, 1200, 240],
                 window: str = "hann_window"):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = getattr(torch, window)
        
    def stft_loss(self, x: torch.Tensor, y: torch.Tensor, 
                  fft_size: int, hop_size: int, win_length: int) -> torch.Tensor:
        """単一スケールのSTFT損失を計算する."""
        # STFT計算
        x_stft = torch.stft(
            x, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
            window=self.window(win_length).to(x.device), return_complex=True
        )
        y_stft = torch.stft(
            y, n_fft=fft_size, hop_length=hop_size, win_length=win_length,
            window=self.window(win_length).to(y.device), return_complex=True
        )
        
        # 振幅と位相を計算
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)
        
        # スペクトログラム損失
        spectral_loss = F.l1_loss(x_mag, y_mag)
        
        # ログスペクトログラム損失
        log_spectral_loss = F.l1_loss(
            torch.log(x_mag + 1e-7), 
            torch.log(y_mag + 1e-7)
        )
        
        return spectral_loss + log_spectral_loss
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """マルチスケールSTFT損失を計算する."""
        total_loss = 0.0
        
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            loss = self.stft_loss(x, y, fft_size, hop_size, win_length)
            total_loss += loss
        
        return total_loss / len(self.fft_sizes)


class DiscriminatorLoss(torch.nn.Module):
    """識別器用の損失関数."""
    
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, real_outputs: List[torch.Tensor],
               fake_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """識別器の損失を計算する."""
        real_loss = 0.0
        fake_loss = 0.0
        
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            if self.loss_type == "hinge":
                real_loss += torch.mean(F.relu(1.0 - real_out))
                fake_loss += torch.mean(F.relu(1.0 + fake_out))
            elif self.loss_type == "mse":
                real_loss += F.mse_loss(real_out, torch.ones_like(real_out))
                fake_loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return real_loss, fake_loss


class GeneratorLoss(torch.nn.Module):
    """生成器用の損失関数."""
    
    def __init__(self, loss_type: str = "hinge"):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, fake_outputs: List[torch.Tensor]) -> torch.Tensor:
        """生成器の敵対的損失を計算する."""
        loss = 0.0
        
        for fake_out in fake_outputs:
            if self.loss_type == "hinge":
                loss += -torch.mean(fake_out)
            elif self.loss_type == "mse":
                loss += F.mse_loss(fake_out, torch.ones_like(fake_out))
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


def compute_total_loss(generator_outputs: dict,
                      discriminator_outputs: dict,
                      targets: dict,
                      hparams) -> dict:
    """全ての損失を計算し、重み付き合計を返す."""
    losses = {}
    
    # メルスペクトログラム損失
    if 'y_mel' in targets and 'y_hat_mel' in generator_outputs:
        losses['mel'] = mel_spectrogram_loss(
            targets['y_mel'], 
            generator_outputs['y_hat_mel']
        ) * hparams.train.c_mel
    
    # KL損失
    if all(k in generator_outputs for k in ['z_p', 'logs_q', 'm_p', 'logs_p', 'z_mask']):
        losses['kl'] = kl_loss(
            generator_outputs['z_p'],
            generator_outputs['logs_q'],
            generator_outputs['m_p'],
            generator_outputs['logs_p'],
            generator_outputs['z_mask']
        ) * hparams.train.c_kl
    
    # 特徴マッチング損失
    if 'fmap_r' in discriminator_outputs and 'fmap_g' in discriminator_outputs:
        losses['fm'] = feature_loss(
            discriminator_outputs['fmap_r'],
            discriminator_outputs['fmap_g']
        )
    
    # 生成器敵対的損失
    if 'disc_gen_outputs' in discriminator_outputs:
        losses['gen'], _ = generator_loss(discriminator_outputs['disc_gen_outputs'])
    
    # 音声変換損失
    if 'vc_y_mel' in targets and 'vc_o_hat_mel' in generator_outputs:
        losses['vc'] = voice_conversion_loss(
            targets['vc_y_mel'],
            generator_outputs['vc_o_hat_mel']
        ) * hparams.train.c_mel
    
    # 総損失
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses
