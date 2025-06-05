"""Audio processing utilities for MMVC."""

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window, lfilter
from typing import Optional, Tuple


def load_wav(path: str, sr: int = 24000) -> np.ndarray:
    """音声ファイルを読み込む."""
    wav, _ = librosa.load(path, sr=sr)
    return wav


def save_wav(wav: np.ndarray, path: str, sr: int = 24000) -> None:
    """音声ファイルを保存する."""
    # librosa.output.write_wavは廃止されたため、soundfileを使用
    try:
        import soundfile as sf
        sf.write(path, wav, sr)
    except ImportError:
        # fallback: scipy.io.wavfile
        from scipy.io import wavfile
        wavfile.write(path, sr, (wav * 32767).astype(np.int16))


def preemphasis(wav: np.ndarray, k: float = 0.97) -> np.ndarray:
    """プリエンファシスフィルターを適用する."""
    return np.append(wav[0], wav[1:] - k * wav[:-1])


def inv_preemphasis(wav: np.ndarray, k: float = 0.97) -> np.ndarray:
    """逆プリエンファシスフィルターを適用する."""
    result = lfilter([1], [1, -k], wav)
    return result if isinstance(result, np.ndarray) else result[0]


def get_mel_from_wav(audio: np.ndarray, _stft) -> torch.Tensor:
    """音声からメルスペクトログラムを計算する."""
    audio_tensor = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio_tensor = torch.autograd.Variable(audio_tensor, requires_grad=False)
    melspec = _stft.mel_spectrogram(audio_tensor)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def get_mel_from_wav_torch(audio: torch.Tensor,
                          filter_length: int = 1024,
                          hop_length: int = 256,
                          win_length: int = 1024,
                          n_mel_channels: int = 80,
                          sampling_rate: int = 24000,
                          mel_fmin: float = 0.0,
                          mel_fmax: Optional[float] = None) -> torch.Tensor:
    """PyTorchテンソルから直接メルスペクトログラムを計算する."""
    if mel_fmax is None:
        mel_fmax = sampling_rate // 2
    
    # STFTを計算
    spec = torch.stft(
        audio,
        n_fft=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=audio.device),
        return_complex=True
    )
    
    # 振幅スペクトログラムに変換
    spec = torch.abs(spec)
    
    # メルフィルターバンクを作成
    mel_basis = librosa_mel_fn(
        sr=sampling_rate,
        n_fft=filter_length,
        n_mels=n_mel_channels,
        fmin=mel_fmin,
        fmax=mel_fmax
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(audio.device)
    
    # メルスペクトログラムを計算
    melspec = torch.matmul(mel_basis, spec)
    
    # ログスケールに変換
    melspec = torch.log(torch.clamp(melspec, min=1e-5))
    
    return melspec


def librosa_mel_fn(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Librosaのメルフィルターバンクを作成する."""
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


def dynamic_range_compression_torch(x: torch.Tensor, C: float = 1, clip_val: float = 1e-5) -> torch.Tensor:
    """動的レンジ圧縮を適用する."""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1) -> torch.Tensor:
    """動的レンジ圧縮を解除する."""
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """スペクトログラムの正規化."""
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """スペクトログラムの逆正規化."""
    output = dynamic_range_decompression_torch(magnitudes)
    return output


class MelSpectrogram(torch.nn.Module):
    """メルスペクトログラム変換モジュール."""
    
    def __init__(self,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 sampling_rate: int = 24000,
                 n_mel_channels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: Optional[float] = None):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax or sampling_rate // 2
        
        # メルフィルターバンクを作成
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=self.mel_fmax
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        # ハミング窓を作成
        self.register_buffer('hann_window', torch.hann_window(win_length))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """音声からメルスペクトログラムを計算する."""
        # STFTを計算
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window.to(audio.device),
            return_complex=True
        )
        
        # 振幅スペクトログラムに変換
        spec = torch.abs(spec)
        
        # メルスペクトログラムを計算
        melspec = torch.matmul(self.mel_basis, spec)
        
        # ログスケールに変換
        melspec = spectral_normalize_torch(melspec)
        
        return melspec


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """無音部分をトリミングする."""
    # エネルギーベースでトリミング
    energy = np.abs(audio)
    mask = energy > threshold
    
    if not np.any(mask):
        return audio
    
    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1]) - 1
    
    return audio[start:end+1]


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """音声の振幅を正規化する."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * (target_peak / max_val)
    return audio
