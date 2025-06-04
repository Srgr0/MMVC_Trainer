import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav_to_torch(full_path):
  """
  Loads wavdata into torch array
  """
  sampling_rate, data = read(full_path)
  return torch.from_numpy(data).float(), sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
  """
  PARAMS
  ------
  C: compression factor
  """
  return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
  """
  PARAMS
  ------
  C: compression factor used to compress
  """
  return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
  output = dynamic_range_compression(magnitudes)
  return output


def spectral_de_normalize_torch(magnitudes):
  output = dynamic_range_decompression(magnitudes)
  return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  global hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  wnsize_dtype_device = str(win_size) + '_' + dtype_device
  if wnsize_dtype_device not in hann_window:
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
  y = y.squeeze(1)

  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
  
  spec = torch.sqrt(spec.real.pow(2.0) + spec.imag.pow(2.0) + 1e-6)
  return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
  global mel_basis
  dtype_device = str(spec.dtype) + '_' + str(spec.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device
  if fmax_dtype_device not in mel_basis:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  spec = spectral_normalize_torch(spec)
  return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  global mel_basis, hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device
  wnsize_dtype_device = str(win_size) + '_' + dtype_device
  if fmax_dtype_device not in mel_basis:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
  if wnsize_dtype_device not in hann_window:
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
  y = y.squeeze(1)

  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

  spec = torch.sqrt(spec.real.pow(2.0) + spec.imag.pow(2.0) + 1e-6)

  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  spec = spectral_normalize_torch(spec)

  return spec


def spectrogram_torch_data(y, hps_data):
  return spectrogram_torch(y, hps_data.filter_length, hps_data.sampling_rate, hps_data.hop_length, hps_data.win_length, center=False)


def mel_spectrogram_torch_data(y, hps_data):
  return mel_spectrogram_torch(y, hps_data.filter_length, hps_data.n_mel_channels, hps_data.sampling_rate, hps_data.hop_length, hps_data.win_length, hps_data.mel_fmin, hps_data.mel_fmax, center=False)


def spec_to_mel_torch_data(spec, hps_data):
  return spec_to_mel_torch(spec, hps_data.filter_length, hps_data.n_mel_channels, hps_data.sampling_rate, hps_data.mel_fmin, hps_data.mel_fmax)


class TacotronSTFT(torch.nn.Module):
  def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
               n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
               mel_fmax=8000.0):
    super(TacotronSTFT, self).__init__()
    self.n_mel_channels = n_mel_channels
    self.sampling_rate = sampling_rate
    self.stft_fn = STFT(filter_length, hop_length, win_length)
    mel_basis = librosa_mel_fn(
        sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
    mel_basis = torch.from_numpy(mel_basis).float()
    self.register_buffer('mel_basis', mel_basis)

  def spectral_normalize(self, magnitudes):
    output = dynamic_range_compression(magnitudes)
    return output

  def spectral_de_normalize(self, magnitudes):
    output = dynamic_range_decompression(magnitudes)
    return output

  def mel_spectrogram(self, y):
    """Computes mel-spectrograms from a batch of waves
    PARAMS
    ------
    y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

    RETURNS
    -------
    mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
    """
    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)

    magnitudes, phases = self.stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(self.mel_basis, magnitudes)
    mel_output = self.spectral_normalize(mel_output)
    return mel_output


class STFT(torch.nn.Module):
  """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
  def __init__(self, filter_length=800, hop_length=200, win_length=800,
               window='hann'):
    super(STFT, self).__init__()
    self.filter_length = filter_length
    self.hop_length = hop_length
    self.win_length = win_length
    self.window = window
    self.forward_transform = None
    scale = self.filter_length / self.hop_length
    fourier_basis = np.fft.fft(np.eye(self.filter_length))

    cutoff = int((self.filter_length / 2 + 1))
    fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                               np.imag(fourier_basis[:cutoff, :])])

    forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
    inverse_basis = torch.FloatTensor(
        np.linalg.pinv(scale * fourier_basis).T[:, None, :])

    if window is not None:
      assert(filter_length >= win_length)
      # get window and zero center pad it to filter_length
      fft_window = get_window(window, win_length, fftbins=True)
      fft_window = pad_center(fft_window, size=filter_length)
      fft_window = torch.from_numpy(fft_window).float()

      # window the bases
      forward_basis *= fft_window
      inverse_basis *= fft_window

    self.register_buffer('forward_basis', forward_basis.float())
    self.register_buffer('inverse_basis', inverse_basis.float())

  def transform(self, input_data):
    num_batches = input_data.size(0)
    num_samples = input_data.size(1)

    self.num_samples = num_samples

    # similar to librosa, reflect-pad the input
    input_data = input_data.view(num_batches, 1, num_samples)
    input_data = F.pad(
        input_data.unsqueeze(1),
        (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
        mode='reflect')
    input_data = input_data.squeeze(1)

    forward_transform = F.conv1d(
        input_data,
        self.forward_basis,
        stride=self.hop_length,
        padding=0)

    cutoff = int((self.filter_length / 2) + 1)
    real_part = forward_transform[:, :cutoff, :]
    imag_part = forward_transform[:, cutoff:, :]

    magnitude = torch.sqrt(real_part**2 + imag_part**2)
    phase = torch.autograd.Variable(
        torch.atan2(imag_part.data, real_part.data))

    return magnitude, phase

  def inverse(self, magnitude, phase):
    recombine_magnitude_phase = torch.cat(
        [magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

    inverse_transform = F.conv_transpose1d(
        recombine_magnitude_phase,
        self.inverse_basis,
        stride=self.hop_length,
        padding=0)

    if self.window is not None:
      window_sum = window_sumsquare(
          self.window, magnitude.size(-1), hop_length=self.hop_length,
          win_length=self.win_length, n_fft=self.filter_length,
          dtype=np.float32)
      # remove modulation effects
      approx_nonzero_indices = torch.from_numpy(
          np.where(window_sum > tiny(window_sum))[0])
      window_sum = torch.autograd.Variable(
          torch.from_numpy(window_sum), requires_grad=False)
      window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
      inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

      # scale by hop ratio
      inverse_transform *= float(self.filter_length) / self.hop_length
    
    inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
    inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

    return inverse_transform

  def forward(self, input_data):
    self.magnitude, self.phase = self.transform(input_data)
    reconstruction = self.inverse(self.magnitude, self.phase)
    return reconstruction


def window_sumsquare(window, n_frames, hop_length=200, win_length=800, n_fft=800,
                     dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = normalize(win_sq, norm=norm)**2
    win_sq = pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x
