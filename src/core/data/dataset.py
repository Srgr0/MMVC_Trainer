"""Data loading utilities for MMVC training."""

import os
import random
from typing import List, Tuple, Optional

import torch
import torch.utils.data
import numpy as np
import librosa

from ...text.symbols import text_to_sequence
from ...text.cleaners import clean_text
from ...utils.audio import load_wav, MelSpectrogram


class TextAudioSpeakerDataset(torch.utils.data.Dataset):
    """テキスト、音声、話者IDのデータセット."""
    
    def __init__(self, 
                 audiopaths_and_text: str,
                 hparams,
                 augmentation: bool = True):
        """
        Args:
            audiopaths_and_text: ファイルリストのパス
            hparams: ハイパーパラメータ
            augmentation: データ拡張を行うかどうか
        """
        self.audiopaths_and_text = self._load_filepaths_and_text(audiopaths_and_text)
        self.hparams = hparams
        self.augmentation = augmentation
        self.sampling_rate = hparams.data.sampling_rate
        self.segment_size = hparams.train.segment_size
        self.text_cleaners = hparams.data.text_cleaners
        self.add_blank = hparams.data.add_blank
        self.max_wav_value = hparams.data.max_wav_value
        
        # メルスペクトログラム変換器を初期化
        self.mel_transform = MelSpectrogram(
            n_fft=hparams.data.filter_length,
            hop_length=hparams.data.hop_length,
            win_length=hparams.data.win_length,
            sampling_rate=hparams.data.sampling_rate,
            n_mel_channels=hparams.data.n_mel_channels,
            mel_fmin=hparams.data.mel_fmin,
            mel_fmax=hparams.data.mel_fmax
        )
        
        # 話者情報を構築
        self._build_speaker_mapping()
        
        # キャッシュ用辞書
        self.spec_cache = {}
        
    def _load_filepaths_and_text(self, filename: str) -> List[List[str]]:
        """ファイルパスとテキストのリストを読み込む."""
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split('|') for line in f if line.strip()]
        return filepaths_and_text
    
    def _build_speaker_mapping(self):
        """話者IDのマッピングを構築する."""
        speakers = set()
        for item in self.audiopaths_and_text:
            if len(item) >= 3:  # audio_path|text|speaker
                speakers.add(item[2])
        
        self.speakers = sorted(list(speakers))
        self.speaker_to_id = {speaker: i for i, speaker in enumerate(self.speakers)}
        self.id_to_speaker = {i: speaker for i, speaker in enumerate(self.speakers)}
        
    def get_audio_text_speaker_pair(self, audiopath_and_text: List[str]) -> Tuple:
        """音声、テキスト、話者IDのペアを取得する."""
        # ファイルパスとテキストを分離
        audiopath = audiopath_and_text[0]
        text = audiopath_and_text[1]
        speaker = audiopath_and_text[2] if len(audiopath_and_text) > 2 else "default"
        
        # テキストをクリーニング
        text = clean_text(text, self.text_cleaners)
        
        # テキストを音素IDシーケンスに変換
        text_sequence = text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence)
        
        # 音声を読み込み
        spec, wav = self.get_audio(audiopath)
        
        # 話者IDを取得
        speaker_id = self.speaker_to_id.get(speaker, 0)
        
        return (text_tensor, spec, wav, speaker_id)
    
    def get_audio(self, filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """音声ファイルを読み込み、スペクトログラムを計算する."""
        # キャッシュファイルのパスを生成
        cache_path = filename.replace('.wav', '.spec.pt')
        
        # キャッシュがあるか確認
        if os.path.exists(cache_path) and filename not in self.spec_cache:
            try:
                spec = torch.load(cache_path, map_location='cpu')
                self.spec_cache[filename] = spec
            except:
                # キャッシュファイルが破損している場合は削除
                os.remove(cache_path)
        
        # キャッシュから取得を試行
        if filename in self.spec_cache:
            spec = self.spec_cache[filename]
        else:
            # 音声ファイルを読み込み
            audio = load_wav(filename, sr=self.sampling_rate)
            
            # 正規化
            audio = audio / self.max_wav_value
            
            # メルスペクトログラムを計算
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            spec = self.mel_transform(audio_tensor).squeeze(0)
            
            # キャッシュに保存
            self.spec_cache[filename] = spec
            torch.save(spec, cache_path)
        
        # 音声を再度読み込み（学習時のランダムセグメント用）
        audio = load_wav(filename, sr=self.sampling_rate)
        audio_norm = audio / self.max_wav_value
        audio_tensor = torch.FloatTensor(audio_norm)
        
        return spec, audio_tensor
    
    def __getitem__(self, index: int) -> Tuple:
        """データセットの要素を取得する."""
        return self.get_audio_text_speaker_pair(self.audiopaths_and_text[index])
    
    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.audiopaths_and_text)


class TextAudioSpeakerCollate:
    """バッチデータのコレート関数."""
    
    def __init__(self, return_ids: bool = False):
        self.return_ids = return_ids
        
    def __call__(self, batch) -> Tuple:
        """バッチデータをコレートする."""
        # バッチサイズを取得
        batch_size = len(batch)
        
        # 各要素を分離
        texts = [item[0] for item in batch]
        specs = [item[1] for item in batch]
        wavs = [item[2] for item in batch]
        speaker_ids = [item[3] for item in batch]
        
        # テキストの長さを取得
        text_lengths = torch.LongTensor([len(text) for text in texts])
        
        # スペクトログラムの長さを取得
        spec_lengths = torch.LongTensor([spec.size(1) for spec in specs])
        
        # 音声の長さを取得
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        
        # パディング
        max_text_len = max(text_lengths)
        max_spec_len = max(spec_lengths)
        max_wav_len = max(wav_lengths)
        
        # テキストをパディング
        text_padded = torch.LongTensor(batch_size, max_text_len)
        text_padded.zero_()
        for i, text in enumerate(texts):
            text_padded[i, :text.size(0)] = text
            
        # スペクトログラムをパディング
        spec_padded = torch.FloatTensor(batch_size, specs[0].size(0), max_spec_len)
        spec_padded.zero_()
        for i, spec in enumerate(specs):
            spec_padded[i, :, :spec.size(1)] = spec
            
        # 音声をパディング
        wav_padded = torch.FloatTensor(batch_size, max_wav_len)
        wav_padded.zero_()
        for i, wav in enumerate(wavs):
            wav_padded[i, :wav.size(0)] = wav
            
        # 話者IDをテンソルに変換
        speaker_ids = torch.LongTensor(speaker_ids)
        
        if self.return_ids:
            ids = [f"batch_{i}" for i in range(batch_size)]
            return (text_padded, text_lengths, spec_padded, spec_lengths, 
                   wav_padded, wav_lengths, speaker_ids, ids)
        else:
            return (text_padded, text_lengths, spec_padded, spec_lengths,
                   wav_padded, wav_lengths, speaker_ids)


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    分散学習対応のバケットサンプラー.
    長さが似ているサンプルを同じバッチに入れることで効率化を図る.
    """
    
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None,
                 rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths if hasattr(dataset, 'lengths') else [1] * len(dataset)
        self.batch_size = batch_size
        self.boundaries = boundaries
        
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        
    def _create_buckets(self):
        """長さに基づいてバケットを作成する."""
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        
        for i, length in enumerate(self.lengths):
            # 適切なバケットを見つける
            bucket_id = 0
            for j in range(len(self.boundaries) - 1):
                if self.boundaries[j] <= length < self.boundaries[j + 1]:
                    bucket_id = j
                    break
            buckets[bucket_id].append(i)
        
        # 各バケットのサンプル数を計算
        num_samples_per_bucket = []
        for bucket in buckets:
            num_samples = len(bucket)
            num_samples_per_bucket.append(num_samples)
        
        return buckets, num_samples_per_bucket
    
    def __iter__(self):
        """イテレータを返す."""
        # ランダムシードを設定
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            
            # 各バケット内でシャッフル
            for bucket in self.buckets:
                indices = torch.randperm(len(bucket), generator=g).tolist()
                bucket[:] = [bucket[i] for i in indices]
        
        # 分散学習用のサンプリング
        indices = []
        for bucket in self.buckets:
            # このランクに割り当てられたサンプルを取得
            num_samples = len(bucket)
            num_samples_per_rank = num_samples // self.num_replicas
            start_idx = self.rank * num_samples_per_rank
            end_idx = start_idx + num_samples_per_rank
            
            indices.extend(bucket[start_idx:end_idx])
        
        # 必要に応じて調整
        if len(indices) < self.num_samples:
            indices += indices[:self.num_samples - len(indices)]
        elif len(indices) > self.num_samples:
            indices = indices[:self.num_samples]
        
        return iter(indices)
    
    def __len__(self):
        """サンプル数を返す."""
        return self.num_samples
