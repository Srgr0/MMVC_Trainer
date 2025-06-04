"""
MMVC_Trainer DataLoader Optimization Module
データローディングとI/O処理を最適化する
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import threading
from queue import Queue
from typing import List, Tuple, Optional, Dict, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class OptimizedDataLoader:
    """最適化されたデータローダー"""
    
    def __init__(self, dataset, batch_size: int, num_workers: int = None, 
                 prefetch_factor: int = 2, pin_memory: bool = True,
                 persistent_workers: bool = True):
        
        # CPUコア数に基づいてワーカー数を自動調整
        if num_workers is None:
            num_workers = min(8, mp.cpu_count())
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.persistent_workers = persistent_workers
        
        # メモリマップドファイルキャッシュ
        self._cache = {}
        self._cache_lock = threading.Lock()
        
    def create_dataloader(self, dataset, collate_fn, batch_sampler=None, shuffle=False) -> DataLoader:
        """最適化されたDataLoaderを作成"""
        
        # PyTorch 1.7.0以降でのみpersistent_workersを使用
        kwargs = {
            'dataset': dataset,
            'batch_size': self.batch_size if batch_sampler is None else 1,
            'shuffle': shuffle if batch_sampler is None else False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'collate_fn': collate_fn,
            'prefetch_factor': self.prefetch_factor,
            'drop_last': True,  # バッチサイズを一定に保つ
        }
        
        if batch_sampler is not None:
            kwargs['batch_sampler'] = batch_sampler
            kwargs.pop('batch_size')
            kwargs.pop('shuffle')
        
        # persistent_workersの対応チェック
        try:
            kwargs['persistent_workers'] = self.persistent_workers and self.num_workers > 0
            dataloader = DataLoader(**kwargs)
        except TypeError:
            # persistent_workersがサポートされていない場合
            kwargs.pop('persistent_workers', None)
            dataloader = DataLoader(**kwargs)
        
        return dataloader

class SpecCacheManager:
    """スペクトログラムキャッシュ管理"""
    
    def __init__(self, cache_dir: str = "./cache", max_cache_size_gb: float = 2.0):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_gb * 1024**3  # GB to bytes
        self.current_cache_size = 0
        self.cache_lock = threading.Lock()
        
        os.makedirs(cache_dir, exist_ok=True)
        self._initialize_cache()
    
    def _initialize_cache(self):
        """キャッシュディレクトリの初期化"""
        if os.path.exists(self.cache_dir):
            # 既存のキャッシュサイズを計算
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    self.current_cache_size += os.path.getsize(filepath)
    
    def get_cache_path(self, audio_path: str) -> str:
        """音声ファイルパスからキャッシュパスを生成"""
        # ファイル名のハッシュを使用してキャッシュパスを生成
        import hashlib
        hash_obj = hashlib.md5(audio_path.encode())
        cache_filename = f"{hash_obj.hexdigest()}.spec.pt"
        return os.path.join(self.cache_dir, cache_filename)
    
    def load_cached_spec(self, audio_path: str) -> Optional[torch.Tensor]:
        """キャッシュされたスペクトログラムを読み込み"""
        cache_path = self.get_cache_path(audio_path)
        
        if os.path.exists(cache_path):
            try:
                # ファイルの更新時間をチェック
                audio_mtime = os.path.getmtime(audio_path)
                cache_mtime = os.path.getmtime(cache_path)
                
                if cache_mtime > audio_mtime:
                    return torch.load(cache_path, map_location='cpu')
            except Exception as e:
                logger.warning(f"Failed to load cached spec for {audio_path}: {e}")
        
        return None
    
    def save_spec_cache(self, audio_path: str, spec: torch.Tensor):
        """スペクトログラムをキャッシュに保存"""
        cache_path = self.get_cache_path(audio_path)
        
        with self.cache_lock:
            try:
                torch.save(spec, cache_path)
                cache_size = os.path.getsize(cache_path)
                self.current_cache_size += cache_size
                
                # キャッシュサイズ制限のチェック
                if self.current_cache_size > self.max_cache_size:
                    self._cleanup_cache()
                    
            except Exception as e:
                logger.warning(f"Failed to save spec cache for {audio_path}: {e}")
    
    def _cleanup_cache(self):
        """古いキャッシュファイルを削除"""
        cache_files = []
        
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath) and filename.endswith('.spec.pt'):
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                cache_files.append((filepath, mtime, size))
        
        # 古いファイルから削除
        cache_files.sort(key=lambda x: x[1])  # 更新時間でソート
        
        while self.current_cache_size > self.max_cache_size * 0.8:  # 80%まで削減
            if not cache_files:
                break
            
            filepath, _, size = cache_files.pop(0)
            try:
                os.remove(filepath)
                self.current_cache_size -= size
                logger.info(f"Removed cache file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {filepath}: {e}")

class BatchProcessor:
    """バッチ処理の最適化"""
    
    @staticmethod
    def optimize_batch_size(model: torch.nn.Module, initial_batch_size: int = 8, 
                           target_memory_usage: float = 0.8) -> int:
        """最適なバッチサイズを自動検出"""
        
        if not torch.cuda.is_available():
            return initial_batch_size
        
        device = next(model.parameters()).device
        total_memory = torch.cuda.get_device_properties(device).total_memory
        target_memory = total_memory * target_memory_usage
        
        # テスト用のダミーデータを作成
        dummy_input = {
            'x': torch.randint(0, 100, (initial_batch_size, 100)).to(device),
            'x_lengths': torch.randint(50, 100, (initial_batch_size,)).to(device),
            'spec': torch.randn(initial_batch_size, 513, 100).to(device),
            'spec_lengths': torch.randint(50, 100, (initial_batch_size,)).to(device),
            'speakers': torch.randint(0, 10, (initial_batch_size,)).to(device),
        }
        
        optimal_batch_size = initial_batch_size
        
        try:
            for batch_size in [initial_batch_size * 2, initial_batch_size * 4, initial_batch_size * 8]:
                torch.cuda.empty_cache()
                
                # バッチサイズを調整
                test_input = {}
                for key, value in dummy_input.items():
                    if value.dim() > 0:
                        # バッチ次元を調整
                        current_batch = value.size(0)
                        if batch_size > current_batch:
                            repeat_factor = batch_size // current_batch
                            test_input[key] = value.repeat(repeat_factor, *([1] * (value.dim() - 1)))[:batch_size]
                        else:
                            test_input[key] = value[:batch_size]
                    else:
                        test_input[key] = value
                
                model.eval()
                with torch.no_grad():
                    # メモリ使用量をテスト
                    _ = model(**test_input)
                    current_memory = torch.cuda.memory_allocated(device)
                    
                    if current_memory < target_memory:
                        optimal_batch_size = batch_size
                    else:
                        break
                        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"GPU memory limit reached at batch size {batch_size}")
            else:
                logger.warning(f"Error during batch size optimization: {e}")
        
        finally:
            torch.cuda.empty_cache()
        
        logger.info(f"Optimal batch size determined: {optimal_batch_size}")
        return optimal_batch_size
    
    @staticmethod
    def dynamic_batch_sizing(current_batch_size: int, memory_usage: float, 
                           target_usage: float = 0.8) -> int:
        """動的バッチサイズ調整"""
        
        if memory_usage < target_usage * 0.7:
            # メモリ使用量が少ない場合、バッチサイズを増加
            return min(current_batch_size + 2, current_batch_size * 2)
        elif memory_usage > target_usage * 1.1:
            # メモリ使用量が多い場合、バッチサイズを減少
            return max(current_batch_size - 2, current_batch_size // 2)
        else:
            return current_batch_size

# グローバルキャッシュマネージャーのインスタンス
_global_cache_manager = None

def get_cache_manager() -> SpecCacheManager:
    """グローバルキャッシュマネージャーを取得"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = SpecCacheManager()
    return _global_cache_manager
