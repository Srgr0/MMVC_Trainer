"""
MMVC_Trainer Memory Optimization Module
メモリ使用量を最適化し、より大きなバッチサイズでの学習を可能にする
"""
import torch
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """メモリ使用量を最適化するためのユーティリティクラス"""
    
    def __init__(self):
        self.peak_memory = 0
        self.baseline_memory = 0
        
    @contextmanager
    def memory_efficient_training(self):
        """メモリ効率的な学習のためのコンテキストマネージャー"""
        try:
            # ベースラインメモリ使用量を記録
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.baseline_memory = torch.cuda.memory_allocated()
            
            yield
            
        finally:
            # 学習後のクリーンアップ
            if torch.cuda.is_available():
                self.peak_memory = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
                gc.collect()
    
    @staticmethod
    def optimize_dataloader_memory(dataset_size: int, batch_size: int, num_workers: int) -> Dict[str, int]:
        """データローダーのメモリ使用量を最適化"""
        # メモリ使用量に基づいて最適なnum_workersを計算
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # メモリ使用量に基づいてワーカー数を調整
        if available_memory_gb < 8:
            optimal_workers = min(2, num_workers)
        elif available_memory_gb < 16:
            optimal_workers = min(4, num_workers)
        else:
            optimal_workers = min(8, num_workers)
        
        # バッチサイズの最適化提案
        if available_memory_gb < 8:
            suggested_batch_size = min(batch_size, 4)
        elif available_memory_gb < 16:
            suggested_batch_size = min(batch_size, 8)
        else:
            suggested_batch_size = batch_size
        
        return {
            'optimal_workers': optimal_workers,
            'suggested_batch_size': suggested_batch_size,
            'available_memory_gb': int(available_memory_gb)
        }
    
    @staticmethod
    def enable_memory_efficient_attention():
        """PyTorchのメモリ効率的なAttentionを有効化"""
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    @staticmethod
    def optimize_mixed_precision(model: torch.nn.Module) -> torch.nn.Module:
        """混合精度最適化の設定"""
        # 特定の層でのFP32計算を強制（数値安定性のため）
        for name, module in model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['norm', 'embedding', 'attention']):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data = module.weight.data.float()
        
        return model
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """メモリ使用量の要約を取得"""
        if not torch.cuda.is_available():
            return {"message": "CUDA not available"}
        
        current_memory = torch.cuda.memory_allocated()
        max_memory = torch.cuda.max_memory_allocated()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        return {
            'current_memory_mb': current_memory / (1024**2),
            'peak_memory_mb': max_memory / (1024**2),
            'total_memory_gb': total_memory / (1024**3),
            'memory_utilization_percent': (current_memory / total_memory) * 100
        }

@contextmanager
def gradient_checkpointing(model: torch.nn.Module):
    """グラディエントチェックポイントを有効化"""
    original_states = {}
    
    # グラディエントチェックポイントを有効化
    for name, module in model.named_modules():
        if hasattr(module, 'use_checkpoint'):
            original_states[name] = getattr(module, 'use_checkpoint', False)
            module.use_checkpoint = True
    
    try:
        yield
    finally:
        # 元の状態に復元
        for name, module in model.named_modules():
            if name in original_states:
                module.use_checkpoint = original_states[name]

def optimize_torch_settings():
    """PyTorchの基本設定を最適化"""
    # CuDNNベンチマークを有効化（入力サイズが固定の場合）
    torch.backends.cudnn.benchmark = True
    
    # CuDNNの決定論的動作を無効化（高速化のため）
    torch.backends.cudnn.deterministic = False
    
    # テンソルのデフォルト型を設定
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # メモリ使用量の最適化
    if hasattr(torch.cuda, 'memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)
