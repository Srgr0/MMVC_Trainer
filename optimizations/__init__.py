"""
MMVC_Trainer Optimization Suite
Main optimization module that integrates all optimization components
"""
from .memory_optimizer import MemoryOptimizer, optimize_torch_settings
from .dataloader_optimizer import OptimizedDataLoader, SpecCacheManager, BatchProcessor
from .training_optimizer import TrainingOptimizer, LossComputation, ModelOptimization
from .performance_monitor import PerformanceMonitor, SystemResourceMonitor

__all__ = [
    'MemoryOptimizer',
    'optimize_torch_settings', 
    'OptimizedDataLoader',
    'SpecCacheManager',
    'BatchProcessor',
    'TrainingOptimizer',
    'LossComputation',
    'ModelOptimization',
    'PerformanceMonitor',
    'SystemResourceMonitor'
]

def initialize_optimizations(config=None):
    """最適化の初期化"""
    # PyTorchの基本設定を最適化
    optimize_torch_settings()
    
    # メモリ効率的なattentionを有効化
    try:
        from .memory_optimizer import MemoryOptimizer
        MemoryOptimizer.enable_memory_efficient_attention()
    except Exception:
        pass
    
    print("MMVC_Trainer optimizations initialized successfully!")

def get_optimization_summary():
    """最適化機能の要約を取得"""
    return {
        'memory_optimization': '✓ メモリ使用量の最適化',
        'dataloader_optimization': '✓ データローダーの高速化',
        'training_optimization': '✓ 学習プロセスの最適化',
        'performance_monitoring': '✓ パフォーマンス監視',
        'cache_management': '✓ スペクトログラムキャッシュ',
        'mixed_precision': '✓ 混合精度学習の最適化',
        'gradient_optimization': '✓ 勾配最適化',
    }
