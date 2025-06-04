"""
MMVC_Trainer Performance Monitor
学習プロセスのパフォーマンスを監視し、リアルタイムで最適化提案を行う
"""
import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json
import os
from datetime import datetime, timedelta
import threading
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self, log_interval: int = 100, history_size: int = 1000):
        self.log_interval = log_interval
        self.history_size = history_size
        
        # パフォーマンス履歴
        self.step_times = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.gpu_utilization = deque(maxlen=history_size)
        self.loss_history = deque(maxlen=history_size)
        self.learning_rates = deque(maxlen=history_size)
        
        # 統計情報
        self.start_time = time.time()
        self.total_steps = 0
        self.best_loss = float('inf')
        self.best_step = 0
        
        # GPU情報の取得
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_names = [torch.cuda.get_device_name(i) for i in range(self.gpu_count)]
        
        # システム情報
        self.cpu_count = psutil.cpu_count()
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
    
    def log_step(self, step_time: float, memory_usage: float, loss: float, 
                learning_rate: float, gpu_util: Optional[float] = None):
        """ステップごとの情報をログ記録"""
        self.step_times.append(step_time)
        self.memory_usage.append(memory_usage)
        self.loss_history.append(loss)
        self.learning_rates.append(learning_rate)
        
        if gpu_util is not None:
            self.gpu_utilization.append(gpu_util)
        
        self.total_steps += 1
        
        # ベスト損失の更新
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = self.total_steps
    
    def get_current_stats(self) -> Dict[str, Any]:
        """現在の統計情報を取得"""
        if not self.step_times:
            return {}
        
        recent_size = min(100, len(self.step_times))
        recent_times = list(self.step_times)[-recent_size:]
        recent_losses = list(self.loss_history)[-recent_size:]
        recent_memory = list(self.memory_usage)[-recent_size:]
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        stats = {
            'total_steps': self.total_steps,
            'elapsed_time_hours': elapsed_time / 3600,
            'avg_step_time': np.mean(recent_times),
            'steps_per_second': 1.0 / np.mean(recent_times) if recent_times else 0,
            'current_loss': recent_losses[-1] if recent_losses else 0,
            'avg_recent_loss': np.mean(recent_losses),
            'best_loss': self.best_loss,
            'best_step': self.best_step,
            'loss_improvement': (recent_losses[0] - recent_losses[-1]) / recent_losses[0] if len(recent_losses) > 10 else 0,
            'avg_memory_usage_gb': np.mean(recent_memory),
            'estimated_time_to_completion': self._estimate_completion_time(),
        }
        
        if self.gpu_utilization:
            recent_gpu = list(self.gpu_utilization)[-recent_size:]
            stats['avg_gpu_utilization'] = np.mean(recent_gpu)
        
        return stats
    
    def _estimate_completion_time(self, target_steps: int = 100000) -> Optional[float]:
        """完了予定時間の推定"""
        if len(self.step_times) < 10:
            return None
        
        avg_step_time = np.mean(list(self.step_times)[-100:])
        remaining_steps = target_steps - self.total_steps
        
        if remaining_steps <= 0:
            return 0.0
        
        return remaining_steps * avg_step_time / 3600  # 時間
    
    def generate_optimization_suggestions(self) -> List[str]:
        """最適化提案の生成"""
        suggestions = []
        
        if not self.step_times:
            return suggestions
        
        recent_times = list(self.step_times)[-100:]
        recent_memory = list(self.memory_usage)[-100:]
        
        avg_step_time = np.mean(recent_times)
        avg_memory = np.mean(recent_memory)
        
        # ステップ時間の分析
        if avg_step_time > 2.0:  # 2秒以上
            suggestions.append("ステップ時間が長いです。バッチサイズを小さくするか、モデルサイズを減らすことを検討してください。")
        
        # メモリ使用量の分析
        if self.has_gpu:
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if avg_memory > total_gpu_memory * 0.9:
                suggestions.append("GPU メモリ使用量が高いです。バッチサイズを減らすか、mixed precision training を有効にしてください。")
            elif avg_memory < total_gpu_memory * 0.5:
                suggestions.append("GPU メモリに余裕があります。バッチサイズを増やして学習効率を向上できます。")
        
        # GPU使用率の分析
        if self.gpu_utilization:
            recent_gpu = list(self.gpu_utilization)[-50:]
            avg_gpu_util = np.mean(recent_gpu)
            
            if avg_gpu_util < 70:
                suggestions.append("GPU 使用率が低いです。データローダーのワーカー数を増やすか、プリフェッチを有効にしてください。")
        
        # 損失の収束分析
        if len(self.loss_history) > 500:
            recent_losses = list(self.loss_history)[-500:]
            loss_variance = np.var(recent_losses[-100:])
            
            if loss_variance < 1e-6:
                suggestions.append("損失が収束している可能性があります。学習率を下げるか、学習を停止することを検討してください。")
        
        return suggestions
    
    def save_performance_log(self, filepath: str):
        """パフォーマンスログをファイルに保存"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': self.cpu_count,
                'total_memory_gb': self.total_memory,
                'gpu_count': self.gpu_count if self.has_gpu else 0,
                'gpu_names': self.gpu_names if self.has_gpu else [],
            },
            'performance_stats': self.get_current_stats(),
            'optimization_suggestions': self.generate_optimization_suggestions(),
            'history': {
                'step_times': list(self.step_times),
                'memory_usage': list(self.memory_usage),
                'loss_history': list(self.loss_history),
                'learning_rates': list(self.learning_rates),
                'gpu_utilization': list(self.gpu_utilization) if self.gpu_utilization else [],
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """パフォーマンス要約を出力"""
        stats = self.get_current_stats()
        suggestions = self.generate_optimization_suggestions()
        
        print("\n" + "="*60)
        print("MMVC_Trainer パフォーマンス要約")
        print("="*60)
        
        if stats:
            print(f"総ステップ数: {stats['total_steps']}")
            print(f"実行時間: {stats['elapsed_time_hours']:.2f} 時間")
            print(f"平均ステップ時間: {stats['avg_step_time']:.3f} 秒")
            print(f"ステップ/秒: {stats['steps_per_second']:.2f}")
            print(f"現在の損失: {stats['current_loss']:.6f}")
            print(f"ベスト損失: {stats['best_loss']:.6f} (ステップ {stats['best_step']})")
            print(f"平均メモリ使用量: {stats['avg_memory_usage_gb']:.2f} GB")
            
            if 'avg_gpu_utilization' in stats:
                print(f"平均GPU使用率: {stats['avg_gpu_utilization']:.1f}%")
            
            if stats['estimated_time_to_completion']:
                print(f"完了予定時間: {stats['estimated_time_to_completion']:.2f} 時間")
        
        if suggestions:
            print("\n最適化提案:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"{i}. {suggestion}")
        
        print("="*60)

class SystemResourceMonitor:
    """システムリソース監視クラス"""
    
    def __init__(self, monitor_interval: float = 5.0):
        self.monitor_interval = monitor_interval
        self.is_monitoring = False
        self.monitor_thread = None
        
        # リソース履歴
        self.cpu_usage = deque(maxlen=720)  # 1時間分（5秒間隔）
        self.memory_usage = deque(maxlen=720)
        self.disk_usage = deque(maxlen=720)
        
        if torch.cuda.is_available():
            self.gpu_memory = deque(maxlen=720)
            self.gpu_utilization = deque(maxlen=720)
    
    def start_monitoring(self):
        """リソース監視を開始"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("システムリソース監視を開始しました")
    
    def stop_monitoring(self):
        """リソース監視を停止"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("システムリソース監視を停止しました")
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.is_monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage.append(cpu_percent)
                
                # メモリ使用率
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # ディスク使用率
                disk = psutil.disk_usage('/')
                self.disk_usage.append(disk.percent)
                
                # GPU情報
                if torch.cuda.is_available():
                    gpu_memory_percent = (torch.cuda.memory_allocated() / 
                                        torch.cuda.get_device_properties(0).total_memory) * 100
                    self.gpu_memory.append(gpu_memory_percent)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.warning(f"リソース監視エラー: {e}")
                time.sleep(self.monitor_interval)
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """リソース使用量の要約を取得"""
        summary = {}
        
        if self.cpu_usage:
            summary['cpu'] = {
                'current': self.cpu_usage[-1],
                'average': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage)
            }
        
        if self.memory_usage:
            summary['memory'] = {
                'current': self.memory_usage[-1],
                'average': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage)
            }
        
        if self.disk_usage:
            summary['disk'] = {
                'current': self.disk_usage[-1],
                'average': np.mean(self.disk_usage),
                'max': np.max(self.disk_usage)
            }
        
        if hasattr(self, 'gpu_memory') and self.gpu_memory:
            summary['gpu_memory'] = {
                'current': self.gpu_memory[-1],
                'average': np.mean(self.gpu_memory),
                'max': np.max(self.gpu_memory)
            }
        
        return summary

class ComprehensiveLossTracker:
    """包括的な損失追跡とパフォーマンス分析"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_components = {
            'discriminator': deque(maxlen=window_size),
            'generator': deque(maxlen=window_size),
            'feature_matching': deque(maxlen=window_size),
            'kl_divergence': deque(maxlen=window_size),
            'mel_spectrogram': deque(maxlen=window_size),
            'total_loss': deque(maxlen=window_size)
        }
        self.convergence_metrics = {}
        self.stability_metrics = {}
        
    def update_losses(self, loss_dict: Dict[str, float]):
        """損失コンポーネントを更新"""
        for component, value in loss_dict.items():
            if component in self.loss_components:
                self.loss_components[component].append(value)
    
    def compute_convergence_metrics(self) -> Dict[str, float]:
        """収束メトリクスを計算"""
        metrics = {}
        
        for component, history in self.loss_components.items():
            if len(history) >= 10:
                recent_losses = list(history)[-10:]
                older_losses = list(history)[-20:-10] if len(history) >= 20 else list(history)[:-10]
                
                if older_losses:
                    # 改善率の計算
                    recent_avg = np.mean(recent_losses)
                    older_avg = np.mean(older_losses)
                    improvement_rate = (older_avg - recent_avg) / (older_avg + 1e-8)
                    
                    # 安定性の計算（標準偏差の相対値）
                    stability = np.std(recent_losses) / (np.mean(recent_losses) + 1e-8)
                    
                    metrics[f'{component}_improvement_rate'] = improvement_rate
                    metrics[f'{component}_stability'] = stability
        
        return metrics
    
    def suggest_optimizations(self) -> List[str]:
        """最適化提案を生成"""
        suggestions = []
        metrics = self.compute_convergence_metrics()
        
        # 収束が遅い場合の提案
        for component in ['discriminator', 'generator', 'total_loss']:
            improvement_key = f'{component}_improvement_rate'
            stability_key = f'{component}_stability'
            
            if improvement_key in metrics:
                improvement = metrics[improvement_key]
                stability = metrics.get(stability_key, 0)
                
                if improvement < 0.001 and stability > 0.1:
                    suggestions.append(f"{component}損失の改善が停滞しています。学習率の調整を検討してください。")
                
                if stability > 0.2:
                    suggestions.append(f"{component}損失が不安定です。バッチサイズの増加やグラディエントクリッピングを検討してください。")
        
        return suggestions

class OptimizedPerformanceMonitor(PerformanceMonitor):
    """最適化されたパフォーマンス監視クラス"""
    
    def __init__(self, log_interval: int = 100, history_size: int = 1000):
        super().__init__(log_interval, history_size)
        self.loss_tracker = ComprehensiveLossTracker()
        self.optimization_suggestions = []
        self.performance_alerts = []
        
        # 詳細メトリクス
        self.data_loading_times = deque(maxlen=history_size)
        self.forward_pass_times = deque(maxlen=history_size)
        self.backward_pass_times = deque(maxlen=history_size)
        self.gradient_norms = deque(maxlen=history_size)
        
        # パフォーマンス閾値
        self.thresholds = {
            'memory_warning': 0.9,  # 90% memory usage
            'gpu_util_low': 0.5,    # 50% GPU utilization
            'gradient_explosion': 10.0,
            'training_stagnation': 100  # steps without improvement
        }
        
    def log_detailed_step(self, step_data: Dict[str, Any]):
        """詳細なステップ情報をログ記録"""
        # 基本情報のログ
        super().log_step(
            step_data.get('step_time', 0),
            step_data.get('memory_usage', 0),
            step_data.get('total_loss', 0),
            step_data.get('learning_rate', 0),
            step_data.get('gpu_utilization', None)
        )
        
        # 詳細メトリクスの更新
        if 'data_loading_time' in step_data:
            self.data_loading_times.append(step_data['data_loading_time'])
        if 'forward_pass_time' in step_data:
            self.forward_pass_times.append(step_data['forward_pass_time'])
        if 'backward_pass_time' in step_data:
            self.backward_pass_times.append(step_data['backward_pass_time'])
        if 'gradient_norm' in step_data:
            self.gradient_norms.append(step_data['gradient_norm'])
        
        # 損失トラッキング
        if 'loss_components' in step_data:
            self.loss_tracker.update_losses(step_data['loss_components'])
        
        # パフォーマンス警告のチェック
        self._check_performance_alerts(step_data)
        
    def _check_performance_alerts(self, step_data: Dict[str, Any]):
        """パフォーマンス警告をチェック"""
        alerts = []
        
        # メモリ使用量の警告
        memory_usage = step_data.get('memory_usage', 0)
        if memory_usage > self.thresholds['memory_warning']:
            alerts.append(f"高メモリ使用量警告: {memory_usage:.1%}")
        
        # GPU使用率の警告
        gpu_util = step_data.get('gpu_utilization', 1.0)
        if gpu_util and gpu_util < self.thresholds['gpu_util_low']:
            alerts.append(f"低GPU使用率警告: {gpu_util:.1%}")
        
        # グラディエント爆発の警告
        gradient_norm = step_data.get('gradient_norm', 0)
        if gradient_norm > self.thresholds['gradient_explosion']:
            alerts.append(f"グラディエント爆発警告: norm={gradient_norm:.2f}")
        
        # 学習停滞の警告
        if len(self.loss_history) >= self.thresholds['training_stagnation']:
            recent_losses = list(self.loss_history)[-self.thresholds['training_stagnation']:]
            if max(recent_losses) - min(recent_losses) < 0.001:
                alerts.append("学習停滞警告: 損失の改善が見られません")
        
        self.performance_alerts.extend(alerts)
        
    def generate_optimization_report(self) -> Dict[str, Any]:
        """最適化レポートを生成"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_duration': time.time() - self.start_time,
            'total_steps': self.total_steps,
            'current_performance': self._get_current_performance_stats(),
            'convergence_analysis': self.loss_tracker.compute_convergence_metrics(),
            'optimization_suggestions': self.loss_tracker.suggest_optimizations(),
            'performance_alerts': self.performance_alerts[-10:],  # 最新の10個の警告
            'efficiency_metrics': self._compute_efficiency_metrics()
        }
        
        return report
    
    def _get_current_performance_stats(self) -> Dict[str, float]:
        """現在のパフォーマンス統計を取得"""
        stats = {}
        
        if self.step_times:
            stats['avg_step_time'] = np.mean(list(self.step_times)[-100:])
            stats['steps_per_second'] = 1.0 / stats['avg_step_time']
        
        if self.data_loading_times:
            stats['avg_data_loading_time'] = np.mean(list(self.data_loading_times)[-100:])
        
        if self.forward_pass_times:
            stats['avg_forward_time'] = np.mean(list(self.forward_pass_times)[-100:])
        
        if self.backward_pass_times:
            stats['avg_backward_time'] = np.mean(list(self.backward_pass_times)[-100:])
        
        if self.memory_usage:
            stats['avg_memory_usage'] = np.mean(list(self.memory_usage)[-100:])
        
        if self.gpu_utilization:
            stats['avg_gpu_utilization'] = np.mean(list(self.gpu_utilization)[-100:])
        
        return stats
    
    def _compute_efficiency_metrics(self) -> Dict[str, float]:
        """効率性メトリクスを計算"""
        metrics = {}
        
        if self.data_loading_times and self.step_times:
            # データ読み込み効率
            avg_data_time = np.mean(list(self.data_loading_times)[-100:])
            avg_step_time = np.mean(list(self.step_times)[-100:])
            metrics['data_loading_efficiency'] = 1.0 - (avg_data_time / avg_step_time)
        
        if self.forward_pass_times and self.backward_pass_times:
            # 計算効率
            avg_forward = np.mean(list(self.forward_pass_times)[-100:])
            avg_backward = np.mean(list(self.backward_pass_times)[-100:])
            metrics['forward_backward_ratio'] = avg_forward / (avg_backward + 1e-8)
        
        if self.gpu_utilization:
            # GPU効率
            avg_gpu_util = np.mean(list(self.gpu_utilization)[-100:])
            metrics['gpu_efficiency'] = avg_gpu_util
        
        # 学習効率（損失改善率）
        if len(self.loss_history) >= 50:
            recent_loss = np.mean(list(self.loss_history)[-10:])
            older_loss = np.mean(list(self.loss_history)[-50:-40])
            if older_loss > 0:
                metrics['learning_efficiency'] = (older_loss - recent_loss) / older_loss
        
        return metrics
    
    def save_performance_log(self, filepath: str):
        """パフォーマンスログをファイルに保存"""
        report = self.generate_optimization_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"パフォーマンスレポートを保存しました: {filepath}")
    
    def reset_alerts(self):
        """警告履歴をリセット"""
        self.performance_alerts.clear()

# グローバルな監視インスタンス
_global_performance_monitor = None
_global_resource_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """グローバルパフォーマンス監視インスタンスを取得"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor

def get_resource_monitor() -> SystemResourceMonitor:
    """グローバルリソース監視インスタンスを取得"""
    global _global_resource_monitor
    if _global_resource_monitor is None:
        _global_resource_monitor = SystemResourceMonitor()
    return _global_resource_monitor
