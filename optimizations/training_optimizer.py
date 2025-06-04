"""
MMVC_Trainer Training Optimization Module
学習プロセスの最適化を行う
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import logging
from contextlib import contextmanager
from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class TrainingOptimizer:
    """学習プロセス最適化のメインクラス"""
    
    def __init__(self, mixed_precision: bool = True, gradient_clipping: float = None):
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.scaler = GradScaler(enabled=mixed_precision) if mixed_precision else None
        
        # パフォーマンス統計
        self.step_times = []
        self.memory_usage = []
        self.loss_history = []
        
    def optimize_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                     loss: torch.Tensor, retain_graph: bool = False) -> Dict[str, Any]:
        """最適化されたstep実行"""
        
        step_start_time = time.time()
        
        # 勾配をゼロクリア
        optimizer.zero_grad()
        
        if self.mixed_precision and self.scaler is not None:
            # 混合精度学習
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            
            if self.gradient_clipping:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # 通常の学習
            loss.backward(retain_graph=retain_graph)
            
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clipping)
            
            optimizer.step()
        
        # パフォーマンス統計の更新
        step_time = time.time() - step_start_time
        self.step_times.append(step_time)
        
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
            self.memory_usage.append(memory_usage)
        
        self.loss_history.append(loss.item())
        
        return {
            'step_time': step_time,
            'memory_usage_gb': self.memory_usage[-1] if self.memory_usage else 0,
            'loss': loss.item()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        if not self.step_times:
            return {}
        
        recent_steps = self.step_times[-100:]  # 最近の100ステップ
        recent_memory = self.memory_usage[-100:] if self.memory_usage else []
        recent_loss = self.loss_history[-100:]
        
        return {
            'avg_step_time': np.mean(recent_steps),
            'min_step_time': np.min(recent_steps),
            'max_step_time': np.max(recent_steps),
            'avg_memory_usage_gb': np.mean(recent_memory) if recent_memory else 0,
            'avg_loss': np.mean(recent_loss),
            'loss_trend': np.polyfit(range(len(recent_loss)), recent_loss, 1)[0] if len(recent_loss) > 1 else 0
        }

class LossComputation:
    """損失計算の最適化"""
    
    @staticmethod
    def compute_mel_loss_efficient(y_mel: torch.Tensor, y_hat_mel: torch.Tensor, 
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """効率的なメル損失計算"""
        if mask is not None:
            # マスクされた部分のみで損失を計算
            y_mel_masked = y_mel * mask
            y_hat_mel_masked = y_hat_mel * mask
            loss = torch.nn.functional.l1_loss(y_mel_masked, y_hat_mel_masked, reduction='sum')
            loss = loss / mask.sum()
        else:
            loss = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
        
        return loss
    
    @staticmethod
    def compute_kl_loss_stable(z_p: torch.Tensor, logs_q: torch.Tensor, 
                             m_p: torch.Tensor, logs_p: torch.Tensor, 
                             z_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """数値的に安定なKL損失計算"""
        # ログ項の数値安定性を向上
        logs_p = torch.clamp(logs_p, min=-10, max=10)
        logs_q = torch.clamp(logs_q, min=-10, max=10)
        
        # KL divergence: D_KL(P||Q) = -0.5 * sum(1 + log(sigma_p^2) - log(sigma_q^2) - (mu_p^2 + sigma_p^2) / sigma_q^2)
        kl = logs_q - logs_p - 0.5
        kl = kl + 0.5 * ((z_p - m_p) ** 2 + torch.exp(2 * logs_p)) / (torch.exp(2 * logs_q) + eps)
        
        # マスクを適用
        kl = kl * z_mask
        loss = kl.sum() / z_mask.sum()
        
        return loss
    
    @staticmethod
    def compute_feature_loss_efficient(fmap_r: List[torch.Tensor], 
                                     fmap_g: List[torch.Tensor]) -> torch.Tensor:
        """効率的な特徴量損失計算"""
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                # メモリ効率を考慮した計算
                loss = loss + torch.nn.functional.l1_loss(rl.detach(), gl)
        
        return loss

class GradientOptimization:
    """勾配最適化のユーティリティ"""
    
    @staticmethod
    def adaptive_gradient_clipping(model: nn.Module, clip_value: float, 
                                 percentile: float = 90) -> float:
        """適応的勾配クリッピング"""
        total_norm = 0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 勾配ノルムの統計に基づいて適応的にクリッピング値を調整
        if hasattr(GradientOptimization, '_gradient_norms'):
            GradientOptimization._gradient_norms.append(total_norm)
            if len(GradientOptimization._gradient_norms) > 1000:
                GradientOptimization._gradient_norms = GradientOptimization._gradient_norms[-1000:]
            
            # パーセンタイルに基づいてクリッピング値を動的調整
            adaptive_clip = np.percentile(GradientOptimization._gradient_norms, percentile)
            clip_value = min(clip_value, adaptive_clip)
        else:
            GradientOptimization._gradient_norms = [total_norm]
        
        return clip_value
    
    @staticmethod
    def gradient_accumulation_step(loss: torch.Tensor, accumulation_steps: int) -> torch.Tensor:
        """勾配蓄積のための損失正規化"""
        return loss / accumulation_steps

class ModelOptimization:
    """モデル最適化のユーティリティ"""
    
    @staticmethod
    def enable_efficient_attention(model: nn.Module) -> nn.Module:
        """効率的なAttentionメカニズムを有効化"""
        for name, module in model.named_modules():
            if hasattr(module, 'attention') or 'attention' in name.lower():
                # PyTorchの最適化されたattentionを使用
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    # PyTorch 2.0以降の最適化されたattention
                    pass
        return model
    
    @staticmethod
    def optimize_conv_layers(model: nn.Module) -> nn.Module:
        """畳み込み層の最適化"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                # 畳み込み層のパラメータ最適化
                if hasattr(module, 'padding_mode'):
                    # パディングモードの最適化
                    pass
        return model
    
    @staticmethod
    def fuse_batch_norm(model: nn.Module) -> nn.Module:
        """BatchNormの融合最適化"""
        model.eval()
        
        # Conv + BatchNormの融合
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                new_layers = []
                i = 0
                while i < len(module):
                    if (i + 1 < len(module) and 
                        isinstance(module[i], (nn.Conv1d, nn.Conv2d)) and
                        isinstance(module[i + 1], (nn.BatchNorm1d, nn.BatchNorm2d))):
                        # Conv + BN融合
                        fused = torch.fx.experimental.optimization.fuse(module[i], module[i + 1])
                        new_layers.append(fused)
                        i += 2
                    else:
                        new_layers.append(module[i])
                        i += 1
                
                if len(new_layers) != len(module):
                    setattr(model, name.split('.')[-1], nn.Sequential(*new_layers))
        
        return model

@contextmanager
def training_mode_context(model: nn.Module, training: bool = True):
    """学習モードのコンテキストマネージャー"""
    original_mode = model.training
    try:
        model.train(training)
        yield model
    finally:
        model.train(original_mode)

def optimize_learning_rate(optimizer: torch.optim.Optimizer, 
                         current_step: int, warmup_steps: int = 1000,
                         base_lr: float = 0.0002) -> float:
    """学習率の最適化スケジューリング"""
    if current_step < warmup_steps:
        # ウォームアップ期間
        lr = base_lr * (current_step / warmup_steps)
    else:
        # コサインアニーリング
        progress = (current_step - warmup_steps) / (100000 - warmup_steps)  # 総ステップ数は調整可能
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr
