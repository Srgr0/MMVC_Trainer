"""Common utility functions for MMVC Trainer."""

import glob
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """ロガーのセットアップ."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """JSONファイルを読み込む."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """JSONファイルに保存する."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint(checkpoint_path: Union[str, Path], model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """チェックポイントを読み込む."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # モデルの重みを読み込み
    model.load_state_dict(checkpoint['model'])
    
    # オプティマイザーの状態を読み込み
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # グローバルステップを返す
    return checkpoint.get('global_step', 0)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   learning_rate: float, global_step: int, 
                   checkpoint_path: Union[str, Path]) -> None:
    """チェックポイントを保存する."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'global_step': global_step
    }
    
    torch.save(checkpoint, checkpoint_path)


def get_hparams_from_file(config_path: Union[str, Path]) -> object:
    """設定ファイルからハイパーパラメータを取得する."""
    config = load_json(config_path)
    
    class HParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, HParams(**v))
                else:
                    setattr(self, k, v)
    
    return HParams(**config)


def create_hparams(**kwargs) -> object:
    """ハイパーパラメータオブジェクトを作成する."""
    class HParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    setattr(self, k, HParams(**v))
                else:
                    setattr(self, k, v)
    
    return HParams(**kwargs)


def scan_checkpoint(cp_dir: Union[str, Path], prefix: str) -> Optional[str]:
    """最新のチェックポイントファイルを検索する."""
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_filepaths_and_text(filename: Union[str, Path]) -> list:
    """ファイルパスとテキストのリストを読み込む."""
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split('|') for line in f]
    return filepaths_and_text


def get_checkpoint_path(model_dir: Union[str, Path], model_name: str, 
                       global_step: Optional[int] = None) -> str:
    """チェックポイントのパスを生成する."""
    if global_step is not None:
        return os.path.join(model_dir, f"{model_name}_{global_step:06d}.pth")
    else:
        return os.path.join(model_dir, f"{model_name}_latest.pth")


def ensure_dir(path: Union[str, Path]) -> None:
    """ディレクトリが存在しない場合は作成する."""
    os.makedirs(path, exist_ok=True)


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """予測値と目標値のメトリクスを計算する."""
    with torch.no_grad():
        l1_loss = F.l1_loss(pred, target).item()
        mse_loss = F.mse_loss(pred, target).item()
        
        return {
            'l1_loss': l1_loss,
            'mse_loss': mse_loss,
            'rmse_loss': np.sqrt(mse_loss)
        }


def set_random_seed(seed: int) -> None:
    """ランダムシードを設定する."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 決定論的な動作を保証
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    """辞書のキーを属性としてアクセス可能にするクラス."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
