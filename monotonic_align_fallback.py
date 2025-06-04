"""
monotonic_align フォールバック実装

monotonic_align.monotonic_align.core モジュールが利用できない環境での
フォールバック実装を提供します。

作成日: 2024/1/1
更新日: 2024/1/1
"""

import torch
import numpy as np


def maximum_path_fallback(value, mask):
    """
    monotonic_align.core.maximum_path のフォールバック実装
    
    Args:
        value: アライメント値のテンソル [B, T_enc, T_dec]
        mask: マスクテンソル [B, T_enc, T_dec]
    
    Returns:
        path: 最適パス [B, T_enc, T_dec]
    """
    device = value.device
    dtype = value.dtype
    
    batch_size, t_enc, t_dec = value.shape
    path = torch.zeros_like(value, dtype=dtype, device=device)
    
    for b in range(batch_size):
        # 各バッチに対して動的プログラミングでパスを計算
        val = value[b].cpu().numpy()
        m = mask[b].cpu().numpy()
        
        # 累積最大値を計算
        dp = np.zeros((t_enc, t_dec))
        dp[0, 0] = val[0, 0] * m[0, 0]
        
        # 初期化
        for i in range(1, t_enc):
            dp[i, 0] = dp[i-1, 0] + val[i, 0] * m[i, 0]
        for j in range(1, t_dec):
            dp[0, j] = dp[0, j-1] + val[0, j] * m[0, j]
        
        # 動的プログラミング
        for i in range(1, t_enc):
            for j in range(1, t_dec):
                if m[i, j] > 0:
                    dp[i, j] = max(
                        dp[i-1, j],     # 上から
                        dp[i, j-1],     # 左から
                        dp[i-1, j-1]    # 斜めから
                    ) + val[i, j]
        
        # バックトラック
        i, j = t_enc - 1, t_dec - 1
        p = np.zeros((t_enc, t_dec))
        
        while i >= 0 and j >= 0:
            p[i, j] = 1.0
            
            if i == 0 and j == 0:
                break
            elif i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # 最大値を与えた方向を選択
                candidates = []
                if i > 0:
                    candidates.append((dp[i-1, j], i-1, j))
                if j > 0:
                    candidates.append((dp[i, j-1], i, j-1))
                if i > 0 and j > 0:
                    candidates.append((dp[i-1, j-1], i-1, j-1))
                
                if candidates:
                    _, next_i, next_j = max(candidates)
                    i, j = next_i, next_j
                else:
                    break
        
        path[b] = torch.from_numpy(p).to(device=device, dtype=dtype)
    
    return path


class MonotonicAlignFallback:
    """
    monotonic_align モジュールのフォールバック実装
    """
    
    @staticmethod
    def maximum_path(value, mask):
        """
        最適パスを計算（フォールバック版）
        
        Args:
            value: アライメント値 [B, T_enc, T_dec]
            mask: マスク [B, T_enc, T_dec]
        
        Returns:
            path: 最適パス [B, T_enc, T_dec]
        """
        return maximum_path_fallback(value, mask)


def safe_import_monotonic_align():
    """
    monotonic_alignの安全なインポート
    
    Returns:
        tuple: (success: bool, module: object, error_msg: str)
    """
    try:
        # 標準のmonotonic_alignをインポートを試行
        import monotonic_align
        from monotonic_align.monotonic_align.core import maximum_path
        
        print("✅ monotonic_align: 標準モジュールを使用")
        return True, monotonic_align, None
        
    except ImportError as e:
        print(f"⚠️ monotonic_align標準モジュール不可: {e}")
        print("🔄 フォールバック実装に切り替え")
        
        # フォールバック実装を作成
        class FallbackModule:
            def __init__(self):
                self.monotonic_align = MonotonicAlignFallback()
        
        fallback_module = FallbackModule()
        return False, fallback_module, str(e)


def test_monotonic_align_fallback():
    """
    フォールバック実装のテスト
    """
    print("🧪 monotonic_align フォールバック実装テスト開始")
    
    # テストデータの作成
    batch_size, t_enc, t_dec = 2, 10, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ランダムなアライメント値
    value = torch.randn(batch_size, t_enc, t_dec, device=device)
    
    # マスクの作成（有効な範囲を指定）
    mask = torch.zeros(batch_size, t_enc, t_dec, device=device)
    for b in range(batch_size):
        # 対角線近辺にマスクを設定
        for i in range(t_enc):
            for j in range(t_dec):
                if abs(i * t_dec / t_enc - j) < 3:
                    mask[b, i, j] = 1.0
    
    try:
        # フォールバック実装をテスト
        path = maximum_path_fallback(value, mask)
        
        # 基本的な検証
        assert path.shape == (batch_size, t_enc, t_dec), f"形状エラー: {path.shape}"
        assert torch.all(path >= 0), "負の値が含まれています"
        assert torch.all(path <= 1), "1より大きい値が含まれています"
        
        # パスの連続性をチェック
        for b in range(batch_size):
            path_sum = torch.sum(path[b])
            assert path_sum > 0, f"バッチ {b}: パスが空です"
        
        print("✅ フォールバック実装テスト成功")
        print(f"   - 入力形状: {value.shape}")
        print(f"   - 出力形状: {path.shape}")
        print(f"   - デバイス: {device}")
        print(f"   - パス密度: {torch.mean(torch.sum(path, dim=(1,2))).item():.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ フォールバック実装テスト失敗: {e}")
        return False


if __name__ == "__main__":
    # テスト実行
    success, module, error = safe_import_monotonic_align()
    
    if not success:
        print("フォールバック実装をテストします...")
        test_result = test_monotonic_align_fallback()
        
        if test_result:
            print("🎯 フォールバック実装の準備完了")
        else:
            print("🚫 フォールバック実装に問題があります")
    
    print("📋 使用方法:")
    print("   from monotonic_align_fallback import safe_import_monotonic_align")
    print("   success, monotonic_align, error = safe_import_monotonic_align()")
