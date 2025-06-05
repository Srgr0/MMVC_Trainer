# MMVC_Trainer

**Multi-Modal Voice Conversion Trainer** - 高品質な音声変換モデルのためのトレーニングフレームワーク

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 概要

MMVC_Trainerは、VITS（Variational Inference with adversarial learning for end-to-end Text-to-Speech）アーキテクチャをベースとした音声変換システムです。高品質な音声合成と変換を実現するためのエンドツーエンドのトレーニングフレームワークを提供します。

### 主な特徴

- 🎯 **VITS アーキテクチャ**: 最先端のテキスト音声合成技術
- ⚡ **最適化された単調アライメント**: Cython実装による高速な音韻アライメント
- 🖥️ **CLI ベース**: コマンドラインインターフェースによる使いやすい操作
- 🔄 **音声変換**: 異なる話者間の音声変換機能
- 📊 **モデルエクスポート**: ONNX形式でのモデル出力対応
- 🧩 **モジュラー設計**: 拡張しやすい設計アーキテクチャ

## システム要件

### 最小要件
- Python 3.8以上
- PyTorch 2.0以上
- 8GB以上のRAM
- 10GB以上の空きストレージ

### 推奨要件
- Python 3.9以上
- PyTorch 2.0以上（CUDA対応）
- NVIDIA GPU（8GB VRAM以上）
- 16GB以上のRAM
- 50GB以上の空きストレージ

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-username/MMVC_Trainer-my.git
cd MMVC_Trainer-my
```

### 2. 仮想環境の作成

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール

#### 開発環境（CPU版）
```bash
pip install -r requirements-dev.txt
```

#### 本番環境（GPU版）
```bash
pip install -r requirements.txt
```

### 4. Cythonモジュールのビルド

```bash
cd src/core/alignment
python setup.py build_ext --inplace
cd ../../..
```

## クイックスタート

### 1. データの準備

音声データを`data/raw/`ディレクトリに配置します：

```
data/raw/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── speaker2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── metadata.txt
```

### 2. 設定ファイルの準備

`configs/`ディレクトリに設定ファイルを作成します：

```bash
cp configs/base_config.json configs/my_config.json
# my_config.jsonを編集
```

### 3. データの前処理

```bash
python -m src.cli.preprocess \
    --input data/raw \
    --output data/processed \
    --config configs/my_config.json
```

### 4. モデルの訓練

```bash
python -m src.cli.train \
    --config configs/my_config.json \
    --model models/my_model
```

### 5. 音声変換

```bash
python -m src.cli.convert \
    --config configs/my_config.json \
    --checkpoint models/my_model/checkpoint_best.pth \
    --input input.wav \
    --output output.wav \
    --target-speaker 1
```

## 詳細な使用方法

### トレーニング

```bash
python -m src.cli.train [OPTIONS]

オプション:
  -c, --config CONFIG     設定ファイルのパス（必須）
  -m, --model MODEL       モデル保存ディレクトリ（必須）
  --resume RESUME         学習再開用のチェックポイント
  -h, --help              ヘルプを表示
```

### 音声変換

```bash
python -m src.cli.convert [OPTIONS]

主要オプション:
  -c, --config CONFIG           設定ファイルのパス（必須）
  -ckpt, --checkpoint CHECKPOINT チェックポイントファイル（必須）
  -i, --input INPUT             入力音声ファイル（必須）
  -o, --output OUTPUT           出力音声ファイル（必須）
  -ts, --target-speaker ID      ターゲット話者ID（デフォルト: 0）
  -ns, --noise-scale SCALE      ノイズスケール（デフォルト: 0.667）
  -ls, --length-scale SCALE     長さスケール（デフォルト: 1.0）
  --device {auto,cpu,cuda}      使用デバイス（デフォルト: auto）
```

### モデルエクスポート

```bash
python -m src.cli.export [OPTIONS]

主要オプション:
  -c, --config CONFIG           設定ファイルのパス（必須）
  -ckpt, --checkpoint CHECKPOINT チェックポイントファイル（必須）
  -o, --output OUTPUT           出力ONNXファイル（必須）
  --export-generator-only       ジェネレーターのみエクスポート
  --export-encoder-only         エンコーダーのみエクスポート
  --opset-version VERSION       ONNXオペセットバージョン（デフォルト: 11）
  --dynamic-axes                動的軸を使用
  --optimize                    モデルを最適化
```

## 設定ファイル

設定ファイルはJSON形式で、以下の主要セクションを含みます：

```json
{
  "train": {
    "log_interval": 200,
    "eval_interval": 1000,
    "epochs": 10000,
    "learning_rate": 2e-4,
    "betas": [0.8, 0.99],
    "eps": 1e-9,
    "batch_size": 16,
    "fp16_run": true,
    "lr_decay": 0.999875
  },
  "data": {
    "training_files": "data/processed/train.txt",
    "validation_files": "data/processed/val.txt",
    "text_cleaners": ["japanese_cleaners"],
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": null,
    "add_blank": true,
    "n_speakers": 0,
    "cleaned_text": true
  },
  "model": {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_vocabularies": 256,
    "segment_size": 8192
  }
}
```

## プロジェクト構造

```
MMVC_Trainer-my/
├── README.md                   # プロジェクト説明
├── requirements.txt            # 本番環境依存関係
├── requirements-dev.txt        # 開発環境依存関係
├── pyproject.toml             # パッケージ設定
├── configs/                   # 設定ファイル
│   └── base_config.json       # 基本設定
├── data/                      # データディレクトリ
│   ├── raw/                   # 生データ
│   └── processed/             # 前処理済みデータ
├── models/                    # モデル保存
│   ├── checkpoints/           # チェックポイント
│   └── pretrained/            # 事前訓練モデル
├── logs/                      # ログファイル
├── src/                       # ソースコード
│   ├── cli/                   # CLIスクリプト
│   │   ├── train.py           # 訓練スクリプト
│   │   ├── convert.py         # 変換スクリプト
│   │   ├── export.py          # エクスポートスクリプト
│   │   └── preprocess.py      # 前処理スクリプト
│   ├── core/                  # コアモジュール
│   │   ├── alignment/         # 単調アライメント
│   │   ├── data/              # データローダー
│   │   ├── models/            # モデル定義
│   │   └── training/          # 訓練ユーティリティ
│   ├── text/                  # テキスト処理
│   └── utils/                 # ユーティリティ
└── scripts/                   # 補助スクリプト
```

## アーキテクチャ

MMVC_Trainerは以下の主要コンポーネントで構成されています：

### 1. VITS モデル
- **Text Encoder**: テキスト入力をエンコード
- **Posterior Encoder**: 音響特徴量をエンコード
- **Generator**: HiFi-GANベースの音声生成
- **Discriminator**: Multi-Period & Multi-Scale識別器

### 2. 単調アライメント
- Cython最適化による高速なモノトニックアライメント検索
- Python フォールバック実装

### 3. 損失関数
- Generator Loss（敵対的学習）
- Feature Matching Loss
- Mel-spectrogram Loss
- KL Divergence Loss

## パフォーマンス

### モデルサイズ
- **総パラメータ数**: 約56.6M
- **訓練可能パラメータ数**: 約56.6M

### 速度（RTX 3080基準）
- **訓練速度**: ~0.5秒/step（バッチサイズ16）
- **推論速度**: ~0.1秒/audio（22.05kHz、5秒音声）

## トラブルシューティング

### よくある問題

#### 1. Cythonモジュールのビルドエラー
```bash
# 解決方法
pip install Cython numpy
cd src/core/alignment
python setup.py build_ext --inplace
```

#### 2. CUDA out of memory
```bash
# バッチサイズを減らす
# configs/my_config.jsonで "batch_size" を小さく設定
```

#### 3. 音声品質が悪い
```bash
# 訓練データの品質を確認
# サンプリングレートを統一
# ノイズを除去
```

### ログの確認

訓練ログは`logs/`ディレクトリに保存されます：

```bash
# TensorBoardでの確認
tensorboard --logdir logs/
```

## 貢献

プロジェクトへの貢献を歓迎します！

1. フォークしてください
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. コミット（`git commit -m 'Add amazing feature'`）
4. プッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 謝辞

- [VITS](https://github.com/jaywalnut310/vits) - ベースとなるアーキテクチャ
- [HiFi-GAN](https://github.com/jik876/hifi-gan) - ジェネレーターの実装
- PyTorchコミュニティ

## 更新履歴

### v1.0.0 (2025-06-05)
- 初回リリース
- VITS モデルの完全実装
- CLI ベースのトレーニング・変換システム
- Cython最適化された単調アライメント
- ONNX エクスポート機能

## サポート

質問や問題がある場合は、以下の方法でお気軽にお問い合わせください：

- [Issues](https://github.com/your-username/MMVC_Trainer-my/issues) - バグ報告や機能要求
- [Discussions](https://github.com/your-username/MMVC_Trainer-my/discussions) - 一般的な質問や議論

---

**MMVC_Trainer** で高品質な音声変換を体験してください！ 🎵
