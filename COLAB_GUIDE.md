# Google Colab 実行ガイド

MMVC_TrainerをGoogle Colabで実行するための詳細な手順です。

## 🚀 事前準備

### 1. Google Driveの準備

まず、Google Driveに作業用フォルダを作成し、データセットを準備します。

```python
# Google Driveのマウント
from google.colab import drive
drive.mount('/content/drive')

# 作業ディレクトリの作成
import os
work_dir = '/content/drive/MyDrive/MMVC_Trainer'
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)
print(f"作業ディレクトリ: {os.getcwd()}")
```

### 2. データセットのアップロード

以下のいずれかの方法でデータセットを準備してください：

#### 方法A: ZIPファイルのアップロード

```python
from google.colab import files
import zipfile

# ZIPファイルのアップロード
print("データセットのZIPファイルを選択してください")
uploaded = files.upload()

# 解凍
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        print(f"解凍中: {filename}")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('dataset/')
        os.remove(filename)  # ZIPファイルを削除
        print("解凍完了")
```

#### 方法B: Google Driveからコピー

```python
# Google Drive上のデータセットフォルダをコピー
import shutil

source_path = '/content/drive/MyDrive/your_dataset_folder'  # 実際のパスに変更
target_path = './dataset'

if os.path.exists(source_path):
    shutil.copytree(source_path, target_path)
    print(f"データセットをコピーしました: {source_path} → {target_path}")
else:
    print(f"データセットフォルダが見つかりません: {source_path}")
```

## 📋 ステップバイステップ実行

### Step 1: リポジトリのクローンと環境セットアップ

```python
# GPU確認
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# リポジトリのクローン
!git clone https://github.com/your-username/MMVC_Trainer.git temp_repo
!rsync -av temp_repo/ ./
!rm -rf temp_repo

print("リポジトリのクローン完了")
```

### Step 2: 依存関係のインストール

```python
# 必要なパッケージのインストール
!pip install torch==1.13.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
!pip install -r requirements.txt

# Cythonモジュールのビルド
os.chdir('monotonic_align')
!python setup.py build_ext --inplace
os.chdir('..')

print("依存関係のインストール完了")
```

### Step 3: データセットの確認と前処理

```python
# データセット構造の確認
def check_dataset_structure(dataset_path='dataset'):
    if not os.path.exists(dataset_path):
        print(f"警告: データセットフォルダが見つかりません: {dataset_path}")
        return False
    
    speakers = []
    for item in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, item)
        if os.path.isdir(speaker_path):
            wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
            speakers.append((item, len(wav_files)))
            print(f"話者: {item}, 音声ファイル数: {len(wav_files)}")
    
    if not speakers:
        print("警告: 音声ファイルが見つかりません")
        return False
    
    total_files = sum([count for _, count in speakers])
    print(f"\\n総話者数: {len(speakers)}")
    print(f"総音声ファイル数: {total_files}")
    
    return len(speakers), total_files

speakers_count, files_count = check_dataset_structure()

if speakers_count == 0:
    print("データセットを正しく配置してください:")
    print("dataset/")
    print("├── speaker1/")
    print("│   ├── audio1.wav")
    print("│   └── audio2.wav")
    print("└── speaker2/")
    print("    ├── audio1.wav")
    print("    └── audio2.wav")
```

### Step 4: 設定ファイルの生成

```python
import json

# GPUメモリに基づく自動設定
def create_config(num_speakers, total_files):
    # GPU メモリチェック
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 8:
            batch_size = 8
            segment_size = 4096
        elif gpu_memory < 12:
            batch_size = 12
            segment_size = 6144
        else:
            batch_size = 16
            segment_size = 8192
    else:
        batch_size = 4
        segment_size = 4096
    
    # エポック数の調整（データ量に基づく）
    if total_files < 100:
        epochs = 500
    elif total_files < 500:
        epochs = 750
    else:
        epochs = 1000
    
    config = {
        "model_name": f"MMVC_VITS_{num_speakers}speakers",
        "sampling_rate": 22050,
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mel_channels": 80,
        "n_speakers": num_speakers if num_speakers > 1 else 0,
        "segment_size": segment_size,
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
            "gin_channels": 256
        },
        "train": {
            "epochs": epochs,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": batch_size,
            "lr_decay": 0.999875,
            "segment_size": segment_size,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0,
            "use_sr": True,
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None
        },
        "data": {
            "training_files": "filelists/train.txt",
            "validation_files": "filelists/val.txt",
            "text_cleaners": ["japanese_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": num_speakers if num_speakers > 1 else 0,
            "cleaned_text": True
        }
    }
    
    # 設定ファイルの保存
    os.makedirs('configs', exist_ok=True)
    with open('configs/config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"設定ファイルを生成しました: configs/config.json")
    print(f"- 話者数: {num_speakers}")
    print(f"- バッチサイズ: {batch_size}")
    print(f"- セグメントサイズ: {segment_size}")
    print(f"- エポック数: {epochs}")
    
    return config

config = create_config(speakers_count, files_count)
```

### Step 5: データセットの前処理

```python
# 日本語音素変換とファイルリスト生成
!python create_dataset_jtalk.py --input_dir dataset --output_dir processed

# ファイルリストの生成
import random
from pathlib import Path

def create_filelists():
    os.makedirs('filelists', exist_ok=True)
    
    # 処理済みファイルの収集
    processed_files = []
    for txt_file in Path('processed').rglob('*.txt'):
        # テキストファイルと対応する音声ファイルの確認
        wav_file = txt_file.with_suffix('.wav')
        if wav_file.exists():
            processed_files.append((str(wav_file), str(txt_file)))
    
    if not processed_files:
        print("警告: 処理済みファイルが見つかりません")
        return
    
    # ランダムシャッフル
    random.shuffle(processed_files)
    
    # 訓練/検証分割（90%/10%）
    split_idx = int(len(processed_files) * 0.9)
    train_files = processed_files[:split_idx]
    val_files = processed_files[split_idx:]
    
    # 訓練用ファイルリスト
    with open('filelists/train.txt', 'w', encoding='utf-8') as f:
        for wav_path, txt_path in train_files:
            with open(txt_path, 'r', encoding='utf-8') as txt_f:
                text = txt_f.read().strip()
            f.write(f"{wav_path}|{text}\\n")
    
    # 検証用ファイルリスト
    with open('filelists/val.txt', 'w', encoding='utf-8') as f:
        for wav_path, txt_path in val_files:
            with open(txt_path, 'r', encoding='utf-8') as txt_f:
                text = txt_f.read().strip()
            f.write(f"{wav_path}|{text}\\n")
    
    print(f"ファイルリストを生成しました:")
    print(f"- 訓練用: {len(train_files)} ファイル")
    print(f"- 検証用: {len(val_files)} ファイル")

create_filelists()
```

### Step 6: 訓練実行

```python
# 訓練開始
import subprocess
import time
from datetime import datetime

def start_training():
    print(f"訓練開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 単一話者または マルチ話者の選択
    if config.get('n_speakers', 0) > 0:
        cmd = ["python", "train_ms.py", "-c", "configs/config.json", "-m", "mmvc_model"]
        print("マルチ話者モデルで訓練を開始します")
    else:
        cmd = ["python", "train.py", "-c", "configs/config.json", "-m", "mmvc_model"]
        print("単一話者モデルで訓練を開始します")
    
    # 訓練プロセスの開始
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              universal_newlines=True, bufsize=1)
    
    # リアルタイムログ表示
    try:
        for line in process.stdout:
            print(line.rstrip())
            
            # 一定時間ごとにGoogle Driveに進捗を保存
            if "INFO" in line and "step" in line:
                # 進捗をファイルに保存
                with open('/content/drive/MyDrive/mmvc_progress.log', 'a') as f:
                    f.write(f"{datetime.now()}: {line}")
                    
    except KeyboardInterrupt:
        print("\\n訓練を中断しています...")
        process.terminate()
        process.wait()
        print("訓練が中断されました")
    
    return process.returncode

# 訓練実行
training_result = start_training()
```

### Step 7: 訓練監視とチェックポイント管理

```python
# 訓練進捗の監視
import matplotlib.pyplot as plt
import glob

def monitor_training():
    # ログファイルの確認
    log_files = glob.glob('logs/*/train.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"最新のログファイル: {latest_log}")
        
        # 最後の数行を表示
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            print("最新のログ:")
            for line in lines[-10:]:
                print(line.rstrip())
    
    # チェックポイントの確認
    checkpoint_files = glob.glob('logs/*/G_*.pth')
    if checkpoint_files:
        print(f"\\n保存されたチェックポイント数: {len(checkpoint_files)}")
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"最新のチェックポイント: {os.path.basename(latest_checkpoint)}")
        
        # チェックポイントをGoogle Driveにバックアップ
        backup_dir = '/content/drive/MyDrive/MMVC_Checkpoints'
        os.makedirs(backup_dir, exist_ok=True)
        
        import shutil
        backup_path = os.path.join(backup_dir, os.path.basename(latest_checkpoint))
        shutil.copy2(latest_checkpoint, backup_path)
        print(f"チェックポイントをバックアップ: {backup_path}")

# 監視実行
monitor_training()
```

### Step 8: 推論テスト

```python
# 訓練済みモデルでテスト推論
def test_inference():
    # 最新のチェックポイントを取得
    checkpoint_files = glob.glob('logs/*/G_*.pth')
    if not checkpoint_files:
        print("チェックポイントが見つかりません")
        return
    
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    print(f"使用するチェックポイント: {latest_checkpoint}")
    
    # モデルの読み込み
    from models import SynthesizerTrn
    from text import text_to_sequence
    from text.symbols import symbols
    import soundfile as sf
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデル初期化
    net_g = SynthesizerTrn(
        len(symbols),
        config["filter_length"] // 2 + 1,
        config["segment_size"] // config["hop_length"],
        n_speakers=config.get("n_speakers", 0),
        **config["model"]
    ).to(device)
    
    # チェックポイント読み込み
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    net_g.load_state_dict(checkpoint['model'])
    net_g.eval()
    
    print("モデル読み込み完了")
    
    # テスト音声生成
    test_texts = [
        "こんにちは、これはテスト音声です。",
        "音声合成の品質を確認しています。",
        "MMVC_Trainerによる音声生成テストです。"
    ]
    
    os.makedirs('test_output', exist_ok=True)
    
    for i, text in enumerate(test_texts):
        print(f"\\n生成中: {text}")
        
        # テキストを音素に変換
        stn_tst = text_to_sequence(text, ["japanese_cleaners"])
        
        with torch.no_grad():
            x_tst = torch.LongTensor(stn_tst).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(stn_tst)]).to(device)
            
            if config.get("n_speakers", 0) > 0:
                sid = torch.LongTensor([0]).to(device)  # 最初の話者
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid)[0][0, 0].cpu().numpy()
            else:
                audio = net_g.infer(x_tst, x_tst_lengths)[0][0, 0].cpu().numpy()
        
        # 音声保存
        output_path = f'test_output/test_{i+1}.wav'
        sf.write(output_path, audio, config["sampling_rate"])
        print(f"保存: {output_path}")
        
        # Google Drive にもコピー
        backup_path = f'/content/drive/MyDrive/test_audio_{i+1}.wav'
        shutil.copy2(output_path, backup_path)
        print(f"バックアップ: {backup_path}")
    
    print("\\nテスト推論完了！")

# 推論テスト実行
test_inference()
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

1. **GPU メモリ不足**
```python
# メモリ使用量を確認
if torch.cuda.is_available():
    print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU メモリ最大: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
# メモリをクリア
torch.cuda.empty_cache()

# バッチサイズを小さくする
# configs/config.json の batch_size を 8 → 4 → 2 に変更
```

2. **接続タイムアウト**
```python
# 定期的にダミー出力でセッション維持
import time
import threading

def keep_alive():
    while True:
        print(".", end="", flush=True)
        time.sleep(300)  # 5分ごと

# バックグラウンドで実行
thread = threading.Thread(target=keep_alive, daemon=True)
thread.start()
```

3. **ランタイム切断対策**
```python
# 重要なファイルを定期的にGoogle Driveに保存
def backup_important_files():
    import shutil
    backup_dir = '/content/drive/MyDrive/MMVC_Backup'
    os.makedirs(backup_dir, exist_ok=True)
    
    # 設定ファイル
    if os.path.exists('configs/config.json'):
        shutil.copy2('configs/config.json', f'{backup_dir}/config.json')
    
    # 最新のチェックポイント
    checkpoint_files = glob.glob('logs/*/G_*.pth')
    if checkpoint_files:
        latest = max(checkpoint_files, key=os.path.getctime)
        shutil.copy2(latest, f'{backup_dir}/latest_model.pth')
    
    print(f"バックアップ完了: {backup_dir}")

# 定期バックアップ（30分ごと）
import threading
import time

def periodic_backup():
    while True:
        time.sleep(1800)  # 30分
        backup_important_files()

backup_thread = threading.Thread(target=periodic_backup, daemon=True)
backup_thread.start()
```

## 📊 パフォーマンス最適化

### GPU最適化設定

```python
# 最適化設定
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# 混合精度訓練の有効化（PyTorch 1.6+）
import torch.cuda.amp as amp

# CUDA メモリの事前割り当て
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # GPU メモリの90%を使用
```

### メモリ監視

```python
def monitor_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"GPU メモリ:")
        print(f"  割り当て済み: {allocated:.2f} GB")
        print(f"  キャッシュ: {cached:.2f} GB")
        print(f"  最大割り当て: {max_allocated:.2f} GB")
        
        if allocated > 10:  # 10GB以上使用時は警告
            print("⚠️ メモリ使用量が多いです。バッチサイズを小さくすることを検討してください。")

# 定期的にメモリ使用量をチェック
monitor_memory()
```

## 🎯 完了チェックリスト

訓練完了時に以下を確認してください：

- [ ] 訓練が正常に完了した
- [ ] チェックポイントが生成された
- [ ] テスト音声が生成できた
- [ ] 重要なファイルがGoogle Driveにバックアップされた
- [ ] 音声品質が満足できるレベルである

## 📞 サポート

問題が発生した場合は、以下の情報と共にIssueを作成してください：

1. エラーメッセージの全文
2. 使用したGPU型番とVRAM容量
3. データセットのサイズと構造
4. 実行した設定（config.json）

---

このガイドに従って実行すれば、Google ColabでMMVC_Trainerを成功に実行できるはずです。何か問題がある場合は、遠慮なくお知らせください！
