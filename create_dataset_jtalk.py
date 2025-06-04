#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from pathlib import Path

import pyopenjtalk
from tqdm import tqdm


def extract_phonemes(text, cleaner_names=['japanese_cleaners']):
    """
    Extract phonemes from Japanese text using pyopenjtalk
    """
    if 'japanese_cleaners' in cleaner_names:
        try:
            # pyopenjtalkを使用して音素抽出
            phonemes = pyopenjtalk.g2p(text)
            return phonemes
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            return ""
    else:
        return text


def process_text_file(text_file_path, audio_dir, output_path, speaker_id=0):
    """
    Process text file and create filelist with phonemes
    """
    processed_lines = []
    
    with open(text_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Processing {text_file_path.name}"):
        line = line.strip()
        if not line:
            continue
            
        # フォーマット: filename|text または filename|speaker_id|text
        parts = line.split('|')
        if len(parts) < 2:
            print(f"Skipping invalid line: {line}")
            continue
            
        if len(parts) == 2:
            filename, text = parts
            sid = speaker_id
        else:
            filename, sid, text = parts[0], int(parts[1]), '|'.join(parts[2:])
        
        # 音声ファイルの存在確認
        audio_path = os.path.join(audio_dir, f"{filename}.wav")
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
        
        # 音素抽出
        phonemes = extract_phonemes(text)
        if not phonemes:
            print(f"Failed to extract phonemes for: {text}")
            continue
        
        # 結果を保存
        if len(parts) == 2:
            processed_lines.append(f"{audio_path}|{phonemes}")
        else:
            processed_lines.append(f"{audio_path}|{sid}|{phonemes}")
    
    # ファイルに書き込み
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed_lines))
    
    return len(processed_lines)


def create_config_template(config_path, dataset_path):
    """
    Create a configuration template file
    """
    config_template = {
        "train": {
            "log_interval": 200,
            "eval_interval": 1000,
            "seed": 1234,
            "epochs": 10000,
            "learning_rate": 2e-4,
            "betas": [0.8, 0.99],
            "eps": 1e-9,
            "batch_size": 16,
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 8192,
            "init_lr_ratio": 1,
            "warmup_epochs": 0,
            "c_mel": 45,
            "c_kl": 1.0
        },
        "data": {
            "training_files": f"{dataset_path}/train.txt",
            "validation_files": f"{dataset_path}/val.txt",
            "text_cleaners": ["japanese_cleaners"],
            "max_wav_value": 32768.0,
            "sampling_rate": 22050,
            "filter_length": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mel_channels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": null,
            "add_blank": True,
            "n_speakers": 0,
            "cleaned_text": True
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
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 256
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Create MMVC dataset with Japanese phonemes')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Path to dataset directory containing textful folder')
    parser.add_argument('--output_dir', type=str, default='filelists',
                        help='Output directory for processed filelists')
    parser.add_argument('--config_template', type=str, default='configs/dataset_config.json',
                        help='Path to save configuration template')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data (0.0-1.0)')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    # 出力ディレクトリ作成
    output_dir.mkdir(exist_ok=True)
    
    # textfulディレクトリの確認
    textful_dir = dataset_dir / 'textful'
    if not textful_dir.exists():
        print(f"Error: {textful_dir} does not exist")
        sys.exit(1)
    
    all_processed_lines = []
    
    # 各話者フォルダを処理
    for speaker_dir in textful_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
            
        print(f"Processing speaker: {speaker_dir.name}")
        
        # transcript.txtファイルを探す
        transcript_file = speaker_dir / 'transcript.txt'
        if not transcript_file.exists():
            print(f"Warning: {transcript_file} not found, skipping...")
            continue
        
        # 音声ファイルディレクトリ（wavs/）
        audio_dir = speaker_dir / 'wavs'
        if not audio_dir.exists():
            audio_dir = speaker_dir  # wavファイルが直接speaker_dirにある場合
        
        # 話者IDを抽出（フォルダ名から）
        try:
            speaker_id = int(speaker_dir.name.split('_')[0])
        except:
            speaker_id = hash(speaker_dir.name) % 1000  # フォールバック
        
        # 一時ファイルに処理
        temp_output = output_dir / f"temp_{speaker_dir.name}.txt"
        count = process_text_file(transcript_file, audio_dir, temp_output, speaker_id)
        
        if count > 0:
            with open(temp_output, 'r', encoding='utf-8') as f:
                all_processed_lines.extend(f.readlines())
            temp_output.unlink()  # 一時ファイル削除
            print(f"Processed {count} lines for speaker {speaker_dir.name}")
        else:
            print(f"No valid data found for speaker {speaker_dir.name}")
    
    if not all_processed_lines:
        print("Error: No valid data found in any speaker directory")
        sys.exit(1)
    
    # データをシャッフル
    import random
    random.seed(1234)
    random.shuffle(all_processed_lines)
    
    # 訓練・検証データに分割
    total_lines = len(all_processed_lines)
    train_count = int(total_lines * args.train_ratio)
    
    train_lines = all_processed_lines[:train_count]
    val_lines = all_processed_lines[train_count:]
    
    # ファイル保存
    train_file = output_dir / 'train.txt'
    val_file = output_dir / 'val.txt'
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    print(f"\nDataset creation completed!")
    print(f"Training data: {len(train_lines)} samples -> {train_file}")
    print(f"Validation data: {len(val_lines)} samples -> {val_file}")
    
    # 設定ファイルテンプレート作成
    config_dir = Path(args.config_template).parent
    config_dir.mkdir(exist_ok=True)
    create_config_template(args.config_template, str(output_dir))
    print(f"Configuration template saved to: {args.config_template}")
    
    # 統計情報
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_lines}")
    print(f"Unique speakers: {len(set(line.split('|')[1] for line in all_processed_lines if len(line.split('|')) > 2))}")
    print(f"Average phoneme length: {sum(len(line.split('|')[-1].strip()) for line in all_processed_lines) / total_lines:.1f}")


if __name__ == '__main__':
    # pyopenjtalkの動作確認
    try:
        test_phonemes = pyopenjtalk.g2p("こんにちは、世界")
        print(f"pyopenjtalk test successful: {test_phonemes}")
    except Exception as e:
        print(f"Error: pyopenjtalk is not working properly: {e}")
        print("Please install pyopenjtalk: pip install pyopenjtalk")
        sys.exit(1)
    
    main()
