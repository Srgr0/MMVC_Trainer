#!/usr/bin/env python3
"""Data preprocessing CLI for MMVC."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
from tqdm import tqdm

# プロジェクトルートをパスに追加
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.text.cleaners import japanese_cleaners
from src.utils.common import setup_logger
from src.utils.audio import load_wav, normalize_audio, trim_silence


def process_text_file(text_path: str) -> str:
    """テキストファイルを処理する."""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # テキストをクリーニングして音素に変換
        phonemes = japanese_cleaners(text)
        return phonemes
    except Exception as e:
        print(f"テキスト処理エラー {text_path}: {e}")
        return ""


def process_audio_file(audio_path: str, target_sr: int = 24000) -> bool:
    """音声ファイルを処理する."""
    try:
        # 音声を読み込み
        audio = load_wav(audio_path, sr=target_sr)
        
        # 無音部分をトリミング
        audio = trim_silence(audio)
        
        # 正規化
        audio = normalize_audio(audio)
        
        # 処理済み音声を保存
        processed_path = audio_path.replace('.wav', '_processed.wav')
        import soundfile as sf
        sf.write(processed_path, audio, target_sr)
        
        # 元のファイルを置き換え
        os.replace(processed_path, audio_path)
        
        return True
    except Exception as e:
        print(f"音声処理エラー {audio_path}: {e}")
        return False


def create_file_lists(data_dir: str, output_dir: str, train_ratio: float = 0.9) -> Tuple[List[str], List[str]]:
    """学習・検証用のファイルリストを作成する."""
    filepaths_and_text = []
    
    # データディレクトリを走査
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        
        if not os.path.isdir(speaker_path):
            continue
        
        print(f"話者 {speaker_dir} を処理中...")
        
        # 音声ファイルとテキストファイルのペアを検索
        audio_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
        
        for audio_file in tqdm(audio_files, desc=f"処理中 {speaker_dir}"):
            audio_path = os.path.join(speaker_path, audio_file)
            text_path = audio_path.replace('.wav', '.txt')
            
            # テキストファイルが存在するかチェック
            if not os.path.exists(text_path):
                print(f"警告: テキストファイルが見つかりません: {text_path}")
                continue
            
            # 音声を処理
            if not process_audio_file(audio_path):
                continue
            
            # テキストを処理
            phonemes = process_text_file(text_path)
            if not phonemes:
                continue
            
            # ファイルリストに追加
            filepaths_and_text.append(f"{audio_path}|{phonemes}|{speaker_dir}")
    
    # ランダムシャッフル
    import random
    random.shuffle(filepaths_and_text)
    
    # 学習・検証に分割
    split_idx = int(len(filepaths_and_text) * train_ratio)
    train_files = filepaths_and_text[:split_idx]
    val_files = filepaths_and_text[split_idx:]
    
    # ファイルリストを保存
    os.makedirs(output_dir, exist_ok=True)
    
    train_list_path = os.path.join(output_dir, 'train.txt')
    val_list_path = os.path.join(output_dir, 'val.txt')
    
    with open(train_list_path, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(train_files))
    
    with open(val_list_path, 'w', encoding='utf-8') as f:
        f.write('\\n'.join(val_files))
    
    return train_files, val_files


def create_speaker_json(data_dir: str, output_path: str):
    """話者情報のJSONファイルを作成する."""
    speakers = []
    
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        
        if os.path.isdir(speaker_path):
            speakers.append(speaker_dir)
    
    speakers.sort()
    
    speaker_info = {
        "speakers": speakers,
        "speaker_to_id": {speaker: i for i, speaker in enumerate(speakers)},
        "n_speakers": len(speakers)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_info, f, indent=2, ensure_ascii=False)
    
    return speaker_info


def create_config_file(base_config_path: str, output_path: str, speaker_info: dict, data_info: dict):
    """設定ファイルを作成する."""
    # ベース設定を読み込み
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 話者数を更新
    config['data']['n_speakers'] = speaker_info['n_speakers']
    config['model']['n_speakers'] = speaker_info['n_speakers']
    
    # データパスを更新
    config['data']['training_files'] = data_info['train_list']
    config['data']['validation_files'] = data_info['val_list']
    
    # 設定を保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config


def validate_dataset(file_list_path: str) -> dict:
    """データセットの妥当性をチェックする."""
    stats = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'total_duration': 0.0,
        'speakers': set()
    }
    
    with open(file_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc="データセット検証中"):
        parts = line.strip().split('|')
        if len(parts) != 3:
            continue
        
        audio_path, phonemes, speaker = parts
        stats['total_files'] += 1
        stats['speakers'].add(speaker)
        
        # 音声ファイルの存在チェック
        if not os.path.exists(audio_path):
            stats['invalid_files'] += 1
            print(f"警告: 音声ファイルが見つかりません: {audio_path}")
            continue
        
        # 音声の長さを取得
        try:
            audio = load_wav(audio_path)
            duration = len(audio) / 24000  # 24kHzと仮定
            stats['total_duration'] += duration
            stats['valid_files'] += 1
        except:
            stats['invalid_files'] += 1
            print(f"警告: 音声ファイルの読み込みに失敗: {audio_path}")
    
    stats['speakers'] = list(stats['speakers'])
    
    print(f"\\n=== データセット統計 ===")
    print(f"総ファイル数: {stats['total_files']}")
    print(f"有効ファイル数: {stats['valid_files']}")
    print(f"無効ファイル数: {stats['invalid_files']}")
    print(f"総再生時間: {stats['total_duration']:.2f}秒 ({stats['total_duration']/3600:.2f}時間)")
    print(f"話者数: {len(stats['speakers'])}")
    print(f"話者: {', '.join(stats['speakers'])}")
    
    return stats


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description='MMVC Data Preprocessing')
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                       help='データディレクトリのパス')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='出力ディレクトリのパス')
    parser.add_argument('-c', '--base_config', type=str, 
                       default='configs/base_config.json',
                       help='ベース設定ファイルのパス')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                       help='学習データの割合 (0.0-1.0)')
    parser.add_argument('--validate', action='store_true',
                       help='データセットの妥当性をチェック')
    
    args = parser.parse_args()
    
    # ロガーの設定
    logger = setup_logger("preprocess")
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("データ前処理を開始します...")
    
    if args.validate:
        # データセットの妥当性チェック
        train_list = os.path.join(args.output_dir, 'train.txt')
        val_list = os.path.join(args.output_dir, 'val.txt')
        
        if os.path.exists(train_list):
            logger.info("学習データセットを検証中...")
            validate_dataset(train_list)
        
        if os.path.exists(val_list):
            logger.info("検証データセットを検証中...")
            validate_dataset(val_list)
        
        return
    
    # データディレクトリの存在チェック
    if not os.path.exists(args.data_dir):
        logger.error(f"データディレクトリが見つかりません: {args.data_dir}")
        return
    
    # ファイルリストを作成
    logger.info("ファイルリストを作成中...")
    train_files, val_files = create_file_lists(args.data_dir, args.output_dir, args.train_ratio)
    
    logger.info(f"学習データ: {len(train_files)}ファイル")
    logger.info(f"検証データ: {len(val_files)}ファイル")
    
    # 話者情報を作成
    logger.info("話者情報を作成中...")
    speaker_info = create_speaker_json(args.data_dir, os.path.join(args.output_dir, 'speakers.json'))
    
    # 設定ファイルを作成
    logger.info("設定ファイルを作成中...")
    data_info = {
        'train_list': os.path.join(args.output_dir, 'train.txt'),
        'val_list': os.path.join(args.output_dir, 'val.txt')
    }
    
    config = create_config_file(
        args.base_config, 
        os.path.join(args.output_dir, 'train_config.json'),
        speaker_info,
        data_info
    )
    
    # データセットの妥当性チェック
    logger.info("データセットを検証中...")
    validate_dataset(data_info['train_list'])
    
    logger.info("前処理が完了しました！")
    logger.info(f"出力ディレクトリ: {args.output_dir}")
    logger.info("次のコマンドで学習を開始できます:")
    logger.info(f"mmvc-train -c {os.path.join(args.output_dir, 'train_config.json')} -m logs/experiment")


if __name__ == "__main__":
    main()
