"""Text cleaning functions for Japanese."""

import re
import unicodedata
from typing import List

import pyopenjtalk


def japanese_cleaners(text: str) -> str:
    """日本語テキストのクリーニングと音素変換."""
    # 基本的な正規化
    text = unicodedata.normalize('NFKC', text)
    
    # 音素変換
    phonemes = pyopenjtalk.g2p(text, kana=False)
    
    # 無音記号を追加
    phonemes = "sil " + phonemes + " sil"
    
    # スペースをハイフンに変換
    phonemes = phonemes.replace(' ', '-')
    
    return phonemes


def basic_cleaners(text: str) -> str:
    """基本的なテキストクリーニング."""
    # 先頭・末尾の空白を除去
    text = text.strip()
    
    # 連続する句読点を正規化
    text = re.sub(r'[。．]+', '。', text)
    text = re.sub(r'[、，]+', '、', text)
    
    # 連続する空白を一つにまとめる
    text = re.sub(r'\s+', ' ', text)
    
    return text


def normalize_numbers(text: str) -> str:
    """数字を読み方に変換する."""
    # TODO: 数字の読み上げ変換を実装
    return text


def expand_abbreviations(text: str) -> str:
    """略語を展開する."""
    # TODO: 略語の展開を実装
    return text


def clean_text(text: str, cleaners: List[str]) -> str:
    """指定されたクリーナーでテキストをクリーニングする."""
    for cleaner_name in cleaners:
        cleaner = globals().get(cleaner_name)
        if cleaner:
            text = cleaner(text)
    return text
