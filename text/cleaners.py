"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import re
from unidecode import unidecode
import pyopenjtalk

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text

def expand_numbers(text):
  return normalize_numbers(text)

def lowercase(text):
  return text.lower()

def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)

def convert_to_ascii(text):
  return unidecode(text)

def basic_cleaners(text):
  """Basic pipeline that lowercases and collapses whitespace without transliteration."""
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def transliteration_cleaners(text):
  """Pipeline for non-English text that transliterates to ASCII."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def english_cleaners(text):
  """Pipeline for English text, including abbreviation expansion."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_abbreviations(text)
  text = expand_numbers(text)
  text = collapse_whitespace(text)
  return text

def japanese_cleaners(text):
  """Pipeline for Japanese text using pyopenjtalk."""
  # pyopenjtalkを使って日本語テキストを音素に変換
  try:
    # フルコンテキストラベルから音素を抽出
    phonemes = pyopenjtalk.g2p(text, kana=False)
    # スペースで区切られた音素列を処理
    phonemes = phonemes.replace(' ', '-')
    # 無音記号を追加
    phonemes = 'sil-' + phonemes + '-sil'
  except Exception as e:
    print(f"Error in japanese_cleaners: {e}")
    # フォールバック: 基本クリーナーを使用
    phonemes = basic_cleaners(text)
  
  return phonemes

def japanese_cleaners2(text):
  """Alternative Japanese cleaner with different processing."""
  try:
    # pyopenjtalkでカナ変換してから音素変換
    kana = pyopenjtalk.g2p(text, kana=True)
    phonemes = pyopenjtalk.g2p(kana, kana=False)
    phonemes = phonemes.replace(' ', '-')
    phonemes = 'sil-' + phonemes + '-sil'
  except Exception as e:
    print(f"Error in japanese_cleaners2: {e}")
    phonemes = basic_cleaners(text)
  
  return phonemes

def korean_cleaners(text):
  """Pipeline for Korean text."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def chinese_cleaners(text):
  """Pipeline for Chinese text."""
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text

def normalize_numbers(text):
  """Normalize numbers in text."""
  import inflect
  _inflect = inflect.engine()
  
  def _expand_number(m):
    num = int(m.group(0))
    if num < 1000000000:
      return _inflect.number_to_words(num)
    else:
      return str(num)
  
  text = re.sub(r'\d+', _expand_number, text)
  return text

# Map cleaner names to functions
CLEANERS = {
    'basic_cleaners': basic_cleaners,
    'transliteration_cleaners': transliteration_cleaners,
    'english_cleaners': english_cleaners,
    'japanese_cleaners': japanese_cleaners,
    'japanese_cleaners2': japanese_cleaners2,
    'korean_cleaners': korean_cleaners,
    'chinese_cleaners': chinese_cleaners,
}
