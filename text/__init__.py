""" from https://github.com/keithito/tacotron """

import re
from text import cleaners
from text.symbols import symbols, symbol_to_id, id_to_symbol


# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = symbol_to_id
_id_to_symbol = id_to_symbol

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def text_to_sequence(text, cleaner_names):
  """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
  """
  sequence = []

  # Check for curly braces and use ARPAbet if they're found
  while len(text):
    m = _curly_re.match(text)
    if not m:
      sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
      break
    sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
    sequence += _arpabet_to_sequence(m.group(2))
    text = m.group(3)

  return sequence


def sequence_to_text(sequence):
  """Converts a sequence of IDs back to a string"""
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    if name in cleaners.CLEANERS:
      cleaner = cleaners.CLEANERS[name]
      if not cleaner:
        raise Exception('Unknown cleaner: %s' % name)
      text = cleaner(text)
    else:
      raise Exception('Unknown cleaner: %s' % name)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _should_keep_symbol(s):
  return s in _symbol_to_id and s != '_' and s != '~'


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def get_text(text, hps):
  """Convert text to phoneme sequence."""
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
    text_norm = intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm


# 音素変換用のユーティリティ関数
def japanese_text_to_phonemes(text):
  """Convert Japanese text to phonemes using pyopenjtalk."""
  import pyopenjtalk
  try:
    phonemes = pyopenjtalk.g2p(text, kana=False)
    # 区切り文字を統一
    phonemes = phonemes.replace(' ', '-')
    # 無音記号を追加
    phonemes = 'sil-' + phonemes + '-sil'
    return phonemes
  except Exception as e:
    print(f"Error converting text to phonemes: {e}")
    return 'sil'


def phonemes_to_sequence(phonemes):
  """Convert phoneme string to sequence of IDs."""
  phoneme_list = phonemes.split('-')
  sequence = []
  for phoneme in phoneme_list:
    if phoneme in _symbol_to_id:
      sequence.append(_symbol_to_id[phoneme])
    else:
      # Unknown phoneme - use UNK token
      sequence.append(_symbol_to_id.get('UNK', 0))
  return sequence


# Import necessary torch for tensor operations
try:
    import torch
except ImportError:
    print("Warning: torch not available, some functions may not work")
    torch = None
