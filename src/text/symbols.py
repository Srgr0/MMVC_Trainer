"""Japanese text processing with phoneme symbols."""

# 日本語音素記号の定義
_pad = '_'
_punctuation = ',.!?-'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ'

# 記号のリスト
symbols = [_pad] + list(_punctuation) + list(_letters)

# 記号からIDへのマッピング
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, symbols=symbols):
    """テキストを音素IDのシーケンスに変換する."""
    sequence = []
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    
    for symbol in text:
        if symbol in symbol_to_id:
            sequence.append(symbol_to_id[symbol])
        else:
            # 未知の記号は無視
            continue
    
    return sequence


def sequence_to_text(sequence):
    """音素IDのシーケンスをテキストに変換する."""
    result = []
    for symbol_id in sequence:
        if symbol_id < len(symbols):
            result.append(symbols[symbol_id])
    return ''.join(result)


def get_symbols():
    """音素記号のリストを取得する."""
    return symbols


def symbol_to_id():
    """記号からIDへのマッピングを取得する."""
    return _symbol_to_id


def id_to_symbol():
    """IDから記号への マッピングを取得する."""
    return _id_to_symbol
